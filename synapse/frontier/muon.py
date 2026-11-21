"""Orthogonalised-momentum optimiser (Muon-style) with an internal AdamW, plus a neural-graft benchmark.

Inspired by (citations verified 2026-11-17 and re-checked 2026-11-20, see
``docs/frontier/modules/muon_optimizer.md``):

* K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, J. Bernstein, "Muon: An optimizer for hidden
  layers in neural networks", blog post, 8 December 2024, https://kellerjordan.github.io/posts/muon/
* J. Liu, J. Su, X. Yao, Z. Jiang, G. Lai, Y. Du, ... Z. Yang (Moonshot AI), "Muon is Scalable for LLM
  Training", technical report, arXiv:2502.16982, 24 February 2025, https://arxiv.org/abs/2502.16982

For a matrix parameter ``W`` of shape ``A x B`` with gradient ``G``, momentum ``mu``, learning rate ``lr`` and
decoupled weight decay ``wd`` one step is::

    M_t = mu * M_{t-1} + (1 - mu) * G_t          momentum (EMA form of Jordan's reference code)
    U_t = (1 - mu) * G_t + mu * M_t               Nesterov input (or U_t = M_t without Nesterov)
    O_t = NS5(U_t)                                quintic Newton-Schulz, 5 steps, Frobenius-normalised input
    W_t = W_{t-1} - lr * (0.2 * sqrt(max(A, B)) * O_t + wd * W_{t-1})          (Liu et al., Eq. 4)

``NS5`` iterates ``X <- a X + b (X X^T) X + c (X X^T)^2 X`` with ``(a, b, c) = (3.4445, -4.7750, 2.0315)`` on the
wide orientation of the matrix (transposed when rows > cols). It does not return the exact polar factor
``U V^T``: it maps each singular value ``s`` of the Frobenius-normalised input through ``p^5`` with
``p(x) = a x + b x^3 + c x^5``. Iterating that scalar map (measured 2026-11-17 on a dense grid) gives outputs in
[0.6818, 1.1344] for every start ``s >= 0.01`` and in (0, 1.2024] for every ``s`` in (0, 1] (see the tests). The
EMA momentum differs from Liu et al.'s ``M_t = mu M_{t-1} + G_t`` only by the factor ``1 - mu``, which the
Frobenius normalisation removes. Everything that is not a 2-D weight matrix (biases, norm gains, embeddings,
other shapes) is updated by an internal AdamW whose arithmetic follows ``torch.optim.AdamW``. The
0.2 * sqrt(max(A, B)) scale lets Muon and the AdamW part share one learning rate.

The module is a synthetic computational abstraction: an optimiser and a small, honest benchmark on the
Sonapse neural-graft architecture. It does not reproduce any published training result.
"""
from __future__ import annotations

import copy
import math
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import harness
from .spec import GRAFT_OPTIMIZERS

__all__ = ['MODULE_ID', 'NS_COEFFICIENTS', 'zeropower_via_newtonschulz5', 'MuonWithAuxAdam', 'muon_param_groups',
           'graft_output_layers', 'graft_input_layers', 'make_graft_optimizer', 'run_experiment', 'EXPERIMENTS',
           'ARMS', 'DEFAULT_CONFIG', 'QUICK_CONFIG']

MODULE_ID = 'muon_optimizer'
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_NS_EPS = 1e-7
_HYPER_KEYS = ('lr', 'momentum', 'nesterov', 'ns_steps', 'weight_decay', 'betas', 'eps', 'update_scale')
_GROUP_KEYS = frozenset(('params', 'use_muon') + _HYPER_KEYS)
_MUON_STATE_KEYS = frozenset(('momentum_buffer',))
_ADAM_STATE_KEYS = frozenset(('step', 'exp_avg', 'exp_avg_sq'))


# ====================================================================== Newton-Schulz orthogonalisation
def _newton_schulz(matrix: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """Unvalidated core of :func:`zeropower_via_newtonschulz5` (callers check shape, dtype and finiteness)."""
    a, b, c = NS_COEFFICIENTS
    x = matrix if matrix.dtype in (torch.float32, torch.float64) else matrix.float()
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.mT
    norm = torch.linalg.matrix_norm(x)
    if not bool(torch.isfinite(norm)):          # the squares overflowed (huge finite entries): rescale first
        x = x / x.abs().amax()
        norm = torch.linalg.matrix_norm(x)
    x = x / (norm + eps)
    for _ in range(steps):
        gram = x @ x.mT
        poly = torch.addmm(gram, gram, gram, beta=b, alpha=c)       # b * A + c * A @ A
        x = torch.addmm(x, poly, x, beta=a)                         # a * X + B @ X
    if transpose:
        x = x.mT
    return x.to(matrix.dtype)


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5, eps: float = _NS_EPS) -> torch.Tensor:
    """Approximate the orthogonal polar factor ``U V^T`` of a 2-D matrix ``U S V^T`` (quintic Newton-Schulz).

    The input is Frobenius-normalised (so every singular value starts in ``(0, 1]``), the iteration runs on the
    wide orientation (transposed when rows > cols) and the result keeps the input's singular vectors with
    singular values pushed into a band around 1. The input is not modified. Computation is float32 (or float64
    for float64 inputs); the reference code uses bfloat16 on GPUs, which this CPU build does not. Entries so
    large that the Frobenius norm overflows are rescaled by the largest magnitude first (the reference code
    would return zeros there).
    """
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or matrix.numel() == 0:
        raise ValueError('matrix must be a non-empty 2-D tensor')
    if not matrix.is_floating_point():
        raise ValueError('matrix must have a floating-point dtype')
    if type(steps) is not int or not 0 <= steps <= 64:
        raise ValueError('steps must be an int in [0, 64]')
    if not (isinstance(eps, float) and math.isfinite(eps) and eps > 0.0):
        raise ValueError('eps must be a positive finite float')
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError('matrix must be finite')
    return _newton_schulz(matrix, steps, eps)


# ====================================================================== validation helpers
def _real(name: str, value: Any, *, low: float, high: float, low_open: bool = False, high_open: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a real number')
    v = float(value)
    if (not math.isfinite(v) or v < low or v > high or (low_open and v == low) or (high_open and v == high)):
        lb, hb = '(' if low_open else '[', ')' if high_open else ']'
        raise ValueError(f'{name} must be finite and in {lb}{low}, {high}{hb}')
    return v


def _check_hyper(h: Mapping[str, Any]) -> dict[str, Any]:
    out = {'lr': _real('lr', h['lr'], low=0.0, high=10.0),
           'momentum': _real('momentum', h['momentum'], low=0.0, high=1.0, high_open=True),
           'weight_decay': _real('weight_decay', h['weight_decay'], low=0.0, high=10.0),
           'eps': _real('eps', h['eps'], low=0.0, high=1.0, low_open=True),
           'update_scale': _real('update_scale', h['update_scale'], low=0.0, high=100.0, low_open=True)}
    if type(h['nesterov']) is not bool:
        raise ValueError('nesterov must be a bool')
    if type(h['ns_steps']) is not int or not 0 <= h['ns_steps'] <= 64:
        raise ValueError('ns_steps must be an int in [0, 64]')
    betas = h['betas']
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError('betas must be a pair')
    out['betas'] = tuple(_real(f'betas[{i}]', bv, low=0.0, high=1.0, high_open=True) for i, bv in enumerate(betas))
    out['nesterov'] = h['nesterov']
    out['ns_steps'] = h['ns_steps']
    return out


def _as_param_list(params: Any) -> list[torch.Tensor]:
    if isinstance(params, torch.Tensor):
        return [params]
    if isinstance(params, (set, frozenset)):
        raise ValueError('parameters must be given in a deterministic order (not a set)')
    try:
        out = list(params)
    except TypeError as exc:
        raise ValueError('parameters must be a tensor or an iterable of tensors') from exc
    if not all(isinstance(p, torch.Tensor) for p in out):
        raise ValueError('parameter groups must contain only tensors')
    return out


# ====================================================================== parameter routing
def muon_param_groups(module: nn.Module, *, aux_modules: Iterable[nn.Module] = (), **hyper: Any) -> list[dict[str, Any]]:
    """Split ``module.parameters()`` (in order) into a Muon group and an AdamW group.

    Muon gets every 2-D parameter that is not an ``nn.Embedding``/``nn.EmbeddingBag`` weight and not owned by one
    of ``aux_modules``; everything else (biases, norm gains, embeddings, non-2-D tensors, ``aux_modules``) goes to
    the AdamW group. Extra keyword arguments are copied into both groups. Empty groups are omitted."""
    if not isinstance(module, nn.Module):
        raise ValueError('module must be a torch.nn.Module')
    unknown = set(hyper) - set(_HYPER_KEYS)
    if unknown:
        raise ValueError(f'unknown optimiser options: {sorted(unknown)}')
    aux_list = list(aux_modules)
    if not all(isinstance(m, nn.Module) for m in aux_list):
        raise ValueError('aux_modules must be torch.nn.Module instances')
    embed = {id(p) for m in module.modules() if isinstance(m, (nn.Embedding, nn.EmbeddingBag))
             for p in m.parameters(recurse=False)}
    aux = {id(p) for m in aux_list for p in m.parameters()}
    muon, adam = [], []
    for p in module.parameters():
        (muon if p.ndim == 2 and id(p) not in embed and id(p) not in aux else adam).append(p)
    groups = []
    if muon:
        groups.append({'params': muon, 'use_muon': True, **hyper})
    if adam:
        groups.append({'params': adam, 'use_muon': False, **hyper})
    return groups


def _normalise_groups(params: Any) -> list[dict[str, Any]]:
    if isinstance(params, nn.Module):
        groups = muon_param_groups(params)
        if not groups:
            raise ValueError('optimizer got an empty parameter list')
        return groups
    if isinstance(params, torch.Tensor):
        raise ValueError('params must be an iterable of tensors or dicts, or an nn.Module (got a single tensor)')
    if isinstance(params, (set, frozenset)):
        raise ValueError('parameters must be given in a deterministic order (not a set)')
    try:
        items = list(params)
    except TypeError as exc:
        raise ValueError('params must be an nn.Module, an iterable of tensors or an iterable of dicts') from exc
    if not items:
        raise ValueError('optimizer got an empty parameter list')
    if all(isinstance(i, Mapping) for i in items):
        groups: list[dict[str, Any]] = []
        for item in items:
            g = dict(item)
            if 'params' not in g:
                raise ValueError('every parameter group needs a "params" entry')
            unknown = set(g) - _GROUP_KEYS
            if unknown:
                raise ValueError(f'unknown parameter-group keys: {sorted(unknown)}')
            ps = _as_param_list(g.pop('params'))
            if 'use_muon' in g:
                groups.append({'params': ps, **g})          # checked by add_param_group
            else:
                muon = [p for p in ps if p.ndim == 2]
                adam = [p for p in ps if p.ndim != 2]
                if muon:
                    groups.append({'params': muon, 'use_muon': True, **g})
                if adam or not muon:
                    groups.append({'params': adam, 'use_muon': False, **g})
        return groups
    if all(isinstance(i, torch.Tensor) for i in items):
        muon = [p for p in items if p.ndim == 2]
        adam = [p for p in items if p.ndim != 2]
        return ([{'params': muon, 'use_muon': True}] if muon else []) + \
               ([{'params': adam, 'use_muon': False}] if adam else [])
    raise ValueError('params must be an nn.Module, an iterable of tensors or an iterable of dicts')


def _all_finite(tensors: list[torch.Tensor]) -> bool:
    """True when every entry of every tensor is finite.

    Fast path: one fused 2-norm call (a finite 2-norm proves every entry finite). Only when a 2-norm is not
    finite, which also happens when the squares of large finite values overflow, the exact per-tensor check
    decides. (The fused max-abs norm would avoid the overflow but measured 10-100x slower on this CPU build.)"""
    if not tensors:
        return True
    try:
        norms = torch._foreach_norm(tensors)
        if bool(torch.isfinite(torch.stack([n.to(torch.float64) for n in norms])).all()):
            return True
    except (AttributeError, RuntimeError, NotImplementedError, TypeError):     # pragma: no cover - old torch
        pass
    return all(bool(torch.isfinite(t).all()) for t in tensors)


# ====================================================================== optimiser
class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for 2-D weight matrices plus an internal AdamW for everything else, in one optimiser.

    ``params`` may be an ``nn.Module`` (routed by :func:`muon_param_groups`), an iterable of tensors (2-D tensors
    go to Muon, the rest to AdamW) or an iterable of group dicts. A group dict may set ``use_muon`` explicitly
    (``True`` needs ``ndim >= 2``; tensors with more than two dimensions are flattened to ``(shape[0], -1)``)
    or leave it out to be split by ``ndim``. Groups may override any hyper-parameter. Groups added later with
    :meth:`add_param_group` must set ``use_muon`` or hold only 2-D (Muon) or only non-2-D (AdamW) tensors.

    Defaults follow Liu et al. (2025): one learning rate for both parts, momentum 0.95 with Nesterov, five
    Newton-Schulz steps, update scale ``0.2 * sqrt(max(A, B))`` and decoupled weight decay. The AdamW part uses
    ``torch.optim.AdamW``'s defaults (betas 0.9/0.999, eps 1e-8). Each step is computed in full before it is
    applied: a non-finite gradient, or an update that would make a parameter or state value non-finite
    (numeric overflow), raises ``ValueError`` and leaves every parameter and all state unchanged.
    """

    def __init__(self, params: Any, lr: float = 1e-3, *, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0, betas: Sequence[float] = (0.9, 0.999),
                 eps: float = 1e-8, update_scale: float = 0.2) -> None:
        defaults = _check_hyper({'lr': lr, 'momentum': momentum, 'nesterov': nesterov, 'ns_steps': ns_steps,
                                 'weight_decay': weight_decay, 'betas': betas, 'eps': eps,
                                 'update_scale': update_scale})
        groups = _normalise_groups(params)
        super().__init__(groups, defaults)          # every group passes through add_param_group (validated)

    def add_param_group(self, param_group: Mapping[str, Any]) -> None:
        """Validate a group (keys, routing, hyper-parameters) before registering it."""
        if not isinstance(param_group, Mapping) or 'params' not in param_group:
            raise ValueError('a parameter group must be a mapping with a "params" entry')
        g = dict(param_group)
        unknown = set(g) - _GROUP_KEYS
        if unknown:
            raise ValueError(f'unknown parameter-group keys: {sorted(unknown)}')
        ps = _as_param_list(g['params'])
        if 'use_muon' not in g:
            kinds = {p.ndim == 2 for p in ps}
            if len(kinds) > 1:
                raise ValueError('a group without use_muon must hold only 2-D or only non-2-D tensors')
            g['use_muon'] = kinds == {True}
        if type(g['use_muon']) is not bool:
            raise ValueError('use_muon must be a bool')
        if g['use_muon'] and any(p.ndim < 2 for p in ps):
            raise ValueError('use_muon=True needs parameters with at least 2 dimensions')
        _check_hyper({**self.defaults, **{k: v for k, v in g.items() if k in _HYPER_KEYS}})
        g['params'] = ps
        super().add_param_group(g)
        self.param_groups[-1].update(_check_hyper(self.param_groups[-1]))

    # ------------------------------------------------------------------ introspection
    def muon_parameters(self) -> list[torch.Tensor]:
        return [p for g in self.param_groups if g['use_muon'] for p in g['params']]

    def adam_parameters(self) -> list[torch.Tensor]:
        return [p for g in self.param_groups if not g['use_muon'] for p in g['params']]

    def state_numbers(self) -> int:
        """Stored optimiser numbers (momentum / moment buffers; step counters excluded), for budget reports."""
        return _moment_numbers(self)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grads = []
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise ValueError('MuonWithAuxAdam does not support sparse gradients')
                    grads.append(p.grad)
        if not _all_finite(grads):
            raise ValueError('non-finite gradient: optimiser step refused (parameters and state unchanged)')
        plan: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []
        for group in self.param_groups:
            (self._plan_muon if group['use_muon'] else self._plan_adam)(group, plan)
        produced = [t for _, new_p, new_state in plan
                    for t in (new_p, *(v for v in new_state.values() if isinstance(v, torch.Tensor)))]
        if not _all_finite(produced):
            raise ValueError('non-finite update (numeric overflow): optimiser step refused '
                             '(parameters and state unchanged)')
        for p, new_p, new_state in plan:                    # commit
            self.state[p].update(new_state)
            p.copy_(new_p)
        return loss

    def _plan_muon(self, group: Mapping[str, Any], plan: list) -> None:
        lr, mu, wd = group['lr'], group['momentum'], group['weight_decay']
        for p in group['params']:
            grad = p.grad
            if grad is None:
                continue
            buf = self.state.get(p, {}).get('momentum_buffer')
            if buf is None:
                buf = torch.zeros_like(p, memory_format=torch.preserve_format)
            new_buf = torch.lerp(buf, grad, 1.0 - mu)
            update = torch.lerp(grad, new_buf, mu) if group['nesterov'] else new_buf
            mat = update if update.ndim == 2 else update.reshape(update.shape[0], -1)
            ortho = _newton_schulz(mat, group['ns_steps'], _NS_EPS)
            scale = group['update_scale'] * math.sqrt(max(mat.shape[0], mat.shape[1]))
            new_p = p.mul(1.0 - lr * wd) if wd != 0.0 else p.clone()
            new_p.add_(ortho.reshape(p.shape), alpha=-lr * scale)
            plan.append((p, new_p, {'momentum_buffer': new_buf}))

    def _plan_adam(self, group: Mapping[str, Any], plan: list) -> None:
        # Same arithmetic, in the same order, as torch.optim.AdamW's single-tensor path (out of place).
        lr, wd, eps = group['lr'], group['weight_decay'], group['eps']
        beta1, beta2 = group['betas']
        for p in group['params']:
            grad = p.grad
            if grad is None:
                continue
            state = self.state.get(p, {})
            step = int(state.get('step', 0)) + 1
            exp_avg = state.get('exp_avg')
            exp_avg_sq = state.get('exp_avg_sq')
            if exp_avg is None:
                exp_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
            if exp_avg_sq is None:
                exp_avg_sq = torch.zeros_like(p, memory_format=torch.preserve_format)
            new_p = p.mul(1.0 - lr * wd) if wd != 0.0 else p.clone()
            new_m = torch.lerp(exp_avg, grad, 1.0 - beta1)
            new_v = exp_avg_sq.mul(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = lr / bias_correction1
            denom = (new_v.sqrt() / bias_correction2 ** 0.5).add_(eps)
            new_p.addcdiv_(new_m, denom, value=-step_size)
            plan.append((p, new_p, {'step': step, 'exp_avg': new_m, 'exp_avg_sq': new_v}))

    # ------------------------------------------------------------------ persistence
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Validate fully (group layout, routing, hyper-parameters, state shapes and finiteness), then load a
        private deep copy, so the result never aliases tensors of the given dict."""
        if not isinstance(state_dict, Mapping) or 'state' not in state_dict or 'param_groups' not in state_dict:
            raise ValueError('optimizer state must be a mapping with "state" and "param_groups"')
        saved_groups = state_dict['param_groups']
        if not isinstance(saved_groups, (list, tuple)) or len(saved_groups) != len(self.param_groups):
            raise ValueError('optimizer state has a different number of parameter groups')
        index_to_param: dict[int, tuple[torch.Tensor, bool]] = {}
        for mine, saved in zip(self.param_groups, saved_groups):
            if not isinstance(saved, Mapping) or set(saved) - _GROUP_KEYS or 'params' not in saved:
                raise ValueError('optimizer state has malformed parameter groups')
            if saved.get('use_muon') is not mine['use_muon']:
                raise ValueError('optimizer state routes parameters differently (use_muon mismatch)')
            if not isinstance(saved['params'], (list, tuple)) or len(saved['params']) != len(mine['params']):
                raise ValueError('optimizer state group sizes do not match')
            _check_hyper({**self.defaults, **{k: v for k, v in saved.items() if k in _HYPER_KEYS}})
            for idx, p in zip(saved['params'], mine['params']):
                if type(idx) is not int or idx in index_to_param:
                    raise ValueError('optimizer state has invalid parameter indices')
                index_to_param[idx] = (p, mine['use_muon'])
        state = state_dict['state']
        if not isinstance(state, Mapping):
            raise ValueError('optimizer state must contain a mapping "state"')
        for idx, entry in state.items():
            if idx not in index_to_param or not isinstance(entry, Mapping):
                raise ValueError('optimizer state refers to unknown parameters')
            p, use_muon = index_to_param[idx]
            allowed = _MUON_STATE_KEYS if use_muon else _ADAM_STATE_KEYS
            if set(entry) - allowed or (not use_muon and entry and set(entry) != _ADAM_STATE_KEYS):
                raise ValueError('optimizer state entries do not match the parameter routing')
            for key, value in entry.items():
                if key == 'step':
                    step = int(value.item()) if isinstance(value, torch.Tensor) and value.numel() == 1 else value
                    if type(step) is not int or not 0 <= step < 2**62:
                        raise ValueError('optimizer step count must be a non-negative int')
                    continue
                if (not isinstance(value, torch.Tensor) or value.shape != p.shape or not value.is_floating_point()
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError(f'optimizer state {key!r} must be a finite tensor shaped like its parameter')
                if key == 'exp_avg_sq' and bool((value < 0).any()):
                    raise ValueError('exp_avg_sq must be non-negative')
        private = copy.deepcopy({'state': dict(state), 'param_groups': [dict(g) for g in saved_groups]})
        for entry in private['state'].values():
            if 'step' in entry and isinstance(entry['step'], torch.Tensor):
                entry['step'] = int(entry['step'].item())
        super().load_state_dict(private)
        for g in self.param_groups:
            g.update(_check_hyper(g))


def _moment_numbers(optimizer: torch.optim.Optimizer) -> int:
    """Numbers held in an optimiser's per-parameter state tensors, step counters excluded."""
    return int(sum(v.numel() for s in optimizer.state.values() for k, v in s.items()
                   if k != 'step' and isinstance(v, torch.Tensor)))


# ====================================================================== graft helper
def _graft_paths(network: nn.Module) -> list[list[nn.Linear]] | None:
    paths = []
    for name in ('actor_encoder', 'value', 'dynamics'):
        seq = getattr(network, name, None)
        linears = [m for m in seq if isinstance(m, nn.Linear)] if isinstance(seq, nn.Sequential) else []
        if not linears:
            return None
        paths.append(linears)
    return paths


def graft_output_layers(network: nn.Module) -> tuple[nn.Module, ...]:
    """The output (head) layers of a neural-graft ``_ResidualNetwork``: the actor head and the last linear layer
    of the value and dynamics paths. Returns ``()`` for modules without that structure."""
    paths = _graft_paths(network)
    actor = getattr(network, 'actor_head', None)
    if paths is None or not isinstance(actor, nn.Linear):
        return ()
    return (actor, paths[1][-1], paths[2][-1])


def graft_input_layers(network: nn.Module) -> tuple[nn.Module, ...]:
    """The input layers of a neural-graft ``_ResidualNetwork``: the first linear layer of the actor encoder and of
    the value and dynamics paths (the layers that read the graft's raw inputs). ``()`` without that structure."""
    paths = _graft_paths(network)
    if paths is None or not isinstance(getattr(network, 'actor_head', None), nn.Linear):
        return ()
    return tuple(p[0] for p in paths)


def make_graft_optimizer(params_or_module: Any, kind: str = 'adam', lr: float = 3e-4, *,
                         heads_on_adam: bool = True, inputs_on_adam: bool | None = None,
                         **muon_options: Any) -> torch.optim.Optimizer:
    """Optimiser for the live neural-graft fit, selected by ``FrontierSpecV1.graft_optimizer``.

    ``'adam'`` returns exactly ``torch.optim.Adam(params, lr=lr)`` (the graft's current optimiser; no other
    option is accepted). ``'muon'`` returns :class:`MuonWithAuxAdam`. For a module with the graft's structure the
    three output layers stay on AdamW (``heads_on_adam``; both Jordan et al. and Liu et al. keep output heads on
    AdamW) and so do the three input layers (``inputs_on_adam``; Jordan et al.: input and output layers belong on
    AdamW even though they are 2-D), which leaves Muon the hidden-to-hidden matrices. ``inputs_on_adam=None`` (the
    default) means ``True`` for a graft-structured module and ``False`` otherwise; ``inputs_on_adam=False`` puts
    the input layers on Muon too (the benchmark's routing ablation). Other modules are routed by
    :func:`muon_param_groups`, parameter iterables by ``ndim``; ``inputs_on_adam=True`` needs a graft module."""
    if kind not in GRAFT_OPTIMIZERS:
        raise ValueError(f'kind must be one of {GRAFT_OPTIMIZERS}')
    _real('lr', lr, low=0.0, high=10.0, low_open=True)
    if type(heads_on_adam) is not bool or (inputs_on_adam is not None and type(inputs_on_adam) is not bool):
        raise ValueError('heads_on_adam must be a bool and inputs_on_adam a bool or None')
    if kind == 'adam':
        if muon_options or not heads_on_adam or inputs_on_adam is True:
            raise ValueError("options other than lr apply only to kind='muon'")
        params = params_or_module.parameters() if isinstance(params_or_module, nn.Module) else params_or_module
        return torch.optim.Adam(params, lr=lr)
    unknown = set(muon_options) - (set(_HYPER_KEYS) - {'lr'})
    if unknown:
        raise ValueError(f'unknown muon options: {sorted(unknown)}')
    if isinstance(params_or_module, nn.Module):
        inputs = graft_input_layers(params_or_module)
        inputs_aux = bool(inputs) if inputs_on_adam is None else inputs_on_adam
        if inputs_aux and not inputs:
            raise ValueError('inputs_on_adam=True needs a module with the neural-graft structure')
        aux = (graft_output_layers(params_or_module) if heads_on_adam else ()) + (inputs if inputs_aux else ())
        return MuonWithAuxAdam(muon_param_groups(params_or_module, aux_modules=aux), lr=lr, **muon_options)
    if inputs_on_adam:
        raise ValueError('inputs_on_adam=True needs a module with the neural-graft structure')
    return MuonWithAuxAdam(params_or_module, lr=lr, **muon_options)


# ====================================================================== benchmark
ARMS = ('adam', 'muon', 'muon_with_inputs')
DEFAULT_CONFIG: dict[str, Any] = {
    'hidden': 64, 'steps': 300, 'batch_size': 128, 'n_train': 2048, 'n_val': 512, 'n_test': 1024,
    'lr_grid': (3e-4, 1e-3, 3e-3, 1e-2, 3e-2), 'clip_grad_norm': 1.0, 'teacher_gain': 2.0, 'momentum': 0.95,
    'nesterov': True, 'ns_steps': 5, 'weight_decay': 0.0, 'heads_on_adam': True, 'routing_ablation': True,
}
QUICK_CONFIG: dict[str, Any] = {'hidden': 32, 'steps': 40, 'batch_size': 128, 'n_train': 512, 'n_val': 256,
                                'n_test': 256, 'lr_grid': (1e-3, 1e-2)}
_DATA_OFFSET, _TEACHER_OFFSET, _STUDENT_OFFSET, _BATCH_OFFSET = 52_001, 52_103, 52_207, 52_309
_N_STATE, _N_OBS, _N_ACT = 256, 155, 5
_HEADS = ('policy', 'value', 'dynamics')


@contextmanager
def _one_torch_thread() -> Iterator[None]:
    """Run with one intra-op thread (stable float results, no oversubscription); restore the caller's count."""
    before = torch.get_num_threads()
    try:
        if before != 1:
            torch.set_num_threads(1)
        yield
    finally:
        if torch.get_num_threads() != before:
            torch.set_num_threads(before)


def _seeds(seeds: Any) -> tuple[int, ...]:
    out = tuple(seeds)
    if (not out or any(type(s) is not int or not 0 <= s <= 2**31 - 1 for s in out) or len(set(out)) != len(out)):
        raise ValueError('seeds must be a non-empty sequence of unique ints in [0, 2**31 - 1]')
    return out


def _int(cfg: Mapping[str, Any], name: str, low: int, high: int) -> int:
    v = cfg[name]
    if type(v) is not int or not low <= v <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return v


def _config(quick: bool, overrides: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(overrides) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f'unknown muon_optimizer options: {sorted(unknown)}')
    cfg = {**DEFAULT_CONFIG, **(QUICK_CONFIG if quick else {}), **dict(overrides)}
    _int(cfg, 'hidden', 32, 1024)               # NeuralResidualGraft's own hidden_dim range
    _int(cfg, 'steps', 1, 100_000)
    _int(cfg, 'batch_size', 1, 4096)
    _int(cfg, 'n_train', 8, 1_000_000)
    _int(cfg, 'n_val', 8, 1_000_000)
    _int(cfg, 'n_test', 8, 1_000_000)
    grid = cfg['lr_grid']
    if not isinstance(grid, (list, tuple)) or not grid or len(grid) > 32:
        raise ValueError('lr_grid must be a non-empty sequence of at most 32 values')
    cfg['lr_grid'] = tuple(_real('lr_grid entry', v, low=0.0, high=1.0, low_open=True) for v in grid)
    if len(set(cfg['lr_grid'])) != len(cfg['lr_grid']):
        raise ValueError('lr_grid entries must be unique')
    cfg['clip_grad_norm'] = _real('clip_grad_norm', cfg['clip_grad_norm'], low=0.0, high=1e6)
    cfg['teacher_gain'] = _real('teacher_gain', cfg['teacher_gain'], low=0.0, high=10.0, low_open=True)
    cfg['weight_decay'] = _real('weight_decay', cfg['weight_decay'], low=0.0, high=10.0)
    _check_hyper({'lr': 1e-3, 'momentum': cfg['momentum'], 'nesterov': cfg['nesterov'], 'ns_steps': cfg['ns_steps'],
                  'weight_decay': cfg['weight_decay'], 'betas': (0.9, 0.999), 'eps': 1e-8, 'update_scale': 0.2})
    cfg['momentum'] = float(cfg['momentum'])
    for name in ('heads_on_adam', 'routing_ablation'):
        if type(cfg[name]) is not bool:
            raise ValueError(f'{name} must be a bool')
    return cfg


def _graft_network(hidden: int, generator: torch.Generator, *, teacher_gain: float | None = None) -> nn.Module:
    """A ``_ResidualNetwork`` initialised from ``generator`` only (the global torch RNG is left untouched).

    Students get PyTorch's default ``nn.Linear`` distribution (uniform +-1/sqrt(fan_in)) and the architecture's
    zero actor head; teachers get every linear layer, the actor head included, at ``teacher_gain`` times that."""
    from ..neural_graft import _ResidualNetwork
    with torch.random.fork_rng(devices=[]):         # construction draws from the global RNG: restore it after
        net = _ResidualNetwork(hidden)
    gain = 1.0 if teacher_gain is None else float(teacher_gain)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Linear):
                bound = gain / math.sqrt(m.in_features)
                m.weight.copy_((torch.rand(m.weight.shape, generator=generator) * 2.0 - 1.0) * bound)
                m.bias.copy_((torch.rand(m.bias.shape, generator=generator) * 2.0 - 1.0) * bound)
            elif isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
                m.bias.zero_()
        if teacher_gain is None:
            net.actor_head.weight.zero_()
            net.actor_head.bias.zero_()
    return net


def _sample_inputs(rng: np.random.Generator, n: int) -> dict[str, torch.Tensor]:
    obs = rng.standard_normal((n, _N_OBS))
    features = rng.standard_normal((n, 64))
    baseline = rng.uniform(-1.0, 1.0, (n, _N_ACT))
    prior = rng.uniform(-1.0, 1.0, (n, 32))
    action = rng.uniform(-1.0, 1.0, (n, _N_ACT))
    state = np.clip(np.concatenate([obs, features, baseline, prior], axis=1), -5.0, 5.0)
    return {'state': torch.from_numpy(state.astype(np.float32)),
            'obs': torch.from_numpy(state[:, :_N_OBS].astype(np.float32)),
            'dyn': torch.from_numpy(np.concatenate([state[:, :_N_OBS], action], axis=1).astype(np.float32))}


def _outputs(net: nn.Module, x: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {'policy': net.policy(x['state']), 'value': net.value(x['obs']), 'dynamics': net.dynamics(x['dyn'])}


def _task(seed: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Inputs shaped like the graft's (state 256, observation 155, observation+action 160) and standardised
    targets from a fixed, never-trained random teacher of the same architecture."""
    rng = np.random.default_rng(seed + _DATA_OFFSET)
    splits = {name: _sample_inputs(rng, cfg[f'n_{name}']) for name in ('train', 'val', 'test')}
    teacher = _graft_network(cfg['hidden'], torch.Generator().manual_seed(seed + _TEACHER_OFFSET),
                             teacher_gain=cfg['teacher_gain'])
    with torch.no_grad():
        raw = {name: _outputs(teacher, x) for name, x in splits.items()}
    mean = {h: raw['train'][h].mean(0, keepdim=True) for h in _HEADS}
    std = {h: raw['train'][h].std(0, keepdim=True).clamp_min(1e-8) for h in _HEADS}
    targets = {name: {h: (raw[name][h] - mean[h]) / std[h] for h in _HEADS} for name in splits}
    batches = np.random.default_rng(seed + _BATCH_OFFSET).integers(0, cfg['n_train'],
                                                                   size=(cfg['steps'], cfg['batch_size']))
    return {'inputs': splits, 'targets': targets, 'batches': torch.from_numpy(batches),
            'teacher_raw_std': {h: float(raw['train'][h].std()) for h in _HEADS}}


def _head_mse(net: nn.Module, x: Mapping[str, torch.Tensor], y: Mapping[str, torch.Tensor]) -> dict[str, float]:
    with torch.no_grad():
        out = _outputs(net, x)
        return {h: float(F.mse_loss(out[h], y[h])) for h in _HEADS}


def _optimizer(arm: str, net: nn.Module, lr: float, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    if arm == 'adam':
        if cfg['weight_decay'] > 0.0:       # the Muon arms decay their weights: give the baseline the same decay
            return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
        return make_graft_optimizer(net, 'adam', lr)
    hyper = dict(momentum=cfg['momentum'], nesterov=cfg['nesterov'], ns_steps=cfg['ns_steps'],
                 weight_decay=cfg['weight_decay'])
    if arm == 'muon':                       # the live opt-in's routing (input layers on AdamW)
        return make_graft_optimizer(net, 'muon', lr, heads_on_adam=cfg['heads_on_adam'], **hyper)
    return make_graft_optimizer(net, 'muon', lr, heads_on_adam=cfg['heads_on_adam'], inputs_on_adam=False, **hyper)


def _train(arm: str, lr: float, seed: int, task: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    net = _graft_network(cfg['hidden'], torch.Generator().manual_seed(seed + _STUDENT_OFFSET))
    opt = _optimizer(arm, net, lr, cfg)
    x, y = task['inputs']['train'], task['targets']['train']
    params = list(net.parameters())
    clip = cfg['clip_grad_norm'] if cfg['clip_grad_norm'] > 0 else float('inf')
    diverged, last = False, None
    for t in range(cfg['steps']):
        idx = task['batches'][t]
        out = _outputs(net, {k: v[idx] for k, v in x.items()})
        loss = sum(F.mse_loss(out[h], y[h][idx]) for h in _HEADS)
        if not bool(torch.isfinite(loss)):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(params, clip)
        if not bool(torch.isfinite(norm)):
            diverged = True
            break
        try:
            opt.step()
        except ValueError:                  # MuonWithAuxAdam refuses non-finite updates
            diverged = True
            break
        last = float(loss.detach())
    val = _head_mse(net, task['inputs']['val'], task['targets']['val'])
    test = _head_mse(net, task['inputs']['test'], task['targets']['test'])
    bad = diverged or not all(math.isfinite(v) for v in (*val.values(), *test.values()))
    return {'seed': seed, 'lr': lr, 'diverged': bool(bad), 'final_train_loss': last,
            'val_mse': float('inf') if bad else float(np.mean(list(val.values()))),
            'test_mse': float('inf') if bad else float(np.mean(list(test.values()))),
            'test_mse_by_head': test, 'optimizer_state_numbers': _moment_numbers(opt)}


def _routing(arms: Sequence[str], cfg: Mapping[str, Any]) -> dict[str, Any]:
    net = _graft_network(cfg['hidden'], torch.Generator().manual_seed(0))
    names = {id(p): n for n, p in net.named_parameters()}
    out: dict[str, Any] = {}
    for arm in arms:
        if arm == 'adam':
            continue
        opt = _optimizer(arm, net, 1e-3, cfg)
        muon, adam = opt.muon_parameters(), opt.adam_parameters()
        out[arm] = {'muon': [names[id(p)] for p in muon], 'adamw': [names[id(p)] for p in adam],
                    'muon_numbers': int(sum(p.numel() for p in muon)),
                    'adamw_numbers': int(sum(p.numel() for p in adam)),
                    'muon_shapes': [list(p.shape) for p in muon]}
    out['parameters'] = int(sum(p.numel() for p in net.parameters()))
    out['linear_macs_per_sample'] = int(sum(m.in_features * m.out_features for m in net.modules()
                                            if isinstance(m, nn.Linear)))
    return out


def _ns_flops(shapes: Sequence[Sequence[int]], ns_steps: int) -> int:
    """Dense-matmul FLOPs of the Newton-Schulz iterations per optimiser step (m <= n after orientation)."""
    return int(ns_steps * sum(4 * min(s) ** 2 * max(s) + 2 * min(s) ** 3 for s in shapes))


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Muon (candidate) vs Adam (baseline) on the neural-graft architecture, held-out MSE (lower is better).

    For every seed a fixed random teacher of the same architecture defines standardised regression targets for
    the policy, value and dynamics heads. Every optimiser trains the same student initialisation on the same
    minibatch sequence for the same number of steps (graft-style loss sum and gradient clipping). Each arm's
    learning rate is chosen from ``lr_grid`` by mean validation MSE over seeds; the verdict is the paired 95 % t
    interval of Muon minus Adam test MSE at the selected learning rates. The ``muon`` arm uses the live opt-in's
    routing (input and output layers on AdamW, as Jordan et al. recommend; Muon on the hidden-to-hidden
    matrices). With ``routing_ablation`` a third arm, ``muon_with_inputs`` (the input layers on Muon too), is
    reported against Adam as a secondary comparison that does not set the verdict."""
    seeds = _seeds(seeds)
    if type(quick) is not bool:
        raise ValueError('quick must be a bool')
    cfg = _config(quick, overrides)
    arms_run = ARMS if cfg['routing_ablation'] else ARMS[:2]
    t0 = time.perf_counter()
    walls: dict[str, float] = {arm: 0.0 for arm in arms_run}
    runs: dict[str, dict[float, list[dict[str, Any]]]] = {arm: {} for arm in arms_run}
    with _one_torch_thread():
        routing = _routing(arms_run, cfg)
        teacher_std = {}
        for seed in seeds:
            task = _task(seed, cfg)
            teacher_std[str(seed)] = task['teacher_raw_std']
            for arm in arms_run:
                for lr in cfg['lr_grid']:
                    t1 = time.perf_counter()
                    runs[arm].setdefault(lr, []).append(_train(arm, lr, seed, task, cfg))
                    walls[arm] += time.perf_counter() - t1
    arms: dict[str, Any] = {}
    for arm in arms_run:
        grid = []
        for lr in cfg['lr_grid']:
            vals = [r['val_mse'] for r in runs[arm][lr]]
            grid.append({'lr': lr, 'mean_val_mse': float(np.mean(vals)) if all(map(math.isfinite, vals)) else None,
                         'diverged_seeds': [r['seed'] for r in runs[arm][lr] if r['diverged']]})
        finite = [g for g in grid if g['mean_val_mse'] is not None]
        best = min(finite, key=lambda g: g['mean_val_mse'])['lr'] if finite else cfg['lr_grid'][0]
        per_seed = runs[arm][best]
        edge = len(cfg['lr_grid']) > 1 and best in (min(cfg['lr_grid']), max(cfg['lr_grid']))
        arms[arm] = {'selected_lr': best, 'selected_at_grid_edge': bool(edge), 'lr_grid': grid, 'per_seed': per_seed,
                     'diverged_seeds_at_selected_lr': [r['seed'] for r in per_seed if r['diverged']],
                     'summary': {'test_mse': harness._mean(r['test_mse'] for r in per_seed),
                                 'val_mse': harness._mean(r['val_mse'] for r in per_seed),
                                 'test_mse_by_head': {h: harness._mean(r['test_mse_by_head'][h] for r in per_seed)
                                                      for h in _HEADS}},
                     'all_runs': {str(lr): [{k: r[k] for k in ('seed', 'val_mse', 'test_mse', 'diverged')}
                                            for r in runs[arm][lr]] for lr in cfg['lr_grid']}}

    def test_mse(arm: str) -> list[float]:
        return [r['test_mse'] for r in arms[arm]['per_seed']]
    comparisons: dict[str, Any] = {'muon_vs_adam': {'role': 'verdict', **harness.compare(
        test_mse('muon'), test_mse('adam'), higher_is_better=False)}}
    if 'muon_with_inputs' in arms:
        comparisons['muon_with_inputs_vs_adam'] = {'role': 'routing_ablation (does not set the verdict)',
                                                   **harness.compare(test_mse('muon_with_inputs'), test_mse('adam'),
                                                                     higher_is_better=False)}
    vs = comparisons['muon_vs_adam']
    verdict, basis = vs['verdict'], 'muon_vs_adam_paired_t95_on_heldout_test_mse_at_validation_selected_lr'
    bad = {arm: arms[arm]['diverged_seeds_at_selected_lr'] for arm in ('adam', 'muon')}
    if bad['muon'] or bad['adam']:          # paired stats drop non-finite pairs: never let that hide a failure
        verdict = 'worse' if bad['muon'] and not bad['adam'] else 'no_detectable_difference'
        basis = 'training_diverged_at_selected_lr'
    receipt = harness.base_receipt(MODULE_ID, 'optimizer_comparison', {
        'quick': quick, 'seeds': list(seeds), **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        'arms': list(arms_run),
        'metric': 'heldout_test_mse_mean_over_heads (standardised targets; lower is better)',
        'selection': 'per arm: lr in lr_grid with the lowest mean validation MSE over seeds',
        'baseline': ('adam: torch.optim.Adam(params, lr) == make_graft_optimizer(kind="adam")'
                     if cfg['weight_decay'] == 0.0 else
                     'adam: torch.optim.AdamW(params, lr, weight_decay) (same decoupled weight decay as the Muon arms)'),
        'candidate': ('muon: MuonWithAuxAdam via make_graft_optimizer(kind="muon", heads_on_adam=heads_on_adam) '
                      '(input layers on AdamW, the live opt-in routing)'),
        'ablation': ('muon_with_inputs: make_graft_optimizer(kind="muon", heads_on_adam=heads_on_adam, '
                     'inputs_on_adam=False) (input layers on Muon too)' if cfg['routing_ablation'] else None),
        'architecture': 'synapse.neural_graft._ResidualNetwork (policy, value and dynamics heads)'})
    receipt['parameter_routing'] = routing
    receipt['arms'] = arms
    receipt['comparisons'] = comparisons
    receipt['verdict'] = verdict
    receipt['verdict_basis'] = basis
    receipt['budget'] = {
        'parameters': routing['parameters'], 'training_steps_per_run': cfg['steps'],
        'runs_per_arm': len(seeds) * len(cfg['lr_grid']),
        'per_arm': {arm: {'muon_parameters': routing[arm]['muon_numbers'] if arm in routing else 0,
                          'adamw_or_adam_parameters': (routing[arm]['adamw_numbers'] if arm in routing
                                                       else routing['parameters']),
                          'optimizer_state_numbers': arms[arm]['per_seed'][0]['optimizer_state_numbers']}
                    for arm in arms_run}}
    receipt['estimated_cost'] = {
        'forward_backward_flops_per_step': int(6 * routing['linear_macs_per_sample'] * cfg['batch_size']),
        'newton_schulz_flops_per_step': {arm: _ns_flops(routing[arm]['muon_shapes'], cfg['ns_steps'])
                                         for arm in arms_run if arm in routing},
        'note': 'analytic estimates (dense matmul FLOPs), not measurements'}
    receipt['task'] = {'teacher_raw_output_std': teacher_std,
                       'targets': 'per-output standardised with training-split mean and std'}
    receipt['notes'] = ('Synthetic teacher-student regression on the graft architecture; inspired by Muon (Jordan '
                        'et al., 2024) with the update scale and weight decay of Liu et al. (2025); not a '
                        'reproduction. Equal steps, shared initialisation and minibatches; the candidate keeps the '
                        f'output heads on AdamW (heads_on_adam={cfg["heads_on_adam"]}) and the input layers on '
                        'AdamW (Jordan et al.), Muon on the hidden-to-hidden matrices; the muon_with_inputs ablation '
                        '(when present) puts the input layers on Muon too and is reported, not used for the '
                        'verdict. Quick mode is a pipeline check with very few steps; its verdict is not evidence. Three seeds give a wide t interval. Results may not transfer '
                        'to the live graft, whose targets come from executed transitions.')
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t0,
                                       **{f'{arm}_runs': walls[arm] for arm in arms_run}}
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment}
