"""Dynamic sparse training with RigL-style prune-and-regrow for ``nn.Linear`` layers, plus an MLP benchmark.

Inspired by (citation verified 2026-11-17, see ``docs/frontier/modules/rigl_sparse.md``):

* U. Evci, T. Gale, J. Menick, P. S. Castro, E. Elsen, "Rigging the Lottery: Making All Tickets Winners",
  Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119:2943-2952,
  July 2020, https://proceedings.mlr.press/v119/evci20a.html (reference code: github.com/google-research/rigl)

The sibling project ``sonapse-evo`` (``src/sonapse_evo/evaluation/rigl.py``, ``AuthenticRigL``) applies the same
drop/regrow idea to a recurrent edge vector. This module follows the paper more closely: seeded per-layer masks
with a uniform or Erdos-Renyi-Kernel (ERK) allocation, zero-initialised regrowth and no optimiser step on
connectivity-update iterations (Algorithm 1 of the paper).

For a managed layer ``l`` with ``N_l`` weights of which ``a_l`` are active, training iterations ``t = 1, 2, ...``::

    update iteration:   t % delta_t == 0  and  t < t_end
    f(t) = alpha / 2 * (1 + cos(pi * t / t_end))                 cosine-decayed update fraction
    k_l  = floor(f(t) * a_l)
    drop = the k_l active weights with the smallest |w|                          (ties: lower flat index)
    grow = the k_l weights with the largest |dL/dw| among all weights not kept   (ties: lower flat index)
    mask = (active - drop) | grow                                                (a_l never changes)

Newly activated weights start at exactly zero (so the network output is unchanged); a weight dropped and regrown
in the same update keeps its value, as in the reference code's default. On ordinary iterations the gradients are
masked before ``optimizer.step()`` and the weights are masked after it, so inactive weights stay exactly zero.
Optimiser state tensors shaped like a managed weight (momentum, Adam moments) are zeroed at inactive and newly
activated positions on every update. Biases stay dense, as in the paper.

This is a synthetic computational abstraction: masks over dense tensors (no sparse kernels, so no measured
speed-up) and a small procedural benchmark. It does not reproduce any published result.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import math
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import harness

__all__ = ['MODULE_ID', 'DISTRIBUTIONS', 'STATE_SCHEMA', 'cosine_drop_fraction', 'layer_densities',
           'layer_active_counts', 'RigLScheduler', 'run_experiment', 'EXPERIMENTS', 'DEFAULT_CONFIG', 'QUICK_CONFIG']

MODULE_ID = 'rigl_sparse'
DISTRIBUTIONS = ('uniform', 'erk')
STATE_SCHEMA = 'RigLSchedulerStateV1'
_MAX_HISTORY = 4096
_MASK_OFFSET, _DATA_OFFSET, _TEACHER_OFFSET, _MODEL_OFFSET, _BATCH_OFFSET = 61_001, 61_103, 61_207, 61_301, 61_409
_GUARD = 1e-9          # protects floor() against floating-point round-down of exact products


# ====================================================================== validation helpers
def _real(name: str, value: Any, *, low: float, high: float, low_open: bool = False, high_open: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a real number')
    v = float(value)
    if not math.isfinite(v) or v < low or v > high or (low_open and v == low) or (high_open and v == high):
        lb, hb = '(' if low_open else '[', ')' if high_open else ']'
        raise ValueError(f'{name} must be finite and in {lb}{low}, {high}{hb}')
    return v


def _int(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return value


# ====================================================================== schedule and allocation
def cosine_drop_fraction(t: int, alpha: float, t_end: int) -> float:
    """RigL's ``f_decay(t; alpha, T_end) = alpha / 2 * (1 + cos(pi * t / T_end))``; 0 from ``t_end`` on."""
    _int('t', t, 0, 2**62)
    a = _real('alpha', alpha, low=0.0, high=1.0)
    _int('t_end', t_end, 1, 2**62)
    if t >= t_end:
        return 0.0
    return a / 2.0 * (1.0 + math.cos(math.pi * t / t_end))


def _shapes(shapes: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    out = []
    for s in shapes:
        s = tuple(s)
        if len(s) != 2 or any(type(v) is not int or v < 1 for v in s):
            raise ValueError('layer shapes must be (out_features, in_features) pairs of positive ints')
        out.append(s)
    if not out:
        raise ValueError('at least one layer shape is needed')
    return out


def layer_densities(shapes: Iterable[Sequence[int]], sparsity: float, distribution: str = 'erk') -> list[float]:
    """Fraction of active weights per layer for a target overall ``sparsity``.

    ``'uniform'`` gives every layer density ``1 - sparsity``. ``'erk'`` (Erdos-Renyi-Kernel; for linear layers the
    Erdos-Renyi rule) makes density proportional to ``(n_in + n_out) / (n_in * n_out)`` with one scale ``eps``
    chosen so the total active count matches the uniform budget; a layer whose density would exceed 1 is made
    dense and ``eps`` is recomputed for the rest (as in the reference implementation)."""
    shp = _shapes(shapes)
    s = _real('sparsity', sparsity, low=0.0, high=1.0, high_open=True)
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f'distribution must be one of {DISTRIBUTIONS}')
    if distribution == 'uniform':
        return [1.0 - s] * len(shp)
    sizes = [o * i for o, i in shp]
    raw = [(o + i) / (o * i) for o, i in shp]
    dense: set[int] = set()
    eps = 0.0
    while True:
        rhs = divisor = 0.0
        for j, n in enumerate(sizes):
            zeros = math.floor(s * n + _GUARD)
            if j in dense:
                rhs -= zeros
            else:
                rhs += n - zeros
                divisor += raw[j] * n
        if divisor <= 0.0:                      # every layer had to be dense
            return [1.0] * len(shp)
        eps = rhs / divisor
        free = [j for j in range(len(shp)) if j not in dense]
        top = max(raw[j] for j in free)
        if eps * top <= 1.0:
            break
        dense.update(j for j in free if raw[j] == top)
    return [1.0 if j in dense else min(1.0, max(0.0, eps * raw[j])) for j in range(len(shp))]


def layer_active_counts(shapes: Iterable[Sequence[int]], sparsity: float, distribution: str = 'erk') -> list[int]:
    """Active weights per layer: ``N - floor((1 - density) * N)``, at least 1 and at most ``N``."""
    shp = _shapes(shapes)
    counts = []
    for (o, i), d in zip(shp, layer_densities(shp, sparsity, distribution)):
        n = o * i
        counts.append(n if d >= 1.0 else min(n, max(1, n - math.floor((1.0 - d) * n + _GUARD))))
    return counts


# ====================================================================== scheduler
class RigLScheduler:
    """Masks, schedule and prune/regrow updates for the ``nn.Linear`` layers of a model.

    Usage, once per training iteration after ``loss.backward()``::

        scheduler.step(optimizer)       # returns False on connectivity-update iterations (no optimiser step)

    ``layers`` defaults to every ``nn.Linear`` of ``model`` (in ``model.modules()`` order). The initial masks are
    drawn from ``numpy.random.default_rng(seed + offset)`` and applied to the weights at construction.
    ``dynamic=False`` keeps the initial masks for the whole run (static sparse training with identical masks).
    At most ``history`` update records are kept.
    """

    def __init__(self, model: nn.Module, sparsity: float, delta_t: int = 100, alpha: float = 0.3, t_end: int = 1000,
                 distribution: str = 'erk', *, layers: Sequence[nn.Linear] | None = None, seed: int = 0,
                 dynamic: bool = True, history: int = 256) -> None:
        if not isinstance(model, nn.Module):
            raise ValueError('model must be a torch.nn.Module')
        sparsity = _real('sparsity', sparsity, low=0.0, high=1.0, high_open=True)
        _int('delta_t', delta_t, 1, 10**9)
        alpha = _real('alpha', alpha, low=0.0, high=1.0)
        _int('t_end', t_end, 1, 10**12)
        if distribution not in DISTRIBUTIONS:
            raise ValueError(f'distribution must be one of {DISTRIBUTIONS}')
        _int('seed', seed, 0, 2**31 - 1)
        if type(dynamic) is not bool:
            raise ValueError('dynamic must be a bool')
        _int('history', history, 1, _MAX_HISTORY)
        modules = list(model.modules())
        if layers is None:
            chosen, explicit = [m for m in modules if isinstance(m, nn.Linear)], False
        else:
            if isinstance(layers, nn.Module):
                raise ValueError('layers must be a sequence of nn.Linear modules')
            chosen, explicit = list(layers), True
            known = {id(m) for m in modules}
            if not all(isinstance(m, nn.Linear) and id(m) in known for m in chosen):
                raise ValueError('layers must be nn.Linear modules of model')
        seen: set[int] = set()
        managed: list[nn.Linear] = []
        for m in chosen:
            if id(m.weight) in seen:
                if explicit:
                    raise ValueError('layers must not repeat a weight')
                continue                               # a module reused in the graph is managed once
            seen.add(id(m.weight))
            managed.append(m)
        if not managed:
            raise ValueError('model has no nn.Linear layer to sparsify')
        if not all(m.weight.is_floating_point() for m in managed):
            raise ValueError('managed weights must be floating point')
        shapes = [tuple(int(v) for v in m.weight.shape) for m in managed]
        self.sparsity, self.delta_t, self.alpha, self.t_end = sparsity, delta_t, alpha, t_end
        self.distribution, self.seed, self.dynamic = distribution, seed, dynamic
        self.layers: list[nn.Linear] = managed
        self.shapes: list[tuple[int, int]] = shapes
        self.densities: list[float] = layer_densities(shapes, sparsity, distribution)
        self.target_counts: list[int] = layer_active_counts(shapes, sparsity, distribution)
        rng = np.random.default_rng(seed + _MASK_OFFSET)
        masks = []
        for m, count in zip(managed, self.target_counts):
            flat = np.zeros(m.weight.numel(), dtype=bool)
            flat[rng.permutation(flat.size)[:count]] = True
            masks.append(torch.from_numpy(flat.reshape(m.weight.shape)).to(m.weight.device))
        self._masks: list[torch.Tensor] = masks
        self._sparse = [c < m.weight.numel() for m, c in zip(managed, self.target_counts)]
        self.t = 0
        self.n_updates = 0
        self._history: collections.deque[dict[str, Any]] = collections.deque(maxlen=history)
        self.apply_masks()

    # ------------------------------------------------------------------ introspection
    @property
    def masks(self) -> list[torch.Tensor]:
        """Copies of the boolean masks (``True`` = active), one per managed layer."""
        return [m.clone() for m in self._masks]

    @property
    def history(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self._history))

    def active_counts(self) -> list[int]:
        return [int(m.sum()) for m in self._masks]

    def is_update_step(self, t: int) -> bool:
        """Whether training iteration ``t`` (1-based) is a connectivity update."""
        _int('t', t, 0, 2**62)
        return bool(self.dynamic and any(self._sparse) and t >= 1 and t % self.delta_t == 0 and t < self.t_end)

    def drop_fraction(self, t: int) -> float:
        return cosine_drop_fraction(t, self.alpha, self.t_end)

    def parameter_budget(self) -> dict[str, Any]:
        """Active managed weights, managed totals and dense biases of the managed layers."""
        active = self.active_counts()
        total = [m.weight.numel() for m in self.layers]
        biases = int(sum(m.bias.numel() for m in self.layers if m.bias is not None))
        return {'active_weights': int(sum(active)), 'managed_weights': int(sum(total)), 'biases': biases,
                'density': float(sum(active) / sum(total)), 'per_layer_active': active,
                'per_layer_total': total, 'per_layer_density': [a / n for a, n in zip(active, total)]}

    def summary(self) -> dict[str, Any]:
        return {'sparsity': self.sparsity, 'distribution': self.distribution, 'delta_t': self.delta_t,
                'alpha': self.alpha, 't_end': self.t_end, 'dynamic': self.dynamic, 'seed': self.seed, 't': self.t,
                'n_updates': self.n_updates, 'shapes': [list(s) for s in self.shapes],
                'target_counts': list(self.target_counts), **self.parameter_budget(),
                'last_update': copy.deepcopy(self._history[-1]) if self._history else None}

    # ------------------------------------------------------------------ masking
    @torch.no_grad()
    def apply_masks(self) -> None:
        """Set every inactive managed weight to exactly zero."""
        for layer, mask in zip(self.layers, self._masks):
            layer.weight.masked_fill_(~mask, 0.0)

    @torch.no_grad()
    def mask_gradients(self) -> None:
        """Zero the gradients of inactive weights (call after ``backward``, before ``optimizer.step``)."""
        for layer, mask in zip(self.layers, self._masks):
            grad = layer.weight.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise ValueError('sparse gradients are not supported')
            grad.masked_fill_(~mask, 0.0)

    # ------------------------------------------------------------------ prune and regrow
    @torch.no_grad()
    def update_connections(self, fraction: float, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
        """Drop ``floor(fraction * a_l)`` smallest-magnitude active weights per layer and grow as many by dense
        gradient magnitude. Needs the dense gradients of the current batch (before :meth:`mask_gradients`).
        Everything is validated before any mask, weight or optimiser state changes."""
        f = _real('fraction', fraction, low=0.0, high=1.0)
        if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError('optimizer must be a torch.optim.Optimizer or None')
        grads = []
        for layer in self.layers:
            grad = layer.weight.grad
            if grad is None or grad.is_sparse or grad.shape != layer.weight.shape:
                raise ValueError('a RigL update needs the dense gradient of every managed layer')
            if not bool(torch.isfinite(grad).all()):
                raise ValueError('non-finite gradient: connectivity update refused (nothing changed)')
            grads.append(grad)
        plans: list[tuple[int, torch.Tensor] | None] = []
        for layer, mask, grad, sparse in zip(self.layers, self._masks, grads, self._sparse):
            flat = mask.reshape(-1)
            n_active = int(flat.sum())
            k = int(math.floor(f * n_active + _GUARD))
            if k == 0 or not sparse:
                plans.append(None)
                continue
            active_idx = torch.nonzero(flat).reshape(-1)                   # ascending flat indices
            weight = layer.weight.detach().reshape(-1)
            drop = active_idx[torch.sort(weight[active_idx].abs(), stable=True).indices[:k]]
            keep = flat.clone()
            keep[drop] = False
            candidates = torch.nonzero(~keep).reshape(-1)                   # inactive + just dropped
            score = grad.detach().reshape(-1)[candidates].abs()
            grow = candidates[torch.sort(score, descending=True, stable=True).indices[:k]]
            new = keep.clone()
            new[grow] = True
            plans.append((k, new.reshape(mask.shape)))
        records = []
        for i, (layer, plan) in enumerate(zip(self.layers, plans)):
            if plan is None:
                records.append({'k': 0, 'activated': 0, 'regrown_same': 0})
                continue
            k, new_mask = plan
            activated = new_mask & ~self._masks[i]
            layer.weight.masked_fill_(activated, 0.0)                    # newly activated weights start at zero
            layer.weight.masked_fill_(~new_mask, 0.0)                    # dropped weights are removed
            self._masks[i] = new_mask
            if optimizer is not None:
                _reset_optimizer_state(optimizer, layer.weight, new_mask & ~activated)
            n_new = int(activated.sum())
            records.append({'k': k, 'activated': n_new, 'regrown_same': k - n_new})
        self.n_updates += 1
        record = {'t': self.t + 1, 'fraction': f, 'layers': records}
        self._history.append(record)
        return copy.deepcopy(record)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Process one training iteration after ``loss.backward()``.

        On an update iteration the connectivity is updated and the optimiser step is skipped (RigL Algorithm 1);
        otherwise gradients are masked and ``optimizer.step()`` runs. Masks are re-applied to the weights either
        way. Returns ``True`` when the optimiser stepped."""
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError('optimizer must be a torch.optim.Optimizer')
        t = self.t + 1
        if self.is_update_step(t):
            self.update_connections(self.drop_fraction(t), optimizer)
            stepped = False
        else:
            self.mask_gradients()
            optimizer.step()
            stepped = True
        self.apply_masks()
        self.t = t
        return stepped

    # ------------------------------------------------------------------ persistence
    def _config(self) -> dict[str, Any]:
        return {'sparsity': self.sparsity, 'delta_t': self.delta_t, 'alpha': self.alpha, 't_end': self.t_end,
                'distribution': self.distribution, 'seed': self.seed, 'dynamic': self.dynamic,
                'shapes': [list(s) for s in self.shapes]}

    def state_dict(self) -> dict[str, Any]:
        return {'schema': STATE_SCHEMA, 'config': self._config(), 't': self.t, 'n_updates': self.n_updates,
                'masks': [m.detach().cpu().clone() for m in self._masks], 'history': self.history}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate everything (configuration, mask shapes and per-layer budgets), then load and re-apply masks."""
        if not isinstance(state, Mapping) or state.get('schema') != STATE_SCHEMA:
            raise ValueError('not a RigL scheduler state')
        if state.get('config') != self._config():
            raise ValueError('RigL state was saved with a different configuration or model')
        t, n_updates = state.get('t'), state.get('n_updates')
        if type(t) is not int or not 0 <= t <= 2**62 or type(n_updates) is not int or not 0 <= n_updates <= 2**62:
            raise ValueError('RigL state has invalid step counters')
        masks = state.get('masks')
        if not isinstance(masks, (list, tuple)) or len(masks) != len(self.layers):
            raise ValueError('RigL state has the wrong number of masks')
        for mask, layer, count in zip(masks, self.layers, self.target_counts):
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != layer.weight.shape
                    or int(mask.sum()) != count):
                raise ValueError('RigL masks must be boolean, shaped like their weights and keep the layer budget')
        history = state.get('history')
        if (not isinstance(history, (list, tuple)) or len(history) > _MAX_HISTORY
                or not all(isinstance(h, Mapping) for h in history)):
            raise ValueError('RigL state has an invalid history')
        self._masks = [m.detach().clone().to(layer.weight.device) for m, layer in zip(masks, self.layers)]
        self.t, self.n_updates = t, n_updates
        self._history = collections.deque(copy.deepcopy(list(history)), maxlen=self._history.maxlen)
        self.apply_masks()


def _reset_optimizer_state(optimizer: torch.optim.Optimizer, weight: torch.Tensor, keep: torch.Tensor) -> None:
    """Zero every per-element optimiser state tensor of ``weight`` outside ``keep``."""
    state = optimizer.state.get(weight)
    if not state:
        return
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.shape == weight.shape and value.is_floating_point():
            value.masked_fill_(~keep, 0.0)


# ====================================================================== benchmark
DEFAULT_CONFIG: dict[str, Any] = {
    'n_inputs': 64, 'n_informative': 8, 'n_classes': 4, 'teacher_hidden': 16, 'teacher_gain': 2.0,
    'hidden': 128, 'n_train': 4096, 'n_val': 1024, 'n_test': 2048, 'steps': 2000, 'batch_size': 64,
    'lr': 0.05, 'momentum': 0.9, 'weight_decay': 1e-4, 'sparsity': 0.9, 'distribution': 'erk',
    'delta_t': 50, 'alpha': 0.3, 't_end_fraction': 0.75,
}
QUICK_CONFIG: dict[str, Any] = {'hidden': 64, 'n_train': 1024, 'n_val': 256, 'n_test': 512, 'steps': 300,
                                'delta_t': 25}
ARMS = ('dense', 'static_sparse', 'rigl')


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
    if not out or any(type(s) is not int or not 0 <= s <= 2**31 - 1 for s in out) or len(set(out)) != len(out):
        raise ValueError('seeds must be a non-empty sequence of unique ints in [0, 2**31 - 1]')
    return out


def _config(quick: bool, overrides: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(overrides) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f'unknown rigl_sparse options: {sorted(unknown)}')
    cfg = {**DEFAULT_CONFIG, **(QUICK_CONFIG if quick else {}), **dict(overrides)}
    _int('n_inputs', cfg['n_inputs'], 2, 4096)
    _int('n_informative', cfg['n_informative'], 1, cfg['n_inputs'])
    _int('n_classes', cfg['n_classes'], 2, 64)
    _int('teacher_hidden', cfg['teacher_hidden'], 1, 1024)
    cfg['teacher_gain'] = _real('teacher_gain', cfg['teacher_gain'], low=0.0, high=100.0, low_open=True)
    _int('hidden', cfg['hidden'], 2, 4096)
    for name in ('n_train', 'n_val', 'n_test'):
        _int(name, cfg[name], 8, 1_000_000)
    _int('steps', cfg['steps'], 1, 1_000_000)
    _int('batch_size', cfg['batch_size'], 1, 65_536)
    cfg['lr'] = _real('lr', cfg['lr'], low=0.0, high=10.0, low_open=True)
    cfg['momentum'] = _real('momentum', cfg['momentum'], low=0.0, high=1.0, high_open=True)
    cfg['weight_decay'] = _real('weight_decay', cfg['weight_decay'], low=0.0, high=1.0)
    cfg['sparsity'] = _real('sparsity', cfg['sparsity'], low=0.0, high=1.0, high_open=True)
    if cfg['distribution'] not in DISTRIBUTIONS:
        raise ValueError(f'distribution must be one of {DISTRIBUTIONS}')
    _int('delta_t', cfg['delta_t'], 1, 10**9)
    cfg['alpha'] = _real('alpha', cfg['alpha'], low=0.0, high=1.0)
    cfg['t_end_fraction'] = _real('t_end_fraction', cfg['t_end_fraction'], low=0.0, high=1.0, low_open=True)
    return cfg


def _task(seed: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Procedural ``sparse_teacher`` classification: Gaussian inputs, of which only ``n_informative`` (a seeded
    random subset) drive the label; the label is the argmax of a fixed random tanh teacher on those inputs, with
    class logits centred on the training split. Every other input is a distractor."""
    rng = np.random.default_rng(seed + _DATA_OFFSET)
    d, m, c, h = cfg['n_inputs'], cfg['n_informative'], cfg['n_classes'], cfg['teacher_hidden']
    informative = np.sort(rng.choice(d, size=m, replace=False))
    n = {k: cfg[f'n_{k}'] for k in ('train', 'val', 'test')}
    x = {k: rng.standard_normal((n[k], d)) for k in ('train', 'val', 'test')}
    trng = np.random.default_rng(seed + _TEACHER_OFFSET)
    a = trng.standard_normal((m, h)) * (cfg['teacher_gain'] / math.sqrt(m))
    b = trng.standard_normal(h) * 0.5
    w = trng.standard_normal((h, c)) / math.sqrt(h)
    logits = {k: np.tanh(v[:, informative] @ a + b) @ w for k, v in x.items()}
    centre = logits['train'].mean(axis=0, keepdims=True)
    y = {k: np.argmax(v - centre, axis=1) for k, v in logits.items()}
    batches = np.random.default_rng(seed + _BATCH_OFFSET).integers(0, n['train'], size=(cfg['steps'], cfg['batch_size']))
    freq = np.bincount(y['train'], minlength=c) / n['train']
    return {'x': {k: torch.from_numpy(v.astype(np.float32)) for k, v in x.items()},
            'y': {k: torch.from_numpy(v.astype(np.int64)) for k, v in y.items()},
            'batches': torch.from_numpy(batches), 'informative': informative.tolist(),
            'class_frequencies_train': freq.tolist(),
            'majority_rate_test': float(np.bincount(y['test'], minlength=c).max() / n['test'])}


def _mlp(cfg: Mapping[str, Any], generator: torch.Generator) -> nn.Sequential:
    """``n_inputs -> hidden -> hidden -> n_classes`` ReLU MLP with PyTorch's default ``nn.Linear`` distribution,
    drawn from ``generator`` only (the global torch RNG is left untouched)."""
    with torch.random.fork_rng(devices=[]):     # nn.Linear construction draws from the global RNG: restore it
        net = nn.Sequential(nn.Linear(cfg['n_inputs'], cfg['hidden']), nn.ReLU(),
                            nn.Linear(cfg['hidden'], cfg['hidden']), nn.ReLU(),
                            nn.Linear(cfg['hidden'], cfg['n_classes']))
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Linear):
                bound = 1.0 / math.sqrt(m.in_features)
                m.weight.copy_((torch.rand(m.weight.shape, generator=generator) * 2.0 - 1.0) * bound)
                m.bias.copy_((torch.rand(m.bias.shape, generator=generator) * 2.0 - 1.0) * bound)
    return net


def _accuracy(net: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((net(x).argmax(dim=1) == y).double().mean())


def _informative_share(first: nn.Linear, mask: torch.Tensor | None, informative: Sequence[int]) -> float:
    """Share of the first layer's active connections that read an informative input (chance: m / d)."""
    m = torch.ones_like(first.weight, dtype=torch.bool) if mask is None else mask
    return float(m[:, list(informative)].sum() / m.sum())


def _train_arm(arm: str, seed: int, task: Mapping[str, Any], cfg: Mapping[str, Any], t_end: int) -> dict[str, Any]:
    net = _mlp(cfg, torch.Generator().manual_seed(seed + _MODEL_OFFSET))
    opt = torch.optim.SGD(net.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    sched = None
    if arm != 'dense':
        sched = RigLScheduler(net, cfg['sparsity'], cfg['delta_t'], cfg['alpha'], t_end, cfg['distribution'],
                              seed=seed, dynamic=(arm == 'rigl'), history=_MAX_HISTORY)
    first = net[0]
    initial_masks = None if sched is None else sched.masks
    x, y = task['x']['train'], task['y']['train']
    optimizer_steps, last_loss, diverged = 0, None, False
    for t in range(cfg['steps']):
        idx = task['batches'][t]
        loss = F.cross_entropy(net(x[idx]), y[idx])
        if not bool(torch.isfinite(loss)):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if sched is None:
            opt.step()
            optimizer_steps += 1
        else:
            optimizer_steps += int(sched.step(opt))
        last_loss = float(loss.detach())
    linears = [m for m in net if isinstance(m, nn.Linear)]
    out: dict[str, Any] = {
        'seed': seed, 'diverged': diverged, 'final_train_loss': last_loss, 'optimizer_steps': optimizer_steps,
        'train_accuracy': _accuracy(net, x, y),
        'val_accuracy': _accuracy(net, task['x']['val'], task['y']['val']),
        'test_accuracy': _accuracy(net, task['x']['test'], task['y']['test'])}
    if sched is None:
        active = [m.weight.numel() for m in linears]
        out['informative_share_initial'] = out['informative_share_final'] = _informative_share(
            first, None, task['informative'])
    else:
        masks = sched.masks
        inactive_zero = all(not bool(layer.weight.detach()[~mask].any()) for layer, mask in zip(sched.layers, masks))
        active = sched.active_counts()
        hist = sched.history
        out.update({
            'informative_share_initial': _informative_share(first, initial_masks[0], task['informative']),
            'informative_share_final': _informative_share(first, masks[0], task['informative']),
            'initial_mask_digest': _mask_digest(initial_masks), 'final_mask_digest': _mask_digest(masks),
            'connectivity_updates': sched.n_updates,
            'connections_activated': int(sum(r['activated'] for h in hist for r in h['layers'])),
            'update_trace': [{'t': h['t'], 'fraction': h['fraction'], 'k': [r['k'] for r in h['layers']],
                              'activated': [r['activated'] for r in h['layers']]} for h in hist],
            'budget_constant': bool(active == sched.target_counts), 'inactive_weights_zero': inactive_zero})
    out['active_weights_per_layer'] = [int(a) for a in active]
    out['parameters'] = int(sum(active) + sum(m.bias.numel() for m in linears))
    return out


def _mask_digest(masks: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for m in masks:
        h.update(np.packbits(m.detach().cpu().numpy().reshape(-1)).tobytes())
    return h.hexdigest()


def _flops(shapes: Sequence[tuple[int, int]], active: Sequence[int], batch: int, delta_t: int, t_end: int,
           steps: int) -> dict[str, Any]:
    """Analytic training FLOPs per iteration, assuming ideal sparse kernels (this module uses dense matmuls)."""
    dense = sum(o * i for o, i in shapes)
    act = int(sum(active))
    updates = sum(1 for t in range(1, steps + 1) if t % delta_t == 0 and t < t_end)
    # ordinary iteration: forward 2BA + input-gradient 2BA + weight-gradient 2BA (A = active weights);
    # update iteration: forward and input-gradient sparse, weight-gradient dense (2BN), no optimiser step.
    return {'dense_per_iteration': 6 * batch * dense, 'static_sparse_per_iteration': 6 * batch * act,
            'rigl_mean_per_iteration': int(round(((steps - updates) * 6 * batch * act
                                                   + updates * (4 * batch * act + 2 * batch * dense)) / steps)),
            'rigl_update_iterations': updates,
            'note': 'analytic estimates (multiply-adds x2) for ideal sparse kernels; not measured'}


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """RigL (candidate) vs static random sparse (same initial masks and budget) vs dense, held-out accuracy.

    For every seed a procedural ``sparse_teacher`` task is generated; the three arms share the network
    initialisation, the minibatch sequence, the optimiser (SGD with momentum) and the number of training
    iterations. The verdict is the paired 95 % t interval of RigL minus static-sparse test accuracy."""
    if type(quick) is not bool:
        raise ValueError('quick must be a bool')
    seeds = _seeds(seeds)
    cfg = _config(quick, overrides)
    t_end = max(1, int(math.floor(cfg['t_end_fraction'] * cfg['steps'] + _GUARD)))
    t0 = time.perf_counter()
    walls = {arm: 0.0 for arm in ARMS}
    per_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    tasks = {}
    with _one_torch_thread():
        for seed in seeds:
            task = _task(seed, cfg)
            tasks[str(seed)] = {'informative_inputs': task['informative'],
                                'class_frequencies_train': task['class_frequencies_train'],
                                'majority_rate_test': task['majority_rate_test']}
            for arm in ARMS:
                t1 = time.perf_counter()
                per_arm[arm].append(_train_arm(arm, seed, task, cfg, t_end))
                walls[arm] += time.perf_counter() - t1
        probe = RigLScheduler(_mlp(cfg, torch.Generator().manual_seed(0)), cfg['sparsity'], cfg['delta_t'],
                              cfg['alpha'], t_end, cfg['distribution'], seed=0)
    acc = {arm: [r['test_accuracy'] for r in per_arm[arm]] for arm in ARMS}
    diverged = {arm: [r['seed'] for r in per_arm[arm] if r['diverged']] for arm in ARMS}
    comparisons = {'rigl_vs_static_sparse': harness.compare(acc['rigl'], acc['static_sparse']),
                   'rigl_vs_dense': harness.compare(acc['rigl'], acc['dense']),
                   'static_sparse_vs_dense': harness.compare(acc['static_sparse'], acc['dense'])}
    verdict = comparisons['rigl_vs_static_sparse']['verdict']
    basis = 'rigl_vs_static_sparse_paired_t95_on_heldout_test_accuracy'
    if diverged['rigl'] or diverged['static_sparse']:
        verdict = 'worse' if diverged['rigl'] and not diverged['static_sparse'] else 'no_detectable_difference'
        basis = 'training_diverged'
    receipt = harness.base_receipt(MODULE_ID, 'sparse_training_comparison', {
        'quick': quick, 'seeds': list(seeds), **cfg, 't_end': t_end,
        'task': 'sparse_teacher_classification (seeded informative-input subset; others are distractors)',
        'model': f"MLP {cfg['n_inputs']}-{cfg['hidden']}-{cfg['hidden']}-{cfg['n_classes']} (ReLU)",
        'optimizer': 'torch.optim.SGD(lr, momentum, weight_decay), identical in every arm',
        'metric': 'heldout_test_accuracy (higher is better)', 'arms': list(ARMS),
        'candidate': 'rigl', 'reference': 'static_sparse (same initial masks, same budget)'})
    receipt['arms'] = {arm: {'per_seed': per_arm[arm], 'diverged_seeds': diverged[arm], 'summary': {
        'test_accuracy': harness._mean(acc[arm]),
        'val_accuracy': harness._mean(r['val_accuracy'] for r in per_arm[arm]),
        'train_accuracy': harness._mean(r['train_accuracy'] for r in per_arm[arm]),
        'informative_share_final': harness._mean(r['informative_share_final'] for r in per_arm[arm]),
        'parameters': per_arm[arm][0]['parameters']}} for arm in ARMS}
    receipt['comparisons'] = comparisons
    receipt['verdict'] = verdict
    receipt['verdict_basis'] = basis
    budget = probe.parameter_budget()
    receipt['budget'] = {'sparse_parameters': budget['active_weights'] + budget['biases'],
                         'dense_parameters': budget['managed_weights'] + budget['biases'],
                         'active_weights': budget['active_weights'], 'managed_weights': budget['managed_weights'],
                         'realised_density': budget['density'], 'per_layer_active': budget['per_layer_active'],
                         'per_layer_density': budget['per_layer_density'],
                         'static_and_rigl_share_initial_masks': all(
                             a['initial_mask_digest'] == b['initial_mask_digest']
                             for a, b in zip(per_arm['static_sparse'], per_arm['rigl'])),
                         'rigl_budget_constant': all(r['budget_constant'] for r in per_arm['rigl']),
                         'inactive_weights_zero': all(r['inactive_weights_zero']
                                                      for arm in ('static_sparse', 'rigl') for r in per_arm[arm])}
    receipt['estimated_cost'] = _flops(probe.shapes, budget['per_layer_active'], cfg['batch_size'], cfg['delta_t'],
                                       t_end, cfg['steps'])
    receipt['task'] = tasks
    receipt['notes'] = ('Synthetic procedural classification in which only a seeded subset of inputs carries the '
                        'label, chosen so that which connections exist matters; inspired by RigL (Evci et al., '
                        '2020), not a reproduction. Equal training iterations, shared initialisation and minibatches; '
                        'RigL skips the optimiser step on its connectivity-update iterations (Algorithm 1), so it '
                        'takes slightly fewer optimiser steps. Masks are applied to dense tensors, so there is no '
                        'measured speed-up. Quick mode is a pipeline check; its verdict is weak evidence. Results may '
                        'not transfer to other tasks, sparsities or optimisers.')
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t0, **{f'{arm}_runs': walls[arm] for arm in ARMS}}
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment}
