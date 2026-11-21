"""Sparse dictionary of brain activity: online TopK sparse autoencoders and causal feature ablation.

Inspired by -- not a reproduction of -- the TopK sparse autoencoders of Gao et al., *Scaling and
evaluating sparse autoencoders* (arXiv:2406.04093, June 2024; ICLR 2025). What is taken from that paper:

* the TopK encoder ``z = ReLU(TopK_k(W_enc (x - b_pre) + b_enc))`` and decoder ``x_hat = W_dec z + b_pre``
  (``L0 <= k``; ``b_enc`` is optional: the paper's main equation omits it and its Appendix C.3 reports it
  does not change the converged MSE, while the authors' reference code keeps it as ``latent_bias``);
* unit-norm decoder columns (at initialisation and after every step) with the gradient component
  parallel to each decoder column projected out before the optimiser step;
* the encoder initialised to the transpose of the decoder and ``b_pre`` to the geometric median of a
  data sample (Weiszfeld iterations);
* the AuxK auxiliary loss for dead latents: a latent is *dead* once it has not fired (``z > 1e-3``) for
  ``dead_after`` training samples; the top-``k_aux`` dead latents (by pre-activation, ReLU applied) model
  the main residual ``e = x - x_hat`` and the loss is
  ``mse_scale * MSE(x, x_hat) + aux_coef * mean_i ||e_i - e_hat_i||^2 / ||e_i||^2``
  (main normalisation computed once from the initial data sample, AuxK normalised per sample, a
  non-finite AuxK term is dropped; ``aux_coef = 1/32`` as in the paper);
* Adam (beta1 0.9, beta2 0.999, eps 6.25e-10) with a constant learning rate; normalised MSE, i.e. the
  fraction of variance unexplained (FVU), as the reconstruction metric.

Gao et al. describe AuxK as similar to "ghost grads" (Jermyn & Templeton, *Ghost Grads: An improvement
on resampling*, Transformer Circuits Thread, January 2024); TopK itself follows the k-sparse autoencoders
of Makhzani & Frey (arXiv:1312.5663, 2013). The firing threshold ``1e-3`` follows the authors' reference
code (github.com/openai/sparse_autoencoder). Not implemented: weight EMA, encoder-magnitude matching at
initialisation, power-of-two ``k_aux`` (here ``d // 2``), Multi-TopK.

Scale, data and purpose differ completely from the paper: float64 NumPy on one CPU thread, tens of
units and tens of latents, online updates on live NEUROWEAVE / Sonapse unit activity. Global-norm
gradient clipping (default 1.0) and parameter clipping are added for bounded online behaviour.

Three entry points:

* :class:`TopKSAE` -- the autoencoder (online ``partial_fit`` or epoch ``fit``, Adam or SGD,
  ``evaluate`` -> FVU / L0 / dead fraction / usage histogram, validated ``export_state`` / ``import_state``).
* :class:`SparseDictionaryObserver` -- read-only live observer (frontier observer contract): copies
  activity into a bounded ring buffer and trains the dictionary online with replay mini-batches.
* :func:`run_experiment` -- lab experiment: a dictionary of a NEUROWEAVE engine's unit outputs on a temporal
  task, the features most correlated with the task target on the *training* stream, and causal ablation
  of those features' decoder contributions on frozen engine clones over held-out streams, against
  random-feature ablation of the same size and no ablation.

Everything is a synthetic computational abstraction (``evidence_tier='synthetic_abstraction'``):
dictionary features are directions in a toy model's activity space, not biological cell types.
"""
from __future__ import annotations

import math
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..neuroweave.engine import NeuroWeaveEngine
from ..neuroweave.experiments import (_LOG_EPS, ReadoutHead, _budget, _estimated_cost, _mean, _state_digest,
                                      clone_engine, make_spec, run_task_phase)
from ..neuroweave.spec import TOPOLOGIES
from ..neuroweave.tasks import TASKS, TaskStream, chance_accuracy, make_stream
from .harness import base_receipt, compare, seal
from .spec import EVIDENCE_TIER, FrontierSpecV1

__all__ = ['MODULE_ID', 'EXPERIMENTS', 'DEFAULTS', 'QUICK', 'TopKSAE', 'SparseDictionaryObserver', 'topk_relu',
           'geometric_median', 'fraction_of_variance_unexplained', 'usage_histogram', 'select_predictive_features',
           'refit_readout', 'ablation_phase', 'run_experiment', 'USAGE_BINS', 'MAX_SAE_WEIGHTS']

MODULE_ID = 'sparse_dictionary'
SAE_STATE_SCHEMA = 'TopKSAEStateV1'
OBSERVER_STATE_SCHEMA = 'SparseDictionaryObserverStateV1'
MAX_ABS_INPUT = 1e6                 # inputs beyond this magnitude are rejected (keeps squares finite)
MAX_SAE_WEIGHTS = 1 << 20           # n_inputs * n_features cap per dictionary (memory / state-size bound)
USAGE_BINS = ('never', '(0, 1e-3]', '(1e-3, 1e-2]', '(1e-2, 1e-1]', '(1e-1, 1]')
_SINCE_CAP = 1 << 40
_COUNT_CAP = 1 << 53
_SAE_INIT_OFFSET = 51_001
_SAE_SHUFFLE_OFFSET = 51_013
_OBS_SAE_OFFSET = 51_029
_OBS_REPLAY_OFFSET = 51_031
_EXP_SAE_OFFSET = 51_043
_EXP_RANDOM_OFFSET = 51_047
_PARAMS = ('W_enc', 'b_enc', 'W_dec', 'b_pre')


# ====================================================================== validation helpers
def _int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _float(name: str, value: Any, low: float, high: float, *, open_low: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number')
    v = float(value)
    if not math.isfinite(v) or v < low or v > high or (open_low and v == low):
        raise ValueError(f'{name} must be a finite number in {"(" if open_low else "["}{low}, {high}]')
    return v


def _array(name: str, value: Any, shape: tuple[int, ...], *, max_abs: float | None = None) -> np.ndarray:
    try:
        a = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be a finite array of shape {shape}')
    if max_abs is not None and a.size and float(np.max(np.abs(a))) > max_abs:
        raise ValueError(f'{name} exceeds the bound {max_abs}')
    return a


def _int_array(name: str, value: Any, shape: tuple[int, ...], low: int, high: int) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f'{name} must be a list of ints')
    raw = np.asarray(value, dtype=object).ravel()
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in raw):
        raise ValueError(f'{name} must contain ints')
    a = np.asarray(value, dtype=np.int64)
    if a.shape != shape or (a.size and (int(a.min()) < low or int(a.max()) > high)):
        raise ValueError(f'{name} must be ints in [{low}, {high}] of shape {shape}')
    return a


def _batch(name: str, value: Any, dim: int | None = None, *, max_rows: int = 1_000_000) -> np.ndarray:
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1 or a.shape[0] > max_rows:
        raise ValueError(f'{name} must be a vector or a (rows <= {max_rows}, dim) matrix')
    if dim is not None and a.shape[1] != dim:
        raise ValueError(f'{name} must have {dim} columns, got {a.shape[1]}')
    if not np.all(np.isfinite(a)) or float(np.max(np.abs(a))) > MAX_ABS_INPUT:
        raise ValueError(f'{name} must be finite with |values| <= {MAX_ABS_INPUT:g}')
    return a


# ====================================================================== primitives
def _topk(pre: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-pre, axis=1, kind='stable')[:, :k]       # descending; ties -> lower index first
    z = np.zeros_like(pre)
    np.put_along_axis(z, idx, np.maximum(np.take_along_axis(pre, idx, axis=1), 0.0), axis=1)
    return z, idx


def topk_relu(pre: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """TopK activation of Gao et al.: keep the ``k`` largest pre-activations of every row, apply ReLU to them
    and zero the rest. Ties are broken towards the lower index. Returns ``(z, idx)`` with ``idx`` the selected
    columns per row in descending pre-activation order (so ``L0 <= k``)."""
    p = _batch('pre', pre)
    return _topk(p, _int('k', k, 1, p.shape[1]))


def geometric_median(points: Any, *, max_iter: int = 256, tol: float = 1e-10) -> np.ndarray:
    """Geometric median (minimiser of the summed Euclidean distance) by Weiszfeld iterations from the mean.

    Distances are floored at 1e-12, so an iterate that lands on a data point stays there (adequate for the
    initialisation of ``b_pre``, which is its only use here)."""
    x = _batch('points', points)
    max_iter = _int('max_iter', max_iter, 1, 100_000)
    y = x.mean(axis=0)
    for _ in range(max_iter):
        w = 1.0 / np.maximum(np.sqrt(((x - y) ** 2).sum(axis=1)), 1e-12)
        nxt = (w[:, None] * x).sum(axis=0) / w.sum()
        shift = float(np.sqrt(((nxt - y) ** 2).sum()))
        y = nxt
        if shift <= tol * (1.0 + float(np.sqrt((y ** 2).sum()))):
            break
    return y


def _fvu(x: np.ndarray, x_hat: np.ndarray) -> float | None:
    num = float(((x - x_hat) ** 2).sum())
    den = float(((x - x.mean(axis=0)) ** 2).sum())
    if den <= 1e-300:
        return 0.0 if num <= 1e-300 else None
    return num / den


def fraction_of_variance_unexplained(x: Any, x_hat: Any) -> float | None:
    """Normalised MSE of Gao et al.: reconstruction error divided by the error of always predicting the
    per-dimension mean (``None`` when the data has no variance but the reconstruction has error)."""
    a = _batch('x', x)
    b = _batch('x_hat', x_hat, a.shape[1])
    if a.shape != b.shape:
        raise ValueError('x and x_hat must have the same shape')
    return _fvu(a, b)


def usage_histogram(rates: Any) -> dict[str, Any]:
    """Number of features per firing-rate bin (``USAGE_BINS``); rates are fractions of samples in [0, 1]."""
    r = np.asarray(rates, dtype=np.float64).ravel()
    if not np.all(np.isfinite(r)) or (r.size and (float(r.min()) < 0.0 or float(r.max()) > 1.0)):
        raise ValueError('rates must be finite fractions in [0, 1]')
    counts = [int(np.sum(r <= 0.0)), int(np.sum((r > 0.0) & (r <= 1e-3))), int(np.sum((r > 1e-3) & (r <= 1e-2))),
              int(np.sum((r > 1e-2) & (r <= 1e-1))), int(np.sum(r > 1e-1))]
    return {'bins': list(USAGE_BINS), 'counts': counts}


# ====================================================================== TopK sparse autoencoder
class TopKSAE:
    """Online TopK sparse autoencoder (NumPy, float64, deterministic, bounded).

    Shapes: ``W_enc`` (features, inputs), ``b_enc`` (features,), ``W_dec`` (inputs, features) with unit-norm
    columns, ``b_pre`` (inputs,). All randomness comes from ``default_rng(seed + offset)`` streams (decoder
    directions) and per-epoch generators seeded by ``(seed + offset, epoch)`` (shuffles), so the object holds
    no generator state and export/import round-trips exactly."""

    def __init__(self, n_inputs: int, n_features: int, k: int, *, seed: int = 0, optimizer: str = 'adam',
                 lr: float = 1e-3, k_aux: int | None = None, aux_coef: float = 1.0 / 32.0, dead_after: int = 1024,
                 encoder_bias: bool = True, grad_clip: float | None = 1.0, betas: Sequence[float] = (0.9, 0.999),
                 eps: float = 6.25e-10, fire_threshold: float = 1e-3, max_param: float = 1e3) -> None:
        n_inputs = _int('n_inputs', n_inputs, 1, MAX_SAE_WEIGHTS)
        n_features = _int('n_features', n_features, 1, MAX_SAE_WEIGHTS)
        if n_inputs * n_features > MAX_SAE_WEIGHTS:
            raise ValueError(f'n_inputs * n_features must be <= {MAX_SAE_WEIGHTS}')
        k = _int('k', k, 1, n_features)
        if optimizer not in ('adam', 'sgd'):
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        if type(encoder_bias) is not bool:
            raise ValueError('encoder_bias must be a boolean')
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError('betas must be a pair')
        self.config: dict[str, Any] = {
            'n_inputs': n_inputs, 'n_features': n_features, 'k': k, 'seed': _int('seed', seed, 0, 2**31 - 1),
            'optimizer': optimizer, 'lr': _float('lr', lr, 0.0, 10.0, open_low=True),
            'k_aux': max(1, min(n_features, n_inputs // 2)) if k_aux is None else _int('k_aux', k_aux, 1, n_features),
            'aux_coef': _float('aux_coef', aux_coef, 0.0, 100.0),
            'dead_after': _int('dead_after', dead_after, 1, _SINCE_CAP), 'encoder_bias': encoder_bias,
            'grad_clip': None if grad_clip is None else _float('grad_clip', grad_clip, 0.0, 1e6, open_low=True),
            'betas': [_float('beta1', betas[0], 0.0, 0.999999), _float('beta2', betas[1], 0.0, 0.999999)],
            'eps': _float('eps', eps, 0.0, 1.0, open_low=True),
            'fire_threshold': _float('fire_threshold', fire_threshold, 0.0, 1e3),
            'max_param': _float('max_param', max_param, 1.0, 1e6),
        }
        self._reset()

    # ------------------------------------------------------------------ configuration accessors
    n_inputs = property(lambda self: self.config['n_inputs'])
    n_features = property(lambda self: self.config['n_features'])
    k = property(lambda self: self.config['k'])
    k_aux = property(lambda self: self.config['k_aux'])
    seed = property(lambda self: self.config['seed'])
    dead_after = property(lambda self: self.config['dead_after'])
    fire_threshold = property(lambda self: self.config['fire_threshold'])

    def _reset(self) -> None:
        d, f = self.n_inputs, self.n_features
        rng = np.random.default_rng(self.seed + _SAE_INIT_OFFSET)
        w = rng.standard_normal((d, f))
        self.W_dec = w / np.maximum(np.sqrt((w ** 2).sum(axis=0, keepdims=True)), 1e-12)
        self.W_enc = self.W_dec.T.copy()           # encoder = decoder transpose (Gao et al., Sec. 2.4)
        self.b_enc = np.zeros(f)
        self.b_pre = np.zeros(d)
        self.mse_scale = 1.0
        self.initialized = False
        self.updates = 0
        self.samples_seen = 0
        self.fit_epochs = 0
        self.adam_t = 0
        self.since_fired = np.zeros(f, dtype=np.int64)
        self.fire_count = np.zeros(f, dtype=np.int64)
        adam = self.config['optimizer'] == 'adam'
        self._m = {p: np.zeros_like(getattr(self, p)) for p in _PARAMS} if adam else None
        self._v = {p: np.zeros_like(getattr(self, p)) for p in _PARAMS} if adam else None

    # ------------------------------------------------------------------ accounting
    @property
    def parameter_count(self) -> int:
        bias = self.b_enc.size if self.config['encoder_bias'] else 0
        return int(self.W_enc.size + self.W_dec.size + self.b_pre.size + bias)

    @property
    def state_numbers(self) -> int:
        moments = 2 * sum(getattr(self, p).size for p in _PARAMS) if self._m is not None else 0
        return int(sum(getattr(self, p).size for p in _PARAMS) + moments + 2 * self.n_features + 6)

    def decoder_norms(self) -> np.ndarray:
        return np.sqrt((self.W_dec ** 2).sum(axis=0))

    def dead_mask(self) -> np.ndarray:
        """Latents that have not fired for ``dead_after`` training samples (all False before any training)."""
        return self.since_fired >= self.dead_after

    def dead_fraction(self) -> float:
        return float(np.mean(self.dead_mask()))

    # ------------------------------------------------------------------ forward
    def _pre(self, x: np.ndarray) -> np.ndarray:
        pre = (x - self.b_pre) @ self.W_enc.T
        return pre + self.b_enc if self.config['encoder_bias'] else pre

    def pre_activations(self, x: Any) -> np.ndarray:
        return self._pre(_batch('x', x, self.n_inputs))

    def encode(self, x: Any) -> np.ndarray:
        """Sparse codes ``z`` (same leading shape as ``x``: a vector gives a vector)."""
        a = np.asarray(x) if isinstance(x, np.ndarray) else x
        xb = _batch('x', a, self.n_inputs)
        z = _topk(self._pre(xb), self.k)[0]
        return z[0] if np.ndim(a) == 1 else z

    def decode(self, z: Any) -> np.ndarray:
        zb = _batch('z', z, self.n_features)
        out = zb @ self.W_dec.T + self.b_pre
        return out[0] if np.ndim(z) == 1 else out

    def reconstruct(self, x: Any) -> np.ndarray:
        xb = _batch('x', x, self.n_inputs)
        out = _topk(self._pre(xb), self.k)[0] @ self.W_dec.T + self.b_pre
        return out[0] if np.ndim(x) == 1 else out

    # ------------------------------------------------------------------ training
    def init_from_data(self, x: Any) -> None:
        """``b_pre`` <- geometric median of the sample; main-loss scale <- 1 / (per-element variance)."""
        xb = _batch('x', x, self.n_inputs)
        if xb.shape[0] < 2:
            raise ValueError('init_from_data needs at least 2 samples')
        b_pre = np.clip(geometric_median(xb), -self.config['max_param'], self.config['max_param'])
        var = float(np.mean((xb - xb.mean(axis=0)) ** 2))
        self.b_pre = b_pre
        self.mse_scale = float(min(1e12, 1.0 / max(var, 1e-12)))
        self.initialized = True

    def partial_fit(self, x: Any) -> dict[str, Any]:
        """One optimiser step on a mini-batch (initialises from the batch first if needed, which takes >= 2 rows).

        Returns pre-update batch metrics plus the batch codes ``z`` and per-row MSE ``row_mse`` (arrays)."""
        xb = _batch('x', x, self.n_inputs, max_rows=65_536)
        if not self.initialized:
            self.init_from_data(xb)
        cfg = self.config
        b, d = xb.shape
        c = xb - self.b_pre
        pre = c @ self.W_enc.T
        if cfg['encoder_bias']:
            pre += self.b_enc
        z = _topk(pre, self.k)[0]
        err = xb - (z @ self.W_dec.T + self.b_pre)
        # firing statistics first (the AuxK dead mask uses them, as in the reference implementation)
        fired = z > cfg['fire_threshold']
        anyf = fired.any(axis=0)
        self.since_fired = np.where(anyf, np.argmax(fired[::-1], axis=0),
                                    np.minimum(self.since_fired + b, _SINCE_CAP)).astype(np.int64)
        self.fire_count = np.minimum(self.fire_count + fired.sum(axis=0), _COUNT_CAP).astype(np.int64)
        self.samples_seen = int(min(self.samples_seen + b, _COUNT_CAP))
        dead = self.since_fired >= cfg['dead_after']
        # main loss  mse_scale * MSE(x, x_hat)
        g_out = (-2.0 * self.mse_scale / (b * d)) * err
        g_dec = g_out.T @ z
        g_pre_bias = g_out.sum(axis=0)
        dpre = (g_out @ self.W_dec) * (z > 0.0)
        g_enc = dpre.T @ c
        g_benc = dpre.sum(axis=0)
        g_pre_bias -= (dpre @ self.W_enc).sum(axis=0)
        loss = self.mse_scale * float(np.mean(err ** 2))
        aux_loss = 0.0
        n_dead = int(dead.sum())
        if cfg['aux_coef'] > 0.0 and n_dead:
            didx = np.flatnonzero(dead)
            za = np.zeros_like(z)
            za[:, didx] = _topk(pre[:, didx], min(cfg['k_aux'], n_dead))[0]
            enorm = np.maximum((err ** 2).sum(axis=1), 1e-12)
            resid = za @ self.W_dec.T - err
            aux = float(np.mean((resid ** 2).sum(axis=1) / enorm))
            g_aux = (2.0 * cfg['aux_coef'] / b) * resid / enorm[:, None]
            if math.isfinite(aux) and np.all(np.isfinite(g_aux)):     # a non-finite AuxK term is dropped
                aux_loss = aux
                if np.any(za > 0.0):
                    g_dec += g_aux.T @ za
                    dpa = (g_aux @ self.W_dec) * (za > 0.0)
                    g_enc += dpa.T @ c
                    g_benc += dpa.sum(axis=0)
                    g_pre_bias -= (dpa @ self.W_enc).sum(axis=0)
        if not cfg['encoder_bias']:
            g_benc = np.zeros_like(g_benc)
        g_dec -= self.W_dec * np.sum(g_dec * self.W_dec, axis=0, keepdims=True)   # project out parallel part
        grads = {'W_enc': g_enc, 'b_enc': g_benc, 'W_dec': g_dec, 'b_pre': g_pre_bias}
        if cfg['grad_clip'] is not None:
            norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
            if norm > cfg['grad_clip']:
                grads = {n: g * (cfg['grad_clip'] / norm) for n, g in grads.items()}
        self._apply(grads)
        self.updates += 1
        return {'loss': loss, 'aux_loss': aux_loss, 'mse': float(np.mean(err ** 2)),
                'l0': float(np.mean(np.count_nonzero(z, axis=1))), 'n_dead': n_dead, 'z': z,
                'row_mse': (err ** 2).mean(axis=1)}

    def _apply(self, grads: Mapping[str, np.ndarray]) -> None:
        cfg = self.config
        lr = cfg['lr']
        if self._m is None:
            for name, g in grads.items():
                getattr(self, name).__isub__(lr * g)
        else:
            self.adam_t += 1
            b1, b2 = cfg['betas']
            c1, c2 = 1.0 - b1 ** self.adam_t, 1.0 - b2 ** self.adam_t
            for name, g in grads.items():
                m, v = self._m[name], self._v[name]
                m *= b1
                m += (1.0 - b1) * g
                v *= b2
                v += (1.0 - b2) * g * g
                getattr(self, name).__isub__(lr * (m / c1) / (np.sqrt(v / c2) + cfg['eps']))
        self.W_dec /= np.maximum(self.decoder_norms()[None, :], 1e-12)
        mp = cfg['max_param']
        for name in ('W_enc', 'b_enc', 'b_pre'):
            np.clip(getattr(self, name), -mp, mp, out=getattr(self, name))
        if not cfg['encoder_bias']:
            self.b_enc[:] = 0.0

    def fit(self, x: Any, *, epochs: int = 1, batch_size: int = 32) -> dict[str, Any]:
        """Epochs of shuffled mini-batch ``partial_fit`` calls (shuffle generator seeded by ``(seed + offset, epoch)``)."""
        xb = _batch('x', x, self.n_inputs)
        epochs = _int('epochs', epochs, 1, 100_000)
        batch_size = _int('batch_size', batch_size, 1, 65_536)
        if not self.initialized:
            self.init_from_data(xb)
        n = xb.shape[0]
        last: dict[str, float] = {}
        for _ in range(epochs):
            order = np.random.default_rng((self.seed + _SAE_SHUFFLE_OFFSET, self.fit_epochs)).permutation(n)
            tot = {'loss': 0.0, 'aux_loss': 0.0, 'mse': 0.0, 'l0': 0.0}
            for s in range(0, n, batch_size):
                rows = order[s:s + batch_size]
                r = self.partial_fit(xb[rows])
                for key in tot:
                    tot[key] += r[key] * rows.size
            self.fit_epochs += 1
            last = {f'last_epoch_{key}': val / n for key, val in tot.items()}
        return {'epochs': epochs, 'updates': self.updates, **last}

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, x: Any) -> dict[str, Any]:
        """FVU, MSE, L0, never-fired fraction and usage histogram on ``x`` (plus the training dead fraction)."""
        xb = _batch('x', x, self.n_inputs)
        z = _topk(self._pre(xb), self.k)[0]
        rates = (z > self.fire_threshold).mean(axis=0)
        return {'samples': int(xb.shape[0]), 'fvu': _fvu(xb, z @ self.W_dec.T + self.b_pre),
                'mse': float(np.mean((xb - z @ self.W_dec.T - self.b_pre) ** 2)),
                'l0': float(np.mean(np.count_nonzero(z, axis=1))), 'never_fired_fraction': float(np.mean(rates <= 0.0)),
                'usage': rates.tolist(), 'usage_histogram': usage_histogram(rates),
                'dead_fraction_training': self.dead_fraction()}

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            'schema': SAE_STATE_SCHEMA, 'config': {**self.config, 'betas': list(self.config['betas'])},
            'initialized': bool(self.initialized), 'mse_scale': float(self.mse_scale), 'updates': int(self.updates),
            'samples_seen': int(self.samples_seen), 'fit_epochs': int(self.fit_epochs), 'adam_t': int(self.adam_t),
            'since_fired': self.since_fired.tolist(), 'fire_count': self.fire_count.tolist(),
            'adam_m': None if self._m is None else {p: self._m[p].tolist() for p in _PARAMS},
            'adam_v': None if self._v is None else {p: self._v[p].tolist() for p in _PARAMS},
        }
        for p in _PARAMS:
            state[p] = getattr(self, p).tolist()
        return state

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate ``state`` completely, then replace this autoencoder's state (configuration must match)."""
        if not isinstance(state, Mapping) or state.get('schema') != SAE_STATE_SCHEMA:
            raise ValueError('not a TopKSAE state')
        cfg = state.get('config')
        mine = {**self.config, 'betas': list(self.config['betas'])}
        if not isinstance(cfg, Mapping) or dict(cfg) != mine:
            raise ValueError('state configuration does not match this autoencoder')
        d, f, mp = self.n_inputs, self.n_features, self.config['max_param']
        shapes = {'W_enc': (f, d), 'b_enc': (f,), 'W_dec': (d, f), 'b_pre': (d,)}
        params = {p: _array(p, state.get(p), shapes[p], max_abs=None if p == 'W_dec' else mp) for p in _PARAMS}
        if float(np.max(np.abs(np.sqrt((params['W_dec'] ** 2).sum(axis=0)) - 1.0))) > 1e-6:
            raise ValueError('decoder columns must have unit norm')
        if not self.config['encoder_bias'] and np.any(params['b_enc'] != 0.0):
            raise ValueError('b_enc must be zero without an encoder bias')
        initialized = state.get('initialized')
        if type(initialized) is not bool:
            raise ValueError('initialized must be a boolean')
        mse_scale = _float('mse_scale', state.get('mse_scale'), 0.0, 1e12, open_low=True)
        counters = {n: _int(n, state.get(n), 0, _COUNT_CAP) for n in ('updates', 'samples_seen', 'fit_epochs', 'adam_t')}
        since = _int_array('since_fired', state.get('since_fired'), (f,), 0, _SINCE_CAP)
        count = _int_array('fire_count', state.get('fire_count'), (f,), 0, _COUNT_CAP)
        moments: dict[str, dict[str, np.ndarray] | None] = {'adam_m': None, 'adam_v': None}
        for key in moments:
            raw = state.get(key)
            if self._m is None:
                if raw is not None:
                    raise ValueError(f'{key} must be null for SGD')
                continue
            if not isinstance(raw, Mapping) or set(raw) != set(_PARAMS):
                raise ValueError(f'{key} must map every parameter')
            moments[key] = {p: _array(f'{key}.{p}', raw[p], shapes[p]) for p in _PARAMS}
            if key == 'adam_v' and any(float(a.min()) < 0.0 for a in moments[key].values() if a.size):
                raise ValueError('adam_v must be non-negative')
        for p in _PARAMS:
            setattr(self, p, params[p])
        self.initialized, self.mse_scale = initialized, mse_scale
        self.updates, self.samples_seen = counters['updates'], counters['samples_seen']
        self.fit_epochs, self.adam_t = counters['fit_epochs'], counters['adam_t']
        self.since_fired, self.fire_count = since, count
        if self._m is not None:
            self._m, self._v = moments['adam_m'], moments['adam_v']

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> 'TopKSAE':
        if not isinstance(state, Mapping) or not isinstance(state.get('config'), Mapping):
            raise ValueError('not a TopKSAE state')
        cfg = dict(state['config'])
        try:
            sae = cls(cfg.pop('n_inputs'), cfg.pop('n_features'), cfg.pop('k'), **cfg)
        except (KeyError, TypeError) as exc:
            raise ValueError('incomplete or unknown TopKSAE configuration') from exc
        sae.import_state(state)
        return sae


# ====================================================================== live observer
class SparseDictionaryObserver:
    """Read-only live observer: an online TopK dictionary of the observed unit activity.

    ``observe`` copies the activity vector into a bounded ring buffer (``history`` rows) and, once ``warmup``
    rows exist (``min(history, 32)``), initialises the dictionary (``b_pre`` = geometric median of the buffer)
    and takes one optimiser step per ``train_every`` observations on a mini-batch made of the new vector plus
    ``batch_size - 1`` replayed buffer rows. FVU, L0 and per-feature usage are exponential moving averages
    over *new* vectors, measured before the update on them (prequential). Nothing is written back anywhere."""

    module_id = MODULE_ID
    TOP_FEATURES = 8
    TOP_UNITS = 5

    def __init__(self, n_units: int, *, seed: int = 7, history: int = 256, features: int = 64, k: int = 8,
                 lr: float = 3e-3, batch_size: int = 8, train_every: int = 1, dead_after: int | None = None,
                 aux_coef: float = 1.0 / 32.0, ema: float = 0.02) -> None:
        n_units = _int('n_units', n_units, 1, 4096)
        history = _int('history', history, 16, 8192)
        features = _int('features', features, 4, 4096)
        batch_size = _int('batch_size', batch_size, 1, 256)
        if n_units * features > MAX_SAE_WEIGHTS:
            raise ValueError(f'n_units * features must be <= {MAX_SAE_WEIGHTS}')
        self.config: dict[str, Any] = {
            'n_units': n_units, 'seed': _int('seed', seed, 0, 2**31 - 1), 'history': history, 'features': features,
            'k': _int('k', k, 1, features), 'lr': _float('lr', lr, 0.0, 1.0, open_low=True), 'batch_size': batch_size,
            'train_every': _int('train_every', train_every, 1, 10_000),
            'dead_after': batch_size * history if dead_after is None else _int('dead_after', dead_after, 1, _SINCE_CAP),
            'aux_coef': _float('aux_coef', aux_coef, 0.0, 100.0), 'ema': _float('ema', ema, 0.0, 1.0, open_low=True),
        }
        self.warmup = min(history, 32)
        self.reset()

    @classmethod
    def from_spec(cls, n_units: int, spec: FrontierSpecV1) -> 'SparseDictionaryObserver':
        """Observer sized from a :class:`FrontierSpecV1` (``seed``, ``history``, ``sae_features``, ``sae_k``)."""
        if not isinstance(spec, FrontierSpecV1):
            raise ValueError('spec must be a FrontierSpecV1')
        return cls(n_units, seed=spec.seed, history=spec.history, features=spec.sae_features, k=spec.sae_k)

    n_units = property(lambda self: self.config['n_units'])
    history = property(lambda self: self.config['history'])

    def _new_sae(self) -> TopKSAE:
        c = self.config
        return TopKSAE(c['n_units'], c['features'], c['k'], seed=c['seed'] + _OBS_SAE_OFFSET, lr=c['lr'],
                       aux_coef=c['aux_coef'], dead_after=c['dead_after'])

    def reset(self) -> None:
        """Return to the freshly constructed state (empty buffer, untrained dictionary, same seed)."""
        n = self.n_units
        self._buf = np.zeros((self.history, n))
        self._head = 0
        self._filled = 0
        self.observed = 0
        self._trained = 0
        self._mean = np.zeros(n)
        self._var = np.zeros(n)
        self._mse_ema: float | None = None
        self._l0_ema: float | None = None
        self._fire_ema = np.zeros(self.config['features'])
        self._sae = self._new_sae()

    def _validate(self, x: Any) -> np.ndarray:
        try:
            v = np.array(x, dtype=np.float64)        # always a copy: the observer never aliases the caller's array
        except (TypeError, ValueError) as exc:
            raise ValueError('activity must be numeric') from exc
        if v.ndim != 1 or v.size != self.n_units:
            raise ValueError(f'activity must be a 1-D vector of length {self.n_units}')
        if not np.all(np.isfinite(v)) or (v.size and float(np.max(np.abs(v))) > MAX_ABS_INPUT):
            raise ValueError(f'activity must be finite with |values| <= {MAX_ABS_INPUT:g}')
        return v

    def _chronological(self) -> np.ndarray:
        return self._buf[(self._head - self._filled + np.arange(self._filled)) % self.history]

    def observe(self, x: np.ndarray) -> None:
        v = self._validate(x)
        h = self.history
        self._buf[self._head] = v
        self._head = (self._head + 1) % h
        self._filled = min(h, self._filled + 1)
        self.observed += 1
        a = max(self.config['ema'], 1.0 / self.observed)        # running mean first, then EMA
        delta = v - self._mean
        self._mean += a * delta
        self._var = (1.0 - a) * (self._var + a * delta * delta)
        sae = self._sae
        if not sae.initialized and self._filled >= self.warmup:
            sae.init_from_data(self._chronological())
        if not sae.initialized or self.observed % self.config['train_every']:
            return
        batch = v[None, :]
        if self.config['batch_size'] > 1:
            rng = np.random.default_rng((self.config['seed'] + _OBS_REPLAY_OFFSET, self.observed))
            pos = rng.integers(0, self._filled, size=self.config['batch_size'] - 1)   # chronological positions
            batch = np.vstack([batch, self._buf[(self._head - self._filled + pos) % h]])
        res = sae.partial_fit(batch)
        self._trained += 1
        b = max(self.config['ema'], 1.0 / self._trained)
        z0 = res['z'][0]
        mse0, l00 = float(res['row_mse'][0]), float(np.count_nonzero(z0))
        self._mse_ema = mse0 if self._mse_ema is None else (1.0 - b) * self._mse_ema + b * mse0
        self._l0_ema = l00 if self._l0_ema is None else (1.0 - b) * self._l0_ema + b * l00
        self._fire_ema = (1.0 - b) * self._fire_ema + b * (z0 > sae.fire_threshold)

    def summary(self) -> dict[str, Any]:
        sae = self._sae
        var = float(np.mean(self._var)) if self.observed else 0.0
        fvu = None if self._mse_ema is None or var <= 1e-12 else float(self._mse_ema / var)
        top = []
        if self._trained:
            usage = self._fire_ema
            for j in [int(i) for i in np.argsort(-usage, kind='stable')[:self.TOP_FEATURES] if usage[i] > 0.0]:
                col = sae.W_dec[:, j]
                units = np.argsort(-np.abs(col), kind='stable')[:self.TOP_UNITS]
                top.append({'feature': j, 'usage': float(usage[j]),
                            'top_units': [{'unit': int(u), 'weight': float(col[u])} for u in units]})
        return {
            'module': MODULE_ID, 'n_units': self.n_units, 'features': self.config['features'], 'k': self.config['k'],
            'history': self.history, 'observed': int(self.observed), 'history_fill': int(self._filled),
            'ready': bool(sae.initialized), 'steps': int(sae.updates), 'samples_trained': int(sae.samples_seen),
            'fvu_ema': fvu, 'mse_ema': self._mse_ema, 'l0_ema': self._l0_ema, 'dead_fraction': sae.dead_fraction(),
            'dead_after': self.config['dead_after'], 'k_aux': sae.k_aux,
            'decoder_norm_max_error': float(np.max(np.abs(sae.decoder_norms() - 1.0))),
            'top_features': top, 'usage_histogram': usage_histogram(self._fire_ema) if self._trained else None,
            'evidence_tier': EVIDENCE_TIER,
        }

    def export_state(self) -> dict[str, Any]:
        return {'module': MODULE_ID, 'schema': OBSERVER_STATE_SCHEMA, 'config': dict(self.config),
                'observed': int(self.observed), 'trained': int(self._trained), 'buffer': self._chronological().tolist(),
                'mean': self._mean.tolist(), 'var': self._var.tolist(), 'mse_ema': self._mse_ema,
                'l0_ema': self._l0_ema, 'fire_ema': self._fire_ema.tolist(), 'sae': self._sae.export_state()}

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate ``state`` completely, then replace this observer's state (configuration must match)."""
        if (not isinstance(state, Mapping) or state.get('module') != MODULE_ID
                or state.get('schema') != OBSERVER_STATE_SCHEMA):
            raise ValueError('not a sparse_dictionary observer state')
        if not isinstance(state.get('config'), Mapping) or dict(state['config']) != self.config:
            raise ValueError('state configuration does not match this observer')
        n, f = self.n_units, self.config['features']
        observed = _int('observed', state.get('observed'), 0, _COUNT_CAP)
        trained = _int('trained', state.get('trained'), 0, observed)
        raw = state.get('buffer')
        if not isinstance(raw, (list, tuple)) or len(raw) != min(self.history, observed):
            raise ValueError('buffer must hold exactly min(history, observed) rows')
        buf = _array('buffer', raw, (len(raw), n), max_abs=MAX_ABS_INPUT) if raw else np.zeros((0, n))
        mean = _array('mean', state.get('mean'), (n,), max_abs=MAX_ABS_INPUT)
        var = _array('var', state.get('var'), (n,), max_abs=4.0 * MAX_ABS_INPUT ** 2)
        fire = _array('fire_ema', state.get('fire_ema'), (f,))
        if float(var.min()) < 0.0 or float(fire.min()) < 0.0 or float(fire.max()) > 1.0:
            raise ValueError('var must be >= 0 and fire_ema in [0, 1]')
        emas = {}
        for key, hi in (('mse_ema', 4.0 * MAX_ABS_INPUT ** 2), ('l0_ema', float(self.config['k']))):
            val = state.get(key)
            emas[key] = None if val is None else _float(key, val, 0.0, hi)
            if (emas[key] is None) != (trained == 0):
                raise ValueError(f'{key} must be null exactly when nothing was trained')
        sae = self._new_sae()
        sae.import_state(state.get('sae'))
        if sae.initialized != (len(raw) >= self.warmup) or sae.updates != trained:
            raise ValueError('dictionary state does not match the buffer or the training count')
        self._buf = np.zeros((self.history, n))
        self._buf[:len(raw)] = buf
        self._filled = len(raw)
        self._head = self._filled % self.history
        self.observed, self._trained = observed, trained
        self._mean, self._var, self._fire_ema = mean, var, fire
        self._mse_ema, self._l0_ema = emas['mse_ema'], emas['l0_ema']
        self._sae = sae

    @property
    def dictionary(self) -> TopKSAE:
        """The underlying autoencoder (read it, do not train it from outside)."""
        return self._sae


# ====================================================================== experiment helpers
def select_predictive_features(codes: Any, target: Any, count: int, *, min_fires: int = 1) -> dict[str, Any]:
    """Rank features by |Pearson correlation| between activation and a binary target (training data only).

    Only features that fired (``z > 0``) on at least ``min_fires`` rows are eligible ("alive"); ties go to the
    lower index. Returns ``targeted`` (rank order), the eligible ``pool`` (ascending), ``corr`` and ``fires``."""
    z = _batch('codes', codes)
    y = np.asarray(target, dtype=np.float64).ravel()
    if y.shape != (z.shape[0],) or not np.all(np.isfinite(y)):
        raise ValueError('target must be a finite vector with one entry per row of codes')
    count = _int('count', count, 1, z.shape[1])
    min_fires = _int('min_fires', min_fires, 1, max(1, z.shape[0]))
    fires = np.count_nonzero(z > 0.0, axis=0)
    zc, yc = z - z.mean(axis=0), y - y.mean()
    den = np.sqrt((zc ** 2).sum(axis=0) * float((yc ** 2).sum()))
    corr = np.divide(zc.T @ yc, den, out=np.zeros(z.shape[1]), where=den > 1e-12)
    pool = np.flatnonzero(fires >= min_fires)
    ranked = sorted(pool.tolist(), key=lambda j: (-abs(float(corr[j])), j))
    return {'targeted': ranked[:count], 'pool': pool.tolist(), 'corr': corr, 'fires': fires}


def refit_readout(head: ReadoutHead, features: Any, targets: Any, *, l2: float = 1e-2, iters: int = 50) -> dict[str, Any]:
    """Refit a lab :class:`ReadoutHead` by L2-regularised logistic regression (damped Newton, warm-started from
    the head; bias unregularised; weights clipped to the head's [-8, 8] bound). Deterministic."""
    x = _batch('features', features, head.dim)
    y = np.asarray(targets, dtype=np.float64).ravel()
    if y.shape != (x.shape[0],) or not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError('targets must be 0/1 with one entry per feature row')
    l2 = _float('l2', l2, 0.0, 1e3, open_low=True)
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    reg = np.full(xb.shape[1], l2)
    reg[-1] = 0.0
    w = np.concatenate([head.w, [head.b]])

    def objective(wv: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.clip(xb @ wv, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z))
        ll = float(np.mean(np.logaddexp(0.0, z) - y * z))
        return ll + 0.5 * float(np.sum(reg * wv * wv)), p

    obj, p = objective(w)
    it = 0
    for it in range(1, _int('iters', iters, 1, 1000) + 1):
        g = xb.T @ (p - y) / x.shape[0] + reg * w
        hess = (xb.T * (p * (1.0 - p))) @ xb / x.shape[0] + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, g)
        t = 1.0
        while True:
            cand = w - t * step
            new_obj, new_p = objective(cand)
            if new_obj <= obj or t < 1e-6:
                break
            t *= 0.5
        converged = abs(obj - new_obj) < 1e-12
        w, obj, p = cand, new_obj, new_p
        if converged:
            break
    head.w = np.clip(w[:-1], -8.0, 8.0)
    head.b = float(np.clip(w[-1], -8.0, 8.0))
    return {'iterations': it, 'objective': obj}


def _constant_log_loss(y: np.ndarray) -> float:
    p = float(np.mean(y)) if y.size else 0.0
    return 0.0 if p <= 0.0 or p >= 1.0 else -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _score_rows(head: ReadoutHead, features: np.ndarray, targets: np.ndarray) -> dict[str, float | None]:
    correct, losses = [], []
    for x, y in zip(features, targets.astype(int)):
        p = head.predict(x)
        correct.append(1.0 if (p >= 0.5) == (y == 1) else 0.0)
        pc = min(max(p, _LOG_EPS), 1.0 - _LOG_EPS)
        losses.append(-(y * math.log(pc) + (1 - y) * math.log(1.0 - pc)))
    return {'accuracy': _mean(correct), 'log_loss': _mean(losses)}


def ablation_phase(engine: NeuroWeaveEngine, head: ReadoutHead, stream: TaskStream, task: str, *,
                   sae: TopKSAE | None = None, ablate: Iterable[int] = (), record: bool = False) -> dict[str, Any]:
    """Frozen evaluation of ``head`` on a **clone** of ``engine`` (the original is never stepped).

    Mirrors ``run_task_phase(..., learn=False)`` step for step. With ``ablate`` features, after every engine step
    the dictionary codes of the first ``head.n0`` unit outputs are computed and those features' decoder
    contributions ``sum_j z_j W_dec[:, j]`` are subtracted from the clone's ``brain.cells.output`` (clipped to
    the cells' [-1, 1] output range) before the readout reads it -- the same place ``run_task_phase``'s
    ``silence`` zeroes units, so the change also reaches the next step's recurrent input."""
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}')
    if type(record) is not bool:
        raise ValueError('record must be a boolean')
    idx = np.asarray(sorted({_int('feature', j, 0, MAX_SAE_WEIGHTS) for j in ablate}), dtype=np.int64)
    if idx.size:
        if not isinstance(sae, TopKSAE):
            raise ValueError('feature ablation needs a TopKSAE')
        if sae.n_inputs != head.n0 or int(idx.max()) >= sae.n_features:
            raise ValueError('dictionary does not match the head or the ablated features')
    eng, hd = clone_engine(engine), head.copy()
    n0, steps = hd.n0, stream.steps
    w_abl = sae.W_dec[:, idx] if idx.size else None
    targets = stream.target(task)
    correct, losses, briers, removed = [], [], [], []
    outs = np.zeros((steps, n0)) if record else None
    feats = np.zeros((steps, hd.dim)) if record else None
    for i in range(steps):
        res = eng.step(stream.inputs[i], stream.sensory[i], reward=0.0, learn=False)
        out = res['output']
        m = min(n0, out.size)
        if w_abl is not None:
            x = np.zeros(n0)
            x[:m] = out[:m]
            z = sae.encode(x)
            new = np.clip(x - w_abl @ z[idx], -1.0, 1.0)
            eng.brain.cells.output[:m] = new[:m]
            out = out.copy()
            out[:m] = new[:m]
            removed.append(float(np.sqrt(np.sum((new - x) ** 2))))
        xf = hd.features(eng, out)
        p = hd.predict(xf)
        y = int(targets[i])
        correct.append(1.0 if (p >= 0.5) == (y == 1) else 0.0)
        pc = min(max(p, _LOG_EPS), 1.0 - _LOG_EPS)
        losses.append(-(y * math.log(pc) + (1 - y) * math.log(1.0 - pc)))
        briers.append((p - y) ** 2)
        if record:
            outs[i, :m] = out[:m]
            feats[i] = xf
    result: dict[str, Any] = {'steps': steps, 'accuracy': _mean(correct), 'log_loss': _mean(losses),
                              'brier': _mean(briers), 'chance_accuracy': chance_accuracy(targets),
                              'constant_log_loss': _constant_log_loss(targets.astype(np.float64)),
                              'removed_norm': _mean(removed) if removed else 0.0, 'ablated': idx.tolist()}
    if record:
        result['outputs'], result['head_features'] = outs, feats
    return result


# ====================================================================== experiment
DEFAULTS: dict[str, Any] = {
    'task': 'delayed_recall', 'heldout_seeds': (100, 101), 'steps': 300, 'heldout_steps': 150, 'neurons': 48,
    'topology': 'clustered', 'mechanisms': (), 'features': 64, 'k': 8, 'k_aux': None, 'aux_coef': 1.0 / 32.0,
    'dead_after': None, 'sae_lr': 3e-3, 'sae_epochs': 60, 'sae_batch': 32, 'n_ablate': 4, 'random_draws': 4,
    'readout': 'refit', 'readout_l2': 1e-2, 'min_fire_rate': 0.02,
}
QUICK: dict[str, Any] = {**DEFAULTS, 'heldout_seeds': (100,), 'steps': 120, 'heldout_steps': 60, 'neurons': 24,
                         'features': 32, 'k': 4, 'n_ablate': 3, 'random_draws': 2, 'sae_epochs': 40, 'sae_batch': 16}


def _config(quick: bool, seeds: Iterable[int], overrides: Mapping[str, Any]) -> dict[str, Any]:
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown overrides: {sorted(unknown)}')
    cfg = {**(QUICK if quick else DEFAULTS), **overrides}
    try:
        seeds_t = tuple(_int('seed', s, 0, 2**31 - 1) for s in seeds)
        held = tuple(_int('heldout seed', s, 0, 2**31 - 1) for s in cfg['heldout_seeds'])
        mechanisms = tuple(cfg['mechanisms'])
    except TypeError as exc:
        raise ValueError('seeds, heldout_seeds and mechanisms must be iterables') from exc
    if not seeds_t or len(set(seeds_t)) != len(seeds_t) or not held or len(set(held)) != len(held):
        raise ValueError('seeds and heldout_seeds must be non-empty and unique')
    if set(seeds_t) & set(held):
        raise ValueError('heldout_seeds must be disjoint from seeds')
    if cfg['task'] not in TASKS:
        raise ValueError(f'task must be one of {TASKS}')
    if cfg['topology'] not in TOPOLOGIES:
        raise ValueError(f'topology must be one of {TOPOLOGIES}')
    if cfg['readout'] not in ('refit', 'online'):
        raise ValueError("readout must be 'refit' or 'online'")
    steps = _int('steps', cfg['steps'], 16, 100_000)
    features = _int('features', cfg['features'], 4, 4096)
    neurons = _int('neurons', cfg['neurons'], 8, 4096)
    if neurons * features > MAX_SAE_WEIGHTS:
        raise ValueError(f'neurons * features must be <= {MAX_SAE_WEIGHTS}')
    out = {
        'task': cfg['task'], 'seeds': seeds_t, 'heldout_seeds': held, 'steps': steps,
        'heldout_steps': _int('heldout_steps', cfg['heldout_steps'], 4, 100_000), 'neurons': neurons,
        'topology': cfg['topology'], 'mechanisms': mechanisms, 'features': features,
        'k': _int('k', cfg['k'], 1, features),
        'k_aux': None if cfg['k_aux'] is None else _int('k_aux', cfg['k_aux'], 1, features),
        'aux_coef': _float('aux_coef', cfg['aux_coef'], 0.0, 100.0),
        'dead_after': steps if cfg['dead_after'] is None else _int('dead_after', cfg['dead_after'], 1, _SINCE_CAP),
        'sae_lr': _float('sae_lr', cfg['sae_lr'], 0.0, 1.0, open_low=True),
        'sae_epochs': _int('sae_epochs', cfg['sae_epochs'], 1, 10_000),
        'sae_batch': _int('sae_batch', cfg['sae_batch'], 1, 4096),
        'n_ablate': _int('n_ablate', cfg['n_ablate'], 1, features),
        'random_draws': _int('random_draws', cfg['random_draws'], 1, 64), 'readout': cfg['readout'],
        'readout_l2': _float('readout_l2', cfg['readout_l2'], 0.0, 1e3, open_low=True),
        'min_fire_rate': _float('min_fire_rate', cfg['min_fire_rate'], 0.0, 1.0), 'quick': quick,
    }
    make_spec(mechanisms, seed=seeds_t[0], neurons=neurons, topology=out['topology'], steps=steps)  # validates mechanisms
    return out


def _arm_metrics(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {key: _mean(r[key] for r in runs) for key in ('accuracy', 'log_loss', 'brier', 'chance_accuracy',
                                                       'constant_log_loss', 'removed_norm')}


def _train_energy(sae: TopKSAE, codes: np.ndarray, feats: Sequence[int]) -> float:
    idx = np.asarray(sorted(feats), dtype=np.int64)
    if not idx.size:
        return 0.0
    contrib = codes[:, idx] @ sae.W_dec[:, idx].T
    return float(np.mean((contrib ** 2).sum(axis=1)))


def _run_seed(seed: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    task = cfg['task']
    spec = make_spec(cfg['mechanisms'], seed=seed, neurons=cfg['neurons'], topology=cfg['topology'], steps=cfg['steps'])
    engine = NeuroWeaveEngine(spec)
    head = ReadoutHead(engine.n, seed)
    stream = make_stream(seed, cfg['steps'])
    online = run_task_phase(engine, {task: head}, stream, learn=True)['tasks'][task]
    digest_before = _state_digest(engine)
    t_train = time.perf_counter()
    # Frozen replay of the *training* stream on a clone: dictionary data, readout refit data, selection data.
    rec = ablation_phase(engine, head, stream, task, record=True)
    x_tr, f_tr = rec['outputs'], rec['head_features']
    y_tr = stream.target(task).astype(np.float64)
    readout_fit = refit_readout(head, f_tr, y_tr, l2=cfg['readout_l2']) if cfg['readout'] == 'refit' else None
    train_frozen = _score_rows(head, f_tr, y_tr)
    sae = TopKSAE(head.n0, cfg['features'], cfg['k'], seed=seed + _EXP_SAE_OFFSET, lr=cfg['sae_lr'],
                  k_aux=cfg['k_aux'], aux_coef=cfg['aux_coef'], dead_after=cfg['dead_after'])
    sae.init_from_data(x_tr)
    fit = sae.fit(x_tr, epochs=cfg['sae_epochs'], batch_size=cfg['sae_batch'])
    t_sae = time.perf_counter()
    z_tr = sae.encode(x_tr)
    train_eval = sae.evaluate(x_tr)
    min_fires = max(1, int(math.ceil(cfg['min_fire_rate'] * x_tr.shape[0])))
    sel = select_predictive_features(z_tr, y_tr, cfg['n_ablate'], min_fires=min_fires)
    targeted = sel['targeted']
    pool = np.asarray(sel['pool'], dtype=np.int64)
    rng = np.random.default_rng(seed + _EXP_RANDOM_OFFSET)
    size = min(len(targeted), pool.size)
    random_sets = [sorted(int(j) for j in rng.choice(pool, size=size, replace=False)) for _ in range(cfg['random_draws'])]
    # Causal ablation on frozen clones over held-out streams (never trained on).
    arms: dict[str, list[dict[str, Any]]] = {'none': [], 'targeted': [], 'random': []}
    heldout_x = []
    for hs in cfg['heldout_seeds']:
        hstream = make_stream(hs, cfg['heldout_steps'])
        base = ablation_phase(engine, head, hstream, task, record=True)
        heldout_x.append(base.pop('outputs'))
        base.pop('head_features')
        arms['none'].append(base)
        arms['targeted'].append(ablation_phase(engine, head, hstream, task, sae=sae, ablate=targeted))
        draws = [ablation_phase(engine, head, hstream, task, sae=sae, ablate=s) for s in random_sets]
        arms['random'].append(_arm_metrics(draws))
    digest_after = _state_digest(engine)
    held_fvu = sae.evaluate(np.vstack(heldout_x))['fvu']
    heldout = {name: _arm_metrics(runs) for name, runs in arms.items()}
    for name in ('targeted', 'random'):
        heldout[name]['delta_log_loss_vs_none'] = heldout[name]['log_loss'] - heldout['none']['log_loss']
        heldout[name]['delta_accuracy_vs_none'] = heldout[name]['accuracy'] - heldout['none']['accuracy']
    t_end = time.perf_counter()
    return {
        'seed': int(seed), 'train_stream_digest': stream.digest, 'engine_state_digest': digest_before,
        'original_engine_unchanged': digest_after == digest_before,
        'readout': {'kind': cfg['readout'],
                    'online_prequential': {k: online[k] for k in ('accuracy', 'log_loss', 'chance_accuracy')},
                    'train_frozen': train_frozen, 'constant_log_loss_train': _constant_log_loss(y_tr),
                    'refit': readout_fit},
        'dictionary': {'train_fvu': train_eval['fvu'], 'heldout_fvu': held_fvu, 'train_l0': train_eval['l0'],
                       'never_fired_train_fraction': train_eval['never_fired_fraction'],
                       'dead_fraction_training': train_eval['dead_fraction_training'],
                       'usage_histogram_train': train_eval['usage_histogram'], 'updates': int(sae.updates),
                       'last_epoch_loss': fit.get('last_epoch_loss'), 'last_epoch_aux_loss': fit.get('last_epoch_aux_loss'),
                       'decoder_norm_max_error': float(np.max(np.abs(sae.decoder_norms() - 1.0)))},
        'selection': {'min_fires': min_fires, 'pool_size': int(pool.size),
                      'targeted': [{'feature': int(j), 'corr': float(sel['corr'][j]), 'fires': int(sel['fires'][j])}
                                   for j in targeted],
                      'random_sets': random_sets, 'targeted_train_energy': _train_energy(sae, z_tr, targeted),
                      'random_train_energy': _mean(_train_energy(sae, z_tr, s) for s in random_sets),
                      'random_overlap_with_targeted': _mean(len(set(s) & set(targeted)) / max(1, len(targeted))
                                                            for s in random_sets)},
        'heldout': heldout,
        'budget': {'dictionary_parameters': sae.parameter_count, 'dictionary_state_numbers': sae.state_numbers,
                   'engine': _budget(engine, head)},
        'estimated_cost': {**_estimated_cost(engine, cfg['steps'], head.dim),
                           'engine_steps': int(2 * cfg['steps'] + len(cfg['heldout_seeds']) * cfg['heldout_steps']
                                               * (2 + cfg['random_draws'])),
                           'dictionary_train_ops': int(6 * sae.n_inputs * sae.n_features * x_tr.shape[0]
                                                       * cfg['sae_epochs']),
                           'ablation_encode_ops': int(2 * sae.n_inputs * sae.n_features * len(cfg['heldout_seeds'])
                                                      * cfg['heldout_steps'] * (1 + cfg['random_draws']))},
        'measured_wallclock_s': {'engine_train': t_train - t0, 'replay_refit_dictionary': t_sae - t_train,
                                 'selection_ablation': t_end - t_sae},
    }


_NOTES = (
    "Protocol per seed: a NEUROWEAVE engine and the lab ReadoutHead are trained prequentially on the training stream "
    "(run_task_phase, learn=True). A frozen clone then replays the training stream to record the first n0 unit outputs "
    "and the head features; with readout='refit' the head is refit on those features by L2 logistic regression (a "
    "competent frozen probe; held-out streams are never used for fitting). A TopK sparse autoencoder (Gao et al. 2024) "
    "is fit to the recorded outputs, and the n_ablate alive features with the largest |Pearson r| with the training "
    "target are 'targeted'. On held-out streams, frozen clones run three arms: no ablation, targeted ablation and random "
    "ablation (random_draws uniform draws of the same number of alive features; draws may overlap the targeted set). "
    "Ablation subtracts the chosen features' decoder contributions from the clone's brain.cells.output after every step "
    "(clipped to [-1, 1]; it also feeds the next step's recurrence). VERDICT MEANING: 'improved' means only that "
    "ablating the targeted features raised held-out log-loss MORE than ablating random features (paired 95% t interval "
    "over seeds excludes zero) -- i.e. the targeted features matter more to the frozen readout than random ones. It does "
    "not mean any model got better. Limits: targeted features may carry more activity energy than random ones "
    "(compare selection.targeted_train_energy, random_train_energy and heldout.*.removed_norm before reading the effect "
    "as feature-specific); tiny engines, few hundred samples, few seeds; dictionary features are directions in a toy "
    "model's activity space, not cell types; synthetic abstraction only.")


def run_experiment(*, quick: bool = False, seeds: Iterable[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Targeted vs random vs no SAE-feature ablation of a frozen NEUROWEAVE readout (sealed frontier receipt)."""
    cfg = _config(quick, seeds, overrides)
    t0 = time.perf_counter()
    per_seed = [_run_seed(s, cfg) for s in cfg['seeds']]
    wall = time.perf_counter() - t0

    def col(arm: str, metric: str) -> list[float]:
        return [p['heldout'][arm][metric] for p in per_seed]

    def cmp(cand: list[float], ref: list[float], higher: bool, meaning: str) -> dict[str, Any]:
        out = compare(cand, ref, higher_is_better=higher)
        sd = out.get('sd')
        out['cohen_dz'] = (out['mean'] / sd) if out.get('mean') is not None and sd else None
        out['improved_means'] = meaning
        return out

    comparisons = {
        'targeted_vs_random_log_loss': cmp(col('targeted', 'log_loss'), col('random', 'log_loss'), True,
                                           'targeted ablation raised held-out log-loss more than random ablation'),
        'targeted_vs_random_accuracy': cmp(col('targeted', 'accuracy'), col('random', 'accuracy'), False,
                                           'targeted ablation lowered held-out accuracy more than random ablation'),
        'targeted_vs_none_log_loss': cmp(col('targeted', 'log_loss'), col('none', 'log_loss'), True,
                                         'targeted ablation raised held-out log-loss'),
        'random_vs_none_log_loss': cmp(col('random', 'log_loss'), col('none', 'log_loss'), True,
                                       'random ablation raised held-out log-loss'),
    }
    head = comparisons['targeted_vs_random_log_loss']
    receipt = base_receipt(MODULE_ID, 'feature_ablation', {k: (list(v) if isinstance(v, tuple) else v)
                                                           for k, v in cfg.items()})
    receipt['title'] = 'Sparse dictionary of brain activity: targeted vs random feature ablation'
    receipt['source'] = ('Inspired by TopK sparse autoencoders with the AuxK dead-latent loss: Gao et al., "Scaling and '
                         'evaluating sparse autoencoders", arXiv:2406.04093 (June 2024), ICLR 2025')
    receipt['per_seed'] = per_seed
    receipt['arms'] = {arm: {'heldout_accuracy': _mean(col(arm, 'accuracy')),
                             'heldout_log_loss': _mean(col(arm, 'log_loss')),
                             'heldout_removed_norm': _mean(col(arm, 'removed_norm'))}
                       for arm in ('none', 'targeted', 'random')}
    receipt['arms']['none']['heldout_constant_log_loss'] = _mean(col('none', 'constant_log_loss'))
    receipt['arms']['none']['heldout_chance_accuracy'] = _mean(col('none', 'chance_accuracy'))
    receipt['comparisons'] = comparisons
    receipt['verdict'] = head['verdict']
    receipt['verdict_basis'] = ('targeted_vs_random_heldout_log_loss: paired 95% t interval over seeds of (held-out '
                                'log-loss with the targeted features ablated) - (with the same number of random alive '
                                "features ablated); 'improved' = the targeted features matter more to the frozen readout "
                                "than random ones (interval above zero), 'worse' = they matter less, nothing else")
    receipt['integrity'] = {
        'original_engines_unchanged': all(p['original_engine_unchanged'] for p in per_seed),
        'decoder_columns_unit_norm': all(p['dictionary']['decoder_norm_max_error'] < 1e-9 for p in per_seed),
        'readout_skill_over_constant': _mean(p['heldout']['none']['constant_log_loss'] - p['heldout']['none']['log_loss']
                                             for p in per_seed),
    }
    receipt['budget'] = per_seed[0]['budget']
    receipt['estimated_cost'] = {k: int(sum(p['estimated_cost'][k] for p in per_seed))
                                 for k in per_seed[0]['estimated_cost']}
    receipt['measured_wallclock_s'] = {'total': wall}
    receipt['notes'] = _NOTES
    return seal(receipt)


EXPERIMENTS = {MODULE_ID: run_experiment}
