"""Synchronisation representation, neuron-level models and internal ticks (frontier module ``synchrony``).

Inspired by *Continuous Thought Machines* (CTM): Luke Darlow, Ciaran Regan, Sebastian Risi, Jeffrey Seely and
Llion Jones (Sakana AI), NeurIPS 2025 (spotlight), arXiv:2505.05522 (v1 8 May 2025, v4 3 October 2025),
https://arxiv.org/abs/2505.05522 ; reference code https://github.com/SakanaAI/continuous-thought-machines .
This module borrows the CTM's ideas at toy scale. It does not reproduce the paper's models, scale or results.

Three pieces
------------
1. :class:`SynchronyExtension` (engine extension ``'synchrony'``): the CTM's recursive, exponentially decayed
   pairwise synchronisation of a NEUROWEAVE brain's unit outputs, exposed as extra readout features. Decay
   rates are fixed per pair (the CTM learns them) and spread log-uniformly over several timescales.
2. :class:`SynchronyObserver`: a read-only live observer. It z-scores every observed unit online and tracks
   multi-timescale synchrony and decayed correlation for a deterministic sample of unit pairs.
3. ``MiniCTM`` (PyTorch): a small CTM with private neuron-level models over pre-activation histories,
   internal ticks, a synapse model, attention over the whole input, and outputs read from a learnable-decay
   synchronisation representation. :func:`run_ctm_benchmark` compares it with parameter-matched GRU and MLP
   baselines on cumulative parity. The PyTorch classes are created on first attribute access, so importing
   this module does not import PyTorch.

Synchronisation (CTM Sec. 3.4, eq. 10; App. H, eqs. 13-17)
----------------------------------------------------------
For a pair (i, j) with decay rate r >= 0 and post-activations z^1..z^t::

    S_ij^t = sum_tau exp(-r (t - tau)) z_i^tau z_j^tau / sqrt( sum_tau exp(-r (t - tau)) )

computed recursively as ``alpha <- exp(-r) alpha + z_i z_j``, ``beta <- exp(-r) beta + 1`` and
``S = alpha / sqrt(beta)``. :func:`decayed_synchrony` is the closed form; tests check that the recursions in
the extension and in ``MiniCTM`` equal it.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .extension import EngineExtension

__all__ = ['MODULE_ID', 'SynchronyExtension', 'SynchronyObserver', 'sample_pairs', 'log_uniform_timescales',
           'decayed_synchrony', 'observer_timescales', 'cumulative_parity_data', 'ctm_tick_loss',
           'run_experiment', 'run_ctm_benchmark', 'EXPERIMENTS']
# Lazy (PyTorch) names, resolved by the module-level __getattr__ below. They are left out of __all__ so a
# star import stays free of PyTorch.
_TORCH_NAMES = ('MiniCTM', 'AttentiveGRU', 'ParityMLP', 'SequenceReader', 'NeuronLevelModels')

MODULE_ID = 'synchrony'
FEATURE_SCALES = ('ctm', 'unit')
OBSERVER_STATE_SCHEMA = 'SynchronyObserverStateV1'
_PAIR_OFFSET = 51_001
_TAU_OFFSET = 51_011
_OBS_OFFSET = 51_021
_CTM_OFFSET = 51_031
_GRU_OFFSET = 51_041
_MLP_OFFSET = 51_051
_DATA_OFFSET = 51_061
_BATCH_OFFSET = 51_071
_READER_OFFSET = 51_081
_NLM_OFFSET = 51_091
_TOP_PAIRS = 8


# ====================================================================== validation helpers
def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _check_float(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number in [{low}, {high}]')
    v = float(value)
    if not (math.isfinite(v) and low <= v <= high):
        raise ValueError(f'{name} must be a finite number in [{low}, {high}]')
    return v


def _check_seeds(seeds: Any) -> tuple[int, ...]:
    try:
        out = tuple(_check_int('seed', s, 0, 2**31 - 1) for s in seeds)
    except TypeError as exc:
        raise ValueError('seeds must be an iterable of ints') from exc
    if not out or len(set(out)) != len(out):
        raise ValueError('seeds must be non-empty and unique')
    return out


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        arr = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be a finite array of shape {shape}')
    return arr


def _int_array(name: str, value: Any, size: int, high: int) -> np.ndarray:
    try:
        arr = np.array(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be an integer array') from exc
    if arr.shape != (size,) or (size and (arr.dtype.kind not in 'iu' or arr.min() < 0 or arr.max() >= high)):
        raise ValueError(f'{name} must be {size} integers in [0, {high})')
    return arr.astype(np.int64)


# ====================================================================== shared maths
def sample_pairs(n: int, pairs: int, self_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic pair sample in the spirit of the CTM's *random pairing* (App. C.2).

    Returns ``(left, right)`` index arrays of length ``pairs``. The first ``self_pairs`` entries are self pairs
    (i, i) on distinct units, as in the CTM, which keeps a snapshot of single-unit activity in the
    representation. The rest are distinct unordered cross pairs i < j. Unlike the CTM code, which samples
    with replacement, cross pairs repeat only when ``pairs - self_pairs`` exceeds n(n-1)/2. With ``n == 1``
    every pair is (0, 0)."""
    n = _check_int('n', n, 1, 10_000_000)
    pairs = _check_int('pairs', pairs, 1, 1_000_000)
    self_pairs = _check_int('self_pairs', self_pairs, 0, pairs)
    s_units = rng.choice(n, size=min(self_pairs, n), replace=False).astype(np.int64)
    if self_pairs > n:
        s_units = np.concatenate([s_units, rng.integers(0, n, self_pairs - n)])
    need = pairs - self_pairs
    if n == 1:
        cl = cr = np.zeros(need, np.int64)
    else:
        max_distinct = n * (n - 1) // 2
        take = min(need, max_distinct)
        if 2 * take >= max_distinct:
            # dense regime (max_distinct <= 2 * pairs, so enumeration is small): seeded order over all pairs
            iu, ju = np.triu_indices(n, 1)
            order = rng.permutation(max_distinct)[:take]
            cl, cr = iu[order].astype(np.int64), ju[order].astype(np.int64)
        else:
            # sparse regime: rejection sampling of unordered distinct pairs (acceptance >= 1/4 per draw)
            seen: set[tuple[int, int]] = set()
            lst_l: list[int] = []
            lst_r: list[int] = []
            while len(lst_l) < take:
                k = 2 * (take - len(lst_l)) + 8
                a, b = rng.integers(0, n, k).tolist(), rng.integers(0, n, k).tolist()
                for x, y in zip(a, b):
                    if x == y:
                        continue
                    key = (x, y) if x < y else (y, x)
                    if key in seen:
                        continue
                    seen.add(key)
                    lst_l.append(key[0])
                    lst_r.append(key[1])
                    if len(lst_l) == take:
                        break
            cl, cr = np.array(lst_l, np.int64), np.array(lst_r, np.int64)
        if need > take:
            extra = rng.integers(0, take, need - take)
            cl, cr = np.concatenate([cl, cl[extra]]), np.concatenate([cr, cr[extra]])
    return np.concatenate([s_units, cl]).astype(np.int64), np.concatenate([s_units, cr]).astype(np.int64)


def log_uniform_timescales(count: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """``count`` timescales spread log-uniformly over [low, high] (stratified), in a seeded random order."""
    count = _check_int('count', count, 1, 1_000_000)
    low = _check_float('low', low, 1e-3, 1e6)
    high = _check_float('high', high, low, 1e6)
    q = (np.arange(count) + 0.5) / count
    tau = low * (high / low) ** q
    return tau[rng.permutation(count)]


def decayed_synchrony(history: Any, left: Any, right: Any, decay: Any) -> np.ndarray:
    """Closed-form CTM synchronisation of a ``(t, n)`` post-activation history for the given pairs.

    ``decay[p] = exp(-r_p)``. Returns ``S_p = sum_tau decay_p^(t - tau) z_i z_j / sqrt(sum_tau decay_p^(t - tau))``."""
    Z = np.asarray(history, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 1:
        raise ValueError('history must be a (t, n) array with t >= 1')
    left, right = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    d = np.asarray(decay, dtype=np.float64)
    t = Z.shape[0]
    w = d[None, :] ** (t - 1 - np.arange(t))[:, None]
    return (w * Z[:, left] * Z[:, right]).sum(0) / np.sqrt(w.sum(0))


def observer_timescales(history: int) -> tuple[float, ...]:
    """Observer timescales ``2 * 4^k`` steps up to ``max(2, history / 2)``: (2, 8, 32, 128) for history 256."""
    history = _check_int('history', history, 4, 65_536)
    out = [2.0]
    while out[-1] * 4.0 <= history / 2.0:
        out.append(out[-1] * 4.0)
    return tuple(out)


# ====================================================================== engine extension
class SynchronyExtension(EngineExtension):
    """CTM-style decayed pairwise synchronisation of brain unit outputs as extra readout features.

    ``pairs`` unit pairs are sampled deterministically from the units present at :meth:`bind`. Units born
    later are ignored, so the shape stays fixed. ``self_pairs`` of them are (i, i). Pair ``p`` has a fixed
    decay ``exp(-1/tau_p)``, with ``tau_p`` spread log-uniformly over ``timescales`` steps. The CTM learns
    these rates; here they are fixed, so nothing adapts and ``ctx['learn']`` changes nothing. The accumulators
    still follow activity, which is inference, not learning.

    ``features()`` returns ``S`` (CTM scale, ``feature_scale='ctm'``, clipped to [-10, 10]) or
    ``S * sqrt(1 - decay)`` (``'unit'``). The 'unit' factor is a fixed per-pair constant, so it does not change
    what a linear readout can represent. It does bound every feature by max|z|^2 <= 1, the same range as the
    random-trace control, so a fixed-learning-rate readout sees comparable scales. Brain outputs already lie in
    [-1, 1]; the clip of z to [-1, 1] only guards other callers."""

    id = 'synchrony'

    def __init__(self, pairs: int = 32, *, seed: int = 0, self_pairs: int | None = None,
                 timescales: Sequence[float] = (1.0, 32.0), feature_scale: str = 'ctm') -> None:
        self.pairs = _check_int('pairs', pairs, 1, 4096)
        self.self_pairs = self.pairs // 8 if self_pairs is None else _check_int('self_pairs', self_pairs, 0, self.pairs)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        try:
            lo, hi = (float(v) for v in timescales)
        except (TypeError, ValueError) as exc:
            raise ValueError('timescales must be a (low, high) pair of numbers') from exc
        lo = _check_float('timescales[0]', lo, 0.05, 1e5)
        hi = _check_float('timescales[1]', hi, lo, 1e5)
        if feature_scale not in FEATURE_SCALES:
            raise ValueError(f'feature_scale must be one of {FEATURE_SCALES}')
        self.timescales = (lo, hi)
        self.feature_scale = feature_scale
        self.feature_dim = self.pairs
        self._tau = log_uniform_timescales(self.pairs, lo, hi, np.random.default_rng(self.seed + _TAU_OFFSET))
        self._decay = np.exp(-1.0 / self._tau)
        self._scale = np.ones(self.pairs) if feature_scale == 'ctm' else np.sqrt(1.0 - self._decay)
        self._n = 0
        self._left = np.zeros(0, np.int64)
        self._right = np.zeros(0, np.int64)
        self._num = np.zeros(self.pairs)
        self._den = np.zeros(self.pairs)
        self._steps = 0

    # ------------------------------------------------------------------ lifecycle
    def bind(self, engine: Any) -> None:
        n = _check_int('engine.n', int(engine.n), 1, 10_000_000)
        left, right = sample_pairs(n, self.pairs, self.self_pairs, np.random.default_rng(self.seed + _PAIR_OFFSET))
        self._n, self._left, self._right = n, left, right
        self.reset()

    def reset(self) -> None:
        self._num = np.zeros(self.pairs)
        self._den = np.zeros(self.pairs)
        self._steps = 0

    # ------------------------------------------------------------------ hooks
    def after_step(self, output: np.ndarray, ctx: Mapping[str, Any] | None = None) -> None:
        if self._n == 0:
            raise ValueError('synchrony extension is not bound to an engine')
        out = np.asarray(output, dtype=np.float64)
        if out.ndim != 1 or not np.all(np.isfinite(out)):
            raise ValueError('output must be a finite 1-D array')
        z = np.zeros(self._n)
        m = min(self._n, out.size)                       # neurons born later are ignored (fixed shape)
        z[:m] = np.clip(out[:m], -1.0, 1.0)
        self._num = self._decay * self._num + z[self._left] * z[self._right]
        self._den = self._decay * self._den + 1.0
        self._steps += 1

    def synchronisation(self) -> np.ndarray:
        """Current CTM synchronisation ``S = alpha / sqrt(beta)`` per pair (zeros before the first step)."""
        if self._steps == 0:
            return np.zeros(self.pairs)
        return self._num / np.sqrt(self._den)

    def features(self) -> np.ndarray:
        return np.clip(self.synchronisation() * self._scale, -10.0, 10.0)

    # ------------------------------------------------------------------ accounting / persistence
    @property
    def pair_indices(self) -> tuple[np.ndarray, np.ndarray]:
        return self._left.copy(), self._right.copy()

    @property
    def decays(self) -> np.ndarray:
        return self._decay.copy()

    def parameter_count(self) -> int:
        return 0                                          # nothing is trained

    def state_numbers(self) -> int:
        return int(self._left.size + self._right.size + self._decay.size + self._num.size + self._den.size)

    def telemetry(self) -> dict[str, Any]:
        s = self.synchronisation()
        return {'id': self.id, 'pairs': self.pairs, 'self_pairs': self.self_pairs, 'n': self._n,
                'steps': self._steps, 'feature_scale': self.feature_scale, 'timescales': list(self.timescales),
                'mean_abs_sync': float(np.abs(s).mean()), 'max_abs_sync': float(np.abs(s).max())}

    def _config(self) -> dict[str, Any]:
        return {'pairs': self.pairs, 'self_pairs': self.self_pairs, 'seed': self.seed,
                'timescales': list(self.timescales), 'feature_scale': self.feature_scale, 'n': self._n}

    def export_state(self) -> dict[str, Any]:
        return {'id': self.id, **self._config(), 'left': self._left.tolist(), 'right': self._right.tolist(),
                'num': self._num.tolist(), 'den': self._den.tolist(), 'steps': self._steps}

    def import_state(self, state: Mapping[str, Any]) -> None:
        super().import_state(state)
        for key, value in self._config().items():
            if state.get(key) != value:
                raise ValueError(f'synchrony state does not match this extension ({key})')
        P = self.pairs
        left = _int_array('left', state.get('left'), P, max(1, self._n))
        right = _int_array('right', state.get('right'), P, max(1, self._n))
        if not (np.array_equal(left, self._left) and np.array_equal(right, self._right)):
            raise ValueError('synchrony state pairs do not match this extension')
        num = _finite_array('num', state.get('num'), (P,))
        den = _finite_array('den', state.get('den'), (P,))
        steps = _check_int('steps', state.get('steps'), 0, 2**62)
        expect = (1.0 - self._decay ** steps) / (1.0 - self._decay)
        if not np.allclose(den, expect, rtol=1e-6, atol=1e-9) or np.any(np.abs(num) > den + 1e-9):
            raise ValueError('synchrony state is inconsistent (den does not match steps or |num| > den)')
        self._num, self._den, self._steps = num, den, steps


# ====================================================================== live observer
class SynchronyObserver:
    """Read-only live observer: online z-scoring per unit plus multi-timescale pairwise synchrony.

    * z-scores use a running mean and variance with window ``history``. They are exact (Welford) for the first
      ``history`` observations and exponentially weighted with rate ``1/history`` afterwards. Values are
      clipped to [-zclip, zclip].
    * ``pairs`` unit pairs, of which ``self_pairs`` are (i, i) and the rest distinct cross pairs, are sampled
      once from ``seed``. At every timescale in :func:`observer_timescales` each pair keeps the CTM recursion
      ``alpha <- d alpha + z_i z_j``, ``beta <- d beta + 1`` with ``d = exp(-1/tau)``, and every unit in a pair
      keeps ``q_u <- d q_u + z_u^2``.
    * ``summary()`` reports, per timescale over the cross pairs, the CTM-scaled synchrony S = alpha/sqrt(beta)
      (it grows like sqrt(window) for a steady product) and the decayed correlation
      C = alpha/sqrt(q_i q_j), which lies in [-1, 1] by Cauchy-Schwarz. It also reports a global synchrony index
      (mean |C| over cross pairs at the slowest timescale), the 8 most synchronised cross pairs (by |C| at the
      slowest timescale), the normalised entropy of |C| across cross pairs, and the number of observed steps.

    The observer never writes to its input. All buffers have fixed size."""

    module_id = MODULE_ID

    def __init__(self, n_units: int, *, seed: int = 7, history: int = 256, pairs: int = 64,
                 self_pairs: int | None = None, zclip: float = 5.0) -> None:
        self.n_units = _check_int('n_units', n_units, 1, 1_000_000)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        self.history = _check_int('history', history, 4, 65_536)
        self.pairs = _check_int('pairs', pairs, 1, 4096)
        self.self_pairs = self.pairs // 8 if self_pairs is None else _check_int('self_pairs', self_pairs, 0, self.pairs)
        self.zclip = _check_float('zclip', zclip, 1.0, 100.0)
        self.timescales = observer_timescales(self.history)
        self._decay = np.exp(-1.0 / np.asarray(self.timescales))
        self._left, self._right = sample_pairs(self.n_units, self.pairs, self.self_pairs,
                                               np.random.default_rng(self.seed + _OBS_OFFSET))
        units, inverse = np.unique(np.concatenate([self._left, self._right]), return_inverse=True)
        self._units = units.astype(np.int64)                          # units that appear in a pair (<= 2 * pairs)
        self._pos_left = inverse[:self.pairs].astype(np.int64)
        self._pos_right = inverse[self.pairs:].astype(np.int64)
        self.reset()

    @classmethod
    def from_spec(cls, n_units: int, spec: Any) -> 'SynchronyObserver':
        """Build from a ``FrontierSpecV1`` (``seed``, ``history``, ``sync_pairs``)."""
        return cls(n_units, seed=int(spec.seed), history=int(spec.history), pairs=int(spec.sync_pairs))

    def reset(self) -> None:
        K = len(self.timescales)
        self._mean = np.zeros(self.n_units)
        self._var = np.zeros(self.n_units)
        self._num = np.zeros((K, self.pairs))
        self._sq = np.zeros((K, self._units.size))
        self._den = np.zeros(K)
        self._steps = 0

    @property
    def observed_steps(self) -> int:
        return self._steps

    @property
    def pair_indices(self) -> tuple[np.ndarray, np.ndarray]:
        return self._left.copy(), self._right.copy()

    def observe(self, x: np.ndarray) -> None:
        v = np.array(x, dtype=np.float64)                # private copy: the caller's array is never touched
        if v.shape != (self.n_units,) or not np.all(np.isfinite(v)):
            raise ValueError(f'observation must be a finite 1-D array of length {self.n_units}')
        a = 1.0 / min(self._steps + 1, self.history)
        delta = v - self._mean
        mean = self._mean + a * delta
        var = (1.0 - a) * (self._var + a * delta * delta)
        z = np.clip((v - mean) / np.sqrt(var + 1e-12), -self.zclip, self.zclip)
        zu = z[self._units]
        d = self._decay[:, None]
        self._mean, self._var = mean, var
        self._num = d * self._num + (z[self._left] * z[self._right])[None, :]
        self._sq = d * self._sq + (zu * zu)[None, :]
        self._den = self._decay * self._den + 1.0
        self._steps += 1

    def synchrony(self) -> tuple[np.ndarray, np.ndarray]:
        """``(S, C)``, each of shape (timescales, pairs): CTM-scaled synchrony and decayed correlation in [-1, 1]."""
        if self._steps == 0:
            z = np.zeros_like(self._num)
            return z, z.copy()
        S = self._num / np.sqrt(self._den)[:, None]
        norm = np.sqrt(self._sq[:, self._pos_left] * self._sq[:, self._pos_right])
        C = np.divide(self._num, norm, out=np.zeros_like(self._num), where=norm > 1e-12)
        return S, np.clip(C, -1.0, 1.0)

    def summary(self) -> dict[str, Any]:
        S, C = self.synchrony()
        cross = np.flatnonzero(self._left != self._right)
        sel = cross if cross.size else np.arange(self.pairs)
        per_ts = [{'tau': float(tau), 'mean_abs_S': float(np.abs(S[k, sel]).mean()),
                   'max_abs_S': float(np.abs(S[k, sel]).max()), 'mean_abs_corr': float(np.abs(C[k, sel]).mean()),
                   'max_abs_corr': float(np.abs(C[k, sel]).max())} for k, tau in enumerate(self.timescales)]
        gsi, entropy, top = 0.0, None, []
        if cross.size and self._steps:
            mag = np.abs(C[-1, cross])
            gsi = float(mag.mean())
            total = float(mag.sum())
            if total > 0.0 and cross.size >= 2:
                q = mag[mag > 0.0] / total
                entropy = float(np.clip(-(q * np.log(q)).sum() / math.log(cross.size), 0.0, 1.0))
            for p in cross[np.argsort(-mag, kind='stable')[:_TOP_PAIRS]]:
                top.append({'i': int(self._left[p]), 'j': int(self._right[p]), 'corr': float(C[-1, p]),
                            'corr_by_tau': [float(c) for c in C[:, p]], 'S_by_tau': [float(s) for s in S[:, p]]})
        return {'module': MODULE_ID, 'n_units': self.n_units, 'pairs': self.pairs, 'self_pairs': self.self_pairs,
                'cross_pairs': int(cross.size), 'timescales': [float(t) for t in self.timescales],
                'zscore_window': self.history, 'observed_steps': self._steps, 'per_timescale': per_ts,
                'global_synchrony_index': gsi, 'synchrony_entropy': entropy, 'top_pairs': top}

    def _config(self) -> dict[str, Any]:
        return {'n_units': self.n_units, 'seed': self.seed, 'history': self.history, 'pairs': self.pairs,
                'self_pairs': self.self_pairs, 'zclip': self.zclip}

    def export_state(self) -> dict[str, Any]:
        return {'module': MODULE_ID, 'schema': OBSERVER_STATE_SCHEMA, **self._config(),
                'left': self._left.tolist(), 'right': self._right.tolist(), 'mean': self._mean.tolist(),
                'var': self._var.tolist(), 'num': self._num.tolist(), 'sq': self._sq.tolist(),
                'den': self._den.tolist(), 'steps': self._steps}

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate the whole state first; nothing changes unless every check passes."""
        if not isinstance(state, Mapping) or state.get('module') != MODULE_ID or state.get('schema') != OBSERVER_STATE_SCHEMA:
            raise ValueError('state is not a synchrony observer state')
        for key, value in self._config().items():
            if state.get(key) != value:
                raise ValueError(f'synchrony observer state does not match this observer ({key})')
        K, P, n, U = len(self.timescales), self.pairs, self.n_units, self._units.size
        left = _int_array('left', state.get('left'), P, n)
        right = _int_array('right', state.get('right'), P, n)
        if not (np.array_equal(left, self._left) and np.array_equal(right, self._right)):
            raise ValueError('synchrony observer state pairs do not match this observer')
        mean = _finite_array('mean', state.get('mean'), (n,))
        var = _finite_array('var', state.get('var'), (n,))
        num = _finite_array('num', state.get('num'), (K, P))
        sq = _finite_array('sq', state.get('sq'), (K, U))
        den = _finite_array('den', state.get('den'), (K,))
        steps = _check_int('steps', state.get('steps'), 0, 2**62)
        expect = (1.0 - self._decay ** steps) / (1.0 - self._decay)
        if np.any(var < 0.0) or np.any(sq < 0.0) or not np.allclose(den, expect, rtol=1e-6, atol=1e-9):
            raise ValueError('synchrony observer state is inconsistent (negative moments or den/steps mismatch)')
        bound = np.sqrt(sq[:, self._pos_left] * sq[:, self._pos_right])
        if np.any(sq > self.zclip ** 2 * den[:, None] * (1.0 + 1e-9) + 1e-9) or np.any(np.abs(num) > bound * (1.0 + 1e-9) + 1e-9):
            raise ValueError('synchrony observer state is out of bounds')
        self._mean, self._var, self._num, self._sq, self._den, self._steps = mean, var, num, sq, den, steps


# ====================================================================== PyTorch models (created lazily)
_TORCH_CLASSES: dict[str, type] = {}


def _torch() -> Any:
    import torch
    return torch


def _generator(seed: int) -> Any:
    """A private seeded ``torch.Generator``. Parameter initialisation never reads or writes the global RNG."""
    return _torch().Generator().manual_seed(int(seed))


def _normalised_entropy(logits: Any) -> Any:
    """Normalised entropy per tick, averaged over output positions: (B, T, L, C) -> (B, T) in [0, 1]."""
    torch = _torch()
    logp = torch.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1) / math.log(logits.shape[-1])
    return ent.mean(-1)


def ctm_tick_loss(logits: Any, targets: Any, certainty: Any = None) -> tuple[Any, dict[str, Any]]:
    """CTM loss (Sec. 3.5, eq. 11; reference ``parity_loss``): cross-entropy averaged over positions at every
    tick, then the mean of the losses at each sample's minimum-loss tick t1 and its maximum-certainty tick t2
    (certainty = 1 - normalised entropy, averaged over positions). With one tick this is plain cross-entropy.

    ``logits``: (B, T, L, C); ``targets``: (B, L) class indices. Returns ``(loss, info)``. ``info`` holds
    ``per_tick_loss`` (B, T) and the selected ``min_loss_tick`` and ``most_certain_tick`` (B,)."""
    torch = _torch()
    if logits.ndim != 4 or targets.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError('logits must be (B, T, L, C) and targets (B, L)')
    B, T, L, C = logits.shape
    tgt = targets[:, None, :].expand(B, T, L).reshape(-1)
    ce = torch.nn.functional.cross_entropy(logits.reshape(B * T * L, C), tgt, reduction='none').view(B, T, L).mean(-1)
    if certainty is None:
        certainty = 1.0 - _normalised_entropy(logits)
    t1 = ce.detach().argmin(1)
    t2 = certainty.detach().argmax(1)
    rows = torch.arange(B)
    loss = 0.5 * (ce[rows, t1].mean() + ce[rows, t2].mean())
    return loss, {'per_tick_loss': ce, 'min_loss_tick': t1, 'most_certain_tick': t2}


def _define_torch_classes() -> dict[str, type]:
    if _TORCH_CLASSES:
        return _TORCH_CLASSES
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    def check_tokens(tokens: Any, seq_len: int, vocab: int) -> None:
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long or tokens.ndim != 2 \
                or tokens.shape[1] != seq_len or tokens.shape[0] < 1:
            raise ValueError(f'tokens must be a LongTensor of shape (batch, {seq_len})')
        if int(tokens.min()) < 0 or int(tokens.max()) >= vocab:
            raise ValueError(f'token values must lie in [0, {vocab})')

    def check_ticks(ticks: Any, default: int) -> int:
        return default if ticks is None else _check_int('ticks', ticks, 1, 1024)

    def uniform(shape: tuple[int, ...], bound: float, g: Any) -> Any:
        return nn.Parameter(torch.empty(shape).uniform_(-bound, bound, generator=g))

    # Layers that own their parameters and draw them from a private generator. They use the distributions of
    # PyTorch's default initialisation, but never touch the global RNG (``nn.Linear`` and friends initialise
    # from it in their constructors). Tests check each one against its ``torch.nn`` counterpart.
    class Linear(nn.Module):
        """``x W^T + b`` with W and b ~ U(+-1/sqrt(fan_in)), as ``nn.Linear``."""

        def __init__(self, fan_in: int, fan_out: int, g: Any) -> None:
            super().__init__()
            self.in_features, self.out_features = fan_in, fan_out
            b = 1.0 / math.sqrt(fan_in)
            self.weight = uniform((fan_out, fan_in), b, g)
            self.bias = uniform((fan_out,), b, g)

        def forward(self, x: Any) -> Any:
            return F.linear(x, self.weight, self.bias)

    class LayerNorm(nn.Module):
        """Layer normalisation over the last axis with a learnable gain (1) and bias (0), as ``nn.LayerNorm``."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))

        def forward(self, x: Any) -> Any:
            return F.layer_norm(x, self.weight.shape, self.weight, self.bias)

    class GRUCell(nn.Module):
        """``torch.gru_cell`` (the kernel behind ``nn.GRUCell``), all weights ~ U(+-1/sqrt(hidden))."""

        def __init__(self, input_size: int, hidden: int, g: Any) -> None:
            super().__init__()
            b = 1.0 / math.sqrt(hidden)
            self.weight_ih = uniform((3 * hidden, input_size), b, g)
            self.weight_hh = uniform((3 * hidden, hidden), b, g)
            self.bias_ih = uniform((3 * hidden,), b, g)
            self.bias_hh = uniform((3 * hidden,), b, g)

        def forward(self, x: Any, h: Any) -> Any:
            return torch.gru_cell(x, h, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

    class SequenceReader(nn.Module):
        """Whole-input feature extractor shared by the tick models (CTM App. F.4 parity setup): token + learned
        position embeddings -> Linear -> LayerNorm give identical keys and values. :meth:`attend` is single-head
        scaled dot-product attention with a query supplied by the core. Token embeddings ~ N(0, 1), as
        ``nn.Embedding``."""

        def __init__(self, seq_len: int, vocab: int, d_input: int, *, seed: int = 0) -> None:
            super().__init__()
            self.scale = 1.0 / math.sqrt(d_input)
            g = _generator(_check_int('seed', seed, 0, 2**31 - 1) + _READER_OFFSET)
            self.token = nn.Parameter(torch.empty(vocab, d_input).normal_(generator=g))
            self.position = uniform((seq_len, d_input), 0.5, g)
            self.proj = Linear(d_input, d_input, g)
            self.norm = LayerNorm(d_input)

        def keys(self, tokens: Any) -> Any:
            return self.norm(self.proj(F.embedding(tokens, self.token) + self.position))

        def attend(self, query: Any, kv: Any, *, weights: bool = False) -> tuple[Any, Any]:
            """``softmax(kv q / sqrt(d)) kv`` for a query (B, d) over identical keys and values (B, L, d), as one
            fused ``scaled_dot_product_attention`` call. Returns ``(o, w)``; the attention weights ``w`` (B, L) are
            computed (separately, same formula) only when ``weights`` is true, else ``None``."""
            o = F.scaled_dot_product_attention(query.unsqueeze(1), kv, kv).squeeze(1)
            if not weights:
                return o, None
            return o, torch.softmax(torch.bmm(kv, query.unsqueeze(-1)).squeeze(-1) * self.scale, dim=-1)

    class NeuronLevelModels(nn.Module):
        """Private per-neuron MLPs (CTM NLMs, Sec. 3.3 and Listing 2; reference ``SuperLinear`` stack).

        Neuron d maps its own last M pre-activations through M -> 2H -> GLU -> H -> 2 -> GLU -> 1 with weights
        ``w1[d]`` (M, 2H) and ``w2[d]`` (H, 2) that belong to it alone. Input and output are neuron-major,
        (D, B, M) -> (D, B). Each layer is one batched einsum over the neuron axis ('dbm,dmh->dbh'): the
        reference's einsum('BDM,MHD->BDH') with the neuron axis first, which maps onto one batched matmul."""

        def __init__(self, d_model: int, memory: int, hidden: int, *, seed: int = 0) -> None:
            super().__init__()
            self.d_model = _check_int('d_model', d_model, 1, 65_536)
            self.memory = _check_int('memory', memory, 1, 1024)
            self.hidden = _check_int('hidden', hidden, 1, 1024)
            b1, b2 = 1.0 / math.sqrt(memory + 2 * hidden), 1.0 / math.sqrt(hidden + 2)
            g = _generator(_check_int('seed', seed, 0, 2**31 - 1) + _NLM_OFFSET)
            self.w1 = uniform((d_model, memory, 2 * hidden), b1, g)
            self.w2 = uniform((d_model, hidden, 2), b2, g)
            self.b1 = nn.Parameter(torch.zeros(d_model, 1, 2 * hidden))
            self.b2 = nn.Parameter(torch.zeros(d_model, 1, 2))

        def forward(self, history: Any) -> Any:
            h = F.glu(torch.einsum('dbm,dmh->dbh', history, self.w1) + self.b1, dim=-1)
            return F.glu(torch.einsum('dbh,dhk->dbk', h, self.w2) + self.b2, dim=-1).squeeze(-1)

    class MiniCTM(nn.Module):
        """A small Continuous Thought Machine for sequence-in, per-position-classes-out tasks.

        Each internal tick follows the reference code's order: action synchrony -> query -> attention over the
        whole input -> synapse model on [z, o] (Linear -> GLU -> LayerNorm, the CTM's depth-1 synapse used for
        parity) -> append the pre-activation to the M-step FIFO history -> private NLMs -> output synchrony ->
        linear readout. Synchrony uses random pairing with ``self_pairs`` self pairs per representation and a
        per-pair decay ``exp(-r)`` with ``r = softplus(rho)`` learnable (the reference clamps ``r`` to [0, 15]
        and starts it at 0; here ``r`` starts log-uniform in ``decay_init``). Action and output pairs share one
        recursion state because both read the same post-activations: the action representation at tick t equals
        the output representation of tick t - 1 on the action pairs, and both start from the learnable start
        state ``start_z`` exactly as in the reference. The start state and the start history are learnable.

        ``forward(tokens, ticks=None, track=False)`` returns ``(logits (B, T, L, C), certainty (B, T))`` plus,
        with ``track``, a dict of detached post-activations (B, T, D), output and action synchrony
        (B, T, n_sync_out) and (B, T, n_sync_action), and attention weights (B, T, L)."""

        def __init__(self, seq_len: int, *, vocab: int = 2, classes: int = 2, d_model: int = 32, memory: int = 4,
                     nlm_hidden: int = 4, ticks: int = 8, d_input: int = 16, n_sync_out: int = 32,
                     n_sync_action: int = 16, self_pairs: int = 4, decay_init: Sequence[float] = (0.03, 1.0),
                     seed: int = 0) -> None:
            super().__init__()
            self.seq_len = _check_int('seq_len', seq_len, 1, 4096)
            self.vocab = _check_int('vocab', vocab, 1, 1_000_000)
            self.classes = _check_int('classes', classes, 2, 1024)
            self.d_model = _check_int('d_model', d_model, 2, 4096)
            self.memory = _check_int('memory', memory, 1, 256)
            self.nlm_hidden = _check_int('nlm_hidden', nlm_hidden, 1, 256)
            self.ticks = _check_int('ticks', ticks, 1, 1024)
            self.d_input = _check_int('d_input', d_input, 2, 4096)
            self.n_sync_out = _check_int('n_sync_out', n_sync_out, 2, 65_536)
            self.n_sync_action = _check_int('n_sync_action', n_sync_action, 2, 65_536)
            self.self_pairs = _check_int('self_pairs', self_pairs, 0, min(self.n_sync_out, self.n_sync_action) - 1)
            try:
                r_lo_raw, r_hi_raw = decay_init
            except (TypeError, ValueError) as exc:
                raise ValueError('decay_init must be a (low, high) pair of rates') from exc
            r_lo = _check_float('decay_init[0]', r_lo_raw, 1e-4, 10.0)
            r_hi = _check_float('decay_init[1]', r_hi_raw, r_lo, 10.0)
            self.seed = _check_int('seed', seed, 0, 2**31 - 1)
            D, L, d, Pa, Po = self.d_model, self.seq_len, self.d_input, self.n_sync_action, self.n_sync_out
            rng = np.random.default_rng(self.seed + _CTM_OFFSET)
            la, ra = sample_pairs(D, Pa, self.self_pairs, rng)
            lo, ro = sample_pairs(D, Po, self.self_pairs, rng)
            rates = 1.0 / log_uniform_timescales(Pa + Po, 1.0 / r_hi, 1.0 / r_lo, rng)
            # pair layout: the first n_sync_action pairs feed the attention query, the rest feed the output
            self.register_buffer('pair_left', torch.as_tensor(np.concatenate([la, lo]), dtype=torch.long))
            self.register_buffer('pair_right', torch.as_tensor(np.concatenate([ra, ro]), dtype=torch.long))
            self.rho = nn.Parameter(torch.as_tensor(np.log(np.expm1(rates)), dtype=torch.float32))
            self.reader = SequenceReader(L, self.vocab, d, seed=self.seed)
            self.nlm = NeuronLevelModels(D, self.memory, self.nlm_hidden, seed=self.seed)
            g = _generator(self.seed + _CTM_OFFSET)
            self.synapse = nn.Sequential(Linear(D + d, 2 * D, g), nn.GLU(), LayerNorm(D))
            self.start_z = uniform((D,), math.sqrt(1.0 / D), g)
            self.start_trace = uniform((D, self.memory), math.sqrt(1.0 / (D + self.memory)), g)
            self.query = Linear(Pa, d, g)
            self.output = Linear(Po, L * self.classes, g)

        def decay_rates(self) -> tuple[Any, Any]:
            """Current ``r = softplus(rho)`` for the action pairs and the output pairs."""
            r = F.softplus(self.rho)
            return r[:self.n_sync_action], r[self.n_sync_action:]

        def forward(self, tokens: Any, ticks: int | None = None, *, track: bool = False) -> Any:
            check_tokens(tokens, self.seq_len, self.vocab)
            T = check_ticks(ticks, self.ticks)
            B, D, Pa = tokens.shape[0], self.d_model, self.n_sync_action
            kv = self.reader.keys(tokens)                                     # identical keys and values
            rates = F.softplus(self.rho)
            decay = torch.exp(-rates)
            # beta_t = sum_{k <= t} exp(-r k) is data independent: one (T + 1, P) table of 1/sqrt(beta)
            k = torch.arange(T + 1, dtype=rates.dtype)
            inv_norm = torch.rsqrt(torch.cumsum(torch.exp(-k[:, None] * rates[None, :]), dim=0))
            inv_act = inv_norm[:T, :Pa]
            z = self.start_z.expand(B, D)
            trace = self.start_trace.unsqueeze(1).expand(D, B, self.memory)    # neuron-major FIFO (D, B, M)
            alpha = z.index_select(1, self.pair_left) * z.index_select(1, self.pair_right)
            alphas, record = [], []
            for t in range(T):
                s_act = alpha[:, :Pa] * inv_act[t]
                o, w = self.reader.attend(self.query(s_act), kv, weights=track)
                a = self.synapse(torch.cat([z, o], dim=-1))                    # pre-activations (B, D)
                trace = torch.cat([trace[:, :, 1:], a.t().unsqueeze(-1)], dim=-1)
                z = self.nlm(trace).t()                                        # post-activations (B, D)
                prod = z.index_select(1, self.pair_left) * z.index_select(1, self.pair_right)
                alpha = torch.addcmul(prod, decay, alpha)                      # alpha <- exp(-r) alpha + z_i z_j
                alphas.append(alpha)
                if track:
                    record.append((z.detach(), s_act.detach(), w.detach()))
            s_out = torch.stack(alphas, dim=1)[:, :, Pa:] * inv_norm[1:, Pa:]   # (B, T, n_sync_out)
            logits = self.output(s_out).view(B, T, self.seq_len, self.classes)
            cert = 1.0 - _normalised_entropy(logits)
            if track:
                zs, acts, ws = zip(*record)
                return logits, cert, {'post_activations': torch.stack(zs, 1), 'sync_out': s_out.detach(),
                                      'sync_action': torch.stack(acts, 1), 'attention': torch.stack(ws, 1)}
            return logits, cert

    class AttentiveGRU(nn.Module):
        """Baseline: a GRU cell unrolled for the same internal ticks. It reads the whole input through the
        same attention reader as :class:`MiniCTM`, with the query taken from the hidden state. This mirrors the
        CTM paper's parameter-matched recurrent baselines (single-layer LSTMs run for the same number of ticks,
        App. F.4), with a GRU in place of the LSTM."""

        def __init__(self, seq_len: int, *, hidden: int, vocab: int = 2, classes: int = 2, ticks: int = 8,
                     d_input: int = 16, seed: int = 0) -> None:
            super().__init__()
            self.seq_len = _check_int('seq_len', seq_len, 1, 4096)
            self.vocab = _check_int('vocab', vocab, 1, 1_000_000)
            self.classes = _check_int('classes', classes, 2, 1024)
            self.hidden = _check_int('hidden', hidden, 1, 8192)
            self.ticks = _check_int('ticks', ticks, 1, 1024)
            self.d_input = _check_int('d_input', d_input, 2, 4096)
            self.seed = _check_int('seed', seed, 0, 2**31 - 1)
            self.reader = SequenceReader(self.seq_len, self.vocab, self.d_input, seed=self.seed)
            g = _generator(self.seed + _GRU_OFFSET)
            self.cell = GRUCell(self.d_input, self.hidden, g)
            self.query = Linear(self.hidden, self.d_input, g)
            self.start_h = uniform((self.hidden,), math.sqrt(1.0 / self.hidden), g)
            self.output = Linear(self.hidden, self.seq_len * self.classes, g)

        def forward(self, tokens: Any, ticks: int | None = None) -> Any:
            check_tokens(tokens, self.seq_len, self.vocab)
            T = check_ticks(ticks, self.ticks)
            B = tokens.shape[0]
            kv = self.reader.keys(tokens)
            h = self.start_h.unsqueeze(0).expand(B, -1).contiguous()
            hs = []
            for _ in range(T):
                o, _w = self.reader.attend(self.query(h), kv)
                h = self.cell(o, h)
                hs.append(h)
            out = self.output(torch.stack(hs, dim=1)).view(B, T, self.seq_len, self.classes)
            return out, 1.0 - _normalised_entropy(out)

    class ParityMLP(nn.Module):
        """Baseline: a one-hidden-layer ReLU MLP that reads the whole sequence (as +/-1 values) at once, with no
        ticks. Its logits have a single tick, so the CTM loss reduces to cross-entropy."""

        def __init__(self, seq_len: int, *, hidden: int, vocab: int = 2, classes: int = 2, seed: int = 0) -> None:
            super().__init__()
            self.seq_len = _check_int('seq_len', seq_len, 1, 4096)
            self.vocab = _check_int('vocab', vocab, 2, 2)
            self.classes = _check_int('classes', classes, 2, 1024)
            self.hidden = _check_int('hidden', hidden, 1, 1_000_000)
            self.seed = _check_int('seed', seed, 0, 2**31 - 1)
            g = _generator(self.seed + _MLP_OFFSET)
            self.net = nn.Sequential(Linear(self.seq_len, self.hidden, g), nn.ReLU(),
                                     Linear(self.hidden, self.seq_len * self.classes, g))

        def forward(self, tokens: Any, ticks: int | None = None) -> Any:
            check_tokens(tokens, self.seq_len, self.vocab)
            B = tokens.shape[0]
            out = self.net(tokens.to(torch.float32) * 2.0 - 1.0).view(B, 1, self.seq_len, self.classes)
            return out, 1.0 - _normalised_entropy(out)

    for cls in (SequenceReader, NeuronLevelModels, MiniCTM, AttentiveGRU, ParityMLP):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__
    for cls in (Linear, LayerNorm, GRUCell):                     # private building blocks (not module attributes)
        cls.__module__ = __name__
        cls.__qualname__ = f'_{cls.__name__}'
    _TORCH_CLASSES.update({'SequenceReader': SequenceReader, 'NeuronLevelModels': NeuronLevelModels,
                           'MiniCTM': MiniCTM, 'AttentiveGRU': AttentiveGRU, 'ParityMLP': ParityMLP,
                           '_Linear': Linear, '_LayerNorm': LayerNorm, '_GRUCell': GRUCell})
    return _TORCH_CLASSES


def __getattr__(name: str) -> Any:
    if name in _TORCH_NAMES:
        cls = _define_torch_classes()[name]
        globals()[name] = cls
        return cls
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_TORCH_NAMES))


# ====================================================================== cumulative-parity benchmark
def cumulative_parity_data(seq_len: int, seed: int, heldout: int) -> dict[str, np.ndarray]:
    """All 2^L bit sequences, split by a seeded permutation into ``heldout`` held-out and the rest for training.

    Inputs are token ids (bit 0/1; the MLP sees them as -1/+1). The target at position l is the parity of bits
    0..l (the CTM paper's *cumulative parity*, Sec. 6 and App. F.3)."""
    seq_len = _check_int('seq_len', seq_len, 2, 16)
    seed = _check_int('seed', seed, 0, 2**31 - 1)
    total = 2 ** seq_len
    heldout = _check_int('heldout', heldout, 1, total - 1)
    ids = np.random.default_rng(seed + _DATA_OFFSET).permutation(total)
    bits = ((ids[:, None] >> np.arange(seq_len)[None, :]) & 1).astype(np.int64)
    parity = np.cumsum(bits, axis=1) % 2
    return {'heldout_x': bits[:heldout], 'heldout_y': parity[:heldout],
            'train_x': bits[heldout:], 'train_y': parity[heldout:]}


# Sized for one CPU thread: the default takes about 3 minutes for three seeds and quick mode about 10-15 s
# (measured 2026-11-17, Snapdragon X Plus, Windows 11 on ARM64, torch 2.11 CPU, under load from other jobs).
_CTM_FULL: dict[str, Any] = {'seq_len': 12, 'ticks': 8, 'd_model': 32, 'memory': 4, 'nlm_hidden': 4, 'd_input': 16,
                             'n_sync_out': 32, 'n_sync_action': 16, 'self_pairs': 4, 'steps': 800, 'batch': 64,
                             'lr': 1e-2, 'weight_decay': 0.01, 'grad_clip': 1.0, 'heldout': 512, 'train_eval': 512}
_CTM_QUICK: dict[str, Any] = {**_CTM_FULL, 'seq_len': 8, 'ticks': 4, 'd_model': 24, 'd_input': 12, 'n_sync_out': 24,
                              'n_sync_action': 12, 'self_pairs': 3, 'steps': 100, 'batch': 32, 'heldout': 64,
                              'train_eval': 192}


def _ctm_config(quick: bool, overrides: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(_CTM_QUICK if quick else _CTM_FULL)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(f'unknown run_ctm_benchmark overrides: {unknown}')
    cfg.update(overrides)
    for key, lo, hi in (('seq_len', 2, 16), ('ticks', 1, 64), ('d_model', 2, 512), ('memory', 1, 64),
                        ('nlm_hidden', 1, 64), ('d_input', 2, 256), ('n_sync_out', 2, 4096), ('n_sync_action', 2, 4096),
                        ('steps', 1, 100_000), ('batch', 1, 4096), ('train_eval', 1, 65_536)):
        cfg[key] = _check_int(key, cfg[key], lo, hi)
    cfg['self_pairs'] = _check_int('self_pairs', cfg['self_pairs'], 0, min(cfg['n_sync_out'], cfg['n_sync_action']) - 1)
    cfg['heldout'] = _check_int('heldout', cfg['heldout'], 1, 2 ** cfg['seq_len'] - 1)
    cfg['lr'] = _check_float('lr', cfg['lr'], 1e-6, 1.0)
    cfg['weight_decay'] = _check_float('weight_decay', cfg['weight_decay'], 0.0, 1.0)
    cfg['grad_clip'] = _check_float('grad_clip', cfg['grad_clip'], 1e-3, 100.0)
    return cfg


def _reader_params(seq_len: int, d_input: int, vocab: int = 2) -> int:
    return vocab * d_input + seq_len * d_input + d_input * d_input + d_input + 2 * d_input


def _ctm_params(cfg: Mapping[str, Any], classes: int = 2) -> int:
    """Analytic trainable-parameter count of :class:`MiniCTM` for a benchmark config."""
    L, D, M, H, d = cfg['seq_len'], cfg['d_model'], cfg['memory'], cfg['nlm_hidden'], cfg['d_input']
    pa, po = cfg['n_sync_action'], cfg['n_sync_out']
    synapse = (D + d) * 2 * D + 2 * D + 2 * D
    nlm = D * M * 2 * H + D * 2 * H + D * H * 2 + D * 2
    return (_reader_params(L, d) + synapse + nlm + D + D * M + (pa + po) + pa * d + d
            + po * L * classes + L * classes)


def _gru_params(seq_len: int, d_input: int, hidden: int, classes: int = 2) -> int:
    return (_reader_params(seq_len, d_input) + 3 * hidden * (d_input + hidden + 2) + hidden * d_input + d_input
            + hidden + hidden * seq_len * classes + seq_len * classes)


def _mlp_params(seq_len: int, hidden: int, classes: int = 2) -> int:
    return seq_len * hidden + hidden + hidden * seq_len * classes + seq_len * classes


def _closest_width(count: Callable[[int], int], target: int, high: int) -> int:
    best, gap = 1, abs(count(1) - target)
    for w in range(2, high + 1):
        g = abs(count(w) - target)
        if g < gap:
            best, gap = w, g
        elif count(w) > target:
            break
    return best


def _trainable(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _ctm_forward_macs(cfg: Mapping[str, Any]) -> int:
    L, T, D, M, H, d = cfg['seq_len'], cfg['ticks'], cfg['d_model'], cfg['memory'], cfg['nlm_hidden'], cfg['d_input']
    po, pa = cfg['n_sync_out'], cfg['n_sync_action']
    per_tick = (pa + po) * 2 + pa * d + 2 * L * d + (D + d) * 2 * D + D * (M * 2 * H + H * 2) + po * L * 2
    return int(L * d * d + T * per_tick)


def _gru_forward_macs(cfg: Mapping[str, Any], hidden: int) -> int:
    L, T, d = cfg['seq_len'], cfg['ticks'], cfg['d_input']
    return int(L * d * d + T * (hidden * d + 2 * L * d + 3 * hidden * (d + hidden) + hidden * L * 2))


def _mlp_forward_macs(cfg: Mapping[str, Any], hidden: int) -> int:
    L = cfg['seq_len']
    return int(L * hidden + hidden * L * 2)


def _evaluate(torch: Any, model: Any, x: Any, y: Any) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits, cert = model(x)
    N = logits.shape[0]
    t_star = cert.argmax(1)
    pred = logits[torch.arange(N), t_star].argmax(-1)
    correct = (pred == y).to(torch.float64)
    last = (logits[:, -1].argmax(-1) == y).to(torch.float64)
    return {'accuracy': float(correct.mean()), 'per_position_accuracy': [float(v) for v in correct.mean(0)],
            'sequence_accuracy': float(correct.min(1).values.mean()), 'last_tick_accuracy': float(last.mean()),
            'mean_most_certain_tick': float(t_star.to(torch.float64).mean()) + 1.0}


def _make_optimizer(torch: Any, params: list[Any], cfg: Mapping[str, Any]) -> tuple[Any, str]:
    """AdamW, fused CPU kernel when this torch build has it (same update rule, fewer kernel launches)."""
    try:
        return torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'], fused=True), 'fused'
    except (RuntimeError, TypeError, ValueError):
        return torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay']), 'default'


def _train_model(torch: Any, model: Any, data: Mapping[str, Any], batches: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = [p for p in model.parameters() if p.requires_grad]
    opt, impl = _make_optimizer(torch, params, cfg)
    xtr, ytr = data['train_x'], data['train_y']
    curve: list[float] = []
    every = max(1, len(batches) // 10)
    t0 = time.perf_counter()
    model.train()
    for step, idx in enumerate(batches):
        ib = torch.as_tensor(idx)
        logits, cert = model(xtr[ib])
        loss, _ = ctm_tick_loss(logits, ytr[ib], cert)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
        opt.step()
        if step % every == 0 or step == len(batches) - 1:
            curve.append(float(loss.detach()))
    train_wall = time.perf_counter() - t0
    k = min(int(cfg['train_eval']), int(xtr.shape[0]))
    return {'heldout': _evaluate(torch, model, data['heldout_x'], data['heldout_y']),
            'train': _evaluate(torch, model, xtr[:k], ytr[:k]), 'loss_curve': curve,
            'final_train_loss': curve[-1], 'optimizer_impl': impl, 'measured_wallclock_s': train_wall}


def run_ctm_benchmark(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """MiniCTM vs parameter-matched GRU and MLP baselines on cumulative parity of L-bit sequences presented whole.

    Per seed, all three models share the train/held-out split, the minibatch sequence, the optimiser (AdamW),
    learning rate, step count, gradient clip and loss (the CTM tick loss; for the one-tick MLP it is plain
    cross-entropy). The GRU (attention reader + ticks) and the MLP are sized to the MiniCTM's trainable-parameter
    count (within 10 %, else ``ValueError``). Tick models are read out at each sample's most certain tick. The
    verdict is ``harness.compare`` on per-seed held-out per-position accuracy, MiniCTM vs the best baseline
    (the higher mean over seeds, chosen after the fact, which favours the baseline). PyTorch runs on one thread
    and the caller's thread count is restored. The global RNG is never used: models draw their initialisation
    from private seeded ``torch.Generator`` objects, and data and minibatches come from NumPy generators."""
    from . import harness
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seeds = _check_seeds(seeds)
    cfg = _ctm_config(quick, overrides)
    torch = _torch()
    classes = _define_torch_classes()
    L, d = cfg['seq_len'], cfg['d_input']
    ctm_kw = {k: cfg[k] for k in ('ticks', 'd_model', 'memory', 'nlm_hidden', 'd_input', 'n_sync_out',
                                  'n_sync_action', 'self_pairs')}
    p_ctm = _ctm_params(cfg)
    h_gru = _closest_width(lambda h: _gru_params(L, d, h), p_ctm, 4096)
    h_mlp = _closest_width(lambda h: _mlp_params(L, h), p_ctm, 1_000_000)
    budget = {'mini_ctm': p_ctm, 'gru': _gru_params(L, d, h_gru), 'mlp': _mlp_params(L, h_mlp)}
    gaps = {k: abs(v - p_ctm) / p_ctm for k, v in budget.items()}
    if max(gaps.values()) > 0.10:
        raise ValueError(f'cannot match parameter counts within 10 %: {budget}')
    makers: dict[str, Callable[[int], Any]] = {
        'mini_ctm': lambda s: classes['MiniCTM'](L, seed=s, **ctm_kw),
        'gru': lambda s: classes['AttentiveGRU'](L, hidden=h_gru, ticks=cfg['ticks'], d_input=d, seed=s),
        'mlp': lambda s: classes['ParityMLP'](L, hidden=h_mlp, seed=s),
    }
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in makers}
    prev_threads = torch.get_num_threads()
    t_start = time.perf_counter()
    try:
        torch.set_num_threads(1)
        for s in seeds:
            raw = cumulative_parity_data(L, s, cfg['heldout'])
            data = {k: torch.as_tensor(v, dtype=torch.long) for k, v in raw.items()}
            n_train = raw['train_x'].shape[0]
            batches = np.random.default_rng(s + _BATCH_OFFSET).integers(0, n_train, (cfg['steps'], cfg['batch']))
            for name, make in makers.items():
                model = make(s)
                count = _trainable(model)
                if count != budget[name]:
                    raise RuntimeError(f'{name} parameter count {count} != analytic {budget[name]}')
                results[name].append({'seed': s, **_train_model(torch, model, data, batches, cfg)})
    finally:
        torch.set_num_threads(prev_threads)
    total_wall = time.perf_counter() - t_start

    def acc(name: str) -> list[float]:
        return [r['heldout']['accuracy'] for r in results[name]]

    def mean(xs: Sequence[float]) -> float:
        return float(np.mean(xs))

    means = {name: mean(acc(name)) for name in results}
    best = max(('gru', 'mlp'), key=lambda name: (means[name], name == 'gru'))
    comparisons = {'mini_ctm_vs_gru': harness.compare(acc('mini_ctm'), acc('gru')),
                   'mini_ctm_vs_mlp': harness.compare(acc('mini_ctm'), acc('mlp'))}
    comparisons['mini_ctm_vs_best_baseline'] = {**comparisons[f'mini_ctm_vs_{best}'], 'best_baseline': best}
    verdict = comparisons['mini_ctm_vs_best_baseline']['verdict']
    models: dict[str, Any] = {}
    for name, runs in results.items():
        L_pos = len(runs[0]['heldout']['per_position_accuracy'])
        models[name] = {
            'trainable_parameters': budget[name], 'per_seed': runs,
            'summary': {'heldout_accuracy': means[name],
                        'heldout_sequence_accuracy': mean([r['heldout']['sequence_accuracy'] for r in runs]),
                        'heldout_last_tick_accuracy': mean([r['heldout']['last_tick_accuracy'] for r in runs]),
                        'heldout_per_position_accuracy': [mean([r['heldout']['per_position_accuracy'][i] for r in runs])
                                                          for i in range(L_pos)],
                        'train_accuracy': mean([r['train']['accuracy'] for r in runs]),
                        'mean_most_certain_tick': mean([r['heldout']['mean_most_certain_tick'] for r in runs])}}
    models['gru']['hidden'] = h_gru
    models['mlp']['hidden'] = h_mlp
    fwd = {'mini_ctm': _ctm_forward_macs(cfg), 'gru': _gru_forward_macs(cfg, h_gru), 'mlp': _mlp_forward_macs(cfg, h_mlp)}
    receipt = harness.base_receipt('synchrony_ctm', 'standalone_model_comparison', {
        **cfg, 'quick': quick, 'seeds': list(seeds), 'task': 'cumulative_parity', 'presentation': 'whole_sequence',
        'optimizer': 'AdamW', 'readout': 'most_certain_tick', 'loss': 'ctm_min_loss_plus_max_certainty',
        'torch_threads': 1, 'torch_version': str(torch.__version__), 'train_pool_size': 2 ** L - cfg['heldout']})
    receipt.update({
        'models': models,
        'parameter_budget': {**budget, 'max_relative_gap': max(gaps.values()), 'matched_within_10pct': True},
        'comparisons': comparisons, 'verdict': verdict,
        'verdict_basis': f'mini_ctm_vs_best_baseline({best})_heldout_per_position_accuracy',
        'estimated_cost': {'forward_macs_per_sequence': fwd,
                           'training_macs': {k: int(3 * v * cfg['batch'] * cfg['steps']) for k, v in fwd.items()}},
        'measured_wallclock_s': {'total': total_wall},
        'notes': ('Toy-scale study inspired by the CTM parity experiment (Darlow et al., NeurIPS 2025, Sec. 6 and '
                  'App. F), which used 64-bit sequences, d_model 1024, up to 100 ticks and 200k AdamW steps. Here '
                  f'L = {L}, d_model = {cfg["d_model"]}, {cfg["ticks"]} ticks and {cfg["steps"]} AdamW steps of '
                  f'batch {cfg["batch"]} at a constant learning rate on one CPU thread, so this measures early '
                  'learning of short parities, not the paper\'s result. The GRU baseline uses the same '
                  'attention reader and the same tick loss (the paper trained its LSTM baselines on the final '
                  'tick); the MLP sees the whole +/-1 sequence at once. Held-out sequences are disjoint from '
                  'training sequences; for parity, memorising neighbours predicts the wrong held-out parity, so '
                  'held-out accuracy can fall below 0.5. The best baseline is chosen after the fact on held-out '
                  'accuracy, which favours the baseline. Accuracy is per position (mean over positions and '
                  'held-out sequences).'),
    })
    return harness.seal(receipt)


# ====================================================================== NEUROWEAVE extension experiment
_EXT_DEFAULTS: dict[str, Any] = {'pairs': 32, 'self_pairs': None, 'timescales': (1.0, 32.0), 'feature_scale': 'unit'}
_RUN_DEFAULTS: dict[str, Any] = {'task': 'rhythm', 'heldout_seeds': (100, 101), 'steps': 300, 'heldout_steps': None,
                                 'neurons': 48, 'topology': 'clustered', 'mechanisms': ()}


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Baseline vs ``SynchronyExtension`` readout features vs the random-trace capacity control on a NEUROWEAVE
    temporal task (default ``rhythm``), through ``harness.run_extension_experiment``.

    Extension overrides: ``pairs``, ``self_pairs``, ``timescales``, ``feature_scale``. Harness overrides:
    ``task``, ``heldout_seeds``, ``steps``, ``heldout_steps``, ``neurons``, ``topology``, ``mechanisms``.
    Quick mode uses the harness ``QUICK_ARGS`` sizes (except the seeds, which are the caller's) and 16 pairs."""
    from . import harness
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seeds = _check_seeds(seeds)
    ext_cfg, run_cfg = dict(_EXT_DEFAULTS), dict(_RUN_DEFAULTS)
    if quick:
        ext_cfg['pairs'] = 16
        run_cfg.update({k: v for k, v in harness.QUICK_ARGS.items() if k != 'seeds'})
    unknown = sorted(set(overrides) - set(ext_cfg) - set(run_cfg))
    if unknown:
        raise ValueError(f'unknown run_experiment overrides: {unknown}')
    for key, value in overrides.items():
        (ext_cfg if key in ext_cfg else run_cfg)[key] = value
    probe = SynchronyExtension(ext_cfg['pairs'], seed=0, self_pairs=ext_cfg['self_pairs'],
                               timescales=ext_cfg['timescales'], feature_scale=ext_cfg['feature_scale'])
    ext_cfg['self_pairs'] = probe.self_pairs
    ext_cfg['timescales'] = list(probe.timescales)

    def factory(seed: int) -> SynchronyExtension:
        return SynchronyExtension(ext_cfg['pairs'], seed=seed, self_pairs=ext_cfg['self_pairs'],
                                  timescales=ext_cfg['timescales'], feature_scale=ext_cfg['feature_scale'])

    t0 = time.perf_counter()
    receipt = harness.run_extension_experiment(
        MODULE_ID, factory, seeds=seeds, control='random_trace', metric='accuracy', notes=(
            'Readout-only extension: decayed pairwise synchronisation of brain unit outputs (CTM-style, fixed '
            'log-uniform per-pair decays, self pairs included) is appended to the shared logistic readout. It never '
            'changes the brain drive. The control gets the same number of random leaky-trace features. '
            "feature_scale='unit' multiplies S by the constant sqrt(1 - decay) per pair, which keeps features in "
            '[-1, 1] like the control. Small synthetic task; the verdict is a paired 95 % t interval over seeds.'),
        **run_cfg)
    receipt = dict(receipt)
    receipt['config'] = {**receipt['config'], 'quick': quick, 'extension': ext_cfg}
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t0}
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment,
                                                          f'{MODULE_ID}_ctm': run_ctm_benchmark}
