"""Continuum memory: linear associative memories updated at geometrically spaced periods.

Inspired by the *continuum memory system* (CMS) of Nested Learning (Behrouz, Razaviyayn, Zhong & Mirrokni,
"Nested Learning: The Illusion of Deep Learning Architectures", NeurIPS 2025; arXiv:2512.24695, Dec 2025) and by
the neural long-term memory of Titans (Behrouz, Zhong & Mirrokni, "Titans: Learning to Memorize at Test Time",
NeurIPS 2025; arXiv:2501.00663, Dec 2024). It is a small synthetic abstraction, not a reproduction of either paper:
linear memories instead of MLP blocks, fixed instead of learned data-dependent gates, no attention and no language
modelling.

Model
-----
``levels`` linear associative memories ``M_l`` (``value_dim x key_dim``) form an *independent (head-wise)* CMS
(Nested Learning Eq. 74) with a sum aggregation. Level ``l`` reads its own key ``k_l`` (one key per level, i.e.
a level-specific key projection, or one key shared by every level)::

    read:  r_l = M_l k_l      (``read`` returns every r_l)      y = sum_l r_l      (``predict``)
    loss:  L_t = ||y_t - v_t||^2        grad_l = dL_t/dM_l = 2 (y_t - v_t) k_{l,t}^T

Level ``l`` has period ``P_l = base_period * 2**l``. It accumulates ``grad_l`` over its chunk of ``P_l`` writes,
each evaluated with the memories as they are at that write, and applies the chunk gradient ``G_l`` only when the
chunk closes (Nested Learning Eq. 71). ``G_l`` is the chunk *sum* (Eq. 71 with a constant learning rate) or, with
``chunk_reduce='mean'``, the chunk mean (Eq. 71 with ``lr / P_l``). The update uses Titans' surprise momentum and
forgetting gate (Titans Eqs. 10 and 13) with constant ``momentum`` (Titans' ``eta``) and ``lr`` (``theta``)::

    S_l <- momentum * S_l - lr * G_l
    M_l <- (1 - alpha_l) * M_l + S_l
    alpha_l = min(max_forget, forget + surprise_forget * mean_chunk_surprise)

Titans learns ``alpha_t`` as a function of the input; here it is a fixed number, optionally raised by this module's
own bounded surprise score ``s_t = e_t / (e_t + ||v_t||^2)``, ``e_t = ||y_t - v_t||^2`` (0 for a perfect read, 1/2
for an empty memory, towards 1 for a confidently wrong read; Titans' "momentary surprise" is the gradient term
itself). ``M`` and ``S`` are clipped to ``[-bound, bound]`` and every array has a fixed shape, so the state stays
finite and bounded.

Engine extension
----------------
``ContinuumMemoryExtension`` stores the predictive association *previous fused task+sensory context -> current
sensory frame* (only when ``ctx['learn']``) and exposes a fixed random projection of the per-level reads for the
current context (a multi-timescale forecast of the next frame) as readout features. It never changes the drive.

Benchmark
---------
``run_recall_benchmark`` streams key -> value associations: some recur after short gaps, some after long gaps,
and one-off associations fill the remaining steps as interference. It compares the CMS with a single-level
Titans-style memory holding the same number of entries (one ``value_dim x (levels * key_dim)`` memory reading the
concatenated key) and with a no-memory predictor (running mean of the values), with hyperparameters tuned per
arm on separate tuning seeds.
"""
from __future__ import annotations

import copy
import itertools
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..neuroweave.spec import SENSORY_DIM, TASK_INPUT_DIM
from .extension import EngineExtension

__all__ = ['MODULE_ID', 'RECALL_ID', 'SOURCE', 'EXPERIMENTS', 'CHUNK_REDUCTIONS', 'KIND_CODES', 'ContinuumMemory',
           'ContinuumMemoryExtension', 'make_recall_stream', 'run_recall_benchmark', 'run_experiment',
           'RECALL_DEFAULTS', 'RECALL_QUICK']

MODULE_ID = 'continuum_memory'
RECALL_ID = 'continuum_memory_recall'
SOURCE = ('Behrouz, Razaviyayn, Zhong & Mirrokni, "Nested Learning: The Illusion of Deep Learning Architectures", '
          'NeurIPS 2025, arXiv:2512.24695 (continuum memory system); Behrouz, Zhong & Mirrokni, "Titans: Learning to '
          'Memorize at Test Time", NeurIPS 2025, arXiv:2501.00663 (surprise momentum and forgetting gate)')
CONTEXT_DIM = TASK_INPUT_DIM + SENSORY_DIM
CHUNK_REDUCTIONS = ('sum', 'mean')
KIND_CODES = {'fresh': 0, 'first': 1, 'short': 2, 'long': 3, 'remap': 4}
_KEY_OFFSET = 51_407
_FEATURE_OFFSET = 51_419
_STREAM_OFFSET = 51_431
_MAX_ABS_INPUT = 1e6
# Far above any accumulator reachable from valid writes (inputs <= 1e6, |M| <= bound <= 1e6, chunks <= 2**23 writes
# give |acc| < 1e30) and far below float64 overflow, so an imported accumulator can never turn the state into inf/NaN.
_MAX_ACCUMULATOR = 1e100


# ====================================================================== validation helpers
def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return value


def _check_float(name: str, value: Any, low: float, high: float, *, high_open: bool = False,
                 low_open: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number')
    v = float(value)
    ok = math.isfinite(v) and (v > low if low_open else v >= low) and (v < high if high_open else v <= high)
    if not ok:
        raise ValueError(f'{name} must be in {"(" if low_open else "["}{low}, {high}{")" if high_open else "]"}')
    return v


def _finite(name: str, value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not np.all(np.isfinite(arr)) or float(np.max(np.abs(arr), initial=0.0)) > _MAX_ABS_INPUT:
        raise ValueError(f'{name} must be finite with magnitude <= {_MAX_ABS_INPUT:g}')
    return arr


def _vector(name: str, value: Any, size: int) -> np.ndarray:
    arr = _finite(name, value)
    if arr.shape != (size,):
        raise ValueError(f'{name} must have shape ({size},), got {arr.shape}')
    return arr


def _array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric array') from exc
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be a finite array of shape {shape}')
    return arr


def _check_seeds(name: str, seeds: Any) -> tuple[int, ...]:
    try:
        out = tuple(seeds)
    except TypeError as exc:
        raise ValueError(f'{name} must be a sequence of ints') from exc
    if not out or len(set(out)) != len(out):
        raise ValueError(f'{name} must be non-empty and unique')
    for s in out:
        _check_int(f'{name} entry', s, 0, 2**31 - 1)
    return out


def _check_pair(name: str, value: Any, low: int, high: int) -> tuple[int, int]:
    try:
        lo, hi = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a (low, high) pair of ints') from exc
    _check_int(f'{name}[0]', lo, low, high)
    _check_int(f'{name}[1]', hi, lo, high)
    return int(lo), int(hi)


def _check_grid(name: str, value: Any, low: float, high: float, *, high_open: bool = False) -> tuple[float, ...]:
    try:
        out = tuple(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a sequence of numbers') from exc
    if not 1 <= len(out) <= 16:
        raise ValueError(f'{name} must hold 1..16 values')
    return tuple(_check_float(f'{name} entry', v, low, high, high_open=high_open) for v in out)


def _grid_edges(chosen: Mapping[str, float], grids: Mapping[str, Sequence[float]],
                natural: Mapping[str, Sequence[float]] | None = None) -> list[str]:
    """Tuned hyperparameters (grids with at least two distinct values) whose chosen value is the smallest or largest
    grid value, so the best setting may lie outside the grid. A value in ``natural[name]`` (a natural bound such as
    momentum 0 = no momentum) is not counted as an edge."""
    natural = natural or {}
    out = []
    for name, grid in grids.items():
        values = sorted({float(v) for v in grid})
        value = float(chosen[name])
        if len(values) >= 2 and value in (values[0], values[-1]) and value not in {float(v) for v in natural.get(name, ())}:
            out.append(name)
    return out


# ====================================================================== memory
class ContinuumMemory:
    """``levels`` linear associative memories with periods ``base_period * 2**l`` (see the module docstring).

    Keys may be one ``(key_dim,)`` vector shared by every level or a ``(levels, key_dim)`` array with one key
    per level. With a shared key and ``chunk_reduce='sum'`` several levels can apply the same error in one
    write, so a smaller ``lr`` (about ``1 / levels`` of the per-level-key value) keeps the joint update stable;
    the ``bound`` clip keeps the state finite either way. The memory has no randomness."""

    def __init__(self, key_dim: int, value_dim: int, levels: int = 4, base_period: int = 1, lr: float = 0.5,
                 momentum: float = 0.0, forget: float = 0.01, *, surprise_forget: float = 0.0,
                 max_forget: float = 0.5, chunk_reduce: str = 'sum', bound: float = 100.0) -> None:
        self.key_dim = _check_int('key_dim', key_dim, 1, 4096)
        self.value_dim = _check_int('value_dim', value_dim, 1, 4096)
        self.levels = _check_int('levels', levels, 1, 12)
        self.base_period = _check_int('base_period', base_period, 1, 4096)
        if self.key_dim * self.value_dim * self.levels > 4_000_000:
            raise ValueError('levels * key_dim * value_dim must be <= 4,000,000')
        self.lr = _check_float('lr', lr, 0.0, 10.0)
        self.momentum = _check_float('momentum', momentum, 0.0, 1.0, high_open=True)
        self.forget = _check_float('forget', forget, 0.0, 1.0, high_open=True)
        self.surprise_forget = _check_float('surprise_forget', surprise_forget, 0.0, 1.0)
        self.max_forget = _check_float('max_forget', max_forget, 0.0, 1.0, high_open=True)
        if self.max_forget < self.forget:
            raise ValueError('max_forget must be >= forget')
        if chunk_reduce not in CHUNK_REDUCTIONS:
            raise ValueError(f'chunk_reduce must be one of {CHUNK_REDUCTIONS}')
        self.chunk_reduce = chunk_reduce
        self.bound = _check_float('bound', bound, 0.0, 1e6, low_open=True)
        self.periods: tuple[int, ...] = tuple(self.base_period * 2 ** level for level in range(self.levels))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Empty memories, momentum and chunk accumulators (a fresh memory)."""
        shape = (self.levels, self.value_dim, self.key_dim)
        self.M = np.zeros(shape)
        self.S = np.zeros(shape)
        self._acc = np.zeros(shape)
        self._acc_surprise = np.zeros(self.levels)
        self._acc_count = np.zeros(self.levels, dtype=np.int64)
        self.writes = 0
        self.update_counts = np.zeros(self.levels, dtype=np.int64)
        self.last_alpha = np.full(self.levels, self.forget)
        self.last_surprise = 0.0
        self.last_error = 0.0

    @property
    def parameter_count(self) -> int:
        """Number of memory entries (the fast weights ``M``)."""
        return int(self.M.size)

    @property
    def state_numbers(self) -> int:
        """All stored numbers: memories, momentum, chunk accumulators and counters."""
        return int(self.M.size + self.S.size + self._acc.size + self._acc_surprise.size + self._acc_count.size
                   + self.update_counts.size + self.last_alpha.size + 3)

    def config(self) -> dict[str, Any]:
        return {'key_dim': self.key_dim, 'value_dim': self.value_dim, 'levels': self.levels,
                'base_period': self.base_period, 'lr': self.lr, 'momentum': self.momentum, 'forget': self.forget,
                'surprise_forget': self.surprise_forget, 'max_forget': self.max_forget,
                'chunk_reduce': self.chunk_reduce, 'bound': self.bound}

    # ------------------------------------------------------------------ read / write
    def _keys(self, name: str, key: Any) -> np.ndarray:
        arr = _finite(name, key)
        if arr.shape == (self.key_dim,):
            return np.broadcast_to(arr, (self.levels, self.key_dim))
        if arr.shape != (self.levels, self.key_dim):
            raise ValueError(f'{name} must have shape ({self.key_dim},) or ({self.levels}, {self.key_dim}), '
                             f'got {arr.shape}')
        return arr

    def read(self, query: Any) -> np.ndarray:
        """Every level's read ``M_l q_l`` as a ``(levels, value_dim)`` array (no state change)."""
        return np.einsum('lvk,lk->lv', self.M, self._keys('query', query))

    def predict(self, query: Any) -> np.ndarray:
        """Combined read ``sum_l M_l q_l``."""
        return self.read(query).sum(axis=0)

    def write(self, key: Any, value: Any) -> dict[str, Any]:
        """One write: score the joint read, accumulate every level's gradient, apply the levels that are due."""
        ks = self._keys('key', key)
        v = _vector('value', value, self.value_dim)
        err = np.einsum('lvk,lk->v', self.M, ks) - v
        e = float(err @ err)
        surprise = e / (e + float(v @ v)) if e > 0.0 else 0.0
        self._acc += 2.0 * err[None, :, None] * ks[:, None, :]
        self._acc_surprise += surprise
        self._acc_count += 1
        self.writes += 1
        updated: list[int] = []
        for level, period in enumerate(self.periods):
            if self.writes % period:
                continue
            n = int(self._acc_count[level])
            grad = self._acc[level] / n if self.chunk_reduce == 'mean' else self._acc[level]
            alpha = min(self.max_forget, self.forget + self.surprise_forget * float(self._acc_surprise[level]) / n)
            self.S[level] = np.clip(self.momentum * self.S[level] - self.lr * grad, -self.bound, self.bound)
            self.M[level] = np.clip((1.0 - alpha) * self.M[level] + self.S[level], -self.bound, self.bound)
            self._acc[level] = 0.0
            self._acc_surprise[level] = 0.0
            self._acc_count[level] = 0
            self.update_counts[level] += 1
            self.last_alpha[level] = alpha
            updated.append(level)
        self.last_surprise = surprise
        self.last_error = e / self.value_dim
        return {'error': self.last_error, 'surprise': surprise, 'updated_levels': updated}

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        return {'config': self.config(), 'M': self.M.tolist(), 'S': self.S.tolist(), 'acc': self._acc.tolist(),
                'acc_surprise': self._acc_surprise.tolist(), 'acc_count': self._acc_count.tolist(),
                'writes': int(self.writes), 'update_counts': self.update_counts.tolist(),
                'last_alpha': self.last_alpha.tolist(), 'last_surprise': float(self.last_surprise),
                'last_error': float(self.last_error)}

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate everything first, then replace the state (the configuration must match)."""
        if not isinstance(state, Mapping):
            raise ValueError('continuum memory state must be a mapping')
        if state.get('config') != self.config():
            raise ValueError('continuum memory state was made with a different configuration')
        shape = (self.levels, self.value_dim, self.key_dim)
        M = _array('M', state.get('M'), shape)
        S = _array('S', state.get('S'), shape)
        acc = _array('acc', state.get('acc'), shape)
        acc_s = _array('acc_surprise', state.get('acc_surprise'), (self.levels,))
        alpha = _array('last_alpha', state.get('last_alpha'), (self.levels,))
        if np.max(np.abs(M), initial=0.0) > self.bound or np.max(np.abs(S), initial=0.0) > self.bound:
            raise ValueError('memory state exceeds the configured bound')
        if np.max(np.abs(acc), initial=0.0) > _MAX_ACCUMULATOR:
            raise ValueError('gradient accumulator is out of range')
        writes = _check_int('writes', state.get('writes'), 0, 2**62)
        raw_counters = (state.get('acc_count'), state.get('update_counts'))
        if not all(isinstance(c, (list, tuple, np.ndarray)) and np.ndim(c) == 1 and all(
                isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) for x in c)
                   for c in raw_counters):
            raise ValueError('counters must be one-dimensional sequences of ints')
        try:
            counts = np.asarray(raw_counters[0], dtype=np.int64)
            upd = np.asarray(raw_counters[1], dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('counters must be integer arrays') from exc
        expected_counts = np.array([writes % p for p in self.periods], dtype=np.int64)
        expected_upd = np.array([writes // p for p in self.periods], dtype=np.int64)
        if counts.shape != (self.levels,) or not np.array_equal(counts, expected_counts):
            raise ValueError('chunk counters do not match the write count')
        if upd.shape != (self.levels,) or not np.array_equal(upd, expected_upd):
            raise ValueError('update counters do not match the write count')
        if np.any(acc_s < 0.0) or np.any(acc_s > counts) or np.any(alpha < 0.0) or np.any(alpha > self.max_forget):
            raise ValueError('surprise accumulators / forgetting gates out of range')
        if np.any(acc[counts == 0] != 0.0):
            raise ValueError('a level with an empty chunk must have a zero gradient accumulator')
        ls = _check_float('last_surprise', state.get('last_surprise'), 0.0, 1.0)
        le = _check_float('last_error', state.get('last_error'), 0.0, 1e300)
        self.M, self.S, self._acc = M.copy(), S.copy(), acc.copy()
        self._acc_surprise, self._acc_count, self.update_counts = acc_s.copy(), counts.copy(), upd.copy()
        self.last_alpha = alpha.copy()
        self.writes = int(writes)
        self.last_surprise, self.last_error = ls, le


# ====================================================================== engine extension
def _fit(vec: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size)
    m = min(size, vec.size)
    out[:m] = vec[:m]
    return out


class ContinuumMemoryExtension(EngineExtension):
    """Readout-only NEUROWEAVE extension around a :class:`ContinuumMemory`.

    Context ``c_t`` = task input (fitted to ``TASK_INPUT_DIM``; zeros when absent) followed by the fused sensory
    frame. The key is a fixed binary random-feature map ``k = sign(R c + b) / sqrt(levels * key_dim)`` split
    into one ``key_dim`` block per level (unit norm overall). After every engine step the extension writes
    ``key(c_{t-1}) -> sensory_t`` (only when ``ctx['learn']``; never before a previous context exists, e.g.
    right after :meth:`reset`) and then reads every level with ``key(c_t)``. ``features()`` are
    ``features_per_level`` fixed unit-norm projections of each level's 16-channel read, clipped to [-10, 10].
    The brain output is not used and the drive is never changed."""

    id = MODULE_ID

    def __init__(self, levels: int = 4, key_dim: int = 8, base_period: int = 1, lr: float = 0.25,
                 momentum: float = 0.0, forget: float = 0.01, *, features_per_level: int = 2,
                 surprise_forget: float = 0.0, max_forget: float = 0.5, chunk_reduce: str = 'sum',
                 seed: int = 0) -> None:
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        self.features_per_level = _check_int('features_per_level', features_per_level, 1, 64)
        self.memory = ContinuumMemory(key_dim, SENSORY_DIM, levels, base_period, lr, momentum, forget,
                                      surprise_forget=surprise_forget, max_forget=max_forget,
                                      chunk_reduce=chunk_reduce)
        self.levels = self.memory.levels
        self.key_dim = self.memory.key_dim
        self.feature_dim = self.levels * self.features_per_level
        if self.feature_dim > 4096:
            raise ValueError('levels * features_per_level must be <= 4096')
        rng = np.random.default_rng(self.seed + _KEY_OFFSET)
        self._R = rng.standard_normal((self.levels * self.key_dim, CONTEXT_DIM))
        self._b = 0.5 * rng.standard_normal(self.levels * self.key_dim)
        prng = np.random.default_rng(self.seed + _FEATURE_OFFSET)
        proj = prng.standard_normal((self.levels, self.features_per_level, SENSORY_DIM))
        self._P = proj / np.linalg.norm(proj, axis=2, keepdims=True)
        self._key_scale = 1.0 / math.sqrt(self.levels * self.key_dim)
        self._prev_key: np.ndarray | None = None
        self._features = np.zeros(self.feature_dim)
        self._steps = 0
        self._frozen_steps = 0

    # ------------------------------------------------------------------ lifecycle
    def bind(self, engine: Any) -> None:
        """Nothing is sized from the engine (the context has a fixed width); start a fresh episode."""
        self.reset()

    def reset(self) -> None:
        self._prev_key = None
        self._features = np.zeros(self.feature_dim)

    # ------------------------------------------------------------------ hooks
    def key(self, context: Any) -> np.ndarray:
        """``(levels, key_dim)`` binary random-feature key of a ``CONTEXT_DIM`` context vector."""
        c = _vector('context', context, CONTEXT_DIM)
        z = self._R @ c + self._b
        return (np.where(z >= 0.0, 1.0, -1.0) * self._key_scale).reshape(self.levels, self.key_dim)

    def after_step(self, output: np.ndarray, ctx: Mapping[str, Any]) -> None:
        if not isinstance(ctx, Mapping):
            raise ValueError('continuum memory needs the engine step context')
        learn = ctx.get('learn')
        if type(learn) is not bool:
            raise ValueError("ctx['learn'] must be a boolean")
        sensory = _vector("ctx['sensory']", ctx.get('sensory'), SENSORY_DIM)
        task = ctx.get('task_input')
        task_vec = np.zeros(TASK_INPUT_DIM) if task is None else _fit(_finite("ctx['task_input']", task).ravel(),
                                                                        TASK_INPUT_DIM)
        key = self.key(np.concatenate([task_vec, sensory]))
        if learn and self._prev_key is not None:
            self.memory.write(self._prev_key, sensory)
        elif not learn:
            self._frozen_steps += 1
        reads = self.memory.read(key)
        self._features = np.clip(np.einsum('lfv,lv->lf', self._P, reads).ravel(), -10.0, 10.0)
        self._prev_key = key
        self._steps += 1

    def features(self) -> np.ndarray:
        return self._features.copy()

    # ------------------------------------------------------------------ accounting / persistence
    def parameter_count(self) -> int:
        return self.memory.parameter_count

    def state_numbers(self) -> int:
        return int(self.memory.state_numbers + self._R.size + self._b.size + self._P.size
                   + self.levels * self.key_dim + self.feature_dim + 2)

    def telemetry(self) -> dict[str, Any]:
        mem = self.memory
        return {'id': self.id, 'levels': self.levels, 'periods': list(mem.periods), 'writes': int(mem.writes),
                'update_counts': mem.update_counts.tolist(), 'steps': self._steps, 'frozen_steps': self._frozen_steps,
                'last_surprise': float(mem.last_surprise), 'last_error': float(mem.last_error),
                'last_alpha': mem.last_alpha.tolist(),
                'level_norms': [float(np.linalg.norm(mem.M[level])) for level in range(self.levels)]}

    def _config(self) -> dict[str, Any]:
        mem = self.memory
        return {'levels': self.levels, 'key_dim': self.key_dim, 'base_period': mem.base_period, 'lr': mem.lr,
                'momentum': mem.momentum, 'forget': mem.forget, 'features_per_level': self.features_per_level,
                'surprise_forget': mem.surprise_forget, 'max_forget': mem.max_forget,
                'chunk_reduce': mem.chunk_reduce, 'seed': self.seed}

    def export_state(self) -> dict[str, Any]:
        return {'id': self.id, 'config': self._config(), 'memory': self.memory.export_state(),
                'prev_key': None if self._prev_key is None else self._prev_key.tolist(),
                'features': self._features.tolist(), 'steps': self._steps, 'frozen_steps': self._frozen_steps}

    def import_state(self, state: Mapping[str, Any]) -> None:
        super().import_state(state)
        if state.get('config') != self._config():
            raise ValueError('continuum memory extension state was made with a different configuration')
        memory = copy.deepcopy(self.memory)
        memory.import_state(state.get('memory'))          # validates fully before anything is replaced
        prev = state.get('prev_key')
        if prev is not None:
            prev = _array('prev_key', prev, (self.levels, self.key_dim))
            if not np.allclose(np.abs(prev), self._key_scale, rtol=0.0, atol=1e-12):
                raise ValueError('prev_key is not a binary random-feature key')
        feats = _array('features', state.get('features'), (self.feature_dim,))
        if np.max(np.abs(feats), initial=0.0) > 10.0:
            raise ValueError('features out of range')
        steps = _check_int('steps', state.get('steps'), 0, 2**62)
        frozen = _check_int('frozen_steps', state.get('frozen_steps'), 0, steps)
        self.memory = memory
        self._prev_key = None if prev is None else np.where(prev > 0.0, self._key_scale, -self._key_scale)
        self._features = feats.copy()
        self._steps, self._frozen_steps = steps, frozen


# ====================================================================== recall benchmark
def make_recall_stream(seed: int, steps: int, *, key_dim: int = 128, value_dim: int = 8, n_short: int = 4,
                       n_long: int = 8, short_gap: Sequence[int] = (3, 10), long_gap: Sequence[int] = (40, 80),
                       remap_prob: float = 0.0) -> dict[str, np.ndarray]:
    """Deterministic key -> value stream with recurring associations and one-off interference.

    ``n_short`` items recur after gaps drawn from ``short_gap`` and ``n_long`` items after gaps from
    ``long_gap`` (inclusive, in steps). When several items are due, the earliest-due one is presented (ties: lowest
    index) and the others wait, so realised gaps can be longer than drawn. Steps without a due item present a fresh
    one-off association. With ``remap_prob`` a short-gap item gets a new value at a recurrence (kind ``remap``:
    a write of a changed association, not a recall query). Keys are random unit vectors, values standard normal.
    Returns ``keys (T, key_dim)``, ``values (T, value_dim)``, ``kind (T,)`` (codes in ``KIND_CODES``),
    ``item (T,)`` (-1 for fresh) and ``gap (T,)`` (0 unless the step is a recurrence)."""
    _check_int('seed', seed, 0, 2**31 - 1)
    _check_int('steps', steps, 1, 1_000_000)
    _check_int('key_dim', key_dim, 1, 65_536)
    _check_int('value_dim', value_dim, 1, 4096)
    _check_int('n_short', n_short, 0, 1024)
    _check_int('n_long', n_long, 0, 1024)
    sg = _check_pair('short_gap', short_gap, 1, 1_000_000)
    lg = _check_pair('long_gap', long_gap, 1, 1_000_000)
    p_remap = _check_float('remap_prob', remap_prob, 0.0, 1.0)
    rng = np.random.default_rng(seed + _STREAM_OFFSET)
    n_items = n_short + n_long
    item_keys = rng.standard_normal((n_items, key_dim))
    item_keys /= np.linalg.norm(item_keys, axis=1, keepdims=True)
    item_values = rng.standard_normal((n_items, value_dim))
    gaps = [sg] * n_short + [lg] * n_long
    due = np.array([int(rng.integers(0, g[1])) for g in gaps], dtype=np.int64)
    last = np.full(n_items, -1, dtype=np.int64)
    keys = np.zeros((steps, key_dim))
    values = np.zeros((steps, value_dim))
    kind = np.zeros(steps, dtype=np.int8)
    item = np.full(steps, -1, dtype=np.int64)
    gap = np.zeros(steps, dtype=np.int64)
    for t in range(steps):
        cand = np.flatnonzero(due <= t) if n_items else np.zeros(0, dtype=np.int64)
        if cand.size:
            i = int(cand[np.argmin(due[cand])])
            if last[i] < 0:
                code = KIND_CODES['first']
            elif i < n_short:
                code = KIND_CODES['short']
                if p_remap > 0.0 and rng.random() < p_remap:
                    item_values[i] = rng.standard_normal(value_dim)
                    code = KIND_CODES['remap']
            else:
                code = KIND_CODES['long']
            keys[t], values[t], kind[t], item[t] = item_keys[i], item_values[i], code, i
            gap[t] = t - last[i] if last[i] >= 0 else 0
            last[i] = t
            due[i] = t + int(rng.integers(gaps[i][0], gaps[i][1] + 1))
        else:
            k = rng.standard_normal(key_dim)
            keys[t] = k / np.linalg.norm(k)
            values[t] = rng.standard_normal(value_dim)
    return {'keys': keys, 'values': values, 'kind': kind, 'item': item, 'gap': gap}


def _score(preds_err: list[float], kinds: list[int]) -> dict[str, Any]:
    err = np.asarray(preds_err)
    kd = np.asarray(kinds)
    out: dict[str, Any] = {}
    for name in ('short', 'long'):
        sel = err[kd == KIND_CODES[name]]
        out[f'{name}_mse'] = float(sel.mean()) if sel.size else None
        out[f'{name}_queries'] = int(sel.size)
    return out


def _run_memory(memory: ContinuumMemory, stream: Mapping[str, np.ndarray], split: bool) -> dict[str, Any]:
    """Prequential: predict the value from the key, score recurring items, then write."""
    errs, kinds = [], []
    shape = (memory.levels, memory.key_dim)
    for key, value, code in zip(stream['keys'], stream['values'], stream['kind']):
        k = key.reshape(shape) if split else key
        pred = memory.predict(k)
        if code in (KIND_CODES['short'], KIND_CODES['long']):
            errs.append(float(np.mean((pred - value) ** 2)))
            kinds.append(int(code))
        memory.write(k, value)
    return _score(errs, kinds)


def _run_mean_predictor(stream: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """No memory: predict the running mean of the values seen so far."""
    mean = np.zeros(stream['values'].shape[1])
    errs, kinds = [], []
    for n, (value, code) in enumerate(zip(stream['values'], stream['kind']), start=1):
        if code in (KIND_CODES['short'], KIND_CODES['long']):
            errs.append(float(np.mean((mean - value) ** 2)))
            kinds.append(int(code))
        mean = mean + (value - mean) / n
    return _score(errs, kinds)


def _overall(score: Mapping[str, Any]) -> float:
    vals = [score[k] for k in ('short_mse', 'long_mse') if score[k] is not None]
    return float(np.mean(vals)) if vals else float('inf')


# The grids bracket both arms' tuned optimum (measured 2026-11-19): learning rate and momentum trade off along a
# ridge of nearly equal effective step lr / (1 - momentum), and a grid that stops short of one arm's optimum on that
# ridge can hand the other arm a spurious win. Receipts flag a chosen value that sits on a grid edge.
RECALL_DEFAULTS: dict[str, Any] = {
    'steps': 800, 'levels': 4, 'key_dim': 32, 'value_dim': 8, 'base_period': 1,
    'n_short': 4, 'n_long': 8, 'short_gap': (3, 10), 'long_gap': (40, 80), 'remap_prob': 0.0,
    'chunk_reduce': 'sum', 'surprise_forget': 0.0, 'max_forget': 0.5,
    'lr_grid': (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5), 'momentum_grid': (0.0, 0.5, 0.75, 0.875, 0.9375),
    'forget_grid': (0.0, 0.01), 'tune_seeds': (1000, 1001)}
RECALL_QUICK: dict[str, Any] = {'steps': 240, 'lr_grid': (0.125, 0.25, 0.5), 'momentum_grid': (0.0, 0.25, 0.5),
                                'forget_grid': (0.0, 0.02), 'tune_seeds': (1000,)}
_NATURAL_BOUNDS = {'lr': (0.0,), 'momentum': (0.0,), 'forget': (0.0,)}


def run_recall_benchmark(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """CMS vs a single-level Titans-style memory with the same number of entries vs no memory.

    Both memories use the same keys (random unit vectors in ``levels * key_dim`` dimensions): CMS level ``l`` reads
    block ``l`` of the key, the single-level memory reads all of it, so both read functions are linear maps of the
    same key with ``value_dim * levels * key_dim`` entries and differ only in the update schedule. For each memory
    arm, ``(lr, momentum, forget)`` is chosen from the same grid by the mean of short- and long-gap recall MSE on
    ``tune_seeds`` (disjoint from ``seeds``); the chosen values are then evaluated on every seed. Verdict: paired
    95 % t interval on long-gap recall MSE (lower is better), CMS vs single-level memory."""
    from . import harness
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seeds = _check_seeds('seeds', seeds)
    cfg = dict(RECALL_DEFAULTS)
    if quick:
        cfg.update(RECALL_QUICK)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(f'unknown run_recall_benchmark overrides: {unknown}')
    cfg.update(overrides)
    _check_int('steps', cfg['steps'], 16, 200_000)
    tune_seeds = _check_seeds('tune_seeds', cfg['tune_seeds'])
    if set(tune_seeds) & set(seeds):
        raise ValueError('tune_seeds must be disjoint from seeds')
    for name, lo, hi, open_hi in (('lr_grid', 0.0, 10.0, False), ('momentum_grid', 0.0, 1.0, True),
                                  ('forget_grid', 0.0, 1.0, True)):
        cfg[name] = _check_grid(name, cfg[name], lo, hi, high_open=open_hi)
    cfg['short_gap'] = _check_pair('short_gap', cfg['short_gap'], 1, 1_000_000)
    cfg['long_gap'] = _check_pair('long_gap', cfg['long_gap'], 1, 1_000_000)
    cfg['max_forget'] = _check_float('max_forget', cfg['max_forget'], 0.0, 1.0, high_open=True)   # used in max()
    cfg['tune_seeds'] = tune_seeds
    levels, key_dim, value_dim = cfg['levels'], cfg['key_dim'], cfg['value_dim']

    def make(arm: str, lr: float, momentum: float, forget: float) -> ContinuumMemory:
        common = dict(surprise_forget=cfg['surprise_forget'], max_forget=max(cfg['max_forget'], forget),
                      chunk_reduce=cfg['chunk_reduce'])
        if arm == 'cms':
            return ContinuumMemory(key_dim, value_dim, levels, cfg['base_period'], lr, momentum, forget, **common)
        return ContinuumMemory(levels * key_dim, value_dim, 1, 1, lr, momentum, forget, **common)

    probe_cms, probe_single = make('cms', 0.5, 0.0, 0.0), make('single', 0.5, 0.0, 0.0)   # validates sizes
    if probe_cms.parameter_count != probe_single.parameter_count:
        raise AssertionError('capacity matching failed')
    stream_kw = dict(key_dim=levels * key_dim, value_dim=value_dim, n_short=cfg['n_short'], n_long=cfg['n_long'],
                     short_gap=cfg['short_gap'], long_gap=cfg['long_gap'], remap_prob=cfg['remap_prob'])
    t0 = time.perf_counter()
    tune_streams = [make_recall_stream(s, cfg['steps'], **stream_kw) for s in tune_seeds]
    grid = list(itertools.product(cfg['lr_grid'], cfg['momentum_grid'], cfg['forget_grid']))
    tuning: dict[str, Any] = {}
    chosen: dict[str, tuple[float, float, float]] = {}
    for arm in ('cms', 'single'):
        table = []
        for lr, momentum, forget in grid:
            score = float(np.mean([_overall(_run_memory(make(arm, lr, momentum, forget), st, arm == 'cms'))
                                   for st in tune_streams]))
            table.append({'lr': lr, 'momentum': momentum, 'forget': forget, 'tune_score': score})
        best = min(range(len(table)), key=lambda i: (table[i]['tune_score'], i))
        chosen[arm] = (table[best]['lr'], table[best]['momentum'], table[best]['forget'])
        picked = dict(zip(('lr', 'momentum', 'forget'), chosen[arm]))
        edges = _grid_edges(picked, {'lr': cfg['lr_grid'], 'momentum': cfg['momentum_grid'],
                                     'forget': cfg['forget_grid']}, _NATURAL_BOUNDS)
        tuning[arm] = {'chosen': picked, 'chosen_on_grid_edge': edges, 'table': table}
    tune_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    per_seed = []
    for s in seeds:
        stream = make_recall_stream(s, cfg['steps'], **stream_kw)
        row = {'seed': int(s),
               'cms': _run_memory(make('cms', *chosen['cms']), stream, True),
               'single': _run_memory(make('single', *chosen['single']), stream, False),
               'none': _run_mean_predictor(stream),
               'kinds': {name: int(np.sum(stream['kind'] == code)) for name, code in KIND_CODES.items()}}
        per_seed.append(row)
    eval_wall = time.perf_counter() - t1

    def vals(arm: str, metric: str) -> list[float]:
        return [float('nan') if r[arm][metric] is None else r[arm][metric] for r in per_seed]

    comparisons = {
        'cms_vs_single_long_gap': harness.compare(vals('cms', 'long_mse'), vals('single', 'long_mse'),
                                                  higher_is_better=False),
        'cms_vs_single_short_gap': harness.compare(vals('cms', 'short_mse'), vals('single', 'short_mse'),
                                                   higher_is_better=False),
        'cms_vs_none_long_gap': harness.compare(vals('cms', 'long_mse'), vals('none', 'long_mse'),
                                                higher_is_better=False),
        'single_vs_none_long_gap': harness.compare(vals('single', 'long_mse'), vals('none', 'long_mse'),
                                                   higher_is_better=False),
    }
    summary = {arm: {m: float(np.nanmean(vals(arm, m))) for m in ('short_mse', 'long_mse')}
               for arm in ('cms', 'single', 'none')}
    entries = probe_cms.parameter_count
    runs = len(grid) * len(tune_seeds) * 2 + 2 * len(seeds)
    receipt = harness.base_receipt(RECALL_ID, 'memory_benchmark', {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        'seeds': list(seeds), 'quick': quick, 'scored_kinds': ['short', 'long'],
        'tuning_objective': 'mean of short- and long-gap recall MSE on tune_seeds'})
    receipt.update({
        'module': MODULE_ID, 'source': SOURCE,
        'arms': {'cms': f'{levels}-level continuum memory, periods {list(probe_cms.periods)}, per-level key blocks',
                 'single': 'one Titans-style memory (momentum + forgetting) reading the full key, period 1',
                 'none': 'running mean of the values seen so far'},
        'tuning': tuning, 'per_seed': per_seed, 'summary': summary, 'comparisons': comparisons,
        'verdict': comparisons['cms_vs_single_long_gap']['verdict'],
        'verdict_basis': ('cms_vs_single_level_memory: paired 95% t interval over seeds on long-gap recall MSE '
                          '(lower is better); hyperparameters tuned per arm on disjoint tune_seeds'),
        'budget': {'cms_memory_entries': entries, 'single_memory_entries': probe_single.parameter_count,
                   'cms_state_numbers': probe_cms.state_numbers, 'single_state_numbers': probe_single.state_numbers},
        'estimated_cost': {'memory_runs': runs, 'steps_per_run': cfg['steps'],
                           'estimated_multiply_adds': int(runs * cfg['steps'] * 4 * entries)},
        'measured_wallclock_s': {'tuning': tune_wall, 'evaluation': eval_wall},
        'notes': ('Synthetic stream of random unit keys and Gaussian values; recall is scored prequentially (read '
                  'before write) at short-gap and long-gap recurrences only. The two memory arms hold the same '
                  'number of entries and read the same keys, so they differ only in the multi-period update '
                  'schedule (and per-level forgetting); the gates are fixed numbers, not the learned '
                  'data-dependent gates of Titans or Hope. Each memory arm gets its best point of the same grid on '
                  'separate tuning seeds; tuning.<arm>.chosen_on_grid_edge names any hyperparameter whose chosen '
                  'value is a grid edge (the optimum may then lie outside the grid). A linear toy: it says nothing '
                  'about language modelling or the published results.'),
    })
    return harness.seal(receipt)


# ====================================================================== NEUROWEAVE extension experiment
_EXT_DEFAULTS: dict[str, Any] = {'levels': 4, 'key_dim': 8, 'base_period': 1, 'lr': 0.25, 'momentum': 0.0,
                                 'forget': 0.01, 'features_per_level': 2, 'surprise_forget': 0.0, 'max_forget': 0.5,
                                 'chunk_reduce': 'sum'}
_RUN_DEFAULTS: dict[str, Any] = {'task': 'delayed_recall', 'heldout_seeds': (100, 101), 'steps': 300,
                                 'heldout_steps': None, 'neurons': 48, 'topology': 'clustered', 'mechanisms': ()}


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Baseline vs ``ContinuumMemoryExtension`` readout features vs the random-trace capacity control on a
    NEUROWEAVE temporal task (default ``delayed_recall``), through ``harness.run_extension_experiment``.

    Extension overrides: the keys of ``_EXT_DEFAULTS``. Harness overrides: ``task``, ``heldout_seeds``, ``steps``,
    ``heldout_steps``, ``neurons``, ``topology``, ``mechanisms``. Quick mode uses the harness ``QUICK_ARGS`` sizes
    (the seeds stay the caller's)."""
    from . import harness
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seeds = _check_seeds('seeds', seeds)
    ext_cfg, run_cfg = dict(_EXT_DEFAULTS), dict(_RUN_DEFAULTS)
    if quick:
        run_cfg.update({k: v for k, v in harness.QUICK_ARGS.items() if k != 'seeds'})
    unknown = sorted(set(overrides) - set(ext_cfg) - set(run_cfg))
    if unknown:
        raise ValueError(f'unknown run_experiment overrides: {unknown}')
    for key, value in overrides.items():
        (ext_cfg if key in ext_cfg else run_cfg)[key] = value
    ContinuumMemoryExtension(seed=0, **ext_cfg)                           # validate before running anything

    def factory(seed: int) -> ContinuumMemoryExtension:
        return ContinuumMemoryExtension(seed=seed, **ext_cfg)

    t0 = time.perf_counter()
    receipt = harness.run_extension_experiment(
        MODULE_ID, factory, seeds=seeds, control='random_trace', metric='accuracy', notes=(
            f'Readout-only extension: a {ext_cfg["levels"]}-level continuum memory stores previous task+sensory '
            'context -> current '
            'sensory frame (writes only while learning) and its per-level reads of the current context (a '
            'multi-timescale forecast of the next frame) are projected to readout features. It never changes the '
            'brain drive and does not read the brain output. The control gets the same number of random '
            'leaky-trace features of the brain output. Small synthetic task; the verdict is a paired 95 % t '
            'interval over seeds.'),
        **run_cfg)
    receipt = dict(receipt)
    receipt.pop('digest', None)
    receipt['config'] = {**receipt['config'], 'quick': quick, 'extension': ext_cfg}
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t0}
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment, RECALL_ID: run_recall_benchmark}
