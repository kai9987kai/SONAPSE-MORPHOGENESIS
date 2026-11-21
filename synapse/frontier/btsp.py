"""One-shot content-addressable memory with binary synapses and BTSP-style plasticity.

Inspired by Wu & Maass, "A simple model for Behavioral Time Scale Synaptic Plasticity (BTSP) provides content
addressable memory with binary synapses and one-shot learning", Nature Communications 16, 342 (published online
2 January 2025), https://doi.org/10.1038/s41467-024-55563-6. It is a small synthetic abstraction of that model, not
a reproduction of the paper's networks, parameters or results, and it says nothing about hippocampal biology.

Model (``BTSPMemory``)
----------------------
``m = input_dim`` binary input units feed ``n = memory_dim`` binary memory units through a fixed random
connectivity mask ``C`` (``n x m``, connection probability ``density``) carrying binary weights ``W`` (initially 0).
Storing a binary pattern ``x`` is one-shot (paper Eq. 1):

* every memory unit receives a plateau potential with probability ``plateau_prob``, independently of its firing;
* for every existing synapse ``(j, i)`` with a plateau in ``j`` and ``x_i = 1``, a weight at 0 flips to 1 with
  probability ``p_ltp`` and a weight at 1 flips to 0 with probability ``p_ltd`` (the paper uses 0.5 for both);
  nothing else changes.

Memory units are McCulloch-Pitts units, ``z_j = 1[(W x)_j > threshold * density * |x|]``, so the threshold is a
fraction of the expected number of connected active inputs. (The paper grid-searches one absolute threshold for its
memory-trace experiments and, for input completion, sets both layers' thresholds by fitted linear functions of the
average fraction of 1's; a threshold proportional to the active count is this module's simplification.)
Reconstruction uses the paper's feedback pathway: a second random mask ``D`` (``m x n``, same density) carries binary
weights ``B`` that are set to 1, right after each store, on every existing synapse whose memory unit is in the new
trace ``z(x)`` and whose input unit is active in ``x`` (one-shot Hebbian rule saturating at 1). Recall of a cue
``c`` gives the trace ``z(c)`` and the reconstruction ``r_i = 1[(B z(c))_i > feedback_threshold * density * |z(c)|]``.

``WillshawMemory`` is an auto-associative form of the classic clipped-Hebbian binary memory (Willshaw, Buneman &
Longuet-Higgins, Nature 222, 960-962, 1969; the paper does not compare BTSP with it, this module does) and
``RandomRecallControl`` returns a seeded random pattern (chance).
``run_capacity_benchmark`` compares them at the same synapse count; ``BTSPExtension`` attaches the memory to a
NEUROWEAVE engine as a readout-only extension.
"""
from __future__ import annotations

import base64
import binascii
import copy
import itertools
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .continuum_memory import _check_float, _check_int, _check_seeds, _grid_edges
from .extension import EngineExtension

__all__ = ['MODULE_ID', 'CAPACITY_ID', 'SOURCE', 'EXPERIMENTS', 'CUE_KINDS', 'BTSPMemory', 'WillshawMemory',
           'RandomRecallControl', 'BTSPExtension', 'overlap', 'hamming', 'reconstruction_error', 'make_cue',
           'random_patterns', 'run_capacity_benchmark', 'run_experiment', 'CAPACITY_DEFAULTS', 'CAPACITY_QUICK']

MODULE_ID = 'btsp_memory'
CAPACITY_ID = 'btsp_memory_capacity'
SOURCE = ('Wu & Maass, "A simple model for Behavioral Time Scale Synaptic Plasticity (BTSP) provides content '
          'addressable memory with binary synapses and one-shot learning", Nature Communications 16, 342 (2025), '
          'doi:10.1038/s41467-024-55563-6; baseline: Willshaw, Buneman & Longuet-Higgins, "Non-holographic '
          'associative memory", Nature 222, 960-962 (1969), doi:10.1038/222960a0')
CUE_KINDS = ('delete', 'flip')
_MAX_MATRIX = 4_000_000
_MASK_OFFSET = 61_403
_PLASTICITY_OFFSET = 61_417
_PROJECTION_OFFSET = 61_429
_PATTERN_OFFSET = 61_441
_TEST_OFFSET = 61_453
_CONTROL_OFFSET = 61_467


# ====================================================================== binary helpers and metrics
def _binary(name: str, value: Any, size: int) -> np.ndarray:
    """Validate a binary vector (bool or numbers in {0, 1}) of length ``size``; returns a new bool array."""
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a binary vector') from exc
    if arr.shape != (size,):
        raise ValueError(f'{name} must have shape ({size},), got {arr.shape}')
    if arr.dtype == np.bool_:
        return arr.copy()
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f'{name} must be a binary vector')
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f'{name} must contain only 0 and 1')
    return arr.astype(bool)


def _pair(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    size = np.shape(a)
    if len(size) != 1:
        raise ValueError('expected two binary vectors of the same length')
    return _binary('a', a, size[0]), _binary('b', b, size[0])


def overlap(a: Any, b: Any) -> float:
    """Normalised overlap ``|a & b| / sqrt(|a| |b|)`` of two binary vectors (cosine similarity; 0 if one is empty).

    This is the module's recall metric. It is not the paper's "overlap ratio" of two memory traces, which divides
    the common 1's by the average number of 1's in a trace."""
    x, y = _pair(a, b)
    nx, ny = int(np.count_nonzero(x)), int(np.count_nonzero(y))
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.count_nonzero(x & y) / math.sqrt(nx * ny))


def hamming(a: Any, b: Any) -> int:
    """Hamming distance between two binary vectors of the same length."""
    x, y = _pair(a, b)
    return int(np.count_nonzero(x ^ y))


def reconstruction_error(target: Any, cue: Any, reconstruction: Any) -> float | None:
    """Wu & Maass's reconstruction error ``HD(r(x'), x) / HD(x, x')``: 0 is perfect, 1 is no better than the cue.
    ``None`` when the cue equals the target (the ratio is undefined)."""
    base = hamming(target, cue)
    return None if base == 0 else hamming(reconstruction, target) / base


def _check_generator(rng: Any) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy.random.Generator (e.g. numpy.random.default_rng(seed))')
    return rng


def make_cue(pattern: Any, fraction: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Degraded copy of a binary pattern: ``round(fraction * |x|)`` active bits are switched off and, for
    ``kind='flip'``, the same number of inactive bits are switched on (the number of ones is kept)."""
    _check_generator(rng)
    if len(np.shape(pattern)) != 1:
        raise ValueError('pattern must be a binary vector')
    x = _binary('pattern', pattern, np.shape(pattern)[0])
    frac = _check_float('fraction', fraction, 0.0, 1.0)
    if kind not in CUE_KINDS:
        raise ValueError(f'kind must be one of {CUE_KINDS}')
    on, off = np.flatnonzero(x), np.flatnonzero(~x)
    k = int(round(frac * on.size))
    if kind == 'flip' and k > off.size:
        raise ValueError('not enough inactive bits to flip')
    cue = x.copy()
    if k:
        cue[rng.choice(on, k, replace=False)] = False
        if kind == 'flip':
            cue[rng.choice(off, k, replace=False)] = True
    return cue


def random_patterns(count: int, size: int, active: int, rng: np.random.Generator) -> np.ndarray:
    """``count`` random binary patterns of length ``size`` with exactly ``active`` ones each, ``(count, size)``."""
    _check_int('count', count, 0, 1_000_000)
    _check_int('size', size, 1, 1_000_000)
    _check_int('active', active, 0, size)
    _check_generator(rng)
    if count * size > 200_000_000:                      # the capacity benchmark's largest workload (100,000 x 2,000)
        raise ValueError('count * size must be <= 200,000,000')
    out = np.zeros((count, size), dtype=bool)
    for row in out:
        row[rng.choice(size, active, replace=False)] = True
    return out


def _pack(bits: np.ndarray) -> str:
    return base64.b64encode(np.packbits(bits.astype(bool)).tobytes()).decode('ascii')


def _unpack(name: str, text: Any, count: int) -> np.ndarray:
    if not isinstance(text, str):
        raise ValueError(f'{name} must be a base64 string')
    try:
        raw = base64.b64decode(text.encode('ascii'), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise ValueError(f'{name} is not valid base64') from exc
    if len(raw) != (count + 7) // 8:
        raise ValueError(f'{name} has the wrong length')
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    if np.any(bits[count:]):
        raise ValueError(f'{name} has non-zero padding bits')
    return bits[:count].astype(bool)


def _rng_state(rng: np.random.Generator) -> dict[str, Any]:
    return copy.deepcopy(rng.bit_generator.state)


def _restore_rng(state: Any) -> np.random.Generator:
    if not isinstance(state, Mapping) or state.get('bit_generator') != 'PCG64':
        raise ValueError('rng state must be a PCG64 state mapping')
    inner = state.get('state')
    if (not isinstance(inner, Mapping) or type(inner.get('state')) is not int or type(inner.get('inc')) is not int
            or type(state.get('has_uint32')) is not int or type(state.get('uinteger')) is not int):
        raise ValueError('malformed PCG64 state')
    rng = np.random.Generator(np.random.PCG64())
    try:
        rng.bit_generator.state = {'bit_generator': 'PCG64', 'state': {'state': inner['state'], 'inc': inner['inc']},
                                   'has_uint32': state['has_uint32'], 'uinteger': state['uinteger']}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid PCG64 state') from exc
    return rng


def _flip_probs(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        if np.ndim(value) != 1 or len(value) != 2:
            raise ValueError('flip_prob must be a number or a (p_ltp, p_ltd) pair')
        return (_check_float('flip_prob[0] (p_ltp)', value[0], 0.0, 1.0),
                _check_float('flip_prob[1] (p_ltd)', value[1], 0.0, 1.0))
    p = _check_float('flip_prob', value, 0.0, 1.0)
    return p, p


# ====================================================================== BTSP memory
class BTSPMemory:
    """Binary feed-forward + feedback memory trained one-shot by plateau-gated stochastic flips (module docstring).

    All randomness comes from ``seed``: the masks from ``default_rng(seed + 61403)`` and the plateau / flip draws
    from ``default_rng(seed + 61417)``, so the same seed and the same store sequence give identical weights."""

    def __init__(self, input_dim: int, memory_dim: int, *, density: float = 0.6, plateau_prob: float = 0.02,
                 flip_prob: float | Sequence[float] = 0.5, threshold: float = 0.35, feedback_threshold: float = 0.5,
                 seed: int = 0) -> None:
        self.input_dim = _check_int('input_dim', input_dim, 1, 1_000_000)
        self.memory_dim = _check_int('memory_dim', memory_dim, 1, 1_000_000)
        if self.input_dim * self.memory_dim > _MAX_MATRIX:
            raise ValueError(f'input_dim * memory_dim must be <= {_MAX_MATRIX:,}')
        self.density = _check_float('density', density, 0.0, 1.0, low_open=True)
        self.plateau_prob = _check_float('plateau_prob', plateau_prob, 0.0, 1.0)
        self.p_ltp, self.p_ltd = _flip_probs(flip_prob)
        self.threshold = _check_float('threshold', threshold, 0.0, 10.0)
        self.feedback_threshold = _check_float('feedback_threshold', feedback_threshold, 0.0, 10.0)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        mask_rng = np.random.default_rng(self.seed + _MASK_OFFSET)
        self._C = mask_rng.random((self.memory_dim, self.input_dim)) < self.density
        self._D = mask_rng.random((self.input_dim, self.memory_dim)) < self.density
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Erase every memory (all weights 0) and restart the plasticity random stream."""
        self.W = np.zeros((self.memory_dim, self.input_dim), dtype=bool)
        self.B = np.zeros((self.input_dim, self.memory_dim), dtype=bool)
        self._rng = np.random.default_rng(self.seed + _PLASTICITY_OFFSET)
        self.stores = 0
        self.plateaus = 0
        self.ltp_flips = 0
        self.ltd_flips = 0

    @property
    def forward_mask(self) -> np.ndarray:
        return self._C.copy()

    @property
    def feedback_mask(self) -> np.ndarray:
        return self._D.copy()

    @property
    def synapse_count(self) -> int:
        """Existing plastic binary synapses (forward plus feedback)."""
        return int(np.count_nonzero(self._C) + np.count_nonzero(self._D))

    @property
    def state_numbers(self) -> int:
        """Stored numbers: binary weights and masks of both pathways plus four counters."""
        return int(self.W.size + self.B.size + self._C.size + self._D.size + 4)

    def weight_fractions(self) -> dict[str, float]:
        """Fraction of existing synapses at weight 1 in each pathway."""
        c, d = max(1, int(np.count_nonzero(self._C))), max(1, int(np.count_nonzero(self._D)))
        return {'forward': float(np.count_nonzero(self.W) / c), 'feedback': float(np.count_nonzero(self.B) / d)}

    def config(self) -> dict[str, Any]:
        return {'input_dim': self.input_dim, 'memory_dim': self.memory_dim, 'density': self.density,
                'plateau_prob': self.plateau_prob, 'flip_prob': [self.p_ltp, self.p_ltd], 'threshold': self.threshold,
                'feedback_threshold': self.feedback_threshold, 'seed': self.seed}

    # ------------------------------------------------------------------ store / recall
    def _code_idx(self, idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.zeros(self.memory_dim, dtype=bool)
        drive = np.count_nonzero(self.W[:, idx], axis=1)
        return drive > self.threshold * self.density * idx.size

    def _reconstruct_idx(self, zi: np.ndarray) -> np.ndarray:
        if zi.size == 0:
            return np.zeros(self.input_dim, dtype=bool)
        drive = np.count_nonzero(self.B[:, zi], axis=1)
        return drive > self.feedback_threshold * self.density * zi.size

    def store(self, pattern: Any, *, plateau_prob: float | None = None) -> dict[str, int]:
        """One-shot store of a binary pattern (``plateau_prob`` overrides the configured probability for this store).

        Returns the number of plateau units, LTP (0 -> 1) and LTD (1 -> 0) flips, the new trace size and the number
        of feedback synapses newly set to 1."""
        x = _binary('pattern', pattern, self.input_dim)
        p = self.plateau_prob if plateau_prob is None else _check_float('plateau_prob', plateau_prob, 0.0, 1.0)
        idx = np.flatnonzero(x)
        rows = np.flatnonzero(self._rng.random(self.memory_dim) < p)
        ltp = ltd = 0
        if rows.size and idx.size:
            ix = np.ix_(rows, idx)
            old = self.W[ix]
            draw = self._rng.random(old.shape)
            flip = self._C[ix] & (draw < np.where(old, self.p_ltd, self.p_ltp))
            ltp = int(np.count_nonzero(flip & ~old))
            ltd = int(np.count_nonzero(flip & old))
            self.W[ix] = old ^ flip
        zi = np.flatnonzero(self._code_idx(idx))
        new_feedback = 0
        if zi.size and idx.size:
            ix = np.ix_(idx, zi)
            before = self.B[ix]
            after = before | self._D[ix]
            new_feedback = int(np.count_nonzero(after & ~before))
            self.B[ix] = after
        self.stores += 1
        self.plateaus += int(rows.size)
        self.ltp_flips += ltp
        self.ltd_flips += ltd
        return {'plateau_units': int(rows.size), 'ltp_flips': ltp, 'ltd_flips': ltd, 'trace_size': int(zi.size),
                'new_feedback_synapses': new_feedback}

    def drive(self, cue: Any) -> np.ndarray:
        """Integer input of every memory unit, ``W c``."""
        idx = np.flatnonzero(_binary('cue', cue, self.input_dim))
        return np.count_nonzero(self.W[:, idx], axis=1).astype(np.int64)

    def code(self, cue: Any) -> np.ndarray:
        """Memory trace ``z(c)`` (bool, length ``memory_dim``)."""
        return self._code_idx(np.flatnonzero(_binary('cue', cue, self.input_dim)))

    def reconstruct(self, code: Any) -> np.ndarray:
        """Feedback reconstruction ``r(z)`` of the input from a memory trace (bool, length ``input_dim``)."""
        return self._reconstruct_idx(np.flatnonzero(_binary('code', code, self.memory_dim)))

    def recall(self, cue: Any) -> dict[str, np.ndarray]:
        """``{'code': z(c), 'reconstruction': r(z(c))}`` (no state change)."""
        z = self.code(cue)
        return {'code': z, 'reconstruction': self._reconstruct_idx(np.flatnonzero(z))}

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        """JSON-safe state: weights on existing synapses (packed bits), counters and the plasticity RNG state."""
        return {'config': self.config(), 'W_bits': _pack(self.W[self._C]), 'B_bits': _pack(self.B[self._D]),
                'stores': int(self.stores), 'plateaus': int(self.plateaus), 'ltp_flips': int(self.ltp_flips),
                'ltd_flips': int(self.ltd_flips), 'rng': _rng_state(self._rng)}

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate everything, then replace weights, counters and RNG (the configuration must match)."""
        if not isinstance(state, Mapping):
            raise ValueError('BTSP memory state must be a mapping')
        if state.get('config') != self.config():
            raise ValueError('BTSP memory state was made with a different configuration')
        w_bits = _unpack('W_bits', state.get('W_bits'), int(np.count_nonzero(self._C)))
        b_bits = _unpack('B_bits', state.get('B_bits'), int(np.count_nonzero(self._D)))
        counters = {k: _check_int(k, state.get(k), 0, 2**62) for k in ('stores', 'plateaus', 'ltp_flips', 'ltd_flips')}
        if counters['plateaus'] > counters['stores'] * self.memory_dim:
            raise ValueError('plateau counter exceeds stores * memory_dim')
        rng = _restore_rng(state.get('rng'))
        W = np.zeros_like(self.W)
        W[self._C] = w_bits
        B = np.zeros_like(self.B)
        B[self._D] = b_bits
        self.W, self.B, self._rng = W, B, rng
        self.stores, self.plateaus = counters['stores'], counters['plateaus']
        self.ltp_flips, self.ltd_flips = counters['ltp_flips'], counters['ltd_flips']


# ====================================================================== comparison memories
class WillshawMemory:
    """Clipped-Hebbian auto-associative binary memory (Willshaw et al., 1969) with ``input_dim**2`` synapses.

    ``store`` sets ``A_ij = 1`` for every co-active pair (including ``i = j``); ``recall`` returns
    ``r_i = 1[(A c)_i >= threshold * |c|]`` (``threshold = 1`` is the classic all-cue-units rule)."""

    def __init__(self, input_dim: int, *, threshold: float = 1.0) -> None:
        self.input_dim = _check_int('input_dim', input_dim, 1, 1_000_000)
        if self.input_dim ** 2 > _MAX_MATRIX:
            raise ValueError(f'input_dim**2 must be <= {_MAX_MATRIX:,}')
        self.threshold = _check_float('threshold', threshold, 0.0, 1.0, low_open=True)
        self.reset()

    def reset(self) -> None:
        self.A = np.zeros((self.input_dim, self.input_dim), dtype=bool)
        self.stores = 0

    @property
    def synapse_count(self) -> int:
        return int(self.A.size)

    def store(self, pattern: Any) -> dict[str, int]:
        idx = np.flatnonzero(_binary('pattern', pattern, self.input_dim))
        new = 0
        if idx.size:
            ix = np.ix_(idx, idx)
            new = int(idx.size * idx.size - np.count_nonzero(self.A[ix]))
            self.A[ix] = True
        self.stores += 1
        return {'new_synapses': new}

    def recall(self, cue: Any) -> np.ndarray:
        idx = np.flatnonzero(_binary('cue', cue, self.input_dim))
        if idx.size == 0:
            return np.zeros(self.input_dim, dtype=bool)
        return np.count_nonzero(self.A[:, idx], axis=1) >= self.threshold * idx.size

    def export_state(self) -> dict[str, Any]:
        return {'config': {'input_dim': self.input_dim, 'threshold': self.threshold}, 'A_bits': _pack(self.A.ravel()),
                'stores': int(self.stores)}

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get('config') != {'input_dim': self.input_dim,
                                                                     'threshold': self.threshold}:
            raise ValueError('Willshaw state does not match this memory')
        bits = _unpack('A_bits', state.get('A_bits'), self.input_dim ** 2).reshape(self.input_dim, self.input_dim)
        if not np.array_equal(bits, bits.T):
            raise ValueError('Willshaw matrix must be symmetric')
        stores = _check_int('stores', state.get('stores'), 0, 2**62)
        self.A, self.stores = bits, stores


class RandomRecallControl:
    """Chance control: 'recalls' a seeded random pattern with ``active`` ones and ignores the cue.
    Its expected normalised overlap with any pattern of ``active`` ones is ``active / input_dim``."""

    def __init__(self, input_dim: int, active: int, seed: int = 0) -> None:
        self.input_dim = _check_int('input_dim', input_dim, 1, 1_000_000)
        self.active = _check_int('active', active, 0, self.input_dim)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        self._rng = np.random.default_rng(self.seed + _CONTROL_OFFSET)

    def store(self, pattern: Any) -> dict[str, int]:
        _binary('pattern', pattern, self.input_dim)
        return {}

    def recall(self, cue: Any) -> np.ndarray:
        _binary('cue', cue, self.input_dim)
        out = np.zeros(self.input_dim, dtype=bool)
        out[self._rng.choice(self.input_dim, self.active, replace=False)] = True
        return out


# ====================================================================== engine extension
class BTSPExtension(EngineExtension):
    """Readout-only NEUROWEAVE extension around a :class:`BTSPMemory` over binarised brain output.

    After each engine step the brain output (fixed at the first ``n`` units seen at :meth:`bind`; units born later
    are ignored) is binarised by top-``k`` (``k = max(1, round(active_fraction * n))``, ties to the lower index).
    The current pattern is used as a recall cue *before* any store, giving the features

    * ``recall_overlap`` = ``overlap(x_t, r(z(x_t)))`` (familiarity of the current state),
    * ``recall_confidence`` = ``min(1, |z(x_t)| / max(1, plateau_prob * memory_dim))`` (trace size relative to the
      expected size of a stored trace),
    * ``projection_dim`` fixed ``+-1`` projections of the trace, ``P z / sqrt(max(1, |z|))``.

    Then, only when ``ctx['learn']`` is true, the pattern is stored one-shot if its salience
    ``max(|ctx['reward']|, 1 - recall_overlap)`` exceeds ``gate``, with plateau probability
    ``plateau_prob * min(1, salience)``. The drive is never changed."""

    id = MODULE_ID

    def __init__(self, memory_dim: int = 64, *, active_fraction: float = 0.25, density: float = 0.6,
                 plateau_prob: float = 0.1, flip_prob: float | Sequence[float] = 0.5, threshold: float = 0.35,
                 feedback_threshold: float = 0.5, gate: float = 0.5, projection_dim: int = 6, seed: int = 0) -> None:
        self.memory_dim = _check_int('memory_dim', memory_dim, 1, 4096)
        self.active_fraction = _check_float('active_fraction', active_fraction, 0.0, 1.0, low_open=True)
        self.density = _check_float('density', density, 0.0, 1.0, low_open=True)
        self.plateau_prob = _check_float('plateau_prob', plateau_prob, 0.0, 1.0)
        self.flip_prob = list(_flip_probs(flip_prob))
        self.threshold = _check_float('threshold', threshold, 0.0, 10.0)
        self.feedback_threshold = _check_float('feedback_threshold', feedback_threshold, 0.0, 10.0)
        self.gate = _check_float('gate', gate, 0.0, 1.0)
        self.projection_dim = _check_int('projection_dim', projection_dim, 0, 64)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        self.feature_dim = 2 + self.projection_dim
        prng = np.random.default_rng(self.seed + _PROJECTION_OFFSET)
        self._P = np.where(prng.random((self.projection_dim, self.memory_dim)) < 0.5, -1.0, 1.0)
        self._expected_trace = max(1.0, self.plateau_prob * self.memory_dim)
        self._n = 0
        self._k = 0
        self.memory: BTSPMemory | None = None
        self._clear_counters()

    def _clear_counters(self) -> None:
        self._features = np.zeros(self.feature_dim)
        self._steps = 0
        self._frozen_steps = 0
        self._stores = 0
        self._last = {'overlap': 0.0, 'confidence': 0.0, 'salience': 0.0, 'stored': False}

    def _new_memory(self, n: int) -> BTSPMemory:
        return BTSPMemory(n, self.memory_dim, density=self.density, plateau_prob=self.plateau_prob,
                          flip_prob=tuple(self.flip_prob), threshold=self.threshold,
                          feedback_threshold=self.feedback_threshold, seed=self.seed)

    def _build(self, n: int) -> None:
        self._n = n
        self._k = max(1, int(round(self.active_fraction * n)))
        self.memory = self._new_memory(n)

    # ------------------------------------------------------------------ lifecycle
    def bind(self, engine: Any) -> None:
        """Size the memory from ``engine.n`` (copied); the memory starts empty."""
        n = getattr(engine, 'n', None)
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise ValueError('engine.n must be an int')
        n = int(n)
        if not 1 <= n <= 4096 or n * self.memory_dim > _MAX_MATRIX:
            raise ValueError('engine size is out of range for the BTSP extension')
        self._build(n)
        self._clear_counters()

    def reset(self) -> None:
        """Episode boundary: clear the current features (stored memories are kept)."""
        self._features = np.zeros(self.feature_dim)

    # ------------------------------------------------------------------ hooks
    def binarise(self, output: Any) -> np.ndarray:
        """Top-``k`` binary pattern of the first ``n`` output values (zero-padded if the output is shorter)."""
        if self.memory is None:
            raise ValueError('BTSP extension is not bound to an engine')
        try:
            o = np.asarray(output, dtype=np.float64).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError('brain output must be numeric') from exc
        if not np.all(np.isfinite(o)):
            raise ValueError('brain output must be finite')
        v = np.zeros(self._n)
        m = min(self._n, o.size)
        v[:m] = o[:m]
        x = np.zeros(self._n, dtype=bool)
        x[np.argsort(-v, kind='stable')[:self._k]] = True
        return x

    def after_step(self, output: np.ndarray, ctx: Mapping[str, Any]) -> None:
        if not isinstance(ctx, Mapping):
            raise ValueError('BTSP extension needs the engine step context')
        learn = ctx.get('learn')
        if type(learn) is not bool:
            raise ValueError("ctx['learn'] must be a boolean")
        reward = ctx.get('reward', 0.0)
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, (int, float, np.integer, np.floating)) \
                or not math.isfinite(float(reward)):
            raise ValueError("ctx['reward'] must be a finite number")
        x = self.binarise(output)
        mem = self.memory
        z = mem._code_idx(np.flatnonzero(x))
        zi = np.flatnonzero(z)
        rec = mem._reconstruct_idx(zi)
        ov = overlap(x, rec)
        conf = min(1.0, zi.size / self._expected_trace)
        proj = self._P @ z.astype(np.float64) / math.sqrt(max(1, zi.size)) if self.projection_dim else np.zeros(0)
        self._features = np.clip(np.concatenate([[ov, conf], proj]), -10.0, 10.0)
        salience = max(min(abs(float(reward)), 1.0), 1.0 - ov)
        stored = False
        if learn:
            if salience > self.gate:
                mem.store(x, plateau_prob=self.plateau_prob * min(1.0, salience))
                self._stores += 1
                stored = True
        else:
            self._frozen_steps += 1
        self._last = {'overlap': float(ov), 'confidence': float(conf), 'salience': float(salience), 'stored': stored}
        self._steps += 1

    def features(self) -> np.ndarray:
        return self._features.copy()

    # ------------------------------------------------------------------ accounting / persistence
    def parameter_count(self) -> int:
        """Plastic binary synapses of the memory (0 before :meth:`bind`)."""
        return 0 if self.memory is None else self.memory.synapse_count

    def state_numbers(self) -> int:
        mem = 0 if self.memory is None else self.memory.state_numbers
        return int(mem + self._P.size + self.feature_dim + 4)

    def telemetry(self) -> dict[str, Any]:
        out: dict[str, Any] = {'id': self.id, 'n': self._n, 'k': self._k, 'memory_dim': self.memory_dim,
                               'steps': self._steps, 'frozen_steps': self._frozen_steps, 'stores': self._stores,
                               'last_overlap': self._last['overlap'], 'last_confidence': self._last['confidence'],
                               'last_salience': self._last['salience'], 'last_stored': self._last['stored']}
        if self.memory is not None:
            out['weight_fractions'] = self.memory.weight_fractions()
            out['synapses'] = self.memory.synapse_count
        return out

    def config(self) -> dict[str, Any]:
        return {'memory_dim': self.memory_dim, 'active_fraction': self.active_fraction, 'density': self.density,
                'plateau_prob': self.plateau_prob, 'flip_prob': list(self.flip_prob), 'threshold': self.threshold,
                'feedback_threshold': self.feedback_threshold, 'gate': self.gate,
                'projection_dim': self.projection_dim, 'seed': self.seed}

    def export_state(self) -> dict[str, Any]:
        return {'id': self.id, 'config': self.config(), 'n': self._n,
                'memory': None if self.memory is None else self.memory.export_state(),
                'features': self._features.tolist(), 'steps': self._steps, 'frozen_steps': self._frozen_steps,
                'stores': self._stores, 'last': dict(self._last)}

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate fully, then restore. An unbound extension is sized from the state's ``n``."""
        super().import_state(state)
        if state.get('config') != self.config():
            raise ValueError('BTSP extension state was made with a different configuration')
        n = _check_int('n', state.get('n'), 0, 4096)
        if self._n and n != self._n:
            raise ValueError('BTSP extension state was made for a different engine size')
        if n == 0:
            if state.get('memory') is not None:
                raise ValueError('unbound BTSP extension state cannot hold a memory')
            memory = None
        else:
            if n * self.memory_dim > _MAX_MATRIX:
                raise ValueError('engine size is out of range for the BTSP extension')
            memory = self._new_memory(n)
            memory.import_state(state.get('memory'))           # validates fully before anything is replaced
        try:
            feats = np.asarray(state.get('features'), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError('features must be numeric') from exc
        if feats.shape != (self.feature_dim,) or not np.all(np.isfinite(feats)) \
                or np.max(np.abs(feats), initial=0.0) > 10.0:
            raise ValueError('features must be finite, in [-10, 10] and of length feature_dim')
        steps = _check_int('steps', state.get('steps'), 0, 2**62)
        frozen = _check_int('frozen_steps', state.get('frozen_steps'), 0, steps)
        stores = _check_int('stores', state.get('stores'), 0, steps - frozen)
        last = state.get('last')
        if not isinstance(last, Mapping) or set(last) != {'overlap', 'confidence', 'salience', 'stored'} \
                or type(last['stored']) is not bool:
            raise ValueError('last must hold overlap, confidence, salience and stored')
        last = {'overlap': _check_float('last.overlap', last['overlap'], 0.0, 1.0),
                'confidence': _check_float('last.confidence', last['confidence'], 0.0, 1.0),
                'salience': _check_float('last.salience', last['salience'], 0.0, 1.0), 'stored': last['stored']}
        if n == 0 and (steps or feats.any() or last['stored']):
            raise ValueError('an unbound BTSP extension cannot have stepped')
        if n:
            self._n, self._k = n, max(1, int(round(self.active_fraction * n)))
        self.memory = memory
        self._features = feats.copy()
        self._steps, self._frozen_steps, self._stores, self._last = steps, frozen, stores, last


# ====================================================================== capacity benchmark
# Both arms' tuned optimum lies inside its grid at these sizes (measured 2026-11-19); receipts flag a chosen value
# that sits on a grid edge, because an under-tuned arm would make the comparison unfair.
CAPACITY_DEFAULTS: dict[str, Any] = {
    'input_dim': 1000, 'active': 50, 'patterns': 400, 'checkpoints': 4, 'test_per_checkpoint': 64,
    'cue_fractions': (0.3, 0.5), 'cue_kinds': ('delete', 'flip'), 'density': 0.6, 'flip_prob': 0.5,
    'plateau_grid': (0.005, 0.01, 0.02, 0.04), 'threshold_grid': (0.25, 0.3, 0.35, 0.4),
    'feedback_grid': (0.35, 0.5, 0.65, 0.8), 'willshaw_grid': (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    'tune_seeds': (1000, 1001)}
CAPACITY_QUICK: dict[str, Any] = {
    'input_dim': 400, 'active': 20, 'patterns': 120, 'test_per_checkpoint': 24, 'plateau_grid': (0.02, 0.04, 0.08),
    'threshold_grid': (0.3, 0.35, 0.4, 0.45), 'feedback_grid': (0.5, 0.65, 0.8), 'willshaw_grid': (0.5, 0.6, 0.8, 1.0),
    'tune_seeds': (1000,)}
_BTSP_NATURAL_BOUNDS = {'plateau_prob': (1.0,), 'threshold': (0.0,), 'feedback_threshold': (0.0,)}
_WILLSHAW_NATURAL_BOUNDS = {'threshold': (1.0,)}       # 1 = the classic all-cue-units rule (and the largest allowed)


def _grid(name: str, value: Any, low: float, high: float, *, low_open: bool = False) -> tuple[float, ...]:
    try:
        out = tuple(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a sequence of numbers') from exc
    if not 1 <= len(out) <= 16:
        raise ValueError(f'{name} must hold 1..16 values')
    return tuple(_check_float(f'{name} entry', v, low, high, low_open=low_open) for v in out)


def _test_sets(seed: int, patterns: np.ndarray, loads: Sequence[int], per: int,
               conditions: Sequence[tuple[float, str]]) -> list[dict[str, Any]]:
    """For each checkpoint: tested pattern indices and one cue per (condition, pattern), shared by every arm."""
    sets = []
    for j, load in enumerate(loads):
        rng = np.random.default_rng(seed + _TEST_OFFSET + 7919 * j)
        idx = np.sort(rng.choice(load, min(per, load), replace=False))
        cues = {cond: np.stack([make_cue(patterns[i], cond[0], cond[1], rng) for i in idx]) for cond in conditions}
        sets.append({'load': int(load), 'idx': idx, 'cues': cues})
    return sets


def _cond_name(cond: tuple[float, str]) -> str:
    return f'{cond[1]}_{cond[0]:g}'


def _score_recall(patterns: np.ndarray, test: Mapping[str, Any], recalls: Mapping[Any, Sequence[np.ndarray]]
                  ) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for cond, recs in recalls.items():
        ovs, errs = [], []
        for i, cue, rec in zip(test['idx'], test['cues'][cond], recs):
            ovs.append(overlap(patterns[i], rec))
            err = reconstruction_error(patterns[i], cue, rec)
            if err is not None:
                errs.append(err)
        out[_cond_name(cond)] = {'overlap': float(np.mean(ovs)),
                                 'reconstruction_error': float(np.mean(errs)) if errs else None}
    return out


def _curve(per_ckpt: Sequence[Mapping[str, Mapping[str, Any]]]) -> float:
    return float(np.mean([v['overlap'] for ck in per_ckpt for v in ck.values()]))


def _run_btsp(patterns: np.ndarray, tests: Sequence[Mapping[str, Any]], *, memory_dim: int, density: float,
              flip_prob: float, plateau: float, threshold: float, feedback_grid: Sequence[float], seed: int,
              diagnostics: bool = False) -> dict[str, Any]:
    """Store sequentially; at each checkpoint score every feedback threshold (it acts at recall only)."""
    mem = BTSPMemory(patterns.shape[1], memory_dim, density=density, plateau_prob=plateau, flip_prob=flip_prob,
                     threshold=threshold, feedback_threshold=feedback_grid[0], seed=seed)
    per_fb: dict[float, list[dict[str, Any]]] = {fb: [] for fb in feedback_grid}
    diag = []
    stored = 0
    for test in tests:
        while stored < test['load']:
            mem.store(patterns[stored])
            stored += 1
        codes = {cond: [mem.code(c) for c in cues] for cond, cues in test['cues'].items()}
        for fb in feedback_grid:
            mem.feedback_threshold = fb
            recs = {cond: [mem._reconstruct_idx(np.flatnonzero(z)) for z in zs] for cond, zs in codes.items()}
            per_fb[fb].append(_score_recall(patterns, test, recs))
        if diagnostics:
            full = [mem.code(patterns[i]) for i in test['idx']]
            diag.append({'load': test['load'], 'weight_fractions': mem.weight_fractions(),
                         'mean_trace_size': float(np.mean([np.count_nonzero(z) for z in full])),
                         'trace_overlap': {_cond_name(cond): float(np.mean([overlap(a, b) for a, b in zip(full, zs)]))
                                           for cond, zs in codes.items()}})
    return {'per_feedback': per_fb, 'diagnostics': diag, 'synapses': mem.synapse_count}


def _run_willshaw(patterns: np.ndarray, tests: Sequence[Mapping[str, Any]], grid: Sequence[float]
                  ) -> dict[float, list[dict[str, Any]]]:
    mem = WillshawMemory(patterns.shape[1], threshold=grid[0])
    out: dict[float, list[dict[str, Any]]] = {th: [] for th in grid}
    stored = 0
    for test in tests:
        while stored < test['load']:
            mem.store(patterns[stored])
            stored += 1
        for th in grid:
            mem.threshold = th
            recs = {cond: [mem.recall(c) for c in cues] for cond, cues in test['cues'].items()}
            out[th].append(_score_recall(patterns, test, recs))
    return out


def _run_reference(patterns: np.ndarray, tests: Sequence[Mapping[str, Any]], *, active: int, seed: int
                   ) -> dict[str, list[dict[str, Any]]]:
    control = RandomRecallControl(patterns.shape[1], active, seed=seed)
    rand, cue_only = [], []
    for test in tests:
        rand.append(_score_recall(patterns, test, {cond: [control.recall(c) for c in cues]
                                                   for cond, cues in test['cues'].items()}))
        cue_only.append(_score_recall(patterns, test, {cond: list(cues) for cond, cues in test['cues'].items()}))
    return {'random_recall': rand, 'cue_only': cue_only}


def run_capacity_benchmark(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any
                           ) -> dict[str, Any]:
    """BTSP vs a Willshaw memory with the same synapse count vs a random-recall control.

    ``patterns`` random binary patterns with exactly ``active`` ones are stored one-shot, in order. At
    ``checkpoints`` evenly spaced loads, ``test_per_checkpoint`` stored patterns are recalled from degraded cues
    (every ``cue_fractions`` x ``cue_kinds`` condition; the same cues for every arm). Metric: normalised overlap
    between the stored pattern and its reconstruction, averaged over checkpoints and conditions (the mean of the
    capacity curve). BTSP uses ``memory_dim = round(input_dim / (2 * density))`` memory units, so its forward plus
    feedback synapses match the Willshaw matrix's ``input_dim**2`` in expectation (exact counts are reported).
    Free parameters are tuned per arm on ``tune_seeds`` (disjoint from ``seeds``) by the same metric: plateau
    probability, threshold and feedback threshold for BTSP, the threshold for Willshaw. Verdict: paired 95 % t
    interval over seeds, BTSP vs Willshaw (higher is better)."""
    from . import harness
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seeds = _check_seeds('seeds', seeds)
    cfg = dict(CAPACITY_DEFAULTS)
    if quick:
        cfg.update(CAPACITY_QUICK)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(f'unknown run_capacity_benchmark overrides: {unknown}')
    cfg.update(overrides)
    m = _check_int('input_dim', cfg['input_dim'], 8, 2000)
    a = _check_int('active', cfg['active'], 1, m // 2)
    n_pat = _check_int('patterns', cfg['patterns'], 1, 100_000)
    n_ck = _check_int('checkpoints', cfg['checkpoints'], 1, min(16, n_pat))
    per = _check_int('test_per_checkpoint', cfg['test_per_checkpoint'], 1, 4096)
    density = _check_float('density', cfg['density'], 0.0, 1.0, low_open=True)
    flip = _check_float('flip_prob', cfg['flip_prob'], 0.0, 1.0)
    fracs = _grid('cue_fractions', cfg['cue_fractions'], 0.0, 1.0)
    try:
        kinds = tuple(cfg['cue_kinds'])
    except TypeError as exc:
        raise ValueError(f'cue_kinds must be distinct values from {CUE_KINDS}') from exc
    if not kinds or any(k not in CUE_KINDS for k in kinds) or len(set(kinds)) != len(kinds):
        raise ValueError(f'cue_kinds must be distinct values from {CUE_KINDS}')
    plateau_grid = _grid('plateau_grid', cfg['plateau_grid'], 0.0, 1.0, low_open=True)
    threshold_grid = _grid('threshold_grid', cfg['threshold_grid'], 0.0, 10.0)
    feedback_grid = _grid('feedback_grid', cfg['feedback_grid'], 0.0, 10.0)
    willshaw_grid = _grid('willshaw_grid', cfg['willshaw_grid'], 0.0, 1.0, low_open=True)
    tune_seeds = _check_seeds('tune_seeds', cfg['tune_seeds'])
    if set(tune_seeds) & set(seeds):
        raise ValueError('tune_seeds must be disjoint from seeds')
    memory_dim = max(1, int(round(m / (2.0 * density))))
    if m * memory_dim > _MAX_MATRIX:
        raise ValueError('input_dim / density gives a BTSP memory that is too large')
    conditions = [(f, k) for k in kinds for f in fracs]
    loads = sorted({max(1, int(round(n_pat * (j + 1) / n_ck))) for j in range(n_ck)})
    cfg.update({'input_dim': m, 'active': a, 'patterns': n_pat, 'checkpoints': n_ck, 'test_per_checkpoint': per,
                'density': density, 'flip_prob': flip, 'cue_fractions': fracs, 'cue_kinds': kinds,
                'plateau_grid': plateau_grid, 'threshold_grid': threshold_grid, 'feedback_grid': feedback_grid,
                'willshaw_grid': willshaw_grid, 'tune_seeds': tune_seeds})

    def workload(seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        pats = random_patterns(n_pat, m, a, np.random.default_rng(seed + _PATTERN_OFFSET))
        return pats, _test_sets(seed, pats, loads, per, conditions)

    t0 = time.perf_counter()
    tune_work = {s: workload(s) for s in tune_seeds}
    btsp_table: list[dict[str, Any]] = []
    for plateau, threshold in itertools.product(plateau_grid, threshold_grid):
        runs = [_run_btsp(*tune_work[s], memory_dim=memory_dim, density=density, flip_prob=flip, plateau=plateau,
                          threshold=threshold, feedback_grid=feedback_grid, seed=s)['per_feedback']
                for s in tune_seeds]
        for fb in feedback_grid:
            btsp_table.append({'plateau_prob': plateau, 'threshold': threshold, 'feedback_threshold': fb,
                               'tune_score': float(np.mean([_curve(r[fb]) for r in runs]))})
    will_runs = [_run_willshaw(*tune_work[s], willshaw_grid) for s in tune_seeds]
    will_table = [{'threshold': th, 'tune_score': float(np.mean([_curve(r[th]) for r in will_runs]))}
                  for th in willshaw_grid]
    b_best = btsp_table[max(range(len(btsp_table)), key=lambda i: (btsp_table[i]['tune_score'], -i))]
    w_best = will_table[max(range(len(will_table)), key=lambda i: (will_table[i]['tune_score'], -i))]
    b_chosen = {k: b_best[k] for k in ('plateau_prob', 'threshold', 'feedback_threshold')}
    b_edges = _grid_edges(b_chosen, {'plateau_prob': plateau_grid, 'threshold': threshold_grid,
                                     'feedback_threshold': feedback_grid}, _BTSP_NATURAL_BOUNDS)
    w_edges = _grid_edges({'threshold': w_best['threshold']}, {'threshold': willshaw_grid}, _WILLSHAW_NATURAL_BOUNDS)
    tune_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    per_seed = []
    for s in seeds:
        pats, tests = workload(s)
        b = _run_btsp(pats, tests, memory_dim=memory_dim, density=density, flip_prob=flip,
                      plateau=b_best['plateau_prob'], threshold=b_best['threshold'],
                      feedback_grid=(b_best['feedback_threshold'],), seed=s, diagnostics=True)
        w = _run_willshaw(pats, tests, (w_best['threshold'],))[w_best['threshold']]
        ref = _run_reference(pats, tests, active=a, seed=s)
        arms = {'btsp': b['per_feedback'][b_best['feedback_threshold']], 'willshaw': w,
                'random_recall': ref['random_recall'], 'cue_only': ref['cue_only']}
        per_seed.append({'seed': int(s), 'btsp_synapses': b['synapses'], 'btsp_diagnostics': b['diagnostics'],
                         'curve_mean_overlap': {arm: _curve(v) for arm, v in arms.items()},
                         'per_checkpoint': {arm: [{'load': t['load'], **ck} for t, ck in zip(tests, v)]
                                            for arm, v in arms.items()}})
    eval_wall = time.perf_counter() - t1

    def vals(arm: str) -> list[float]:
        return [r['curve_mean_overlap'][arm] for r in per_seed]

    comparisons = {'btsp_vs_willshaw': harness.compare(vals('btsp'), vals('willshaw')),
                   'btsp_vs_random_recall': harness.compare(vals('btsp'), vals('random_recall')),
                   'willshaw_vs_random_recall': harness.compare(vals('willshaw'), vals('random_recall')),
                   'btsp_vs_cue_only': harness.compare(vals('btsp'), vals('cue_only')),
                   'willshaw_vs_cue_only': harness.compare(vals('willshaw'), vals('cue_only'))}
    arm_names = ('btsp', 'willshaw', 'random_recall', 'cue_only')
    summary = {arm: {'curve_mean_overlap': float(np.mean(vals(arm))),
                     'overlap_by_load': [float(np.mean([r['per_checkpoint'][arm][j][_cond_name(c)]['overlap']
                                                        for r in per_seed for c in conditions]))
                                         for j in range(len(loads))]}
               for arm in arm_names}
    btsp_syn = [r['btsp_synapses'] for r in per_seed]
    receipt = harness.base_receipt(CAPACITY_ID, 'memory_benchmark', {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, 'seeds': list(seeds), 'quick': quick,
        'loads': loads, 'conditions': [_cond_name(c) for c in conditions], 'memory_dim': memory_dim,
        'metric': 'normalised reconstruction overlap |x & r| / sqrt(|x| |r|), mean over checkpoints and conditions',
        'tuning_objective': 'the same metric on tune_seeds'})
    receipt.update({
        'module': MODULE_ID, 'source': SOURCE,
        'arms': {'btsp': f'BTSP memory: {memory_dim} memory units, forward + feedback masks at density {density:g}, '
                         'plateau-gated stochastic flips, Hebbian feedback readout',
                 'willshaw': f'Willshaw clipped-Hebbian auto-associative matrix, {m}x{m} binary synapses',
                 'random_recall': f'seeded random pattern with {a} ones (chance; expected overlap {a / m:.4f})',
                 'cue_only': 'reference without memory: the degraded cue itself is the reconstruction'},
        'tuning': {'btsp': {'chosen': b_chosen, 'chosen_on_grid_edge': b_edges, 'table': btsp_table},
                   'willshaw': {'chosen': {'threshold': w_best['threshold']}, 'chosen_on_grid_edge': w_edges,
                                'table': will_table}},
        'per_seed': per_seed, 'summary': summary, 'comparisons': comparisons,
        'verdict': comparisons['btsp_vs_willshaw']['verdict'],
        'verdict_basis': ('btsp_vs_willshaw: paired 95% t interval over seeds on the capacity-curve mean '
                          'reconstruction overlap (higher is better), both arms tuned on disjoint tune_seeds'),
        'budget': {'willshaw_synapses': m * m, 'btsp_synapses_per_seed': btsp_syn,
                   'btsp_synapses_expected': float(2.0 * density * m * memory_dim),
                   'btsp_to_willshaw_synapse_ratio': float(np.mean(btsp_syn) / (m * m)),
                   'btsp_state_numbers': int(4 * m * memory_dim + 4), 'willshaw_state_numbers': int(m * m + 1)},
        'estimated_cost': {'stores_per_arm_run': n_pat, 'tuning_btsp_runs': len(plateau_grid) * len(threshold_grid)
                           * len(tune_seeds), 'recall_queries_per_run': len(loads) * len(conditions) * per,
                           'estimated_synapse_reads': int((len(plateau_grid) * len(threshold_grid) * len(tune_seeds)
                                                           + len(seeds)) * (n_pat + len(loads) * len(conditions) * per)
                                                          * 2 * a * memory_dim)},
        'measured_wallclock_s': {'tuning': tune_wall, 'evaluation': eval_wall},
        'notes': ('Synthetic random sparse patterns stored once each, in order; recall is one feed-forward + feedback '
                  'pass for BTSP and one pass for Willshaw (no iteration). Thresholds are proportional to the cue / '
                  'trace size (the paper grid-searches an absolute memory threshold and, for input completion, uses '
                  'fitted linear functions of the average fraction of 1\'s). BTSP gets a larger tuning grid than '
                  'Willshaw (it has three free parameters, Willshaw one); tuning.<arm>.chosen_on_grid_edge names any '
                  'parameter whose chosen value is a grid edge. The paper itself does not compare BTSP with a '
                  'Willshaw memory. The comparison is at equal synapse count, not equal bits of '
                  'state: Willshaw has no masks. BTSP feed-forward weights are a palimpsest (stochastic flips), but the '
                  'Hebbian feedback weights only saturate, as in the paper. The cue_only reference shows whether a '
                  'reconstruction beats the degraded cue. A small toy at one scale: it does not test the paper\'s '
                  'network sizes or claims.'),
    })
    return harness.seal(receipt)


# ====================================================================== NEUROWEAVE extension experiment
_EXT_DEFAULTS: dict[str, Any] = {'memory_dim': 64, 'active_fraction': 0.25, 'density': 0.6, 'plateau_prob': 0.1,
                                 'flip_prob': 0.5, 'threshold': 0.35, 'feedback_threshold': 0.5, 'gate': 0.5,
                                 'projection_dim': 6}
_RUN_DEFAULTS: dict[str, Any] = {'task': 'delayed_recall', 'heldout_seeds': (100, 101), 'steps': 300,
                                 'heldout_steps': None, 'neurons': 48, 'topology': 'clustered', 'mechanisms': ()}


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Baseline vs ``BTSPExtension`` readout features vs the random-trace capacity control on a NEUROWEAVE
    temporal task (default ``delayed_recall``), through ``harness.run_extension_experiment``.

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
    probe = BTSPExtension(seed=0, **ext_cfg)                              # validate before running anything
    ext_cfg = {k: v for k, v in probe.config().items() if k != 'seed'}

    def factory(seed: int) -> BTSPExtension:
        return BTSPExtension(seed=seed, **ext_cfg)

    t0 = time.perf_counter()
    receipt = dict(harness.run_extension_experiment(
        MODULE_ID, factory, seeds=seeds, control='random_trace', metric='accuracy', notes=(
            f'Readout-only extension: the brain output is binarised by top-k (fraction {probe.active_fraction:g}) and '
            f'recalled from a {probe.memory_dim}-unit BTSP memory before any store; features are the recall overlap, '
            f'the recall confidence and {probe.projection_dim} fixed projections of the memory trace. While learning, '
            f'a pattern is stored one-shot when max(|reward|, novelty) exceeds {probe.gate:g}, with the plateau '
            'probability scaled by that salience. It never changes the brain drive. The control gets the same number '
            'of random leaky-trace features of the brain output. Small synthetic task; the verdict is a paired 95 % '
            't interval over seeds.'),
        **run_cfg))
    receipt.pop('digest', None)
    receipt['config'] = {**receipt['config'], 'quick': quick, 'extension': ext_cfg}
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t0}
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment, CAPACITY_ID: run_capacity_benchmark}
