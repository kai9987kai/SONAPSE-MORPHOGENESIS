"""Effective information and causal emergence (frontier module ``causal_emergence``).

Inspired by effective information (EI) and causal emergence as defined by Hoel, Albantakis & Tononi (2013) and
Hoel (2017), and by the macro-vs-micro EI experiments of SONAPSE-ORIGIN (kai9987kai,
``sonapse_origin/theories/causal_emergence.py``). GhostInTheMachine (kai9987kai) measures emergence with the
Rosas et al. (2020) Psi criterion instead; its lesson that an emergence statistic must be read against a matched
null is applied here. This is a synthetic computational abstraction over discretised unit activity, not a measure
of consciousness, agency or biological causation.

What is computed
----------------
* **Effective information.** For a row-stochastic transition probability matrix (TPM) ``T`` over ``n`` states and
  maximum-entropy (uniform) interventions on the current state,
  ``EI(T) = (1/n) sum_i D_KL(T_i || E_D)`` in bits, ``E_D = (1/n) sum_i T_i`` (the effect distribution).
  Equivalently ``EI = H(E_D) - (1/n) sum_i H(T_i)``, the mutual information between uniform interventions and
  their effects.
* **Decomposition (Hoel 2017).** ``determinism = (1/n) sum_i D_KL(T_i || U) / log2 n``,
  ``degeneracy = D_KL(E_D || U) / log2 n``, ``effectiveness = determinism - degeneracy`` and
  ``EI = log2(n) * effectiveness`` (``U`` uniform).
* **Estimation.** Transition counts of a discretised state sequence, additive (Laplace, ``alpha = 1``) smoothing:
  ``T_ij = (C_ij + alpha) / (sum_k C_ik + alpha n)``.
* **Coarse-graining.** A partition of micro states into macro states; the macro TPM is obtained by intervening
  uniformly on the micro states inside each macro state and summing their effects over each macro state
  (Hoel et al. 2013). Candidates are all proper set partitions of a small state space (``3 <= n <= 10``), or
  partitions of binary units into groups whose macro variable is the group's count of active units.
* **Causal emergence.** ``CE = max_macro EI(macro) - EI(micro)`` over proper coarse-grainings (the micro
  description itself is excluded, so ``CE`` can be negative).
* **Nulls.** Two references, both re-running the search for the best coarse-graining in every draw (so the null
  includes the selection of the maximum), each reported as ``z = (CE - mean_null) / sd_null`` and the one-sided
  rank p-value ``(1 + #{CE_null >= CE}) / (1 + draws)``:

  - *successor block permutation* (temporal-dependence null): the successor sequence is block-permuted against
    the current-state sequence, which breaks lag-1 dependence but keeps how often each state was visited;
  - *independent units* (multi-unit data only): every unit's bit sequence is circularly shifted by its own random
    offset, which keeps each unit's own occupancy and autocorrelation but makes the units independent.

  Neither null tests "true macro EI > true micro EI". Smoothed EI estimates shrink the many-state micro TPM more
  than a macro TPM, so with sparse counts the CE estimate rises above the successor null for *any* temporally
  dependent data, including independent slow units whose exact CE is negative (a measured case is in the
  receipt). Against the independent-units null, a large CE means the coarse-graining gains from coordination
  between units beyond the units' own dynamics; the known-answer checks show where each null is calibrated.

References (verified 2026-11-20; notes in ``docs/frontier/modules/causal_emergence.md``)
--------------------------------------------------------------------------------------
* E. P. Hoel, L. Albantakis, G. Tononi, "Quantifying causal emergence shows that macro can beat micro",
  PNAS 110(49):19790-19795, December 2013 (online 18 November 2013), https://doi.org/10.1073/pnas.1314922110
* E. P. Hoel, "When the map is better than the territory", Entropy 19(5):188, published 26 April 2017,
  https://doi.org/10.3390/e19050188 (preprint arXiv:1612.09592, whose v1 text was used to check printed values)
* F. E. Rosas, P. A. M. Mediano, H. J. Jensen, A. K. Seth, A. B. Barrett, R. L. Carhart-Harris, D. Bor,
  "Reconciling emergences: an information-theoretic approach to identify causal emergence in multivariate data",
  PLOS Comput. Biol. 16(12):e1008289, December 2020, https://doi.org/10.1371/journal.pcbi.1008289 (the Psi
  criterion used by GhostInTheMachine; not implemented here)
"""
from __future__ import annotations

import math
import time
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .connectomics import block_permutation
from .spec import EVIDENCE_TIER, FrontierSpecV1

__all__ = ['MODULE_ID', 'CausalEmergenceObserver', 'entropy_bits', 'as_tpm', 'effective_information',
           'ei_decomposition', 'transition_counts', 'estimate_tpm', 'median_binarize', 'encode_bits',
           'decode_states', 'coarse_grain_tpm', 'set_partitions', 'unit_sum_labels', 'unit_sum_candidates',
           'state_partition_candidates', 'macro_ei_batch', 'best_state_partition', 'best_unit_coarse_graining',
           'causal_emergence', 'successor_block_null', 'independent_units_null', 'hoel2017_examples',
           'and_gate_network', 'pair_count_copy_tpm', 'noisy_permutation_tpm', 'independent_flip_tpm',
           'PRINTED_VALUES', 'run_experiment', 'EXPERIMENTS']

MODULE_ID = 'causal_emergence'
OBSERVER_STATE_SCHEMA = 'CausalEmergenceObserverStateV1'
_PERM_OFFSET = 64_001
_EST_OFFSET = 64_011
_IID_OFFSET = 64_021
_CHAIN_OFFSET = 64_031
_NULL_OFFSET = 64_041
_OBS_UNITS_OFFSET = 64_051
_OBS_NULL_OFFSET = 64_061
_INDEP_OFFSET = 64_071
_OBS_INDEP_OFFSET = 64_081
_PAIR_OFFSET = 64_091
_SLOW_OFFSET = 64_101
_UNIT_NULL_OFFSET = 64_111
_PERM8_OFFSET = 64_121
_EPS = 1e-12
_MAX_STATES = 2048
_MAX_SEARCH_STATES = 10
_MAX_OBSERVER_UNITS = 6
_CHUNK_CELLS = 1 << 20                  # candidate chunks hold at most ~1M (candidate, i, j) cells
_SPEC_OBSERVER_PARAMS = frozenset({'sync_pairs', 'sae_features', 'sae_k', 'fc_window', 'max_observed_units',
                                   'observe_every'})


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


def _int_vector(name: str, value: Any, *, min_len: int = 1) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.size < min_len or (arr.size and arr.dtype.kind not in 'iu'):
        raise ValueError(f'{name} must be a 1-D integer array with at least {min_len} entries')
    return arr.astype(np.int64)


# ====================================================================== information measures
def _entropy(P: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the last axis (``0 log 0 = 0``)."""
    L = np.zeros_like(P)
    np.log2(P, out=L, where=P > 0.0)
    return -(P * L).sum(axis=-1)


def entropy_bits(p: Any) -> float:
    """Shannon entropy (bits) of a probability vector (finite, non-negative, summing to 1 within 1e-8)."""
    try:
        v = np.array(p, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('p must be numeric') from exc
    if v.ndim != 1 or v.size < 1 or not np.all(np.isfinite(v)) or np.any(v < 0.0) or abs(float(v.sum()) - 1.0) > 1e-8:
        raise ValueError('p must be a finite, non-negative 1-D probability vector')
    return float(max(0.0, float(_entropy(v / v.sum()))))


def as_tpm(tpm: Any, *, atol: float = 1e-8) -> np.ndarray:
    """A validated float64 copy of a row-stochastic ``(n, n)`` matrix, ``1 <= n <= 2048``. Rows must sum to 1
    within ``atol`` and are then renormalised exactly."""
    try:
        T = np.array(tpm, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('tpm must be numeric') from exc
    if T.ndim != 2 or T.shape[0] != T.shape[1] or not 1 <= T.shape[0] <= _MAX_STATES:
        raise ValueError(f'tpm must be a square matrix with 1 to {_MAX_STATES} states')
    if not np.all(np.isfinite(T)) or np.any(T < 0.0):
        raise ValueError('tpm must be finite and non-negative')
    s = T.sum(axis=1)
    if np.any(np.abs(s - 1.0) > _check_float('atol', atol, 0.0, 1e-3)):
        raise ValueError('every tpm row must sum to 1')
    return T / s[:, None]


def _ei(T: np.ndarray) -> float:
    """EI of a validated TPM: mean KL divergence (bits) of the rows from their mean."""
    m = T.mean(axis=0)
    ratio = np.ones_like(T)
    np.divide(T, m[None, :], out=ratio, where=T > 0.0)        # T_ij > 0 implies m_j >= T_ij / n > 0
    return max(0.0, float((T * np.log2(ratio)).sum() / T.shape[0]))


def effective_information(tpm: Any) -> float:
    """``EI = (1/n) sum_i D_KL(T_i || E_D)`` in bits under uniform interventions (Hoel et al., 2013)."""
    return _ei(as_tpm(tpm))


def _decomposition(T: np.ndarray) -> dict[str, Any]:
    n = T.shape[0]
    ei = _ei(T)
    mean_h = float(_entropy(T).mean())
    h_eff = float(_entropy(T.mean(axis=0)))
    size = math.log2(n)
    out: dict[str, Any] = {'states': int(n), 'ei': ei, 'size_bits': size, 'mean_row_entropy': mean_h,
                           'effect_entropy': h_eff, 'determinism': None, 'degeneracy': None, 'effectiveness': None}
    if n > 1:
        det = min(1.0, max(0.0, (size - mean_h) / size))
        deg = min(1.0, max(0.0, (size - h_eff) / size))
        out.update(determinism=det, degeneracy=deg, effectiveness=det - deg)
    return out


def ei_decomposition(tpm: Any) -> dict[str, Any]:
    """EI with Hoel's (2017) decomposition: ``determinism = (log2 n - mean_i H(T_i)) / log2 n``,
    ``degeneracy = (log2 n - H(E_D)) / log2 n``, ``effectiveness = determinism - degeneracy``, so
    ``EI = log2(n) * effectiveness``. The normalised terms are ``None`` for a single state."""
    return _decomposition(as_tpm(tpm))


# ====================================================================== estimation from data
def _state_sequence(states: Any, n_states: int, *, min_len: int = 2) -> np.ndarray:
    s = _int_vector('states', states, min_len=min_len)
    if s.min() < 0 or s.max() >= n_states:
        raise ValueError(f'states must lie in [0, {n_states})')
    return s


def _counts(cur: np.ndarray, nxt: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(cur * n + nxt, minlength=n * n).reshape(n, n)


def transition_counts(states: Any, n_states: int, *, next_states: Any = None) -> np.ndarray:
    """Counts ``C[i, j]`` of transitions ``i -> j`` (values in ``[0, n_states)``).

    Without ``next_states`` the transitions are the consecutive pairs of one state sequence. With ``next_states``
    (same length as ``states``) they are the pairs ``(states[t], next_states[t])``, e.g. interventional samples
    in which the current state was set and its successor observed."""
    n = _check_int('n_states', n_states, 1, _MAX_STATES)
    if next_states is None:
        s = _state_sequence(states, n)
        return _counts(s[:-1], s[1:], n)
    cur = _state_sequence(states, n, min_len=1)
    nxt = _state_sequence(next_states, n, min_len=1)
    if nxt.size != cur.size:
        raise ValueError('next_states must have the same length as states')
    return _counts(cur, nxt, n)


def _smoothed(C: np.ndarray, alpha: float) -> np.ndarray:
    n = C.shape[0]
    A = C + alpha
    tot = A.sum(axis=1, keepdims=True)
    out = np.full(A.shape, 1.0 / n)
    np.divide(A, tot, out=out, where=tot > 0.0)                # unseen rows with alpha = 0 stay uniform
    return out


def estimate_tpm(counts: Any, *, alpha: float = 1.0) -> np.ndarray:
    """Additively smoothed TPM ``(C + alpha) / (row total + alpha n)``; ``alpha = 1`` is Laplace smoothing.
    Rows without counts (possible only with ``alpha = 0``) are uniform: no evidence, maximum entropy."""
    try:
        C = np.array(counts, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('counts must be numeric') from exc
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not 1 <= C.shape[0] <= _MAX_STATES:
        raise ValueError('counts must be a square matrix')
    if not np.all(np.isfinite(C)) or np.any(C < 0.0):
        raise ValueError('counts must be finite and non-negative')
    return _smoothed(C, _check_float('alpha', alpha, 0.0, 1e6))


def _above_median(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``values > median(reference)`` per column. A value tied with the median counts as on when more of the
    reference lies strictly below the median than strictly above it, so a discrete unit whose upper value is the
    majority (e.g. a 0/1 unit that is on 70 % of the time) is binarised as itself rather than as constant 0;
    for continuous data ties have probability zero and the rule is plain ``x > median``."""
    med = np.median(reference, axis=0)
    on = values > med
    tie = values == med
    if np.any(tie):
        on |= tie & ((reference < med).sum(axis=0) > (reference > med).sum(axis=0))
    return on


def median_binarize(X: Any) -> np.ndarray:
    """Offline per-column binarisation ``x > median(column)`` of a finite ``(T, k)`` array (boolean result); values
    tied with the median go to the side of the median that holds less of the column (see the observer)."""
    try:
        A = np.array(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('X must be numeric') from exc
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1 or not np.all(np.isfinite(A)):
        raise ValueError('X must be a finite 2-D array')
    return _above_median(A, A)


def encode_bits(bits: Any) -> np.ndarray:
    """Integer states ``sum_j bit_j 2^j`` of a ``(T, k)`` 0/1 array, ``1 <= k <= 30``."""
    b = np.asarray(bits)
    if b.ndim != 2 or not 1 <= b.shape[1] <= 30 or b.dtype.kind not in 'biu' or np.any((b != 0) & (b != 1)):
        raise ValueError('bits must be a 2-D 0/1 array with 1 to 30 columns')
    return (b.astype(np.int64) << np.arange(b.shape[1], dtype=np.int64)).sum(axis=1)


def decode_states(states: Any, n_bits: int) -> np.ndarray:
    """Inverse of :func:`encode_bits`: a ``(T, n_bits)`` int8 array of bits."""
    k = _check_int('n_bits', n_bits, 1, 30)
    s = _state_sequence(states, 2 ** k, min_len=1)
    return ((s[:, None] >> np.arange(k, dtype=np.int64)) & 1).astype(np.int8)


# ====================================================================== coarse-graining
def _labels(labels: Any, n: int) -> tuple[np.ndarray, int]:
    lab = _int_vector('labels', labels)
    if lab.size != n or lab.min() < 0:
        raise ValueError(f'labels must hold one non-negative macro label per micro state ({n})')
    m = int(lab.max()) + 1
    if np.unique(lab).size != m:
        raise ValueError('labels must use every macro state 0..m-1')
    return lab, m


def coarse_grain_tpm(tpm: Any, labels: Any) -> np.ndarray:
    """Macro TPM of a partition (Hoel et al., 2013): macro state ``A`` is entered by intervening uniformly on its
    micro states, ``T_M[A, B] = (1/|A|) sum_{i in A} sum_{j in B} T[i, j]``."""
    T = as_tpm(tpm)
    lab, m = _labels(labels, T.shape[0])
    Q = np.bincount((lab[:, None] * m + lab[None, :]).ravel(), weights=T.ravel(), minlength=m * m).reshape(m, m)
    return Q / np.bincount(lab, minlength=m)[:, None]


def set_partitions(n: int) -> np.ndarray:
    """Every set partition of ``n`` items (``1 <= n <= 10``) as restricted-growth label rows, shape ``(B_n, n)``
    (Bell number ``B_n``; 115975 rows for ``n = 10``)."""
    n = _check_int('n', n, 1, _MAX_SEARCH_STATES)
    rgs = np.zeros((1, 1), dtype=np.int64)
    mx = np.zeros(1, dtype=np.int64)
    for _ in range(1, n):
        reps = mx + 2                                       # the next label is 0 .. max + 1
        start = np.repeat(np.cumsum(reps) - reps, reps)
        vals = np.arange(int(reps.sum()), dtype=np.int64) - start
        rgs = np.column_stack([np.repeat(rgs, reps, axis=0), vals])
        mx = np.maximum(np.repeat(mx, reps), vals)
    return rgs


def unit_sum_labels(groups: Any) -> tuple[np.ndarray, int]:
    """Macro labels of the ``2^u`` binary micro states when units are grouped by ``groups`` (one group label per
    unit, every group ``0..G-1`` used) and each group's macro variable is its number of active units. Returns
    ``(labels, m)`` with ``m = prod_g (|g| + 1)`` (mixed-radix encoding of the group counts)."""
    g = _int_vector('groups', groups)
    if not 1 <= g.size <= 12 or g.min() < 0 or np.unique(g).size != int(g.max()) + 1:
        raise ValueError('groups must give 1 to 12 units a group label, using every group 0..G-1')
    u, G = g.size, int(g.max()) + 1
    sizes = np.bincount(g, minlength=G)
    bits = (np.arange(2 ** u, dtype=np.int64)[:, None] >> np.arange(u, dtype=np.int64)) & 1
    member = np.zeros((u, G), dtype=np.int64)
    member[np.arange(u), g] = 1
    radix = np.cumprod(np.r_[1, sizes[:-1] + 1]).astype(np.int64)
    return (bits @ member) @ radix, int(np.prod(sizes + 1))


def unit_sum_candidates(n_units: int) -> dict[str, Any]:
    """All partitions of ``n_units`` binary units (2..6) except all-singletons (the micro description), each
    with its count-per-group macro labels. Keys ``groups (K, u)``, ``labels (K, 2^u)``, ``m (K,)``."""
    u = _check_int('n_units', n_units, 2, _MAX_OBSERVER_UNITS)
    parts = set_partitions(u)
    parts = parts[parts.max(axis=1) < u - 1]
    labs, ms = zip(*(unit_sum_labels(p) for p in parts))
    return {'groups': parts, 'labels': np.stack(labs), 'm': np.asarray(ms, dtype=np.int64)}


def state_partition_candidates(n: int) -> dict[str, Any]:
    """All proper partitions (``2 <= m < n`` macro states) of ``3 <= n <= 10`` micro states."""
    n = _check_int('n', n, 3, _MAX_SEARCH_STATES)
    parts = set_partitions(n)
    m = parts.max(axis=1) + 1
    keep = (m >= 2) & (m < n)
    return {'labels': parts[keep], 'm': m[keep]}


class _Candidates:
    """A chunk of ``K`` coarse-grainings of ``n`` micro states. The macro cells of candidate ``k`` form a compact
    ``m_k x m_k`` block; a sparse 0/1 matrix maps the ``n^2`` micro TPM entries onto all blocks at once."""

    def __init__(self, labels: np.ndarray, m: np.ndarray) -> None:
        from scipy.sparse import csr_matrix
        K, n = labels.shape
        m = m.astype(np.int64)
        self.labels, self.K, self.n, self.m = labels, K, n, m.astype(np.float64)
        sq = m * m
        off_cells = np.r_[0, np.cumsum(sq)[:-1]].astype(np.int64)
        off_cols = np.r_[0, np.cumsum(m)[:-1]].astype(np.int64)
        n_cells, n_cols = int(sq.sum()), int(m.sum())
        cell = off_cells[:, None, None] + labels[:, :, None] * m[:, None, None] + labels[:, None, :]
        # micro entry (i, j) of candidate k adds T[i, j] / |A| to macro cell (A, B), A = label of i, B = label of j
        sizes = np.bincount((off_cols[:, None] + labels).ravel(), minlength=n_cols).astype(np.float64)
        inv_size = 1.0 / sizes[off_cols[:, None] + labels]                     # every macro state is non-empty
        self.S = csr_matrix((np.repeat(inv_size.ravel(), n),
                             (cell.ravel(), np.tile(np.arange(n * n, dtype=np.int64), K))), shape=(n_cells, n * n))
        k_cell = np.repeat(np.arange(K, dtype=np.int64), sq)
        m_cell = m[k_cell]
        cell_col = off_cols[k_cell] + (np.arange(n_cells, dtype=np.int64) - off_cells[k_cell]) % m_cell
        # column means of every macro TPM (its effect distribution under uniform macro interventions)
        self.C = csr_matrix((1.0 / m_cell.astype(np.float64), (cell_col, np.arange(n_cells, dtype=np.int64))),
                            shape=(n_cols, n_cells))
        self.cell_starts, self.col_starts = off_cells, off_cols

    def macro_ei(self, T: np.ndarray) -> np.ndarray:
        """``EI_k = H(mean row) - mean_A H(row A)`` (bits) of every candidate's macro TPM."""
        r = self.S @ T.ravel()                                  # macro TPM entries, block by block
        rlogr = np.add.reduceat(r * np.log2(np.where(r > 0.0, r, 1.0)), self.cell_starts)
        col = self.C @ r                                        # mean rows
        h_mean = -np.add.reduceat(col * np.log2(np.where(col > 0.0, col, 1.0)), self.col_starts)
        return np.maximum(h_mean + rlogr / self.m, 0.0)


def _candidate_chunks(labels_batch: Any, n: int, m: Any = None) -> list[_Candidates]:
    L = np.asarray(labels_batch)
    if L.ndim != 2 or L.shape[0] < 1 or L.shape[1] != n or L.dtype.kind not in 'iu':
        raise ValueError(f'labels_batch must be a non-empty 2-D integer array with {n} columns')
    L = L.astype(np.int64)
    if L.min() < 0:
        raise ValueError('labels must be non-negative')
    mk = L.max(axis=1) + 1
    if m is None:
        used = np.zeros((L.shape[0], int(mk.max())), dtype=bool)
        used[np.repeat(np.arange(L.shape[0]), n), L.ravel()] = True
        if not np.array_equal(used.sum(axis=1), mk):
            raise ValueError('every labels row must use each macro state 0..m-1')
    step = max(1, _CHUNK_CELLS // (n * n))
    return [_Candidates(L[i:i + step], mk[i:i + step]) for i in range(0, L.shape[0], step)]


def _batch_ei(T: np.ndarray, chunks: Sequence[_Candidates]) -> np.ndarray:
    return np.concatenate([c.macro_ei(T) for c in chunks])


def macro_ei_batch(tpm: Any, labels_batch: Any) -> np.ndarray:
    """EI of the macro TPM of every coarse-graining in ``labels_batch`` (``(K, n)``, one partition per row)."""
    T = as_tpm(tpm)
    return _batch_ei(T, _candidate_chunks(labels_batch, T.shape[0]))


def best_state_partition(tpm: Any) -> dict[str, Any]:
    """Exhaustive search over every proper partition of a ``3 <= n <= 10`` state TPM for the macro TPM with the
    largest EI (ties: the first partition in restricted-growth order)."""
    T = as_tpm(tpm)
    cand = state_partition_candidates(T.shape[0])
    eis = _batch_ei(T, _candidate_chunks(cand['labels'], T.shape[0], cand['m']))
    k = int(np.argmax(eis))
    micro = _ei(T)
    return {'ei_micro': micro, 'ei_macro': float(eis[k]), 'ce': float(eis[k]) - micro,
            'labels': cand['labels'][k].copy(), 'macro_states': int(cand['m'][k]),
            'partitions_evaluated': int(eis.size), 'macro_tpm': coarse_grain_tpm(T, cand['labels'][k])}


def best_unit_coarse_graining(tpm: Any, n_units: int) -> dict[str, Any]:
    """Best count-per-group coarse-graining of a TPM over ``2^n_units`` binary micro states (``2 <= n_units <= 6``)."""
    T = as_tpm(tpm)
    cand = unit_sum_candidates(n_units)
    if T.shape[0] != 2 ** int(n_units):
        raise ValueError('tpm must have 2^n_units states')
    eis = _batch_ei(T, _candidate_chunks(cand['labels'], T.shape[0], cand['m']))
    k = int(np.argmax(eis))
    micro = _ei(T)
    return {'ei_micro': micro, 'ei_macro': float(eis[k]), 'ce': float(eis[k]) - micro,
            'groups': _groups(cand['groups'][k], np.arange(int(n_units))), 'labels': cand['labels'][k].copy(),
            'macro_states': int(cand['m'][k]), 'partitions_evaluated': int(eis.size)}


def _groups(rgs: np.ndarray, units: np.ndarray) -> list[list[int]]:
    return [[int(units[j]) for j in np.flatnonzero(rgs == g)] for g in range(int(rgs.max()) + 1)]


def causal_emergence(tpm: Any, labels: Any) -> dict[str, Any]:
    """Micro and macro EI (with decompositions) of one coarse-graining and ``CE = EI_macro - EI_micro``."""
    T = as_tpm(tpm)
    M = coarse_grain_tpm(T, labels)
    micro, macro = _decomposition(T), _decomposition(M)
    return {'micro': micro, 'macro': macro, 'ce': macro['ei'] - micro['ei'], 'macro_tpm': M}


# ====================================================================== null distribution
def _default_block(transitions: int) -> int:
    return max(1, int(round(transitions ** (1.0 / 3.0)))) if transitions >= 8 else 1


def _ce_of_pairs(cur: np.ndarray, nxt: np.ndarray, n: int, alpha: float,
                 chunks: Sequence[_Candidates]) -> tuple[float, np.ndarray, int]:
    T = _smoothed(_counts(cur, nxt, n), alpha)
    eis = _batch_ei(T, chunks)
    return _ei(T), eis, int(np.argmax(eis))


def _null_stats(ce: float, null: np.ndarray) -> dict[str, Any]:
    mu = float(null.mean())
    sd = float(null.std(ddof=1)) if null.size > 1 else 0.0
    return {'null_mean': mu, 'null_sd': sd, 'excess': ce - mu, 'z': (ce - mu) / sd if sd > _EPS else None,
            'p': float((1 + int((null >= ce - 1e-12).sum())) / (1 + null.size))}


def successor_block_null(states: Any, n_states: int, labels_batch: Any, *, alpha: float = 1.0, draws: int = 99,
                         block: int | None = None, seed: int = 0) -> dict[str, Any]:
    """CE of a state sequence (best of the coarse-grainings in ``labels_batch``) against a successor
    block-permutation null.

    The TPM is estimated from the pairs ``(s_t, s_{t+1})`` with smoothing ``alpha``. In each of ``draws`` null
    draws the successor sequence ``s_{1..T}`` is block-permuted (blocks of ``block`` steps keep their order,
    default ``round(transitions^(1/3))``) against the unchanged ``s_{0..T-1}``: every state keeps its visit count
    and the successor marginal is kept, but lag-1 dependence is broken. The best coarse-graining is searched again
    per draw. Returns observed EI/CE, the best partition, the null CE values and ``z`` / one-sided ``p``."""
    n = _check_int('n_states', n_states, 2, _MAX_STATES)
    s = _state_sequence(states, n, min_len=3)
    chunks = _candidate_chunks(labels_batch, n)
    alpha = _check_float('alpha', alpha, 0.0, 1e6)
    draws = _check_int('draws', draws, 2, 100_000)
    block = None if block is None else _check_int('block', block, 1, s.size - 1)
    return _successor_block_null(s, n, chunks, alpha, draws, block, _check_int('seed', seed, 0, 2**31 - 1))


def _successor_null_values(cur: np.ndarray, nxt: np.ndarray, n: int, chunks: Sequence[_Candidates], alpha: float,
                           draws: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Null CE values with the successor sequence block-permuted against the current-state sequence."""
    out = np.empty(draws)
    for b in range(draws):
        m_b, e_b, k_b = _ce_of_pairs(cur, nxt[block_permutation(cur.size, block, rng)], n, alpha, chunks)
        out[b] = float(e_b[k_b]) - m_b
    return out


def _successor_block_null(s: np.ndarray, n: int, chunks: Sequence[_Candidates], alpha: float, draws: int,
                          block: int | None, seed: int) -> dict[str, Any]:
    """:func:`successor_block_null` on validated inputs and pre-built candidate chunks."""
    cur, nxt = s[:-1], s[1:]
    block = _default_block(cur.size) if block is None else int(block)
    rng = np.random.default_rng(seed + _NULL_OFFSET)
    micro, eis, k = _ce_of_pairs(cur, nxt, n, alpha, chunks)
    ce = float(eis[k]) - micro
    null = _successor_null_values(cur, nxt, n, chunks, alpha, draws, block, rng)
    labels = np.concatenate([c.labels for c in chunks])
    return {'ei_micro': micro, 'ei_macro': float(eis[k]), 'ce': ce, 'best_index': k, 'labels': labels[k].copy(),
            'null_ce': null, 'draws': draws, 'block': block, 'alpha': alpha, 'transitions': int(cur.size),
            **_null_stats(ce, null)}


def _indep_null_values(states: np.ndarray, n_units: int, chunks: Sequence[_Candidates], alpha: float, draws: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Null CE values with every unit's bit sequence circularly shifted by its own offset in ``[1, L - 1]``."""
    L = int(states.size)
    bits = (states[:, None] >> np.arange(n_units, dtype=np.int64)) & 1
    weights = np.left_shift(1, np.arange(n_units, dtype=np.int64))
    rows, cols = np.arange(L, dtype=np.int64)[:, None], np.arange(n_units, dtype=np.int64)[None, :]
    out = np.empty(draws)
    for b in range(draws):
        shifted = bits[(rows - rng.integers(1, L, n_units)[None, :]) % L, cols] @ weights   # np.roll per unit
        micro, eis, k = _ce_of_pairs(shifted[:-1], shifted[1:], 2 ** n_units, alpha, chunks)
        out[b] = float(eis[k]) - micro
    return out


def independent_units_null(states: Any, n_units: int, labels_batch: Any, *, alpha: float = 1.0, draws: int = 99,
                           seed: int = 0) -> dict[str, Any]:
    """CE of a binary-unit state sequence against an independent-units null.

    ``states`` encode ``n_units`` binary units (bit ``j`` of a state is unit ``j``, see :func:`encode_bits`). In
    each of ``draws`` null draws every unit's bit sequence is circularly shifted by its own random offset in
    ``[1, L - 1]``: each unit keeps its occupancy and autocorrelation (up to one wrap-around transition) while the
    units become independent of each other. The best coarse-graining in ``labels_batch`` (``(K, 2^n_units)``) is
    searched again per draw. A large ``z`` says the coarse-grained description gains from coordination between
    units beyond what the units' own dynamics, and the estimator's finite-sample behaviour on them, give."""
    u = _check_int('n_units', n_units, 2, 11)
    n = 2 ** u
    s = _state_sequence(states, n, min_len=3)
    chunks = _candidate_chunks(labels_batch, n)
    alpha = _check_float('alpha', alpha, 0.0, 1e6)
    draws = _check_int('draws', draws, 2, 100_000)
    rng = np.random.default_rng(_check_int('seed', seed, 0, 2**31 - 1) + _INDEP_OFFSET)
    micro, eis, k = _ce_of_pairs(s[:-1], s[1:], n, alpha, chunks)
    ce = float(eis[k]) - micro
    null = _indep_null_values(s, u, chunks, alpha, draws, rng)
    labels = np.concatenate([c.labels for c in chunks])
    return {'ei_micro': micro, 'ei_macro': float(eis[k]), 'ce': ce, 'best_index': k, 'labels': labels[k].copy(),
            'null_ce': null, 'draws': draws, 'alpha': alpha, 'transitions': int(s.size - 1), **_null_stats(ce, null)}


def _ce_and_nulls(seq: np.ndarray, n_units: int, chunks: Sequence[_Candidates], alpha: float, draws: int,
                  block: int | None, rng_successor: np.random.Generator,
                  rng_independent: np.random.Generator) -> dict[str, Any]:
    """Observed CE of a binary-unit state sequence plus both null summaries (shared by the observer and the lab)."""
    n = 2 ** n_units
    cur, nxt = seq[:-1], seq[1:]
    micro, eis, k = _ce_of_pairs(cur, nxt, n, alpha, chunks)
    ce = float(eis[k]) - micro
    blk = min(_default_block(cur.size) if block is None else int(block), int(cur.size))
    succ = _successor_null_values(cur, nxt, n, chunks, alpha, draws, blk, rng_successor)
    indep = _indep_null_values(seq, n_units, chunks, alpha, draws, rng_independent)
    return {'ei_micro': micro, 'ei_macro': float(eis[k]), 'best_index': k, 'ce': ce, 'block': blk,
            'successor': _null_stats(ce, succ), 'independent': _null_stats(ce, indep)}


# ====================================================================== published toy systems
def hoel2017_examples() -> dict[str, np.ndarray]:
    """The Markov chains printed in Hoel (2017): ``M1``, ``M2``, ``M3`` (EI 2, 1, 0 bits), the 8-state chain ``Sm``
    (seven states mix uniformly among themselves, the eighth is absorbing), its variant with unequal rows, and the
    4-state channel ``X -> Y`` of the channel-capacity section (three states mix uniformly, the fourth is
    absorbing; printed EI 0.81 bits at the micro scale and 1 bit for the macro code {x1, x2, x3} -> 0, x4 -> 1)."""
    third, F = 1.0 / 3.0, Fraction
    chain = np.zeros((8, 8))
    chain[:7, :7] = 1.0 / 7.0
    chain[7, 7] = 1.0
    varied = [[F(1, 5)] * 5 + [0, 0, 0],
              [F(1, 7), F(3, 7), F(1, 7), 0, F(1, 7), 0, F(1, 7), 0],
              [0] + [F(1, 6)] * 6 + [0],
              [F(1, 7), 0, F(1, 7), F(1, 7), F(1, 7), F(1, 7), F(2, 7), 0],
              [F(1, 9), F(2, 9), F(2, 9), F(1, 9), 0, F(2, 9), F(1, 9), 0],
              [F(1, 7)] * 7 + [0],
              [F(1, 6), F(1, 6), 0, F(1, 6), F(1, 6), F(1, 6), F(1, 6), 0],
              [0] * 7 + [1]]
    return {'M1': np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]], dtype=np.float64),
            'M2': np.array([[third, third, third, 0], [third, third, third, 0], [0, 0, 0, 1], [0, 0, 0, 1]]),
            'M3': np.full((4, 4), 0.25), 'chain8': chain,
            'chain8_varied': np.array([[float(x) for x in row] for row in varied]),
            'channel4': np.array([[third, third, third, 0], [third, third, third, 0], [third, third, third, 0],
                                  [0, 0, 0, 1]])}


def and_gate_network(pairs: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ring of ``2 * pairs`` AND gates in the spirit of Hoel et al. (2013, Fig. 4): both elements of
    pair ``k`` compute AND of the two elements of pair ``k + 1`` (mod ``pairs``). Bit ``2k`` and ``2k + 1`` of a
    state belong to pair ``k``. Returns the micro TPM and the macro labels that map each pair to one element that
    is on only when both of its micro elements are on (the macro elements then copy each other around the ring)."""
    p = _check_int('pairs', pairs, 1, 5)
    u = 2 * p
    states = np.arange(2 ** u, dtype=np.int64)
    bits = (states[:, None] >> np.arange(u, dtype=np.int64)) & 1
    pair_and = bits[:, 0::2] & bits[:, 1::2]
    nxt_bits = np.repeat(np.roll(pair_and, -1, axis=1), 2, axis=1)
    nxt = (nxt_bits << np.arange(u, dtype=np.int64)).sum(axis=1)
    T = np.zeros((2 ** u, 2 ** u))
    T[states, nxt] = 1.0
    return T, (pair_and << np.arange(p, dtype=np.int64)).sum(axis=1)


def pair_count_copy_tpm(pairs: int = 3, noise: float = 0.1) -> np.ndarray:
    """Exact micro TPM of a ring of binary-unit pairs whose active-unit *counts* copy each other.

    Pair ``k`` (units ``2k`` and ``2k + 1``) takes the count of pair ``k + 1`` (mod ``pairs``) with probability
    ``1 - noise`` and a uniformly random count in {0, 1, 2} otherwise; count 1 is realised as either unit alone
    with probability 1/2. The pair counts follow near-deterministic macro dynamics over a degenerate, noisy micro
    realisation, so the count-per-pair coarse-graining has higher EI than the micro TPM. This is a constructed
    known-answer system in the spirit of Hoel et al. (2013), not an example printed there."""
    p = _check_int('pairs', pairs, 1, 5)
    eps = _check_float('noise', noise, 0.0, 1.0)
    u, n = 2 * p, 4 ** p
    bits = decode_states(np.arange(n), u).astype(np.int64)
    counts = bits[:, 0::2] + bits[:, 1::2]                              # (n, pairs) counts of the current state
    p_count = (1.0 - eps) * (np.roll(counts, -1, axis=1)[:, :, None] == np.arange(3)) + eps / 3.0
    # P(pair micro state | count) for pair micro index b0 + 2 b1 = 00, 10, 01, 11
    realise = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
    T = np.ones((n, n))
    for k in range(p):
        T *= p_count[:, k, :] @ realise[bits[:, 2 * k] + 2 * bits[:, 2 * k + 1]].T
    return T


def noisy_permutation_tpm(n: int, noise: float, *, seed: int = 0) -> np.ndarray:
    """``(1 - noise) P + noise / n`` for a seeded random ``n x n`` permutation matrix ``P``.

    Every row is a permutation of every other row and all column sums are 1, so the TPM is a weakly symmetric
    channel. Uniform inputs achieve its capacity, and since a coarse-graining only changes the input distribution
    (uniform over macro states) and post-processes inputs and outputs, no coarse-graining has a larger EI: the
    exact CE of every partition is <= 0 (Hoel 2017: no causal emergence without asymmetry)."""
    n = _check_int('n', n, 2, _MAX_STATES)
    eps = _check_float('noise', noise, 0.0, 1.0)
    P = np.eye(n)[np.random.default_rng(_check_int('seed', seed, 0, 2**31 - 1) + _PERM8_OFFSET).permutation(n)]
    return (1.0 - eps) * P + eps / n


def independent_flip_tpm(n_units: int, keep: float) -> np.ndarray:
    """Exact micro TPM of ``n_units`` independent binary units that each keep their state with probability
    ``keep`` and flip otherwise (bit ``j`` of a state is unit ``j``)."""
    u = _check_int('n_units', n_units, 1, 11)
    q = _check_float('keep', keep, 0.0, 1.0)
    bits = decode_states(np.arange(2 ** u), u)
    same = (bits[:, None, :] == bits[None, :, :]).sum(axis=2)
    return q ** same * (1.0 - q) ** (u - same)


def _sample_pair_count_copy(pairs: int, noise: float, steps: int, rng: np.random.Generator) -> np.ndarray:
    c = rng.integers(0, 3, pairs)
    out = np.zeros((steps, 2 * pairs), dtype=np.int64)
    for t in range(steps):
        c = np.where(rng.random(pairs) < noise, rng.integers(0, 3, pairs), np.roll(c, -1))
        second = rng.integers(0, 2, pairs) == 1                         # which unit realises a count of 1
        out[t, 0::2] = (c == 2) | ((c == 1) & ~second)
        out[t, 1::2] = (c == 2) | ((c == 1) & second)
    return out


def _sample_independent_flips(n_units: int, keep: float, steps: int, rng: np.random.Generator) -> np.ndarray:
    start = rng.integers(0, 2, n_units)
    return (start[None, :] + np.cumsum(rng.random((steps, n_units)) >= keep, axis=0)) % 2


PRINTED_VALUES: dict[str, float] = {
    # Hoel (2017), Entropy 19:188, as printed in its preprint arXiv:1612.09592v1 (usually 2 decimals)
    'hoel2017_M1_ei': 2.0, 'hoel2017_M2_ei': 1.0, 'hoel2017_M3_ei': 0.0,
    'hoel2017_chain8_ei_micro': 0.55, 'hoel2017_chain8_effectiveness': 0.18, 'hoel2017_chain8_ei_macro': 1.0,
    'hoel2017_chain8_ce': 0.45, 'hoel2017_chain8_varied_ei_micro': 0.81, 'hoel2017_chain8_varied_ei_macro': 1.0,
    'hoel2017_and_ei_micro': 2.43, 'hoel2017_and_effectiveness': 0.41, 'hoel2017_and_degeneracy': 0.59,
    'hoel2017_and_ei_macro': 3.0, 'hoel2017_channel4_ei_micro': 0.81, 'hoel2017_channel4_conditional_entropy': 1.19,
    'hoel2017_channel4_ei_macro': 1.0,
    # Hoel, Albantakis & Tononi (2013), PNAS 110:19790, Fig. 4 (six AND gates -> three COPY gates)
    'hoel2013_and_ei_micro': 2.43, 'hoel2013_and_effectiveness': 0.4, 'hoel2013_and_determinism': 1.0,
    'hoel2013_and_degeneracy': 0.6, 'hoel2013_and_ei_macro': 3.0, 'hoel2013_and_ce': 0.57,
}


# ====================================================================== live observer
class CausalEmergenceObserver:
    """Read-only live observer: EI and causal emergence of a few discretised units.

    * ``u = min(units, n_units)`` units (``2 <= u <= 6``; a seeded, sorted subset when ``n_units > units``) are
      binarised online: each new value is compared with the median of that unit's last ``history`` raw values
      (including the new one), so a unit is on when it is above its running median (a value tied with the median
      is on when more of the window lies below the median than above it, which keeps discrete 0/1 units intact).
      The ``2^u`` micro states of the last ``history`` observations are kept in a ring buffer (bounded transition
      counts). With ``alpha = 1`` every row of the ``2^u``-state TPM receives ``2^u`` pseudo-counts, so until there
      are many transitions per micro state (``transitions_per_micro_state``) EI estimates are shrunk far below the
      exact values; the nulls below, not the raw ``ce``, carry the evidence.
    * ``summary()`` estimates the micro TPM (smoothing ``alpha``; 1 = Laplace) from the buffered transitions and
      reports micro EI with determinism/degeneracy, the best count-per-group coarse-graining over every partition
      of the units (the micro description excluded) and ``ce = ei_macro - ei_micro``. Once ``min_transitions``
      transitions exist (default ``min(32, history - 1)``) it adds two nulls of ``null_draws`` draws each, seeded
      by ``(seed, observed steps)``: the successor block permutation (``null_ce_mean``, ``null_ce_sd``,
      ``ce_excess``, ``ce_z``, ``ce_p``: is CE above what temporally independent data give?) and the
      independent-units circular shift (``indep_ce_mean``, ``indep_ce_sd``, ``ce_excess_indep``, ``ce_z_indep``,
      ``ce_p_indep``: is CE above what the same units give when they are made independent?). The smallest
      attainable p is ``1 / (null_draws + 1)`` (0.05 at the default 19 draws). With few transitions per micro
      state the successor null flags any temporal dependence (see the module docstring); read emergence claims
      against the independent-units null and the lab's known-answer checks.
    * Results are a pure function of the buffered state and are cached per observed step, so they do not depend
      on how often ``summary()`` is called. Unit indices refer to positions in the observed vector. FrontierSpecV1
      parameters meant for other observers are accepted and ignored."""

    module_id = MODULE_ID
    schema = OBSERVER_STATE_SCHEMA

    def __init__(self, n_units: int, *, seed: int = 7, history: int = 256, units: int = 6, alpha: float = 1.0,
                 null_draws: int = 19, block: int | None = None, min_transitions: int | None = None,
                 **params: Any) -> None:
        unknown = sorted(set(params) - _SPEC_OBSERVER_PARAMS)
        if unknown:
            raise ValueError(f'unknown CausalEmergenceObserver parameters: {unknown}')
        self.n_units = _check_int('n_units', n_units, 2, 1_000_000)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        self.history = _check_int('history', history, 16, 8192)
        self.units = _check_int('units', units, 2, _MAX_OBSERVER_UNITS)
        self.alpha = _check_float('alpha', alpha, 0.0, 100.0)
        self.null_draws = _check_int('null_draws', null_draws, 2, 999)
        self.block = None if block is None else _check_int('block', block, 1, self.history)
        self.min_transitions = (min(32, self.history - 1) if min_transitions is None
                                else _check_int('min_transitions', min_transitions, 8, self.history - 1))
        m = self.m = min(self.units, self.n_units)
        if m == self.n_units:
            sample = np.arange(m, dtype=np.int64)
        else:
            sample = np.sort(np.random.default_rng(self.seed + _OBS_UNITS_OFFSET).choice(self.n_units, m, replace=False))
        self._units = sample.astype(np.int64)
        self.n_states = 2 ** m
        cand = unit_sum_candidates(m)
        self._groups = cand['groups']
        self._chunks = _candidate_chunks(cand['labels'], self.n_states, cand['m'])
        self._bit_weights = np.left_shift(1, np.arange(m, dtype=np.int64))
        self.reset()

    @classmethod
    def from_spec(cls, n_units: int, spec: Any, **params: Any) -> 'CausalEmergenceObserver':
        """Observer sized from a :class:`~synapse.frontier.spec.FrontierSpecV1` (``seed``, ``history``); ``params``
        are the observer's own keyword parameters (``units``, ``alpha``, ...)."""
        if not isinstance(spec, FrontierSpecV1):
            raise ValueError('spec must be a FrontierSpecV1')
        return cls(n_units, seed=int(spec.seed), history=int(spec.history), **params)

    def config(self) -> dict[str, Any]:
        return {'n_units': self.n_units, 'seed': self.seed, 'history': self.history, 'units': self.units,
                'alpha': self.alpha, 'null_draws': self.null_draws, 'block': self.block,
                'min_transitions': self.min_transitions}

    def reset(self) -> None:
        self._raw = np.zeros((self.history, self.m))
        self._states = np.zeros(self.history, dtype=np.int64)
        self._pos = 0
        self._filled = 0
        self._steps = 0
        self._cache: tuple[int, dict[str, Any]] | None = None

    @property
    def observed_steps(self) -> int:
        return int(self._steps)

    @property
    def units_observed(self) -> np.ndarray:
        return self._units.copy()

    def _encode(self, raw: np.ndarray, buffer: np.ndarray) -> int:
        return int((_above_median(raw, buffer).astype(np.int64) * self._bit_weights).sum())

    def observe(self, x: np.ndarray) -> None:
        try:
            v = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'observation must be a finite 1-D array of length {self.n_units}') from exc
        if v.shape != (self.n_units,) or not np.all(np.isfinite(v)):
            raise ValueError(f'observation must be a finite 1-D array of length {self.n_units}')
        raw = v[self._units]                                 # fancy indexing copies; the input is never written
        p = self._pos
        self._raw[p] = raw
        self._filled = min(self.history, self._filled + 1)
        self._states[p] = self._encode(raw, self._raw[:self._filled])
        self._pos = (p + 1) % self.history
        self._steps += 1
        self._cache = None

    def _sequence(self) -> np.ndarray:
        if self._filled < self.history:
            return self._states[:self._filled].copy()
        return np.concatenate([self._states[self._pos:], self._states[:self._pos]])

    def _compute(self) -> dict[str, Any]:
        seq = self._sequence()
        cur, nxt = seq[:-1], seq[1:]
        n = self.n_states
        micro_T = _smoothed(_counts(cur, nxt, n), self.alpha)
        micro = _decomposition(micro_T)
        eis = _batch_ei(micro_T, self._chunks)
        k = int(np.argmax(eis))
        labels = np.concatenate([c.labels for c in self._chunks])[k]
        macro = _decomposition(coarse_grain_tpm(micro_T, labels))
        ce = float(eis[k]) - micro['ei']
        out: dict[str, Any] = {
            'visited_states': int(np.unique(cur).size), 'ei_micro': micro['ei'],
            'determinism_micro': micro['determinism'], 'degeneracy_micro': micro['degeneracy'],
            'effectiveness_micro': micro['effectiveness'], 'ei_macro': float(eis[k]),
            'determinism_macro': macro['determinism'], 'degeneracy_macro': macro['degeneracy'],
            'macro_partition': _groups(self._groups[k], self._units), 'macro_states': int(macro['states']),
            'ce': ce, 'null_ready': False, 'null_block': None, 'null_ce_mean': None, 'null_ce_sd': None,
            'ce_excess': None, 'ce_z': None, 'ce_p': None, 'indep_ce_mean': None, 'indep_ce_sd': None,
            'ce_excess_indep': None, 'ce_z_indep': None, 'ce_p_indep': None}
        if cur.size >= self.min_transitions:
            res = _ce_and_nulls(seq, self.m, self._chunks, self.alpha, self.null_draws, self.block,
                                np.random.default_rng([self.seed + _OBS_NULL_OFFSET, self._steps]),
                                np.random.default_rng([self.seed + _OBS_INDEP_OFFSET, self._steps]))
            st, si = res['successor'], res['independent']
            out.update(null_ready=True, null_block=int(res['block']), null_ce_mean=st['null_mean'],
                       null_ce_sd=st['null_sd'], ce_excess=st['excess'], ce_z=st['z'], ce_p=st['p'],
                       indep_ce_mean=si['null_mean'], indep_ce_sd=si['null_sd'], ce_excess_indep=si['excess'],
                       ce_z_indep=si['z'], ce_p_indep=si['p'])
        return out

    _METRIC_KEYS = ('visited_states', 'ei_micro', 'determinism_micro', 'degeneracy_micro', 'effectiveness_micro',
                    'ei_macro', 'determinism_macro', 'degeneracy_macro', 'macro_partition', 'macro_states', 'ce',
                    'null_ready', 'null_block', 'null_ce_mean', 'null_ce_sd', 'ce_excess', 'ce_z', 'ce_p',
                    'indep_ce_mean', 'indep_ce_sd', 'ce_excess_indep', 'ce_z_indep', 'ce_p_indep')

    def summary(self) -> dict[str, Any]:
        """JSON-safe metrics (``None`` until the first transition; null fields once ``min_transitions`` exist)."""
        transitions = max(0, self._filled - 1)
        out: dict[str, Any] = {'module': MODULE_ID, 'evidence_tier': EVIDENCE_TIER, 'n_units': self.n_units,
                               'analysed_units': int(self.m), 'unit_sample': self._units.tolist(),
                               'history': self.history, 'micro_states': int(self.n_states), 'alpha': self.alpha,
                               'null_draws': self.null_draws, 'min_transitions': self.min_transitions,
                               'observed_steps': int(self._steps), 'buffered': int(self._filled),
                               'transitions': int(transitions),
                               'transitions_per_micro_state': float(transitions / self.n_states)}
        if transitions == 0:
            out.update({k: None for k in self._METRIC_KEYS})
            out['null_ready'] = False
            return out
        if self._cache is None or self._cache[0] != self._steps:
            self._cache = (self._steps, self._compute())
        metrics = self._cache[1]
        out.update({k: ([list(g) for g in v] if k == 'macro_partition' else v) for k, v in metrics.items()})
        return out

    def export_state(self) -> dict[str, Any]:
        return {'module': MODULE_ID, 'schema': self.schema, 'config': self.config(), 'units': self._units.tolist(),
                'raw': self._raw.tolist(), 'states': self._states.tolist(), 'pos': int(self._pos),
                'filled': int(self._filled), 'steps': int(self._steps)}

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate the whole state first (configuration, shapes, counters and, where the raw values are still
        buffered, that every stored state is the median-binarisation of its raw values); then replace."""
        if not isinstance(state, Mapping) or state.get('module') != MODULE_ID or state.get('schema') != self.schema:
            raise ValueError('state does not belong to a CausalEmergenceObserver')
        try:
            cfg = dict(state['config'])
            units = state['units']
            raw = np.array(state['raw'], dtype=np.float64)
            st = np.array(state['states'])
            pos, filled, steps = state['pos'], state['filled'], state['steps']
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('CausalEmergenceObserver state is incomplete or malformed') from exc
        if cfg != self.config() or units != self._units.tolist():
            raise ValueError('CausalEmergenceObserver state was made with a different configuration')
        H = self.history
        if raw.shape != (H, self.m) or not np.all(np.isfinite(raw)):
            raise ValueError('raw buffer has the wrong shape or non-finite values')
        if st.shape != (H,) or (st.size and st.dtype.kind not in 'iu') or np.any(st < 0) or np.any(st >= self.n_states):
            raise ValueError('state buffer must hold one micro state in [0, 2^units) per history row')
        if not all(type(v) is int for v in (pos, filled, steps)) or not (0 <= pos < H and 0 <= filled <= H
                                                                         and filled == min(steps, H)):
            raise ValueError('CausalEmergenceObserver counters are inconsistent')
        if pos != steps % H:
            raise ValueError('CausalEmergenceObserver ring-buffer position is inconsistent')
        st = st.astype(np.int64)
        if filled < H and (np.any(raw[filled:] != 0.0) or np.any(st[filled:] != 0)):
            raise ValueError('buffers hold values beyond the observed steps')
        if steps <= H:                                       # every raw value is still buffered: check them all
            ok = all(st[t] == self._encode(raw[t], raw[:t + 1]) for t in range(filled))
        else:                                                # only the newest state's median is still known
            last = (pos - 1) % H
            ok = st[last] == self._encode(raw[last], raw)
        if not ok:
            raise ValueError('stored states are not the median-binarisation of the raw buffer')
        self._raw, self._states = raw.copy(), st.copy()
        self._pos, self._filled, self._steps = pos, filled, steps
        self._cache = None


# ====================================================================== lab experiment
_MECH_DYNAMICS = ('sparse_code', 'synaptic_delays', 'membrane_dynamics')
_PAIR_GROUPS = [[0, 1], [2, 3], [4, 5]]
_SEVEN_PLUS_ONE = [0] * 7 + [1]
_DEFAULTS: dict[str, Any] = {
    'sizes': (2, 4, 8, 16, 32), 'alpha': 1.0, 'estimation_samples': 16_000, 'estimation_tolerance': 0.05,
    'null_steps': 512, 'null_draws': 199, 'null_replicates': 10, 'emergence_noise': 0.1, 'permutation_noise': 0.3,
    'unit_steps': 1024, 'unit_draws': 49, 'unit_replicates': 4, 'pair_copy_noise': 0.1, 'independent_keep': 0.9,
    # 4 units (16 micro states) for the NEUROWEAVE report: ~11 (quick) / ~34 (default) transitions per micro
    # state, against ~3 / ~9 with the observer's default of 6 units (64 states)
    'units': 4, 'topology': 'clustered', 'neurons': 32, 'nw_steps': 600, 'nw_burn_in': 50, 'nw_null_draws': 49,
    'mechanism_sets': {'baseline': (), 'dynamics': _MECH_DYNAMICS},
}
_QUICK: dict[str, Any] = {'estimation_samples': 4000, 'null_steps': 384, 'null_draws': 39, 'null_replicates': 2,
                          'unit_steps': 512, 'unit_draws': 19, 'unit_replicates': 1, 'neurons': 24, 'nw_steps': 200,
                          'nw_burn_in': 24, 'nw_null_draws': 19}


def _validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    from ..neuroweave.spec import MECHANISM_IDS, TOPOLOGIES
    c = dict(cfg)
    try:
        sizes = tuple(c['sizes'])
    except TypeError as exc:
        raise ValueError('sizes must be a sequence of ints') from exc
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError('sizes must be non-empty and distinct')
    c['sizes'] = [_check_int('size', s, 2, 512) for s in sizes]
    c['alpha'] = _check_float('alpha', c['alpha'], 0.0, 100.0)
    c['estimation_samples'] = _check_int('estimation_samples', c['estimation_samples'], 200, 10_000_000)
    c['estimation_tolerance'] = _check_float('estimation_tolerance', c['estimation_tolerance'], 1e-4, 1.0)
    c['null_steps'] = _check_int('null_steps', c['null_steps'], 64, 1_000_000)
    c['null_draws'] = _check_int('null_draws', c['null_draws'], 19, 100_000)
    c['null_replicates'] = _check_int('null_replicates', c['null_replicates'], 1, 10_000)
    c['emergence_noise'] = _check_float('emergence_noise', c['emergence_noise'], 0.01, 0.3)
    c['permutation_noise'] = _check_float('permutation_noise', c['permutation_noise'], 0.01, 0.5)
    c['unit_steps'] = _check_int('unit_steps', c['unit_steps'], 64, 1_000_000)
    c['unit_draws'] = _check_int('unit_draws', c['unit_draws'], 19, 100_000)
    c['unit_replicates'] = _check_int('unit_replicates', c['unit_replicates'], 1, 10_000)
    c['pair_copy_noise'] = _check_float('pair_copy_noise', c['pair_copy_noise'], 0.0, 0.5)
    c['independent_keep'] = _check_float('independent_keep', c['independent_keep'], 0.6, 0.99)
    c['units'] = _check_int('units', c['units'], 2, _MAX_OBSERVER_UNITS)
    if c['topology'] not in TOPOLOGIES:
        raise ValueError(f'topology must be one of {TOPOLOGIES}')
    c['neurons'] = _check_int('neurons', c['neurons'], 8, 512)
    c['nw_steps'] = _check_int('nw_steps', c['nw_steps'], 64, 100_000)
    c['nw_burn_in'] = _check_int('nw_burn_in', c['nw_burn_in'], 0, c['nw_steps'] - 48)
    c['nw_null_draws'] = _check_int('nw_null_draws', c['nw_null_draws'], 2, 999)
    sets = c['mechanism_sets']
    if not isinstance(sets, Mapping) or not 1 <= len(sets) <= 8:
        raise ValueError('mechanism_sets must map 1 to 8 names to NEUROWEAVE mechanism tuples')
    clean: dict[str, list[str]] = {}
    for name, mechs in sets.items():
        if not isinstance(name, str) or not name or isinstance(mechs, str):
            raise ValueError('mechanism_sets must map names to sequences of mechanism ids')
        try:
            mechs = list(mechs)
        except TypeError as exc:
            raise ValueError('mechanism_sets must map names to sequences of mechanism ids') from exc
        if any(m not in MECHANISM_IDS for m in mechs) or len(set(mechs)) != len(mechs):
            raise ValueError('mechanism_sets values must be distinct NEUROWEAVE mechanism ids')
        clean[name] = mechs
    c['mechanism_sets'] = clean
    return c


def _close(a: float, b: float, tol: float) -> bool:
    return bool(abs(float(a) - float(b)) <= tol)


def _exact_checks(c: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    """Seed-independent checks on exactly known TPMs (no sampling)."""
    rng = np.random.default_rng(_PERM_OFFSET)
    sizes = []
    for n in c['sizes']:
        P = np.eye(n)[rng.permutation(n)]
        d_id, d_perm, d_uni = ei_decomposition(np.eye(n)), ei_decomposition(P), ei_decomposition(np.full((n, n), 1.0 / n))
        sizes.append({'n': n, 'log2_n': math.log2(n), 'ei_identity': d_id['ei'], 'ei_permutation': d_perm['ei'],
                      'ei_uniform': d_uni['ei'], 'identity_determinism': d_id['determinism'],
                      'identity_degeneracy': d_id['degeneracy'], 'uniform_determinism': d_uni['determinism'],
                      'uniform_degeneracy': d_uni['degeneracy']})
    ex = hoel2017_examples()
    dec = {k: ei_decomposition(v) for k, v in ex.items()}
    chain_exact = 3.0 - 7.0 / 8.0 * math.log2(7.0)
    search = {k: best_state_partition(ex[k]) for k in ('chain8', 'chain8_varied')}
    channel = best_state_partition(ex['channel4'])
    T_and, lab_and = and_gate_network(3)
    ce_and = causal_emergence(T_and, lab_and)
    p = PRINTED_VALUES
    hoel = {
        'M_ei': {k: dec[k]['ei'] for k in ('M1', 'M2', 'M3')},
        'chain8': {'ei_micro': dec['chain8']['ei'], 'ei_micro_closed_form': chain_exact,
                   'effectiveness': dec['chain8']['effectiveness'], 'ei_macro': search['chain8']['ei_macro'],
                   'ce': search['chain8']['ce'], 'best_labels': search['chain8']['labels'],
                   'partitions_evaluated': search['chain8']['partitions_evaluated']},
        'chain8_varied': {'ei_micro': dec['chain8_varied']['ei'], 'ei_macro': search['chain8_varied']['ei_macro'],
                          'ce': search['chain8_varied']['ce'], 'best_labels': search['chain8_varied']['labels']},
        'channel4': {'ei_micro': dec['channel4']['ei'], 'conditional_entropy': 2.0 - dec['channel4']['ei'],
                     'ei_macro': channel['ei_macro'], 'ce': channel['ce'], 'best_labels': channel['labels'],
                     'partitions_evaluated': channel['partitions_evaluated']},
        'and_gates': {'ei_micro': ce_and['micro']['ei'], 'determinism_micro': ce_and['micro']['determinism'],
                      'degeneracy_micro': ce_and['micro']['degeneracy'],
                      'effectiveness_micro': ce_and['micro']['effectiveness'], 'ei_macro': ce_and['macro']['ei'],
                      'effectiveness_macro': ce_and['macro']['effectiveness'], 'ce': ce_and['ce'],
                      'ei_micro_closed_form': 3.0 * entropy_bits([0.25, 0.75])},
        'printed': dict(p),
    }
    symmetric = []
    for n in (5, 8):                              # weakly symmetric channels: no partition beats the micro scale
        b = best_state_partition(noisy_permutation_tpm(n, c['permutation_noise']))
        symmetric.append({'n': n, 'noise': c['permutation_noise'], 'ei_micro': b['ei_micro'],
                          'best_ei_macro': b['ei_macro'], 'best_ce': b['ce'], 'best_labels': b['labels'],
                          'partitions_evaluated': b['partitions_evaluated']})
    pair = best_unit_coarse_graining(pair_count_copy_tpm(3, c['pair_copy_noise']), 6)
    indep = best_unit_coarse_graining(independent_flip_tpm(6, c['independent_keep']), 6)
    units = {'pair_copy': {'noise': c['pair_copy_noise'], **{k: pair[k] for k in (
                 'ei_micro', 'ei_macro', 'ce', 'groups', 'macro_states', 'partitions_evaluated')}},
             'independent': {'keep': c['independent_keep'], **{k: indep[k] for k in (
                 'ei_micro', 'ei_macro', 'ce', 'groups', 'macro_states', 'partitions_evaluated')}},
             'independent_ei_micro_closed_form': 6.0 * (1.0 - entropy_bits([c['independent_keep'],
                                                                           1.0 - c['independent_keep']]))}
    every = [*(ex[k] for k in ex), T_and, np.eye(8), np.full((5, 5), 0.2)]
    checks = {
        'identity_ei_equals_log2_n': all(_close(s['ei_identity'], s['log2_n'], 1e-9) and _close(s['identity_determinism'], 1.0, 1e-9)
                                         and _close(s['identity_degeneracy'], 0.0, 1e-9) for s in sizes),
        'permutation_ei_equals_log2_n': all(_close(s['ei_permutation'], s['log2_n'], 1e-9) for s in sizes),
        'uniform_ei_is_zero': all(_close(s['ei_uniform'], 0.0, 1e-9) and _close(s['uniform_determinism'], 0.0, 1e-9)
                                  and _close(s['uniform_degeneracy'], 0.0, 1e-9) for s in sizes),
        'ei_equals_size_times_effectiveness': all(
            _close(d['ei'], d['size_bits'] * d['effectiveness'], 1e-9)
            for d in (ei_decomposition(T) for T in every)),
        'hoel2017_M1_M2_M3_exact': all(_close(hoel['M_ei'][k], p[f'hoel2017_{k}_ei'], 1e-9) for k in ('M1', 'M2', 'M3')),
        'hoel2017_chain8_micro_closed_form': _close(hoel['chain8']['ei_micro'], chain_exact, 1e-9),
        'hoel2017_chain8_matches_printed': (_close(hoel['chain8']['ei_micro'], p['hoel2017_chain8_ei_micro'], 0.01)
                                            and _close(hoel['chain8']['effectiveness'], p['hoel2017_chain8_effectiveness'], 0.005)
                                            and _close(hoel['chain8']['ce'], p['hoel2017_chain8_ce'], 0.01)),
        'hoel2017_chain8_search_finds_1_bit_seven_plus_one': (
            _close(hoel['chain8']['ei_macro'], 1.0, 1e-9) and hoel['chain8']['best_labels'].tolist() == _SEVEN_PLUS_ONE),
        'hoel2017_chain8_varied_matches_printed': (
            _close(hoel['chain8_varied']['ei_micro'], p['hoel2017_chain8_varied_ei_micro'], 0.005)
            and _close(hoel['chain8_varied']['ei_macro'], 1.0, 1e-9)
            and hoel['chain8_varied']['best_labels'].tolist() == _SEVEN_PLUS_ONE),
        'hoel2017_channel4_matches_printed': (
            _close(hoel['channel4']['ei_micro'], p['hoel2017_channel4_ei_micro'], 0.005)
            and _close(hoel['channel4']['conditional_entropy'], p['hoel2017_channel4_conditional_entropy'], 0.005)
            and _close(hoel['channel4']['ei_macro'], p['hoel2017_channel4_ei_macro'], 1e-9)
            and hoel['channel4']['best_labels'].tolist() == [0, 0, 0, 1]),
        'and_gate_network_matches_printed': (
            _close(hoel['and_gates']['ei_micro'], p['hoel2013_and_ei_micro'], 0.005)
            and _close(hoel['and_gates']['determinism_micro'], 1.0, 1e-9)
            and _close(hoel['and_gates']['degeneracy_micro'], p['hoel2017_and_degeneracy'], 0.005)
            and _close(hoel['and_gates']['effectiveness_micro'], p['hoel2017_and_effectiveness'], 0.005)
            and _close(hoel['and_gates']['ei_macro'], 3.0, 1e-9)
            and _close(hoel['and_gates']['ce'], p['hoel2013_and_ce'], 0.005)),
        'weakly_symmetric_tpm_has_no_emergence': all(s['best_ce'] <= 1e-12 for s in symmetric),
        'unit_pair_copy_exact_ce_positive_for_pair_counts': bool(pair['ce'] > 0.0 and pair['groups'] == _PAIR_GROUPS),
        'unit_independent_exact_ce_negative': bool(indep['ce'] < 0.0 and _close(
            indep['ei_micro'], units['independent_ei_micro_closed_form'], 1e-9)),
    }
    return {'sizes': sizes, 'hoel': hoel, 'weakly_symmetric': symmetric, 'units_exact': units}, checks


def _noisy_chain(eps: float) -> np.ndarray:
    T = np.zeros((8, 8))
    T[:7, :7] = (1.0 - eps) / 7.0
    T[:7, 7] = eps
    T[7, :7] = eps / 7.0
    T[7, 7] = 1.0 - eps
    return T


def _sample_chain(T: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    cum = np.cumsum(T, axis=1)
    cum[:, -1] = 1.0
    u = rng.random(steps)
    s = np.empty(steps, dtype=np.int64)
    s[0] = int(rng.integers(T.shape[0]))
    for t in range(1, steps):
        s[t] = int(np.searchsorted(cum[s[t - 1]], u[t], side='right'))
    return s


def _estimation(seed: int, c: Mapping[str, Any]) -> dict[str, Any]:
    """Interventional samples of the Hoel (2017) 8-state chain: uniform current states, successors from the TPM."""
    T = hoel2017_examples()['chain8']
    rng = np.random.default_rng(seed + _EST_OFFSET)
    N = c['estimation_samples']
    cur = rng.integers(0, 8, N)
    cum = np.cumsum(T, axis=1)
    cum[:, -1] = 1.0
    nxt = (rng.random(N)[:, None] >= cum[cur]).sum(axis=1)       # inverse CDF (searchsorted side='right')
    est = _smoothed(transition_counts(cur, 8, next_states=nxt).astype(np.float64), c['alpha'])
    labels = np.array(_SEVEN_PLUS_ONE)
    exact = causal_emergence(T, labels)
    fit = causal_emergence(est, labels)
    return {'seed': seed, 'samples': N, 'ei_micro_exact': exact['micro']['ei'], 'ei_micro_estimated': fit['micro']['ei'],
            'ce_exact': exact['ce'], 'ce_estimated': fit['ce'],
            'abs_error_ei_micro': abs(fit['micro']['ei'] - exact['micro']['ei']),
            'abs_error_ce': abs(fit['ce'] - exact['ce'])}


def _chain_record(out: Mapping[str, Any]) -> dict[str, Any]:
    return {'ei_micro': out['ei_micro'], 'ei_macro': out['ei_macro'], 'ce': out['ce'], 'best_labels': out['labels'],
            'null_mean': out['null_mean'], 'null_sd': out['null_sd'], 'z': out['z'], 'p': out['p'],
            'block': out['block']}


def _null_runs(seed: int, c: Mapping[str, Any], chunks: Sequence[_Candidates]) -> dict[str, Any]:
    """Successor block-permutation null on 8-state sequences: i.i.d. states (calibration), the noisy ergodic
    version of the Hoel (2017) chain (exact CE > 0) and a weakly symmetric noisy permutation of the same length
    (exact CE <= 0 for every partition)."""
    iid = []
    for r in range(c['null_replicates']):
        rng = np.random.default_rng([seed + _IID_OFFSET, r])
        out = _successor_block_null(rng.integers(0, 8, c['null_steps']), 8, chunks, c['alpha'], c['null_draws'],
                                    None, (seed * 1009 + r) % (2**31 - 1))
        iid.append({'replicate': r, 'ce': out['ce'], 'null_mean': out['null_mean'], 'null_sd': out['null_sd'],
                    'z': out['z'], 'p': out['p']})
    eps = c['emergence_noise']
    T = _noisy_chain(eps)
    states = _sample_chain(T, c['null_steps'], np.random.default_rng(seed + _CHAIN_OFFSET))
    out = _successor_block_null(states, 8, chunks, c['alpha'], c['null_draws'], None, seed)
    exact = causal_emergence(T, np.array(_SEVEN_PLUS_ONE))
    emerg = {'noise': eps, 'steps': c['null_steps'], 'ce_exact_seven_plus_one': exact['ce'],
             'ei_micro_exact': exact['micro']['ei'], 'ei_macro_exact': exact['macro']['ei'], **_chain_record(out)}
    Tp = noisy_permutation_tpm(8, c['permutation_noise'], seed=seed)
    sp = _sample_chain(Tp, c['null_steps'], np.random.default_rng([seed + _PERM8_OFFSET, 1]))
    outp = _successor_block_null(sp, 8, chunks, c['alpha'], c['null_draws'], None, seed)
    exact_p = best_state_partition(Tp)
    control = {'noise': c['permutation_noise'], 'steps': c['null_steps'], 'ei_micro_exact': exact_p['ei_micro'],
               'best_ce_exact': exact_p['ce'], **_chain_record(outp)}
    return {'seed': seed, 'iid': iid, 'emergence': emerg, 'permutation_control': control}


def _unit_runs(seed: int, c: Mapping[str, Any], chunks: Sequence[_Candidates], groups: np.ndarray) -> dict[str, Any]:
    """Six binary units sampled from the pair-count copy ring (exact CE > 0) and from independent slow flips
    (exact CE < 0), each scored with the observer's count-per-group search against both nulls."""
    weights = np.left_shift(1, np.arange(6, dtype=np.int64))
    out: dict[str, Any] = {'seed': seed, 'pair_copy': [], 'independent': []}
    for r in range(c['unit_replicates']):
        for name, offset in (('pair_copy', _PAIR_OFFSET), ('independent', _SLOW_OFFSET)):
            rng = np.random.default_rng([seed + offset, r])
            if name == 'pair_copy':
                bits = _sample_pair_count_copy(3, c['pair_copy_noise'], c['unit_steps'], rng)
            else:
                bits = _sample_independent_flips(6, c['independent_keep'], c['unit_steps'], rng)
            seq = bits.astype(np.int64) @ weights
            res = _ce_and_nulls(seq, 6, chunks, c['alpha'], c['unit_draws'], None,
                                np.random.default_rng([seed + _UNIT_NULL_OFFSET, r, 0]),
                                np.random.default_rng([seed + _UNIT_NULL_OFFSET, r, 1]))
            part = _groups(groups[res['best_index']], np.arange(6))
            st, si = res['successor'], res['independent']
            out[name].append({'replicate': r, 'visited_states': int(np.unique(seq[:-1]).size),
                              'ei_micro': res['ei_micro'], 'ei_macro': res['ei_macro'], 'ce': res['ce'],
                              'best_partition': part, 'pairs_recovered': part == _PAIR_GROUPS,
                              'successor_excess': st['excess'], 'successor_z': st['z'], 'successor_p': st['p'],
                              'independent_excess': si['excess'], 'independent_z': si['z'], 'independent_p': si['p']})
    return out


def _neuroweave_activity(mechanisms: Sequence[str], seed: int, c: Mapping[str, Any]) -> np.ndarray:
    from ..neuroweave.engine import NeuroWeaveEngine
    from ..neuroweave.experiments import make_spec
    from ..neuroweave.tasks import make_stream
    steps = c['nw_steps']
    engine = NeuroWeaveEngine(make_spec(mechanisms, seed=seed, neurons=c['neurons'], topology=c['topology'],
                                        steps=steps))
    stream = make_stream(seed, steps)
    outs = [np.array(engine.step(stream.inputs[i], stream.sensory[i], reward=0.0, learn=False)['output'],
                     dtype=np.float64) for i in range(steps)]
    return np.stack(outs)[c['nw_burn_in']:]


_NW_KEYS = ('unit_sample', 'transitions', 'transitions_per_micro_state', 'visited_states', 'ei_micro',
            'determinism_micro', 'degeneracy_micro',
            'ei_macro', 'macro_partition', 'macro_states', 'ce', 'null_ready', 'null_ce_mean', 'null_ce_sd',
            'ce_excess', 'ce_z', 'ce_p', 'indep_ce_mean', 'indep_ce_sd', 'ce_excess_indep', 'ce_z_indep',
            'ce_p_indep')


def _neuroweave_report(name: str, mechanisms: Sequence[str], seed: int, c: Mapping[str, Any]) -> dict[str, Any]:
    X = _neuroweave_activity(mechanisms, seed, c)
    obs = CausalEmergenceObserver(X.shape[1], seed=seed, history=min(8192, max(16, X.shape[0])), units=c['units'],
                                  alpha=c['alpha'], null_draws=c['nw_null_draws'],
                                  min_transitions=min(32, max(8, X.shape[0] - 2)))
    for row in X:
        obs.observe(row)
    s = obs.summary()
    return {'mechanism_set': name, 'seed': seed, 'neurons': int(X.shape[1]), **{k: s[k] for k in _NW_KEYS}}


def _finite_tree(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        return all(_finite_tree(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_finite_tree(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return bool(obj.dtype.kind not in 'fc' or np.all(np.isfinite(obj)))
    if isinstance(obj, (float, np.floating)):
        return math.isfinite(float(obj))
    return True


def _mean_of(values: Sequence[Any]) -> float | None:
    v = [float(x) for x in values if x is not None]
    return float(np.mean(v)) if v else None


def _rate(values: Sequence[Any], threshold: float = 0.05) -> float | None:
    v = [float(x) for x in values if x is not None]
    return float(np.mean([x <= threshold for x in v])) if v else None


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Known-answer validation of EI, its decomposition, coarse-graining search, TPM estimation and both nulls,
    then a descriptive EI/CE report on NEUROWEAVE activity under each mechanism set.

    Exact checks (no sampling): identity and permutation TPMs give ``EI = log2 n`` (determinism 1, degeneracy 0),
    uniform TPMs ``EI = 0``, and ``EI = log2(n) * effectiveness``; the printed values of Hoel (2017) (M1/M2/M3,
    the 8-state chain with its closed form ``3 - (7/8) log2 7`` and the exhaustive search over its 4138 proper
    partitions finding the 1-bit seven-plus-one macro, the unequal-row variant, the 4-state channel) and of the
    six-AND-gate ring (Hoel et al. 2013, Fig. 4); weakly symmetric noisy permutations have no partition with
    ``CE > 0``; the pair-count copy ring has exact ``CE > 0`` for the count-per-pair grouping and independent
    slow units have exact ``CE < 0``.

    Sampled checks (every seed): Laplace-smoothed TPMs from interventional samples recover micro EI and CE within
    ``estimation_tolerance``; the successor null is calibrated on i.i.d. states (at most 25 % of replicates with
    ``p <= 0.05``, ``|mean z| < 1``); on 8-state sequences of the noisy chain the estimated CE is positive and
    above the successor null (``z > 3``, ``p <= 0.05``), while a weakly symmetric noisy permutation of the same
    length gives a negative CE that is not above it; on six binary units the pair-count copy ring is above the
    independent-units null (``z > 3``, ``p <= 0.05``) with the pair grouping recovered, and the independent-units
    null is calibrated on independent slow units (at most half of the replicates at ``p <= 0.05``,
    ``|mean z| < 2``). How often the successor null flags those independent units (exact CE < 0) is reported,
    not checked. The NEUROWEAVE part is descriptive (only finiteness is checked): for each mechanism set and seed the
    frozen engine runs on the task stream and a :class:`CausalEmergenceObserver` over ``units`` (default 4) output
    units reports micro/macro EI, CE and both nulls.
    ``kind='validation'``, ``verdict`` in ``('pass', 'fail')``."""
    from . import harness
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seeds = _check_seeds(seeds)
    cfg = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in _DEFAULTS.items()}
    if quick:
        cfg.update(_QUICK)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(f'unknown run_experiment overrides: {unknown}')
    cfg.update(overrides)
    c = _validate_config(cfg)
    t0 = time.perf_counter()
    exact, checks = _exact_checks(c)
    estimation = [_estimation(s, c) for s in seeds]
    cand = state_partition_candidates(8)
    chunks8 = _candidate_chunks(cand['labels'], 8, cand['m'])
    nulls = [_null_runs(s, c, chunks8) for s in seeds]
    unit_cand = unit_sum_candidates(6)
    unit_chunks = _candidate_chunks(unit_cand['labels'], 64, unit_cand['m'])
    units = [_unit_runs(s, c, unit_chunks, unit_cand['groups']) for s in seeds]
    t_val = time.perf_counter() - t0
    t1 = time.perf_counter()
    nw: dict[str, Any] = {}
    for name, mechs in c['mechanism_sets'].items():
        runs = [_neuroweave_report(name, mechs, s, c) for s in seeds]
        nw[name] = {'mechanisms': list(mechs), 'per_seed': runs,
                    'mean': {k: _mean_of([r[k] for r in runs]) for k in (
                        'ei_micro', 'ei_macro', 'ce', 'ce_excess', 'ce_z', 'ce_excess_indep', 'ce_z_indep',
                        'visited_states')}}
    t_nw = time.perf_counter() - t1
    iid = [r for n in nulls for r in n['iid']]
    z_iid = [r['z'] for r in iid if r['z'] is not None]
    fpr = float(np.mean([r['p'] <= 0.05 for r in iid]))
    pair_rows = [r for u in units for r in u['pair_copy']]
    ind_rows = [r for u in units for r in u['independent']]
    ind_rate = _rate([r['independent_p'] for r in ind_rows])
    ind_mean_z = _mean_of([r['independent_z'] for r in ind_rows])
    unit_summary = {
        'pair_copy_replicates': len(pair_rows),
        'pair_copy_pairs_recovered_fraction': float(np.mean([r['pairs_recovered'] for r in pair_rows])),
        'pair_copy_min_independent_z': min((r['independent_z'] for r in pair_rows if r['independent_z'] is not None),
                                           default=None),
        'pair_copy_mean_ce': _mean_of([r['ce'] for r in pair_rows]),
        'independent_replicates': len(ind_rows),
        'independent_units_null_flag_rate_p_le_0.05': ind_rate,
        'independent_units_null_mean_z': ind_mean_z,
        'independent_successor_null_flag_rate_p_le_0.05': _rate([r['successor_p'] for r in ind_rows]),
        'independent_successor_null_mean_z': _mean_of([r['successor_z'] for r in ind_rows]),
        'independent_mean_ce': _mean_of([r['ce'] for r in ind_rows]),
        'note': ('independent slow units have exact CE < 0, yet the successor null flags them when micro states '
                 'have few transitions each; the independent-units null does not'),
    }
    tol = c['estimation_tolerance']
    checks.update({
        'estimated_tpm_recovers_ei_and_ce': all(e['abs_error_ei_micro'] <= tol and e['abs_error_ce'] <= tol
                                                for e in estimation),
        'iid_null_false_positive_rate_at_most_0.25': bool(fpr <= 0.25),
        'iid_null_mean_z_within_1': bool(z_iid and abs(float(np.mean(z_iid))) < 1.0),
        'noisy_chain_ce_positive_and_above_successor_null': all(
            n['emergence']['ce'] > 0.0 and n['emergence']['z'] is not None and n['emergence']['z'] > 3.0
            and n['emergence']['p'] <= 0.05 for n in nulls),
        'permutation_control_ce_negative_and_not_above_successor_null': all(
            n['permutation_control']['ce'] < 0.0 and n['permutation_control']['p'] > 0.05 for n in nulls),
        'unit_pair_copy_above_independent_units_null_with_pairs_recovered': all(
            r['independent_z'] is not None and r['independent_z'] > 3.0 and r['independent_p'] <= 0.05
            and r['pairs_recovered'] for r in pair_rows),
        'unit_independent_units_null_calibrated': bool(ind_rate is not None and ind_rate <= 0.5
                                                       and ind_mean_z is not None and abs(ind_mean_z) < 2.0),
        'neuroweave_values_finite': _finite_tree(nw),
    })
    receipt = harness.base_receipt(MODULE_ID, 'validation', {**c, 'quick': quick, 'seeds': list(seeds)})
    receipt['validations'] = {
        **exact, 'estimation': estimation,
        'null_iid': {'replicates': len(iid), 'false_positive_rate_p_le_0.05': fpr,
                     'mean_z': float(np.mean(z_iid)) if z_iid else None, 'per_replicate': iid},
        'null_emergence': [n['emergence'] for n in nulls],
        'null_permutation_control': [n['permutation_control'] for n in nulls],
        'candidate_partitions': int(cand['labels'].shape[0]),
        'units': {'per_seed': units, 'summary': unit_summary,
                  'candidate_coarse_grainings': int(unit_cand['labels'].shape[0])}}
    receipt['neuroweave'] = {'per_mechanism_set': nw, 'topology': c['topology'], 'units': c['units'],
                             'note': ('descriptive only: values are reported, no verdict is drawn from them; read CE '
                                      'against the independent-units null (ce_excess_indep, ce_z_indep)')}
    receipt['checks'] = checks
    receipt['verdict'] = 'pass' if all(checks.values()) else 'fail'
    receipt['verdict_basis'] = 'all_known_answer_checks' if receipt['verdict'] == 'pass' else 'failed_checks: ' + ', '.join(
        k for k, v in checks.items() if not v)
    n_null_runs = len(seeds) * (c['null_replicates'] + 2)
    n_unit_runs = len(seeds) * c['unit_replicates'] * 2
    nw_cands = int(unit_sum_candidates(c['units'])['m'].size)
    receipt['estimated_cost'] = {
        'tpm_evaluations_null': int(n_null_runs * (c['null_draws'] + 1) * (cand['labels'].shape[0] + 1)),
        'tpm_evaluations_units': int(n_unit_runs * (2 * c['unit_draws'] + 1) * (unit_cand['labels'].shape[0] + 1)),
        'engine_steps': int(len(c['mechanism_sets']) * len(seeds) * c['nw_steps']),
        'tpm_evaluations_neuroweave': int(len(c['mechanism_sets']) * len(seeds) * (2 * c['nw_null_draws'] + 1)
                                          * (nw_cands + 1))}
    receipt['measured_wallclock_s'] = {'validation': t_val, 'neuroweave': t_nw, 'total': time.perf_counter() - t0}
    receipt['notes'] = (
        'EI and causal emergence follow the definitions of Hoel et al. (2013) and Hoel (2017): uniform '
        '(maximum-entropy) interventions, macro TPMs by uniform interventions on the micro states of each macro '
        'state. The six-AND-gate wiring is reconstructed from the papers\' text (pairs reading the next pair); its '
        'exact values match the printed ones (2.43 bits, effectiveness 0.4/0.41, degeneracy 0.6/0.59, macro 3 bits, '
        'CE 0.57 bits) at their printed precision. The 8-state chain has the closed form 3 - (7/8) log2 7 = 0.5436 '
        'bits (CE 0.4564), printed as 0.55 and 0.45 bits, so it is checked to 0.01. The pair-count copy ring and the '
        'independent slow units are constructed known-answer systems, not examples from the papers. Estimated EI '
        'from short discretised windows is dominated by Laplace smoothing and finite-sample effects (see '
        'transitions_per_micro_state): with few transitions per micro state the successor null flags any temporal '
        'dependence, so NEUROWEAVE CE is only meaningful against the independent-units null; the observer binarises '
        'at a running median and only searches count-per-group coarse-grainings of 2 to 6 units. Nothing here is '
        'evidence of consciousness, agency or biological causation.')
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment}
