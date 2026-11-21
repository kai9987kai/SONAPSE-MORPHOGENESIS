"""Joint-failure ensemble selection (frontier module ``joint_failure``).

Inspired by FAULTLINE (kai9987kai, https://github.com/kai9987kai/FAULTLINE), a CPU-local laboratory for
correlated and higher-order failures in small voting ensembles. This module is a synthetic computational
abstraction: it re-implements the four FAULTLINE selection rules for NumPy failure arrays and adds its own
seeded generator, exact population errors and a NEUROWEAVE application; it reproduces none of FAULTLINE's
recorded numbers. Full description and limits: ``docs/frontier/modules/joint_failure.md``.

Data. ``failures[i, c, t]`` is true when component ``i`` is wrong on trial ``t`` of condition ``c``; every
condition has the same number of trials ``T``. A panel ``P`` of odd size ``k`` votes by majority. With binary
labels every wrong member votes for the one wrong label, so the vote is wrong exactly when at least
``s = (k + 1) / 2`` members are wrong (with more labels this is an upper bound on plurality-vote error)::

    W_P(c, t) = sum_{i in P} F[i, c, t]            Err(P | c) = (1 / T) sum_t 1[W_P(c, t) >= s]

``Err`` depends only on the joint failure pattern (signature) of each trial, so it is computed exactly from
signature counts alone by ``majority_error_from_signatures``.

Selectors (calibration trials only; scores are integer counts, so ties are exact; a tie goes to the next
key and finally to the lexicographically first panel):

* ``accuracy``  min sum_{i in P} e_i (pooled individual error)
* ``pairwise``  min max_c B(P | c), then sum_c B(P | c), then accuracy, where
                ``B(P | c) = sum_{i<j in P} Pr(F_i F_j | c) / C(s, 2) >= Err(P | c)`` is the second
                factorial-moment bound (``1[W >= s] <= C(W, 2) / C(s, 2)``); for ``k = 1`` the individual error
* ``pooled``    min pooled majority error, then worst-condition majority error, then accuracy
* ``joint``     min worst-condition majority error ``max_c Err(P | c)``, then pooled, then accuracy

All randomness comes from ``numpy.random.default_rng(seed + offset)``.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import harness

__all__ = ['MODULE_ID', 'NEUROWEAVE_ID', 'SOURCE', 'EXPERIMENTS', 'POLICIES', 'SCENARIOS', 'validate_failures',
           'majority_threshold', 'majority_error', 'pairwise_bound', 'failure_signatures',
           'majority_error_from_signatures', 'panel_scores', 'select_panels', 'select_panel', 'evaluate_panel',
           'run_selection', 'make_failure_data', 'expected_condition_errors', 'constructed_cases',
           'check_constructed_cases', 'run_experiment', 'run_neuroweave_application']

MODULE_ID = 'joint_failure'
NEUROWEAVE_ID = MODULE_ID + '_neuroweave'
SOURCE = ('FAULTLINE: Joint-Failure Ensemble Selection Laboratory (kai9987kai), '
          'https://github.com/kai9987kai/FAULTLINE (design dated 24 October 2026)')
POLICIES = ('accuracy', 'pairwise', 'pooled', 'joint')
SCENARIOS = ('shared', 'independent', 'shift')

MAX_COMPONENTS = 24
MAX_CONDITIONS = 64
MAX_TRIALS = 100_000
MAX_CELLS = 5_000_000            # components x conditions x trials per failure array
MAX_PANELS = 100_000             # candidate panels enumerated per selection
_CHUNK_CELLS = 2_000_000         # panels x conditions x trials evaluated per chunk (bounded memory)

_STRUCT_OFFSET = 89_101
_CAL_OFFSET = 89_113
_TEST_OFFSET = 89_127
_HIDDEN_OFFSET = 89_137


# ====================================================================== validation helpers
def _int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _num(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number')
    v = float(value)
    if not math.isfinite(v) or v < low or v > high:
        raise ValueError(f'{name} must be a finite number in [{low}, {high}]')
    return v


def _range(name: str, value: Any, low: float, high: float) -> tuple[float, float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f'{name} must be a pair (lo, hi)')
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a pair (lo, hi)') from exc
    if len(pair) != 2:
        raise ValueError(f'{name} must be a pair (lo, hi)')
    lo, hi = _num(name, pair[0], low, high), _num(name, pair[1], low, high)
    if lo > hi:
        raise ValueError(f'{name} must satisfy lo <= hi')
    return lo, hi


def _seeds(seeds: Any) -> tuple[int, ...]:
    if isinstance(seeds, (str, bytes, Mapping)):
        raise ValueError('seeds must be an iterable of ints')
    try:
        out = tuple(_int('seed', s, 0, 2**31 - 1) for s in seeds)
    except TypeError as exc:
        raise ValueError('seeds must be an iterable of ints') from exc
    if not out or len(set(out)) != len(out):
        raise ValueError('seeds must be non-empty and unique')
    return out


def validate_failures(failures: Any, *, name: str = 'failures') -> np.ndarray:
    """Return a boolean copy of a ``(components, conditions, trials)`` failure array (0/1 or bool entries)."""
    if isinstance(failures, (str, bytes, Mapping)):
        raise ValueError(f'{name} must be a 3-D array of booleans or 0/1 values')
    try:
        a = np.asarray(failures)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a 3-D array of booleans or 0/1 values') from exc
    if a.ndim != 3 or a.dtype.kind not in 'biuf':
        raise ValueError(f'{name} must be a 3-D array (components x conditions x trials) of booleans or 0/1 values')
    m, k, t = a.shape
    if not (1 <= m <= MAX_COMPONENTS and 1 <= k <= MAX_CONDITIONS and 1 <= t <= MAX_TRIALS) or a.size > MAX_CELLS:
        raise ValueError(f'{name} shape must be at most ({MAX_COMPONENTS}, {MAX_CONDITIONS}, {MAX_TRIALS}) with '
                         f'every axis non-empty and at most {MAX_CELLS} cells')
    if a.dtype.kind == 'b':
        return a.astype(bool, copy=True)
    if a.dtype.kind == 'f' and not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must contain only 0/1 values')
    if not np.all((a == 0) | (a == 1)):
        raise ValueError(f'{name} must contain only 0/1 values')
    return a == 1


def majority_threshold(panel_size: Any) -> int:
    """``s = (k + 1) / 2`` for an odd panel size ``k``."""
    k = _int('panel_size', panel_size, 1, MAX_COMPONENTS)
    if k % 2 == 0:
        raise ValueError('panel_size must be odd')
    return (k + 1) // 2


def _panel(panel: Any, m: int) -> tuple[int, ...]:
    if isinstance(panel, (str, bytes, Mapping)):
        raise ValueError('panel must be a sequence of component indices')
    try:
        p = tuple(_int('panel member', v, 0, m - 1) for v in panel)
    except TypeError as exc:
        raise ValueError('panel must be a sequence of component indices') from exc
    if not p or len(p) % 2 == 0:
        raise ValueError('panel must have an odd number of members')
    if len(set(p)) != len(p):
        raise ValueError('panel members must be distinct')
    return tuple(sorted(p))


# ====================================================================== exact panel statistics
def majority_error(failures: Any, panel: Any) -> np.ndarray:
    """Per-condition majority-vote error ``Err(P | c)`` of ``panel`` (exact, from the joint failures)."""
    f = validate_failures(failures)
    p = _panel(panel, f.shape[0])
    wrong = f[list(p)].sum(axis=0, dtype=np.int64)
    return (wrong >= (len(p) + 1) // 2).mean(axis=1)


def pairwise_bound(failures: Any, panel: Any) -> np.ndarray:
    """Per-condition second factorial-moment bound ``B(P | c) >= Err(P | c)`` (individual error for ``k = 1``)."""
    f = validate_failures(failures)
    p = _panel(panel, f.shape[0])
    if len(p) == 1:
        return f[p[0]].mean(axis=1)
    sub = f[list(p)].astype(np.float64)
    co = np.einsum('ict,jct->ijc', sub, sub)
    iu = np.triu_indices(len(p), 1)
    s = (len(p) + 1) // 2
    return co[iu].sum(axis=0) / (math.comb(s, 2) * f.shape[2])


def failure_signatures(failures: Any) -> list[dict[str, Any]]:
    """Empirical failure hyperedges: for each condition the distinct sets of failing components and their counts.

    The empty set (no component failed) is included, so the counts of each condition sum to ``T``."""
    f = validate_failures(failures)
    m, k, t = f.shape
    weights = np.left_shift(np.int64(1), np.arange(m, dtype=np.int64))
    out = []
    for c in range(k):
        codes = weights @ f[:, c, :].astype(np.int64)
        uniq, counts = np.unique(codes, return_counts=True)
        sigs = [{'members': [i for i in range(m) if (code >> i) & 1], 'count': int(n)}
                for code, n in zip(uniq.tolist(), counts.tolist())]
        out.append({'condition': c, 'components': int(m), 'trials': int(t), 'signatures': sigs})
    return out


def majority_error_from_signatures(signatures: Any, panel: Any) -> np.ndarray:
    """``Err(P | c)`` computed from signature counts only (``failure_signatures`` output).

    When the entries carry ``components`` (as ``failure_signatures`` output does), every signature member and
    every panel member must be a valid component index; otherwise indices up to ``MAX_COMPONENTS - 1`` are
    accepted and a component that never appears in a signature never failed."""
    if isinstance(signatures, (str, bytes, Mapping)) or not isinstance(signatures, Sequence) or not signatures:
        raise ValueError('signatures must be a non-empty list of per-condition mappings')
    rows = []
    n_comp = MAX_COMPONENTS
    declared = [_int('components', entry['components'], 1, MAX_COMPONENTS) for entry in signatures
                if isinstance(entry, Mapping) and entry.get('components') is not None]
    if declared:
        if len(declared) != len(signatures) or len(set(declared)) != 1:
            raise ValueError('every condition entry must declare the same number of components')
        n_comp = declared[0]
    for entry in signatures:
        if not isinstance(entry, Mapping) or 'trials' not in entry or 'signatures' not in entry:
            raise ValueError('each condition entry needs trials and signatures')
        trials = _int('trials', entry['trials'], 1, MAX_TRIALS)
        sigs = entry['signatures']
        if isinstance(sigs, (str, bytes, Mapping)) or not isinstance(sigs, Sequence):
            raise ValueError('signatures must be a list of {members, count}')
        parsed, total = [], 0
        for sig in sigs:
            if not isinstance(sig, Mapping) or 'members' not in sig or 'count' not in sig:
                raise ValueError('each signature needs members and count')
            members = sig['members']
            if isinstance(members, (str, bytes, Mapping)) or not isinstance(members, Sequence):
                raise ValueError('signature members must be a list of component indices')
            mem = {_int('signature member', v, 0, n_comp - 1) for v in members}
            if len(mem) != len(members):
                raise ValueError('signature members must be distinct')
            count = _int('count', sig['count'], 0, MAX_TRIALS)
            parsed.append((mem, count))
            total += count
        if total != trials:
            raise ValueError('signature counts of a condition must sum to its trials')
        rows.append((trials, parsed))
    p = set(_panel(panel, n_comp))
    s = (len(p) + 1) // 2
    return np.array([sum(n for mem, n in parsed if len(mem & p) >= s) / trials for trials, parsed in rows])


def evaluate_panel(failures: Any, panel: Any) -> dict[str, Any]:
    """Majority-vote error by condition, worst condition, pooled error and unanimous-failure rate."""
    f = validate_failures(failures)
    p = _panel(panel, f.shape[0])
    wrong = f[list(p)].sum(axis=0, dtype=np.int64)
    err = (wrong >= (len(p) + 1) // 2).mean(axis=1)
    return {'panel': list(p), 'condition_error': err.tolist(), 'worst_condition_error': float(err.max()),
            'worst_condition': int(np.argmax(err)), 'pooled_error': float(err.mean()),
            'unanimous_failure_rate': float((wrong == len(p)).mean()),
            'pairwise_bound_worst': float(pairwise_bound(f, p).max()) if len(p) > 1 else float(err.max())}


# ====================================================================== selection
def _panel_table(f: np.ndarray, k: int) -> dict[str, np.ndarray]:
    m, kc, t = f.shape
    if k > m:
        raise ValueError('panel_size must not exceed the number of components')
    n_p = math.comb(m, k)
    if n_p > MAX_PANELS:
        raise ValueError(f'C(components, panel_size) must be at most {MAX_PANELS}')
    panels = np.array(list(itertools.combinations(range(m), k)), dtype=np.int64).reshape(n_p, k)
    s = (k + 1) // 2
    ind = f.sum(axis=(1, 2), dtype=np.int64)
    ind_c = f.sum(axis=2, dtype=np.int64)                                   # (m, K)
    flat = f.reshape(m, kc * t).astype(np.float32)                         # exact for counts <= 2**24
    maj = np.empty((n_p, kc), dtype=np.int64)
    chunk = max(1, _CHUNK_CELLS // max(1, kc * t))
    for start in range(0, n_p, chunk):
        sel = panels[start:start + chunk]
        inc = np.zeros((sel.shape[0], m), dtype=np.float32)
        np.put_along_axis(inc, sel, 1.0, axis=1)
        wrong = (inc @ flat).reshape(sel.shape[0], kc, t)
        maj[start:start + chunk] = (wrong >= s - 0.5).sum(axis=2)
    if k >= 2:
        fc = f.astype(np.float64)
        co = np.einsum('ict,jct->ijc', fc, fc).round().astype(np.int64)  # (m, m, K) co-failure counts
        pair = np.zeros((n_p, kc), dtype=np.int64)
        for a, b in itertools.combinations(range(k), 2):
            pair += co[panels[:, a], panels[:, b], :]
    else:
        pair = ind_c[panels[:, 0], :]
    return {'panels': panels, 'individual_sum': ind[panels].sum(axis=1), 'pair_by_condition': pair,
            'majority_by_condition': maj, 'trials': np.int64(t)}


def _keys(table: Mapping[str, np.ndarray], policy: str) -> list[np.ndarray]:
    ind = table['individual_sum']
    maj, pair = table['majority_by_condition'], table['pair_by_condition']
    if policy == 'accuracy':
        return [ind]
    if policy == 'pairwise':
        return [pair.max(axis=1), pair.sum(axis=1), ind]
    if policy == 'pooled':
        return [maj.sum(axis=1), maj.max(axis=1), ind]
    if policy == 'joint':
        return [maj.max(axis=1), maj.sum(axis=1), ind]
    raise ValueError(f'policy must be one of {POLICIES}')


def _argbest(keys: Sequence[np.ndarray]) -> int:
    n = keys[0].shape[0]
    return int(np.lexsort((np.arange(n),) + tuple(reversed(list(keys))))[0])


def panel_scores(failures: Any, panel_size: Any) -> dict[str, Any]:
    """Every candidate panel (lexicographic order) with the integer scores the selectors rank by."""
    f = validate_failures(failures)
    k = majority_threshold(panel_size) * 2 - 1
    table = _panel_table(f, k)
    return {'panels': table['panels'].tolist(), 'trials_per_condition': int(table['trials']),
            'individual_failures': table['individual_sum'].tolist(),
            'pair_cofailures_by_condition': table['pair_by_condition'].tolist(),
            'majority_failures_by_condition': table['majority_by_condition'].tolist(),
            'pair_bound_denominator': math.comb((k + 1) // 2, 2) if k >= 3 else 1}


def select_panels(failures: Any, panel_size: Any, policies: Sequence[str] = POLICIES) -> dict[str, tuple[int, ...]]:
    """Panel chosen by each policy from the (calibration) failures."""
    f = validate_failures(failures)
    k = majority_threshold(panel_size) * 2 - 1
    if isinstance(policies, str) or any(p not in POLICIES for p in policies):
        raise ValueError(f'policies must be a sequence drawn from {POLICIES}')
    table = _panel_table(f, k)
    return {p: tuple(int(v) for v in table['panels'][_argbest(_keys(table, p))]) for p in policies}


def select_panel(failures: Any, panel_size: Any, policy: str) -> tuple[int, ...]:
    if policy not in POLICIES:
        raise ValueError(f'policy must be one of {POLICIES}')
    return select_panels(failures, panel_size, (policy,))[policy]


def run_selection(calibration: Any, test: Any, panel_size: Any) -> dict[str, Any]:
    """Select with every policy on ``calibration``, evaluate the frozen panels on held-out ``test`` trials.

    ``oracle`` is the panel with the lowest *test* worst-condition error (chosen on the test data: a reference
    for how much any calibration-only rule could achieve, never a selector)."""
    cal = validate_failures(calibration, name='calibration')
    tst = validate_failures(test, name='test')
    if cal.shape[:2] != tst.shape[:2]:
        raise ValueError('calibration and test must have the same components and conditions')
    k = majority_threshold(panel_size) * 2 - 1
    table = _panel_table(cal, k)
    out: dict[str, Any] = {'panel_size': k, 'candidate_panels': int(table['panels'].shape[0]),
                           'calibration_trials': int(cal.shape[2]), 'test_trials': int(tst.shape[2]), 'policies': {}}
    for policy in POLICIES:
        panel = tuple(int(v) for v in table['panels'][_argbest(_keys(table, policy))])
        out['policies'][policy] = {'panel': list(panel), 'calibration': evaluate_panel(cal, panel),
                                   'test': evaluate_panel(tst, panel)}
    test_table = _panel_table(tst, k)
    best = _argbest(_keys(test_table, 'joint'))
    out['oracle'] = {'panel': test_table['panels'][best].tolist(),
                     'test': evaluate_panel(tst, test_table['panels'][best])}
    return out


# ====================================================================== synthetic generator
_GEN_DEFAULTS: dict[str, Any] = {
    'families': 3, 'family_size': 3, 'calibration_trials': 64, 'test_trials': 512,
    'accurate_error': (0.01, 0.03), 'other_error': (0.03, 0.07), 'stress_activation': (0.25, 0.40),
    'stress_hit': 0.85, 'independent_error': (0.04, 0.16), 'independent_multiplier': (1.0, 1.6),
    'shift_activation': 0.6, 'shift_hit': 0.9, 'shift_target': 'hardened'}


def _gen_config(params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(_GEN_DEFAULTS))
    if unknown:
        raise ValueError(f'unknown generator parameters: {unknown}')
    cfg = {**_GEN_DEFAULTS, **params}
    cfg['families'] = _int('families', cfg['families'], 1, MAX_CONDITIONS - 1)
    cfg['family_size'] = _int('family_size', cfg['family_size'], 1, MAX_COMPONENTS)
    if cfg['families'] * cfg['family_size'] > MAX_COMPONENTS:
        raise ValueError(f'families * family_size must be at most {MAX_COMPONENTS}')
    cfg['calibration_trials'] = _int('calibration_trials', cfg['calibration_trials'], 1, MAX_TRIALS)
    cfg['test_trials'] = _int('test_trials', cfg['test_trials'], 1, MAX_TRIALS)
    for key in ('accurate_error', 'other_error', 'stress_activation', 'independent_error'):
        cfg[key] = _range(key, cfg[key], 0.0, 1.0)
    cfg['independent_multiplier'] = _range('independent_multiplier', cfg['independent_multiplier'], 0.0, 10.0)
    if cfg['independent_error'][1] * cfg['independent_multiplier'][1] > 1.0:
        raise ValueError('independent_error hi * independent_multiplier hi must be at most 1')
    for key in ('stress_hit', 'shift_activation', 'shift_hit'):
        cfg[key] = _num(key, cfg[key], 0.0, 1.0)
    if cfg['shift_target'] not in ('hardened', 'random'):
        raise ValueError("shift_target must be 'hardened' or 'random'")
    m, k = cfg['families'] * cfg['family_size'], cfg['families'] + 1
    if m * k * max(cfg['calibration_trials'], cfg['test_trials']) > MAX_CELLS:
        raise ValueError(f'components * conditions * trials must be at most {MAX_CELLS}')
    return cfg


def make_failure_data(seed: int, *, scenario: str = 'shared', **params: Any) -> dict[str, Any]:
    """Seeded synthetic failure observations (calibration and held-out test) with a documented hidden structure.

    Components come in ``families`` of ``family_size`` (shuffled per seed so tie-breaking cannot favour a
    family); conditions are ``nominal`` plus one ``stress_f`` per family.

    * ``shared``: each component fails independently at its base rate (family 0, the "accurate" family:
      ``accurate_error``; others: ``other_error``); in condition ``stress_f`` a shared stressor of family ``f``
      is active with probability ``stress_activation`` and then hits each family member with ``stress_hit``.
    * ``independent`` (negative control): no shared stressor; component ``i`` fails independently with
      ``independent_error_i * multiplier_c`` (one multiplier per condition, so every condition ranks the
      components identically and the individually best panel is optimal everywhere).
    * ``shift`` (unobserved common-cause stress test): calibration identical to ``shared``; on test trials of
      the ``nominal`` condition a common cause never seen in calibration is active with probability
      ``shift_activation`` and hits each member of one cross-family set with ``shift_hit``. ``shift_target``
      ``'hardened'`` puts the lowest-base-error member of every family in that set (adversarial to diversity-based
      selection by construction); ``'random'`` draws one member per family."""
    if scenario not in SCENARIOS:
        raise ValueError(f'scenario must be one of {SCENARIOS}')
    seed = _int('seed', seed, 0, 2**31 - 1)
    cfg = _gen_config(params)
    nf, fs = cfg['families'], cfg['family_size']
    m, k = nf * fs, nf + 1
    rng = np.random.default_rng(seed + _STRUCT_OFFSET)
    family_raw = np.repeat(np.arange(nf), fs)
    acc = rng.uniform(*cfg['accurate_error'], m)
    oth = rng.uniform(*cfg['other_error'], m)
    base_raw = np.where(family_raw == 0, acc, oth)
    activation = rng.uniform(*cfg['stress_activation'], nf)
    ind_raw = rng.uniform(*cfg['independent_error'], m)
    mult = np.concatenate([[1.0], rng.uniform(*cfg['independent_multiplier'], nf)])
    random_pick = rng.integers(0, fs, nf)
    perm = rng.permutation(m)                      # output component j is raw component perm[j]
    if cfg['shift_target'] == 'hardened':
        hidden_raw = np.array([f * fs + int(np.argmin(base_raw[f * fs:(f + 1) * fs])) for f in range(nf)])
    else:
        hidden_raw = np.array([f * fs + int(random_pick[f]) for f in range(nf)])

    def split(offset: int, trials: int, hidden: bool) -> np.ndarray:
        r = np.random.default_rng(seed + offset)
        if scenario == 'independent':
            fails = r.random((m, k, trials)) < (ind_raw[:, None] * mult[None, :])[:, :, None]
        else:
            fails = r.random((m, k, trials)) < base_raw[:, None, None]
            active = r.random((nf, trials)) < activation[:, None]
            hit = r.random((m, trials)) < cfg['stress_hit']
            for f in range(nf):
                mem = family_raw == f
                fails[mem, f + 1, :] |= active[f][None, :] & hit[mem]
        if hidden:
            h = np.random.default_rng(seed + _HIDDEN_OFFSET)
            active = h.random(trials) < cfg['shift_activation']
            hit = h.random((hidden_raw.size, trials)) < cfg['shift_hit']
            fails[hidden_raw, 0, :] |= active[None, :] & hit
        return fails[perm]

    cal = split(_CAL_OFFSET, cfg['calibration_trials'], False)
    test = split(_TEST_OFFSET, cfg['test_trials'], scenario == 'shift')
    inv = np.argsort(perm)
    structure = {'scenario': scenario, 'family': family_raw[perm].tolist(), 'base_error': base_raw[perm].tolist(),
                 'stress_activation': activation.tolist(), 'stress_hit': cfg['stress_hit'],
                 'independent_error': ind_raw[perm].tolist(), 'independent_multiplier': mult.tolist(),
                 'hidden_common_cause': {'members': sorted(int(inv[i]) for i in hidden_raw), 'condition': 0,
                                         'activation': cfg['shift_activation'], 'hit': cfg['shift_hit'],
                                         'target': cfg['shift_target'], 'active_in': 'test split of the shift scenario'}}
    h = hashlib.sha256()
    h.update(f'{scenario}|{seed}|{cal.shape}|{test.shape}'.encode('utf-8'))
    h.update(np.packbits(cal).tobytes())
    h.update(np.packbits(test).tobytes())
    return {'seed': seed, 'scenario': scenario, 'config': cfg,
            'conditions': ['nominal'] + [f'stress_{f}' for f in range(nf)],
            'calibration': cal, 'test': test, 'structure': structure, 'digest': h.hexdigest()}


def _tail(p: np.ndarray, s: int) -> float:
    """``Pr(sum of independent Bernoulli(p_i) >= s)`` (Poisson-binomial tail, exact by recursion)."""
    dist = np.zeros(p.size + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1.0 - q) + dist[:-1] * q
        dist[0] *= 1.0 - q
    return float(dist[s:].sum())


def expected_condition_errors(structure: Mapping[str, Any], panel: Any, *, split: str = 'test') -> np.ndarray:
    """Population (infinite-trial) ``Err(P | c)`` of a panel under a ``make_failure_data`` structure.

    Given the state of the (at most one) channel acting in a condition, components fail independently, so the
    majority error is a mixture of Poisson-binomial tails."""
    if split not in ('calibration', 'test'):
        raise ValueError("split must be 'calibration' or 'test'")
    st = _parse_structure(structure)
    p = list(_panel(panel, st['family'].size))
    s = (len(p) + 1) // 2
    mult = st['independent_multiplier']
    out = np.zeros(mult.size)
    if st['scenario'] == 'independent':
        ind = st['independent_error'][p]
        return np.array([_tail(np.clip(ind * mu, 0.0, 1.0), s) for mu in mult])
    base = st['base_error'][p]
    fam = st['family'][p]
    act, hit = st['stress_activation'], st['stress_hit']
    for c in range(mult.size):
        if c == 0:
            if st['scenario'] == 'shift' and split == 'test':
                members = np.isin(p, st['hidden_members'])
                hit_p = np.where(members, 1.0 - (1.0 - base) * (1.0 - st['hidden_hit']), base)
                a = st['hidden_activation']
                out[c] = a * _tail(hit_p, s) + (1.0 - a) * _tail(base, s)
            else:
                out[c] = _tail(base, s)
        else:
            hit_p = np.where(fam == c - 1, 1.0 - (1.0 - base) * (1.0 - hit), base)
            out[c] = act[c - 1] * _tail(hit_p, s) + (1.0 - act[c - 1]) * _tail(base, s)
    return out


def _parse_structure(structure: Any) -> dict[str, Any]:
    """Validated arrays from a ``make_failure_data`` structure (``ValueError`` on anything malformed)."""
    if not isinstance(structure, Mapping) or structure.get('scenario') not in SCENARIOS:
        raise ValueError('structure must come from make_failure_data')

    def probs(key: str, value: Any, size: int | None) -> np.ndarray:
        a = np.asarray(value, dtype=np.float64)
        if a.ndim != 1 or (size is not None and a.size != size) or not np.all((a >= 0.0) & (a <= 1.0)):
            raise ValueError(f'structure {key} must be a list of probabilities of the right length')
        return a

    try:
        family = np.asarray(structure['family'])
        if family.ndim != 1 or not 1 <= family.size <= MAX_COMPONENTS or family.dtype.kind not in 'iu':
            raise ValueError('structure family must be a list of component family indices')
        family = family.astype(np.int64)
        m, nf = family.size, int(family.max()) + 1
        if family.min() < 0 or nf > MAX_CONDITIONS - 1:
            raise ValueError('structure family indices are out of range')
        mult = np.asarray(structure['independent_multiplier'], dtype=np.float64)
        if mult.shape != (nf + 1,) or not np.all(np.isfinite(mult) & (mult >= 0.0)):
            raise ValueError('structure independent_multiplier must have one finite value per condition')
        hidden = structure['hidden_common_cause']
        members = np.asarray(hidden['members'])
        if members.ndim != 1 or (members.size and (members.dtype.kind not in 'iu' or members.min() < 0
                                                   or members.max() >= m)):
            raise ValueError('structure hidden_common_cause members must be component indices')
        return {'scenario': structure['scenario'], 'family': family, 'independent_multiplier': mult,
                'independent_error': probs('independent_error', structure['independent_error'], m),
                'base_error': probs('base_error', structure['base_error'], m),
                'stress_activation': probs('stress_activation', structure['stress_activation'], nf),
                'stress_hit': float(probs('stress_hit', [structure['stress_hit']], 1)[0]),
                'hidden_members': members.astype(np.int64),
                'hidden_hit': float(probs('hidden hit', [hidden['hit']], 1)[0]),
                'hidden_activation': float(probs('hidden activation', [hidden['activation']], 1)[0])}
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('structure must come from make_failure_data') from exc


# ====================================================================== constructed known-answer cases
def _rows(spec: Sequence[tuple[Sequence[int], int]], m: int, trials: int) -> np.ndarray:
    """One condition: ``spec`` lists (failing components, number of trials); remaining trials have no failure."""
    out = np.zeros((m, trials), dtype=bool)
    t = 0
    for members, count in spec:
        out[list(members), t:t + count] = True
        t += count
    if t > trials:
        raise AssertionError('constructed case overflows its trials')
    return out


def constructed_cases() -> dict[str, dict[str, Any]]:
    """Hand-built failure arrays with known answers (used by tests and recorded in receipts).

    * ``parity``: 3 components; ``even`` holds the four even-weight failure patterns, ``odd`` the four odd-weight
      ones. Marginal (1/2) and pairwise (1/4) failure rates are identical, majority error is 3/4 vs 1/4.
    * ``accuracy_trap``: components 0-2 are individually best (error 0.10) but fail together on 20 % of
      ``stress`` trials; 3-5 (error 0.15) never fail together. Accuracy picks 0-2 (worst error 0.20), the
      joint rule finds a panel with worst error 0.
    * ``pairwise_trap``: one condition; panel (2, 3, 4) fails only unanimously (10 %), so its pairwise bound is
      30 %, while (0, 1, 2) fails in pairs (18 %, bound 18 %); cross pairs make every mixed panel worse.
      Pairwise picks (0, 1, 2), the joint rule (2, 3, 4).
    * ``pooled_trap``: components 0-2 fail together on 30 % of ``b`` trials only (pooled 15 %, worst 30 %);
      3-5 fail together on 20 % of trials in both conditions (pooled and worst 20 %). Pooled picks a panel
      with worst error 0.30, the joint rule one with 0.20."""
    parity = np.zeros((3, 2, 4), dtype=bool)
    for t, pat in enumerate(((), (0, 1), (0, 2), (1, 2))):
        parity[list(pat), 0, t] = True
    for t, pat in enumerate(((0,), (1,), (2,), (0, 1, 2))):
        parity[list(pat), 1, t] = True
    acc = np.stack([_rows([((0, 1, 2), 12), ((3,), 9), ((4,), 9), ((5,), 9)], 6, 60),
                    _rows([((3,), 9), ((4,), 9), ((5,), 9)], 6, 60)], axis=1)
    pair = _rows([((2, 3, 4), 5), ((0, 1), 3), ((0, 2), 3), ((1, 2), 3),
                  ((0, 3), 4), ((0, 4), 4), ((1, 3), 4), ((1, 4), 4)], 5, 50)[:, None, :]
    pooled = np.stack([_rows([((3, 4, 5), 10)], 6, 50), _rows([((0, 1, 2), 15), ((3, 4, 5), 10)], 6, 50)], axis=1)
    return {
        'parity': {'failures': parity, 'conditions': ['even', 'odd'], 'panel_size': 3,
                   'expected_majority_error': [0.75, 0.25]},
        'accuracy_trap': {'failures': acc, 'conditions': ['nominal', 'stress'], 'panel_size': 3,
                          'expected_accuracy_panel': [0, 1, 2], 'expected_accuracy_worst': 0.2,
                          'expected_joint_worst': 0.0},
        'pairwise_trap': {'failures': pair, 'conditions': ['storm'], 'panel_size': 3,
                          'expected_pairwise_panel': [0, 1, 2], 'expected_joint_panel': [2, 3, 4],
                          'expected_pairwise_worst': 0.18, 'expected_joint_worst': 0.10},
        'pooled_trap': {'failures': pooled, 'conditions': ['a', 'b'], 'panel_size': 3,
                        'expected_pooled_worst': 0.30, 'expected_pooled_pooled': 0.15,
                        'expected_joint_worst': 0.20, 'expected_joint_pooled': 0.20},
    }


def check_constructed_cases() -> dict[str, bool]:
    """Named booleans: exactness on the parity case and the expected direction on each trap."""
    cases = constructed_cases()
    checks: dict[str, bool] = {}
    par = cases['parity']['failures']
    err = majority_error(par, (0, 1, 2))
    checks['parity_majority_error_exact'] = bool(np.array_equal(err, [0.75, 0.25]))
    checks['parity_signature_path_exact'] = bool(np.array_equal(
        majority_error_from_signatures(failure_signatures(par), (0, 1, 2)), [0.75, 0.25]))
    marg = par.mean(axis=2)
    co = np.array([[par[i, c] & par[j, c] for c in range(2)] for i, j in ((0, 1), (0, 2), (1, 2))]).mean(axis=2)
    checks['parity_marginals_and_pairs_identical'] = bool(np.all(marg == 0.5) and np.all(co == 0.25))
    for name, policy, other in (('accuracy_trap', 'joint', 'accuracy'), ('pairwise_trap', 'joint', 'pairwise'),
                                ('pooled_trap', 'joint', 'pooled')):
        case = cases[name]
        f = case['failures']
        chosen = select_panels(f, case['panel_size'])
        worst = {p: float(majority_error(f, chosen[p]).max()) for p in chosen}
        checks[f'{name}_joint_lower_worst_condition_error_than_{other}'] = bool(worst[policy] < worst[other])
    acc = cases['accuracy_trap']
    pick = select_panels(acc['failures'], 3)
    checks['accuracy_trap_values'] = bool(list(pick['accuracy']) == acc['expected_accuracy_panel'] and math.isclose(
        majority_error(acc['failures'], pick['accuracy']).max(), acc['expected_accuracy_worst']) and
        majority_error(acc['failures'], pick['joint']).max() == acc['expected_joint_worst'])
    pt = cases['pairwise_trap']
    pick = select_panels(pt['failures'], 3)
    checks['pairwise_trap_values'] = bool(
        list(pick['pairwise']) == pt['expected_pairwise_panel'] and list(pick['joint']) == pt['expected_joint_panel']
        and math.isclose(majority_error(pt['failures'], pick['pairwise'])[0], pt['expected_pairwise_worst'])
        and math.isclose(majority_error(pt['failures'], pick['joint'])[0], pt['expected_joint_worst']))
    po = cases['pooled_trap']
    pick = select_panels(po['failures'], 3)
    e_pool, e_joint = majority_error(po['failures'], pick['pooled']), majority_error(po['failures'], pick['joint'])
    checks['pooled_trap_values'] = bool(
        math.isclose(e_pool.max(), po['expected_pooled_worst']) and math.isclose(e_pool.mean(), po['expected_pooled_pooled'])
        and math.isclose(e_joint.max(), po['expected_joint_worst'])
        and math.isclose(e_joint.mean(), po['expected_joint_pooled']))
    return checks


# ====================================================================== synthetic benchmark
_EXP_DEFAULTS: dict[str, Any] = {'panel_size': 3, **_GEN_DEFAULTS}
_EXP_QUICK: dict[str, Any] = {'calibration_trials': 48, 'test_trials': 256}


def _policy_record(sel: Mapping[str, Any], structure: Mapping[str, Any], policy: str) -> dict[str, Any]:
    rec = sel['policies'][policy]
    expected = expected_condition_errors(structure, rec['panel'], split='test')
    return {'panel': rec['panel'], 'calibration_worst_condition_error': rec['calibration']['worst_condition_error'],
            'test_worst_condition_error': rec['test']['worst_condition_error'],
            'test_pooled_error': rec['test']['pooled_error'], 'test_condition_error': rec['test']['condition_error'],
            'expected_worst_condition_error': float(expected.max()), 'expected_pooled_error': float(expected.mean())}


def _scenario_summary(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for policy in POLICIES + ('oracle',):
        rows = [r['policies'][policy] for r in per_seed]
        out[policy] = {key: float(np.mean([r[key] for r in rows])) for key in
                       ('test_worst_condition_error', 'test_pooled_error', 'expected_worst_condition_error')}
    return out


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Synthetic joint-failure selection benchmark (FAULTLINE-style), held-out worst-condition error.

    For every seed and scenario (``shared``, ``independent``, ``shift``) all four selectors see the same
    calibration trials and candidate panels; frozen panels are scored on fresh test trials. The verdict is
    ``harness.compare`` of joint vs accuracy test worst-condition error in the ``shared`` scenario (lower is
    better). Other comparisons, exact population errors of every chosen panel and the constructed
    known-answer checks are reported alongside."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seed_list = _seeds(seeds)
    unknown = sorted(set(overrides) - set(_EXP_DEFAULTS))
    if unknown:
        raise ValueError(f'unknown run_experiment parameters: {unknown}')
    cfg = {**_EXP_DEFAULTS, **(_EXP_QUICK if quick else {}), **overrides}
    k = majority_threshold(cfg.pop('panel_size')) * 2 - 1
    gen = _gen_config(cfg)
    m = gen['families'] * gen['family_size']
    if k > m:
        raise ValueError('panel_size must not exceed families * family_size')
    if math.comb(m, k) > MAX_PANELS:
        raise ValueError(f'C(components, panel_size) must be at most {MAX_PANELS}')
    t0 = time.perf_counter()
    per: dict[str, list[dict[str, Any]]] = {s: [] for s in SCENARIOS}
    for seed in seed_list:
        for scenario in SCENARIOS:
            data = make_failure_data(seed, scenario=scenario, **gen)
            sel = run_selection(data['calibration'], data['test'], k)
            rec = {'seed': seed, 'data_digest': data['digest'],
                   'policies': {p: _policy_record(sel, data['structure'], p) for p in POLICIES}}
            orc = sel['oracle']
            expected = expected_condition_errors(data['structure'], orc['panel'], split='test')
            rec['policies']['oracle'] = {'panel': orc['panel'],
                                         'test_worst_condition_error': orc['test']['worst_condition_error'],
                                         'test_pooled_error': orc['test']['pooled_error'],
                                         'expected_worst_condition_error': float(expected.max())}
            per[scenario].append(rec)
    checks = check_constructed_cases()
    wall = time.perf_counter() - t0

    def vals(scenario: str, policy: str) -> list[float]:
        return [r['policies'][policy]['test_worst_condition_error'] for r in per[scenario]]

    comparisons = {
        'shared_joint_vs_accuracy': harness.compare(vals('shared', 'joint'), vals('shared', 'accuracy'),
                                                    higher_is_better=False),
        'shared_joint_vs_pairwise': harness.compare(vals('shared', 'joint'), vals('shared', 'pairwise'),
                                                    higher_is_better=False),
        'shared_joint_vs_pooled': harness.compare(vals('shared', 'joint'), vals('shared', 'pooled'),
                                                  higher_is_better=False),
        'independent_joint_vs_accuracy': harness.compare(vals('independent', 'joint'), vals('independent', 'accuracy'),
                                                         higher_is_better=False),
        'shift_joint_vs_accuracy': harness.compare(vals('shift', 'joint'), vals('shift', 'accuracy'),
                                                   higher_is_better=False),
    }
    n_p = math.comb(m, k)
    kc = gen['families'] + 1
    receipt = harness.base_receipt(MODULE_ID, 'selection_benchmark', {
        **gen, 'panel_size': k, 'seeds': list(seed_list), 'quick': quick, 'components': m, 'conditions': kc,
        'scenarios': list(SCENARIOS), 'policies': list(POLICIES),
        'metric': 'held-out (test) worst-condition majority-vote error, lower is better',
        'selection': 'calibration trials only; exhaustive over all C(components, panel_size) panels; integer-count '
                     'keys, ties to the next key then the lexicographically first panel',
        'oracle': 'panel with the lowest test worst-condition error (chosen on test data; reference only)',
        'expected': 'exact population error of the chosen panel under the generator (Poisson-binomial mixture)'})
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['per_seed'] = per
    receipt['summary'] = {s: _scenario_summary(per[s]) for s in SCENARIOS}
    receipt['comparisons'] = comparisons
    receipt['verdict'] = comparisons['shared_joint_vs_accuracy']['verdict']
    receipt['verdict_basis'] = 'joint_vs_accuracy_heldout_worst_condition_error_shared_scenario'
    receipt['constructed_checks'] = checks
    receipt['constructed_checks_pass'] = bool(all(checks.values()))
    receipt['budget'] = {'candidate_panels': n_p, 'calibration_component_calls': m * kc * gen['calibration_trials'],
                         'deployment_component_calls_per_case': k, 'trainable_parameters': 0,
                         'note': 'every selector sees the same calibration trials and candidate panels'}
    receipt['estimated_cost'] = {'panel_trial_evaluations_per_selection': n_p * kc * gen['calibration_trials'],
                                 'pair_cofailure_updates_per_selection': m * m * kc * gen['calibration_trials']}
    receipt['measured_wallclock_s'] = wall
    receipt['notes'] = ('Synthetic benchmark with a documented generator: correlated stressors shared within a '
                        'family (shared), a same-ranking independent-error control (independent), and a common '
                        'cause that is absent from calibration and active on held-out nominal trials (shift; with '
                        "shift_target='hardened' it hits the lowest-error member of every family, which is "
                        'adversarial to diversity-based selection by construction). Joint-failure selection can '
                        'only protect against failure modes that calibration exercised. Paired 95 % t intervals over '
                        'seeds summarise seed-to-seed variability; they are not anytime-valid. Inspired by FAULTLINE; '
                        'none of its recorded numbers are reproduced.')
    return harness.seal(receipt)


# ====================================================================== NEUROWEAVE application
# The unmodified engine plus the six single mechanisms that changed the most training-phase head decisions
# relative to it in a diversity probe on seed 99 (never an evaluation seed here), 16 and 24 neurons, 120 and
# 160 steps (2026-11-20). The probe looked at decision changes only, never at accuracy or at any selector.
_NW_MECHANISM_SETS: tuple[tuple[str, ...], ...] = (
    (), ('sparse_code',), ('homeostatic_governor',), ('astrocyte_regulation',), ('multi_soma',),
    ('reward_plasticity',), ('dendritic_compartments',))
_NW_DEFAULTS: dict[str, Any] = {
    'mechanism_sets': _NW_MECHANISM_SETS, 'tasks': harness.TASKS, 'panel_size': 3, 'neurons': 24,
    'topology': 'clustered', 'train_steps': 160, 'calibration_steps': 96, 'test_steps': 96,
    'calibration_seed_offset': 50_000, 'test_seed_offset': 60_000}
_NW_QUICK: dict[str, Any] = {'mechanism_sets': _NW_MECHANISM_SETS[:5], 'neurons': 16, 'train_steps': 120,
                             'calibration_steps': 32, 'test_steps': 32}


def _nw_config(quick: bool, overrides: Mapping[str, Any]) -> dict[str, Any]:
    from ..neuroweave.spec import MECHANISM_IDS, TOPOLOGIES
    from ..neuroweave.tasks import TASKS as NW_TASKS
    unknown = sorted(set(overrides) - set(_NW_DEFAULTS))
    if unknown:
        raise ValueError(f'unknown run_neuroweave_application parameters: {unknown}')
    cfg = {**_NW_DEFAULTS, **(_NW_QUICK if quick else {}), **overrides}
    sets = cfg['mechanism_sets']
    if isinstance(sets, (str, bytes, Mapping)):
        raise ValueError('mechanism_sets must be a sequence of mechanism-id sequences')
    try:
        sets = tuple(tuple(s) if not isinstance(s, str) else (s,) for s in sets)
    except TypeError as exc:
        raise ValueError('mechanism_sets must be a sequence of mechanism-id sequences') from exc
    if not 1 <= len(sets) <= 12:
        raise ValueError('mechanism_sets must hold 1..12 sets')
    for s in sets:
        bad = [x for x in s if x not in MECHANISM_IDS]
        if bad:
            raise ValueError(f'unknown NEUROWEAVE mechanisms: {bad}')
    if len({tuple(sorted(s)) for s in sets}) != len(sets):
        raise ValueError('mechanism_sets must be distinct')
    cfg['mechanism_sets'] = sets
    tasks = cfg['tasks']
    if isinstance(tasks, (str, bytes, Mapping)):
        raise ValueError('tasks must be a sequence of task ids')
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in NW_TASKS for t in tasks):
        raise ValueError(f'tasks must be distinct ids from {NW_TASKS}')
    cfg['tasks'] = tasks
    k = majority_threshold(cfg['panel_size']) * 2 - 1
    if k > len(sets):
        raise ValueError('panel_size must not exceed the number of mechanism sets')
    cfg['panel_size'] = k
    cfg['neurons'] = _int('neurons', cfg['neurons'], 8, 512)
    if cfg['topology'] not in TOPOLOGIES:
        raise ValueError(f'topology must be one of {TOPOLOGIES}')
    for key in ('train_steps', 'calibration_steps', 'test_steps'):
        cfg[key] = _int(key, cfg[key], 4, 20_000)
    for key in ('calibration_seed_offset', 'test_seed_offset'):
        cfg[key] = _int(key, cfg[key], 1, 2**30)
    if cfg['calibration_seed_offset'] == cfg['test_seed_offset']:
        raise ValueError('calibration_seed_offset and test_seed_offset must differ')
    return cfg


def _nw_component(mechs: tuple[str, ...], seed: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    from ..neuroweave.engine import NeuroWeaveEngine
    from ..neuroweave.experiments import ReadoutHead, clone_engine, make_spec, run_task_phase
    from ..neuroweave.tasks import make_stream
    tasks = cfg['tasks']
    spec = make_spec(mechs, seed=seed, neurons=cfg['neurons'], topology=cfg['topology'], steps=cfg['train_steps'])
    engine = NeuroWeaveEngine(spec)
    heads = {t: ReadoutHead(engine.n, seed) for t in tasks}
    train_stream = make_stream(seed, cfg['train_steps'])
    train = run_task_phase(engine, heads, train_stream, learn=True)
    out: dict[str, Any] = {'mechanisms': list(mechs), 'train_stream_digest': train_stream.digest,
                           'train_accuracy': {t: train['tasks'][t]['accuracy'] for t in tasks},
                           'head_parameters': int(sum(h.parameter_count for h in heads.values()))}
    for split, offset, steps in (('calibration', cfg['calibration_seed_offset'], cfg['calibration_steps']),
                                 ('test', cfg['test_seed_offset'], cfg['test_steps'])):
        stream = make_stream(seed + offset, steps)
        res = run_task_phase(clone_engine(engine), {t: h.copy() for t, h in heads.items()}, stream, learn=False)
        out[f'{split}_failures'] = np.array([[c < 0.5 for c in res['tasks'][t]['correct']] for t in tasks], dtype=bool)
        out[f'{split}_accuracy'] = {t: res['tasks'][t]['accuracy'] for t in tasks}
        out[f'{split}_stream_digest'] = stream.digest
    return out


def run_neuroweave_application(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2),
                               **overrides: Any) -> dict[str, Any]:
    """Joint-failure panel selection over NEUROWEAVE readouts.

    Components are NEUROWEAVE engines that differ only in their mechanism set, each trained (``learn=True``,
    the lab's ``run_task_phase``) with one logistic ``ReadoutHead`` per task; conditions are the temporal
    tasks. Failures are recorded frozen on a calibration stream and on a disjoint test stream (stream seeds
    ``seed + calibration_seed_offset`` / ``seed + test_seed_offset``). The verdict is ``harness.compare`` of
    joint vs accuracy held-out worst-task error over seeds (lower is better)."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seed_list = _seeds(seeds)
    cfg = _nw_config(quick, overrides)
    train_seeds = set(seed_list)
    cal_seeds = {s + cfg['calibration_seed_offset'] for s in seed_list}
    test_seeds = {s + cfg['test_seed_offset'] for s in seed_list}
    if train_seeds & cal_seeds or train_seeds & test_seeds or cal_seeds & test_seeds:
        raise ValueError('training, calibration and test stream seeds must be pairwise disjoint')
    k = cfg['panel_size']
    t0 = time.perf_counter()
    per_seed = []
    for seed in seed_list:
        comps = [_nw_component(m, seed, cfg) for m in cfg['mechanism_sets']]
        cal = np.stack([c['calibration_failures'] for c in comps])
        test = np.stack([c['test_failures'] for c in comps])
        sel = run_selection(cal, test, k)
        flat = cal.reshape(cal.shape[0], -1)
        agreement = (flat[:, None, :] == flat[None, :, :]).mean(axis=2)
        iu = np.triu_indices(cal.shape[0], 1)
        per_seed.append({
            'seed': seed,
            'components': [{key: c[key] for key in ('mechanisms', 'train_stream_digest', 'train_accuracy',
                                                    'calibration_accuracy', 'test_accuracy', 'head_parameters',
                                                    'calibration_stream_digest', 'test_stream_digest')}
                           for c in comps],
            'policies': {p: {'panel': sel['policies'][p]['panel'],
                             'calibration_worst_condition_error': sel['policies'][p]['calibration']['worst_condition_error'],
                             'test_worst_condition_error': sel['policies'][p]['test']['worst_condition_error'],
                             'test_pooled_error': sel['policies'][p]['test']['pooled_error'],
                             'test_condition_error': sel['policies'][p]['test']['condition_error']}
                         for p in POLICIES},
            'oracle': {'panel': sel['oracle']['panel'],
                       'test_worst_condition_error': sel['oracle']['test']['worst_condition_error']},
            'best_single_component_test_worst_error': float(test.mean(axis=2).max(axis=1).min()),
            'calibration_failure_agreement': {'mean_offdiagonal': float(agreement[iu].mean()),
                                              'min_offdiagonal': float(agreement[iu].min()),
                                              'identical_pairs': int(np.sum(agreement[iu] == 1.0))},
        })
    wall = time.perf_counter() - t0

    def vals(policy: str) -> list[float]:
        return [r['policies'][policy]['test_worst_condition_error'] for r in per_seed]

    comparisons = {'joint_vs_accuracy': harness.compare(vals('joint'), vals('accuracy'), higher_is_better=False),
                   'joint_vs_pairwise': harness.compare(vals('joint'), vals('pairwise'), higher_is_better=False),
                   'joint_vs_pooled': harness.compare(vals('joint'), vals('pooled'), higher_is_better=False)}
    n_c, n_t = len(cfg['mechanism_sets']), len(cfg['tasks'])
    steps_per_component = cfg['train_steps'] + cfg['calibration_steps'] + cfg['test_steps']
    receipt = harness.base_receipt(NEUROWEAVE_ID, 'selection_application', {
        **cfg, 'mechanism_sets': [list(s) for s in cfg['mechanism_sets']], 'tasks': list(cfg['tasks']),
        'seeds': list(seed_list), 'quick': quick,
        'protocol': 'per seed and mechanism set: one engine trained learn=True with one ReadoutHead per task '
                    '(run_task_phase; the first task supplies the reward), then frozen calibration and test streams',
        'metric': 'held-out (test) worst-task majority-vote error, lower is better'})
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['per_seed'] = per_seed
    receipt['summary'] = {p: {'test_worst_condition_error': float(np.mean(vals(p)))} for p in POLICIES}
    receipt['summary']['oracle'] = {'test_worst_condition_error': float(np.mean(
        [r['oracle']['test_worst_condition_error'] for r in per_seed]))}
    receipt['comparisons'] = comparisons
    receipt['verdict'] = comparisons['joint_vs_accuracy']['verdict']
    receipt['verdict_basis'] = 'joint_vs_accuracy_heldout_worst_task_error'
    receipt['budget'] = {'components': n_c, 'candidate_panels': math.comb(n_c, k),
                         'calibration_component_calls': n_c * n_t * cfg['calibration_steps'],
                         'deployment_component_calls_per_case': k,
                         'head_parameters_per_component': per_seed[0]['components'][0]['head_parameters']}
    receipt['estimated_cost'] = {'engine_steps': len(seed_list) * n_c * steps_per_component}
    receipt['measured_wallclock_s'] = wall
    receipt['notes'] = ('Synthetic NEUROWEAVE temporal tasks with binary targets, so majority-of-failures equals '
                        'majority-vote error exactly. Components differ only in their mechanism set (same seed, '
                        'topology, training stream and head initialisation), so their failures are strongly '
                        'correlated; calibration_failure_agreement reports how often two components fail on the same '
                        'trials. The default mechanism sets were chosen by a decision-diversity probe on seed 99, '
                        'not by accuracy or by any selector outcome; many other single mechanisms leave the head '
                        'decisions unchanged at these sizes. The heads are close to chance on most tasks, so this '
                        'demonstrates the selection pipeline on real engine outputs; it is not evidence that any '
                        'mechanism or selector helps NEUROWEAVE.')
    return harness.seal(receipt)


EXPERIMENTS = {MODULE_ID: run_experiment, NEUROWEAVE_ID: run_neuroweave_application}
