"""Budget-matched synthetic sequence-variant search assay (Fly-Protein-inspired).

Port of NEUROWEAVE-LAB 0.4.0 ``runVariantSearchAssay``. Three search strategies
receive exactly the same number of fitness queries per run on the same
synthetic landscapes:

``sparse``     sparse random-projection guidance: every observed sequence is
               encoded by a fixed random +/-1 projection of its one-hot code
               followed by k-winners-take-all; mutants of the current top
               sequences are *scored without querying* by the code-overlap
               weighted mean of observed fitness, and only the best-scored
               mutant is queried;
``evolution``  steady-state evolutionary search (tournament selection,
               uniform crossover, point mutation, population 12);
``random``     uniform random sequences.

Each landscape is a synthetic additive-plus-pairwise-epistatic function on
``length`` positions over an ``alphabet``-symbol alphabet with values in
[0, 1]. The empirical top-1 % *hit threshold* of a landscape is the 99th
percentile of a fixed random reference sample (not counted in any query
budget). Every run is paired across strategies by (landscape, seed), and
paired 95 % t-intervals decide the verdicts with the same rule as the
experiment laboratory.

This is **not** protein modelling: there is no structure, no folding, no
biophysics and no ProteinGym-style benchmark. The landscape is a toy fitness
function; results do not generalise to real protein engineering.
``evidence_tier='synthetic_abstraction'``. The NumPy random streams differ from
the browser prototype's ``rng32`` so numbers are not expected to match it.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from .spec import EVIDENCE_TIER

__all__ = ['SEQUENCE_SEARCH_SCHEMA', 'STRATEGIES', 'DEFAULTS', 'QUICK', 'DISCLAIMER', 'SyntheticLandscape',
           'make_landscape', 'run_strategy', 'run_sequence_search_assay']

SEQUENCE_SEARCH_SCHEMA = 'NeuroWeaveSequenceSearchV1'
STRATEGIES = ('sparse', 'evolution', 'random')
STRATEGY_LABELS = {'sparse': 'Sparse-code guided', 'evolution': 'Evolutionary search', 'random': 'Random search'}
DEFAULTS: dict[str, Any] = {'landscapes': 10, 'seeds': 8, 'budget': 50, 'length': 24, 'alphabet': 20,
                            'projection_dim': 32, 'k_winners': 4, 'reference_samples': 600, 'initial': 12,
                            'candidates': 24, 'epistatic_pairs': 18, 'base_seed': 88000}
QUICK: dict[str, Any] = {'landscapes': 3, 'seeds': 3, 'budget': 30, 'reference_samples': 300}
DISCLAIMER = ('Synthetic additive-plus-epistatic sequence landscape; not a protein-folding model, not a '
              'ProteinGym assay and not evidence about real protein engineering.')
_LANDSCAPE_STRIDE = 97
_REF_OFFSET = 500_000
_RUN_OFFSET = 12_000
_STRATEGY_OFFSET = {'sparse': 0, 'evolution': 991, 'random': 1777}
_PROJECTION_OFFSET = 713


class SyntheticLandscape:
    """Additive (match-to-target) + pairwise epistatic bonus fitness in [0, 1]; counts its queries."""

    def __init__(self, seed: int, length: int, alphabet: int, pairs: int) -> None:
        rng = np.random.default_rng(seed)
        self.length, self.alphabet = int(length), int(alphabet)
        self.target = rng.integers(0, alphabet, size=length)
        self.pair_a = rng.integers(0, length, size=pairs)
        self.pair_b = rng.integers(0, length, size=pairs)
        self.bonus = 0.25 + 0.45 * rng.random(pairs)
        self._bonus_total = float(self.bonus.sum())
        self.queries = 0

    def evaluate(self, seq: np.ndarray, *, count: bool = True) -> float:
        match = np.asarray(seq) == self.target
        epi = float(self.bonus[match[self.pair_a] & match[self.pair_b]].sum())
        if count:
            self.queries += 1
        return float(np.clip(0.58 * match.mean() + 0.42 * epi / self._bonus_total, 0.0, 1.0))


def make_landscape(index: int, cfg: dict[str, Any]) -> SyntheticLandscape:
    return SyntheticLandscape(cfg['base_seed'] + index * _LANDSCAPE_STRIDE, cfg['length'], cfg['alphabet'],
                              cfg['epistatic_pairs'])


def _random_seq(rng: np.random.Generator, cfg: dict[str, Any]) -> np.ndarray:
    return rng.integers(0, cfg['alphabet'], size=cfg['length'])


def _mutate(seq: np.ndarray, rng: np.random.Generator, count: int, cfg: dict[str, Any]) -> np.ndarray:
    out = seq.copy()
    for _ in range(count):
        out[int(rng.integers(0, cfg['length']))] = int(rng.integers(0, cfg['alphabet']))
    return out


def _sparse_run(land: SyntheticLandscape, seed: int, cfg: dict[str, Any]) -> float:
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + _PROJECTION_OFFSET)
    length, alphabet, k = cfg['length'], cfg['alphabet'], cfg['k_winners']
    projection = np.where(prng.random((cfg['projection_dim'], length * alphabet)) < 0.5, -1.0, 1.0)
    cols = np.arange(length) * alphabet

    def encode(seq: np.ndarray) -> frozenset:
        sums = projection[:, cols + seq].sum(axis=1)
        return frozenset(np.argsort(-sums, kind='stable')[:k].tolist())

    seqs: list[np.ndarray] = []
    fits: list[float] = []
    codes: list[frozenset] = []
    for _ in range(min(cfg['initial'], cfg['budget'])):
        s = _random_seq(rng, cfg)
        seqs.append(s); fits.append(land.evaluate(s)); codes.append(encode(s))
    while len(seqs) < cfg['budget']:
        order = sorted(range(len(fits)), key=lambda i: -fits[i])[:5]
        prior = float(np.mean(fits))
        best_pred, best = -np.inf, None
        for c in range(cfg['candidates']):
            parent = seqs[order[c % len(order)]]
            cand = _mutate(parent, rng, 1 + int(rng.integers(0, 3)), cfg)
            code = encode(cand)
            weighted = den = 0.0
            for old_code, f in zip(codes, fits):
                sim = len(code & old_code) / k
                if sim > 0:
                    weighted += sim * f
                    den += sim
            pred = weighted / den if den else prior
            if pred > best_pred:            # first maximum wins (stable, like the JS sort)
                best_pred, best = pred, (cand, code)
        seqs.append(best[0]); fits.append(land.evaluate(best[0])); codes.append(best[1])
    return max(fits)


def _evolution_run(land: SyntheticLandscape, seed: int, cfg: dict[str, Any]) -> float:
    rng = np.random.default_rng(seed + _STRATEGY_OFFSET['evolution'])
    pop: list[tuple[float, np.ndarray]] = []
    for _ in range(min(cfg['initial'], cfg['budget'])):
        s = _random_seq(rng, cfg)
        pop.append((land.evaluate(s), s))
    evaluated = len(pop)

    def tournament() -> np.ndarray:
        best = None
        for _ in range(3):
            x = pop[int(rng.integers(0, len(pop)))]
            if best is None or x[0] > best[0]:
                best = x
        return best[1]

    while evaluated < cfg['budget']:
        a, b = tournament(), tournament()
        child = np.where(rng.random(cfg['length']) < 0.5, a, b)
        child = _mutate(child, rng, 1 + int(rng.integers(0, 4)), cfg)
        pop.append((land.evaluate(child), child))
        pop.sort(key=lambda x: -x[0])
        del pop[cfg['initial']:]
        evaluated += 1
    return max(f for f, _ in pop)


def _random_run(land: SyntheticLandscape, seed: int, cfg: dict[str, Any]) -> float:
    rng = np.random.default_rng(seed + _STRATEGY_OFFSET['random'])
    return max(land.evaluate(_random_seq(rng, cfg)) for _ in range(cfg['budget']))


_RUNNERS: dict[str, Callable[[SyntheticLandscape, int, dict[str, Any]], float]] = {
    'sparse': _sparse_run, 'evolution': _evolution_run, 'random': _random_run}


def run_strategy(name: str, landscape: SyntheticLandscape, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """One budgeted run; returns best fitness and the exact number of fitness queries used."""
    if name not in _RUNNERS:
        raise ValueError(f'strategy must be one of {STRATEGIES}')
    before = landscape.queries
    best = _RUNNERS[name](landscape, seed, cfg)
    return {'best': float(best), 'queries': int(landscape.queries - before)}


def _reference_threshold(land: SyntheticLandscape, index: int, cfg: dict[str, Any]) -> float:
    rng = np.random.default_rng(_REF_OFFSET + index * 601)
    ref = np.sort([land.evaluate(_random_seq(rng, cfg), count=False) for _ in range(cfg['reference_samples'])])
    return float(ref[int(np.floor(len(ref) * 0.99))])


def _int_opt(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def run_sequence_search_assay(*, quick: bool = False, landscapes: int | None = None, seeds: int | None = None,
                              budget: int | None = None, length: int | None = None, alphabet: int | None = None,
                              projection_dim: int | None = None, k_winners: int | None = None,
                              reference_samples: int | None = None, base_seed: int | None = None) -> dict[str, Any]:
    """Run the budget-matched assay and return a digest-sealed receipt."""
    from .experiments import _finalise, paired_stats, verdict_from_stats  # lazy: avoids an import cycle
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    cfg: dict[str, Any] = dict(DEFAULTS)
    if quick:
        cfg.update(QUICK)
    given = {'landscapes': landscapes, 'seeds': seeds, 'budget': budget, 'length': length, 'alphabet': alphabet,
             'projection_dim': projection_dim, 'k_winners': k_winners, 'reference_samples': reference_samples,
             'base_seed': base_seed}
    cfg.update({k: v for k, v in given.items() if v is not None})
    _int_opt('landscapes', cfg['landscapes'], 1, 50)
    _int_opt('seeds', cfg['seeds'], 1, 32)
    _int_opt('length', cfg['length'], 4, 64)
    _int_opt('alphabet', cfg['alphabet'], 2, 32)
    _int_opt('budget', cfg['budget'], cfg['initial'] + 1, 500)
    _int_opt('projection_dim', cfg['projection_dim'], 4, 256)
    _int_opt('k_winners', cfg['k_winners'], 1, cfg['projection_dim'])
    _int_opt('reference_samples', cfg['reference_samples'], 100, 5000)
    _int_opt('base_seed', cfg['base_seed'], 0, 2**31 - 2_000_000)
    started = time.perf_counter()
    per_run: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    hits: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    landscape_rows = []
    budget_ok = True
    for a in range(cfg['landscapes']):
        land = make_landscape(a, cfg)
        threshold = _reference_threshold(land, a, cfg)
        row: dict[str, Any] = {'index': a, 'top1pct_threshold': threshold, 'mean_best': {}, 'hit_rate': {}}
        for seed in range(cfg['seeds']):
            run_seed = _RUN_OFFSET + a * 1009 + seed * 31
            for name in STRATEGIES:
                res = run_strategy(name, land, run_seed, cfg)
                budget_ok &= res['queries'] == cfg['budget']
                per_run[name].append(res['best'])
                hits[name].append(int(res['best'] >= threshold))
        n = cfg['seeds']
        for name in STRATEGIES:
            row['mean_best'][name] = float(np.mean(per_run[name][-n:]))
            row['hit_rate'][name] = float(np.mean(hits[name][-n:]))
        landscape_rows.append(row)
    if not budget_ok:  # pragma: no cover - guarded by construction
        raise RuntimeError('query budget was not matched across strategies')
    runs = cfg['landscapes'] * cfg['seeds']
    results = [{'strategy': s, 'label': STRATEGY_LABELS[s], 'mean_best_fitness': float(np.mean(per_run[s])),
                'top1pct_hit_rate': float(np.mean(hits[s])), 'runs': runs, 'queries_per_run': cfg['budget']}
               for s in STRATEGIES]
    comparisons = {}
    for other in ('evolution', 'random'):
        st = paired_stats(per_run['sparse'], per_run[other])
        comparisons[f'sparse_vs_{other}'] = {'metric': 'best_fitness', 'higher_is_better': True, **st,
                                             **verdict_from_stats(st, higher_is_better=True)}
    receipt = {
        'schema': SEQUENCE_SEARCH_SCHEMA, 'experiment_id': 'sequence_search', 'supplementary': True,
        'config': dict(cfg, quick=quick), 'strategies': list(STRATEGIES), 'results': results,
        'landscapes': landscape_rows, 'comparisons': comparisons,
        'verdict': comparisons['sparse_vs_evolution']['verdict'],
        'verdict_basis': {'comparison': 'sparse_vs_evolution', 'metric': 'best_fitness', 'higher_is_better': True,
                          'pairing': 'per (landscape, seed) run',
                          'caveat': 'runs on one landscape share it; the t interval treats runs as independent '
                                    'and may be optimistic'},
        'budget_matched': True, 'reference_samples_counted_in_budget': False,
        'hit_threshold': 'empirical 99th percentile of a fixed random reference sample per landscape (approximate)',
        'disclaimer': DISCLAIMER, 'protein_modelling': False, 'evidence_tier': EVIDENCE_TIER,
        'measured_wallclock_s_total': time.perf_counter() - started,
    }
    return _finalise(receipt)
