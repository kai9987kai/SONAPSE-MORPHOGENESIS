"""NEUROWEAVE lesion assay: targeted / random / pathway lesions on *clones* only.

An intact engine is trained (prequentially) on a temporal task while a second
readout head learns a delayed-recall memory probe on the same input stream.
Each lesion is then applied to an independent clone (the original engine is
digest-checked before and after, and the assay raises if it changed):

``intact``           no change (reference)
``random``           a fraction of all synapses removed
``hub``              the highest-degree neurons silenced
``region``           neurons silenced region by region from a seeded start region
``sensory_pathway``  synapses leaving sensory regions removed
``memory_pathway``   synapses touching memory regions removed
``cross_region``     inter-region synapses removed

A *silenced* neuron loses every incident synapse and its output is clamped to
0 after each step. ``fraction`` is the fraction of the targeted pool. With
``progressive=True`` the lesion sets are nested (one seeded order per kind,
larger fractions extend smaller ones) so structural damage is monotone.

Measured per (seed, kind, fraction): frozen held-out accuracy / log-loss /
Brier, memory-probe retention, connectivity change (``topology_metrics``),
functional redundancy (performance above chance relative to intact) and
recovery time (steps of continued learning until a rolling accuracy reaches
90 % of the intact clone's accuracy on the same recovery stream).

Synthetic abstraction only (``evidence_tier='synthetic_abstraction'``).
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from .engine import NeuroWeaveEngine
from .experiments import (LAB_SCHEMA, ReadoutHead, _finalise, _mean, _state_digest, clone_engine, engine_topology,
                          make_spec, paired_stats, run_task_phase, verdict_from_stats)
from .spec import EVIDENCE_TIER, MECHANISM_IDS, REGIONS, TOPOLOGIES
from .tasks import TASKS, make_stream
from .topology import topology_metrics

__all__ = ['LESION_KINDS', 'LESION_SCHEMA', 'lesion_order', 'apply_lesion', 'run_lesion_assay']

LESION_SCHEMA = 'NeuroWeaveLesionAssayV1'
LESION_KINDS = ('intact', 'random', 'hub', 'region', 'sensory_pathway', 'memory_pathway', 'cross_region')
_NEURON_KINDS = frozenset({'hub', 'region'})
_LESION_OFFSET = 9_601
_RECOVERY_SEED_OFFSET = 70_000
_REGION_KIND = np.array([k for _, k in REGIONS])


def lesion_order(engine: NeuroWeaveEngine, kind: str, rng: np.random.Generator) -> tuple[str, np.ndarray]:
    """Ordered lesion candidates: ``('edges', edge indices)`` or ``('neurons', neuron indices)``."""
    if kind not in LESION_KINDS:
        raise ValueError(f'unknown lesion kind {kind!r}')
    syn, cells = engine.brain.synapses, engine.brain.cells
    n = int(engine.n)
    region = np.asarray(cells.region[:n], dtype=np.int64)
    if kind == 'intact':
        return 'edges', np.zeros(0, dtype=np.int64)
    if kind == 'hub':
        deg = np.bincount(syn.src, minlength=n)[:n] + np.bincount(syn.dst, minlength=n)[:n]
        return 'neurons', np.lexsort((np.arange(n), -deg)).astype(np.int64)
    if kind == 'region':
        present = np.unique(region)
        start = int(present[int(rng.integers(0, present.size))])
        dist = (region - start) % len(REGIONS)
        return 'neurons', np.lexsort((rng.permutation(n), dist)).astype(np.int64)
    rk_src, rk_dst = _REGION_KIND[region[syn.src]], _REGION_KIND[region[syn.dst]]
    if kind == 'random':
        pool = np.arange(syn.size)
    elif kind == 'sensory_pathway':
        pool = np.flatnonzero(rk_src == 'sensory')
    elif kind == 'memory_pathway':
        pool = np.flatnonzero((rk_src == 'memory') | (rk_dst == 'memory'))
    else:  # cross_region
        pool = np.flatnonzero(region[syn.src] != region[syn.dst])
    return 'edges', rng.permutation(pool).astype(np.int64)


def apply_lesion(engine: NeuroWeaveEngine, unit: str, chosen: np.ndarray) -> dict[str, Any]:
    """Lesion ``engine`` in place (callers pass a clone). Returns the silence set and counts."""
    syn = engine.brain.synapses
    chosen = np.asarray(chosen, dtype=np.int64)
    mask = np.zeros(syn.size, dtype=bool)
    silence = np.zeros(0, dtype=np.int64)
    if unit == 'neurons':
        silence = np.sort(chosen)
        if silence.size:
            mask |= np.isin(syn.src, silence) | np.isin(syn.dst, silence)
    elif chosen.size:
        mask[chosen] = True
    removed = int(mask.sum())
    if removed:
        engine.brain.synapses = syn.lesion(mask)
    if silence.size:
        engine.brain.cells.output[silence] = 0.0
    return {'silence': silence, 'edges_removed': removed, 'neurons_silenced': int(silence.size)}


def _rolling_recovery(correct: Sequence[float], target: float, window: int) -> int | None:
    c = np.asarray(correct, dtype=np.float64)
    for t in range(1, c.size + 1):
        lo = max(0, t - window)
        if c[lo:t].mean() >= target:
            return t
    return None


def _check_seq(name: str, value: Any, allowed: Sequence | None = None) -> tuple:
    if isinstance(value, (str, bytes)):
        raise ValueError(f'{name} must be a sequence')
    out = tuple(value)
    if not out or len(set(out)) != len(out):
        raise ValueError(f'{name} must be a non-empty sequence of distinct values')
    if allowed is not None and any(v not in allowed for v in out):
        raise ValueError(f'{name} entries must be in {tuple(allowed)}')
    return out


def run_lesion_assay(*, kinds: Sequence[str] = ('intact', 'random', 'hub', 'region', 'sensory_pathway',
                                                'memory_pathway', 'cross_region'),
                     fractions: Sequence[float] = (0.1, 0.2, 0.3), progressive: bool = True,
                     seeds: Sequence[int] = (0, 1, 2), neurons: int = 48, topology: str = 'multi_region',
                     steps: int = 200, recovery_steps: int = 100, heldout_steps: int | None = None,
                     task: str = 'xor', probe_task: str = 'delayed_recall', mechanisms: Sequence[str] = (),
                     heldout_seed_offset: int = 100) -> dict[str, Any]:
    """Run the lesion assay and return a digest-sealed receipt (see module docstring)."""
    kinds = _check_seq('kinds', kinds, LESION_KINDS)
    fractions = _check_seq('fractions', fractions)
    if any(isinstance(f, bool) or not isinstance(f, (int, float)) or not 0.0 < float(f) <= 1.0 for f in fractions):
        raise ValueError('fractions must be in (0, 1]')
    fractions = tuple(sorted(float(f) for f in fractions))
    seeds = _check_seq('seeds', seeds)
    if any(type(s) is not int or s < 0 or s > 2**31 - 200_000 for s in seeds):
        raise ValueError('seeds must be non-negative ints')
    for name, v, lo in (('neurons', neurons, 8), ('steps', steps, 2), ('recovery_steps', recovery_steps, 1)):
        if type(v) is not int or not lo <= v <= 100_000:
            raise ValueError(f'{name} must be an int in [{lo}, 100000]')
    if type(progressive) is not bool:
        raise ValueError('progressive must be a boolean')
    if topology not in TOPOLOGIES:
        raise ValueError(f'topology must be one of {TOPOLOGIES}')
    if task not in TASKS or probe_task not in TASKS:
        raise ValueError(f'task and probe_task must be in {TASKS}')
    mechanisms = tuple(m for m in MECHANISM_IDS if m in set(mechanisms))
    heldout_steps = max(8, steps // 2) if heldout_steps is None else int(heldout_steps)
    config = {'kinds': list(kinds), 'fractions': list(fractions), 'progressive': progressive, 'seeds': list(seeds),
              'neurons': neurons, 'topology': topology, 'steps': steps, 'recovery_steps': recovery_steps,
              'heldout_steps': heldout_steps, 'task': task, 'probe_task': probe_task,
              'mechanisms': list(mechanisms), 'heldout_seed_offset': heldout_seed_offset}
    started = time.perf_counter()
    window = max(1, min(10, recovery_steps))
    records: list[dict[str, Any]] = []
    originals_untouched = True
    for seed in seeds:
        spec = make_spec(mechanisms, seed=seed, neurons=neurons, topology=topology, steps=steps)
        engine = NeuroWeaveEngine(spec)
        heads = {task: ReadoutHead(engine.n, seed), probe_task: ReadoutHead(engine.n, seed + 1)} \
            if probe_task != task else {task: ReadoutHead(engine.n, seed)}
        stream = make_stream(seed, steps + recovery_steps)
        run_task_phase(engine, heads, stream, learn=True, stop=steps)
        before = _state_digest(engine)
        head_before = [h.w.copy() for h in heads.values()]
        intact_topo = topology_metrics(engine_topology(engine))
        held = make_stream(seed + heldout_seed_offset, heldout_steps)

        def evaluate(e: NeuroWeaveEngine, silence: np.ndarray | None) -> dict[str, Any]:
            hs = {t: h.copy() for t, h in heads.items()}
            return run_task_phase(e, hs, held, learn=False, silence=silence)

        def recover(e: NeuroWeaveEngine, silence: np.ndarray | None) -> list[float]:
            hs = {t: h.copy() for t, h in heads.items()}
            r = run_task_phase(e, hs, stream, learn=True, start=steps, silence=silence)
            return r['tasks'][task]['correct']

        intact_eval = evaluate(clone_engine(engine), None)
        intact_recovery = recover(clone_engine(engine), None)
        intact_acc = intact_eval['tasks'][task]['accuracy']
        intact_probe = intact_eval['tasks'][probe_task]['accuracy']
        chance = intact_eval['tasks'][task]['chance_accuracy']
        target = 0.9 * float(np.mean(intact_recovery))
        for ki, kind in enumerate(kinds):
            fr = (0.0,) if kind == 'intact' else fractions
            order_rng = np.random.default_rng(seed + _LESION_OFFSET + 31 * ki)
            unit, order = lesion_order(engine, kind, order_rng)
            for fi, frac in enumerate(fr):
                if not progressive and kind != 'intact' and fi > 0:
                    unit, order = lesion_order(engine, kind, np.random.default_rng(seed + _LESION_OFFSET + 31 * ki + fi))
                k = int(round(frac * order.size))
                lesioned = clone_engine(engine)
                info = apply_lesion(lesioned, unit, order[:k])
                sil = info['silence'] if info['silence'].size else None
                topo = topology_metrics(engine_topology(lesioned))
                ev = evaluate(clone_engine(lesioned), sil)
                rec = recover(lesioned, sil)
                acc = ev['tasks'][task]['accuracy']
                probe = ev['tasks'][probe_task]['accuracy']
                above = (intact_acc - chance) if intact_acc is not None else 0.0
                redundancy = None
                if above > 1e-9 and acc is not None:
                    redundancy = float(np.clip((acc - chance) / above, 0.0, 1.0))
                records.append({
                    'seed': seed, 'kind': kind, 'fraction': frac, 'unit': unit, 'pool_size': int(order.size),
                    'lesioned_elements': k, 'edges_removed': info['edges_removed'],
                    'neurons_silenced': info['neurons_silenced'],
                    'accuracy': acc, 'log_loss': ev['tasks'][task]['log_loss'], 'brier': ev['tasks'][task]['brier'],
                    'intact_accuracy': intact_acc, 'accuracy_drop': (intact_acc - acc) if acc is not None else None,
                    'prediction': ev['prediction'],
                    'memory_retention': (probe / intact_probe) if (probe is not None and intact_probe) else None,
                    'probe_accuracy': probe,
                    'connectivity': {'edges': topo['edges'], 'global_efficiency': topo['global_efficiency'],
                                     'reachable_pair_fraction': topo['reachable_pair_fraction'],
                                     'delta_global_efficiency': topo['global_efficiency'] - intact_topo['global_efficiency'],
                                     'delta_edges': topo['edges'] - intact_topo['edges']},
                    'functional_redundancy': redundancy,
                    'recovery_time': _rolling_recovery(rec, target, window),
                    'recovery_final_accuracy': float(np.mean(rec[-window:])) if rec else None,
                    'recovery_target': target,
                })
        after = _state_digest(engine)
        unchanged = after == before and all(np.array_equal(a, h.w) for a, h in zip(head_before, heads.values()))
        if not unchanged:
            raise AssertionError('lesion assay mutated the original engine')
        originals_untouched &= unchanged

    summary: list[dict[str, Any]] = []
    for kind in kinds:
        for frac in ((0.0,) if kind == 'intact' else fractions):
            rs = [r for r in records if r['kind'] == kind and r['fraction'] == frac]
            accs = [r['accuracy'] for r in rs]
            st = paired_stats([float('nan') if a is None else a for a in accs],
                              [float('nan') if r['intact_accuracy'] is None else r['intact_accuracy'] for r in rs])
            rt = [r['recovery_time'] for r in rs]
            summary.append({'kind': kind, 'fraction': frac, 'accuracy': _mean(accs),
                            'accuracy_vs_intact': st,
                            'memory_retention': _mean(r['memory_retention'] for r in rs),
                            'functional_redundancy': _mean(r['functional_redundancy'] for r in rs),
                            'delta_global_efficiency': _mean(r['connectivity']['delta_global_efficiency'] for r in rs),
                            'edges_removed': _mean(r['edges_removed'] for r in rs),
                            'recovered_fraction': float(np.mean([t is not None for t in rt])) if rt else None,
                            'recovery_time_mean': _mean(t for t in rt if t is not None)})

    monotone = True
    for seed in seeds:
        for kind in kinds:
            if kind == 'intact':
                continue
            rs = sorted((r for r in records if r['seed'] == seed and r['kind'] == kind), key=lambda r: r['fraction'])
            for a, b in zip(rs, rs[1:]):
                if b['edges_removed'] < a['edges_removed'] or b['neurons_silenced'] < a['neurons_silenced'] \
                        or b['connectivity']['global_efficiency'] > a['connectivity']['global_efficiency'] + 1e-12:
                    monotone = False
    # verdict: the strongest random lesion vs the intact clone (frozen held-out accuracy)
    comparisons: dict[str, Any] = {}
    basis_key = None
    for s in summary:
        if s['kind'] == 'intact':
            continue
        key = f"{s['kind']}_{s['fraction']:g}_vs_intact"
        comparisons[key] = {'primary': {'metric': 'heldout_accuracy', 'higher_is_better': True,
                                        **s['accuracy_vs_intact']},
                            **verdict_from_stats(s['accuracy_vs_intact'])}
    random_keys = [k for k in comparisons if k.startswith('random_')]
    if random_keys:
        basis_key = random_keys[-1]
    elif comparisons:
        basis_key = sorted(comparisons)[-1]
    verdict = comparisons[basis_key]['verdict'] if basis_key else 'no_detectable_difference'
    receipt = {
        'schema': LESION_SCHEMA, 'lab_schema': LAB_SCHEMA, 'config': config, 'records': records, 'summary': summary,
        'comparisons': comparisons, 'verdict': verdict,
        'verdict_basis': {'comparison': basis_key, 'metric': 'heldout_accuracy',
                          'meaning': "'worse' = the lesion detectably reduced held-out accuracy versus the intact clone"},
        'originals_untouched': bool(originals_untouched), 'progressive_monotone_damage': bool(monotone) if progressive else None,
        'evidence_tier': EVIDENCE_TIER, 'measured_wallclock_s_total': time.perf_counter() - started,
    }
    return _finalise(receipt)

