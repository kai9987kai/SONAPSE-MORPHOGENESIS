"""Shared experiment harness for frontier modules.

Every frontier experiment reports the same things so results are comparable
and hard to over-read:

* a **baseline** arm (the unmodified system),
* a **candidate** arm (the module under test),
* where the candidate adds capacity, a **capacity-matched control** arm,
* paired differences over seeds with a 95 % t interval and a verdict that is
  ``improved`` only when the interval excludes zero,
* frozen held-out evaluation, parameter/state budgets, estimated cost kept
  apart from measured wall-clock, and a SHA-256 digest that ignores wall-clock.

The statistics and receipt helpers are the NEUROWEAVE lab's, reused unchanged.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..neuroweave.engine import NeuroWeaveEngine
from ..neuroweave.experiments import (ReadoutHead, _HEAD_OFFSET, _budget, _estimated_cost, _finalise, _jsonable, _mean,
                                      clone_engine, make_spec, paired_stats, receipt_digest, run_task_phase,
                                      verdict_from_stats, verify_receipt, VERDICTS)
from ..neuroweave.tasks import make_stream
from .extension import EngineExtension, RandomTraceControl
from .spec import EVIDENCE_TIER, LAB_SCHEMA

__all__ = ['ExtendedHead', 'clone_with_extensions', 'compare', 'base_receipt', 'seal', 'verify_receipt',
           'receipt_digest', 'run_extension_experiment', 'headline_verdict', 'VERDICTS', 'QUICK_ARGS']

TASKS = ('delayed_recall', 'xor', 'rhythm', 'context_switch')
QUICK_ARGS: dict[str, Any] = {'seeds': (0, 1), 'heldout_seeds': (100,), 'steps': 24, 'neurons': 24,
                              'heldout_steps': 12}
ExtensionFactory = Callable[[int], 'EngineExtension | Sequence[EngineExtension]']


class ExtendedHead(ReadoutHead):
    """The lab's logistic readout plus the attached extensions' features.

    The shared features keep the baseline head's initial weights (same generator
    stream), so the only difference from the baseline readout is the extra block."""

    def __init__(self, n0: int, seed: int, extra_dim: int = 0, lr: float = 0.05) -> None:
        super().__init__(n0, seed, lr)
        self.extra_dim = int(extra_dim)
        if self.extra_dim:
            base = self.w
            rng = np.random.default_rng(int(seed) + _HEAD_OFFSET)   # ReadoutHead's stream: same first draws
            self.w = (rng.random(self.dim + self.extra_dim) - 0.5) * 0.02
            assert np.array_equal(self.w[:self.dim], base)
            self.dim += self.extra_dim

    def features(self, engine: NeuroWeaveEngine, out: np.ndarray) -> np.ndarray:
        x = super().features(engine, out)
        if not self.extra_dim:
            return x
        parts = [np.asarray(e.features(), dtype=np.float64).ravel() for e in engine.extensions]
        extra = np.concatenate(parts) if parts else np.zeros(0)
        if extra.size != self.extra_dim or not np.all(np.isfinite(extra)):
            raise ValueError('extension features do not match the head (size or non-finite values)')
        return np.concatenate([x, np.clip(extra, -10.0, 10.0)])


def clone_with_extensions(engine: NeuroWeaveEngine) -> NeuroWeaveEngine:
    """Independent copy of an engine and deep copies of its attached extensions."""
    other = clone_engine(engine)
    for ext in engine.extensions:
        other.attach_extension(copy.deepcopy(ext), bind=False)
    return other


def compare(candidate: Sequence[float], reference: Sequence[float], *, higher_is_better: bool = True) -> dict[str, Any]:
    """Paired statistics plus the CI-only verdict used across the lab."""
    stats = paired_stats(candidate, reference)
    return {**stats, **verdict_from_stats(stats, higher_is_better=higher_is_better),
            'higher_is_better': bool(higher_is_better)}


def headline_verdict(vs_baseline: Mapping[str, Any], vs_control: Mapping[str, Any] | None) -> dict[str, str]:
    """``improved`` needs a win over the baseline *and* (when present) over the capacity-matched control."""
    if vs_baseline['verdict'] == 'worse':
        return {'verdict': 'worse', 'basis': 'candidate_vs_baseline'}
    if vs_baseline['verdict'] == 'improved' and (vs_control is None or vs_control['verdict'] == 'improved'):
        return {'verdict': 'improved', 'basis': 'candidate_vs_baseline' if vs_control is None
                else 'candidate_vs_baseline_and_control'}
    if vs_baseline['verdict'] == 'improved':
        return {'verdict': 'no_detectable_difference', 'basis': 'not_better_than_capacity_matched_control'}
    return {'verdict': 'no_detectable_difference', 'basis': 'candidate_vs_baseline'}


def base_receipt(experiment_id: str, kind: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {'schema': LAB_SCHEMA, 'schema_version': 1, 'experiment_id': str(experiment_id), 'kind': str(kind),
            'config': _jsonable(dict(config)), 'evidence_tier': EVIDENCE_TIER}


def seal(receipt: dict[str, Any]) -> dict[str, Any]:
    """JSON-normalise the receipt and add its wall-clock-independent SHA-256 digest."""
    if receipt.get('schema') != LAB_SCHEMA:
        raise ValueError('not a frontier lab receipt')
    return _finalise(receipt)


def _as_list(value: Any) -> list[EngineExtension]:
    if value is None:
        return []
    return [value] if isinstance(value, EngineExtension) or not isinstance(value, (list, tuple)) else list(value)


def _arm(factory: ExtensionFactory | None, seed: int, *, task: str, steps: int, heldout_seeds: Sequence[int],
         heldout_steps: int, neurons: int, topology: str, mechanisms: Sequence[str]) -> dict[str, Any]:
    spec = make_spec(mechanisms, seed=seed, neurons=neurons, topology=topology, steps=steps)
    engine = NeuroWeaveEngine(spec)
    exts = _as_list(None if factory is None else factory(seed))
    for ext in exts:
        engine.attach_extension(ext)
    extra = int(sum(int(e.feature_dim) for e in exts))
    head = ExtendedHead(engine.n, seed, extra)
    stream = make_stream(seed, steps)
    t0 = time.perf_counter()
    train = run_task_phase(engine, {task: head}, stream, learn=True)
    train_wall = time.perf_counter() - t0
    held = []
    t1 = time.perf_counter()
    for hs in heldout_seeds:
        held.append(run_task_phase(clone_with_extensions(engine), {task: head.copy()},
                                   make_stream(hs, heldout_steps), learn=False))
    held_wall = time.perf_counter() - t1
    tm = train['tasks'][task]
    budget = _budget(engine, head)
    budget['extension_parameters'] = int(sum(int(e.parameter_count()) for e in exts))
    budget['extension_state_numbers'] = int(sum(int(e.state_numbers()) for e in exts))
    budget['extension_features'] = extra
    return {
        'seed': int(seed), 'extensions': [e.id for e in exts], 'train_stream_digest': stream.digest,
        'train': {k: tm[k] for k in ('accuracy', 'log_loss', 'brier', 'chance_accuracy')},
        'heldout': {k: _mean(h['tasks'][task][k] for h in held)
                    for k in ('accuracy', 'log_loss', 'brier', 'chance_accuracy')},
        'telemetry': [_jsonable(e.telemetry()) for e in exts],
        'budget': budget, 'estimated_cost': _estimated_cost(engine, steps, head.dim),
        'measured_wallclock_s': {'train': train_wall, 'heldout': held_wall},
    }


def _summary(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    return {'train_accuracy': _mean(a['train']['accuracy'] for a in per_seed),
            'heldout_accuracy': _mean(a['heldout']['accuracy'] for a in per_seed),
            'heldout_log_loss': _mean(a['heldout']['log_loss'] for a in per_seed),
            'heldout_chance_accuracy': _mean(a['heldout']['chance_accuracy'] for a in per_seed)}


def run_extension_experiment(experiment_id: str, candidate: ExtensionFactory, *, task: str = 'xor',
                             seeds: Iterable[int] = (0, 1, 2), heldout_seeds: Iterable[int] = (100, 101),
                             steps: int = 300, heldout_steps: int | None = None, neurons: int = 48,
                             topology: str = 'clustered', mechanisms: Iterable[str] = (),
                             control: ExtensionFactory | str | None = 'random_trace',
                             metric: str = 'accuracy', notes: str = '') -> dict[str, Any]:
    """Baseline vs candidate vs capacity-matched control on a NEUROWEAVE temporal task.

    ``candidate(seed)`` returns the extension(s) to attach for that seed. ``control`` is a factory, ``None``
    (no control arm) or ``'random_trace'`` (random leaky traces with the candidate's feature count; skipped
    when the candidate adds no readout features). All arms share seeds, streams, the NEUROWEAVE mechanism
    set and the readout protocol; held-out streams are evaluated frozen."""
    seeds, heldout_seeds = tuple(seeds), tuple(heldout_seeds)
    for group in (seeds, heldout_seeds):
        if not group or any(isinstance(s, bool) or not isinstance(s, (int, np.integer))
                            or not 0 <= s < 2**31 for s in group):
            raise ValueError('seeds and heldout_seeds must contain non-negative 31-bit integers')
        if len(set(group)) != len(group):
            raise ValueError('seeds and heldout_seeds must each be unique')
    seeds, heldout_seeds = tuple(map(int, seeds)), tuple(map(int, heldout_seeds))
    if set(seeds) & set(heldout_seeds):
        raise ValueError('heldout_seeds must be disjoint from seeds')
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}')
    if metric not in ('accuracy', 'log_loss', 'brier'):
        raise ValueError("metric must be 'accuracy', 'log_loss' or 'brier'")
    if type(steps) is not int or not 4 <= steps <= 100_000:
        raise ValueError('steps must be an int in [4, 100000]')
    heldout_steps = max(4, steps // 2) if heldout_steps is None else heldout_steps
    if type(heldout_steps) is not int or not 1 <= heldout_steps <= 1_000_000:
        raise ValueError('heldout_steps must be an int in [1, 1000000]')
    mechanisms = tuple(mechanisms)
    kw = dict(task=task, steps=steps, heldout_seeds=heldout_seeds, heldout_steps=heldout_steps, neurons=int(neurons),
              topology=topology, mechanisms=mechanisms)
    probe = _as_list(candidate(seeds[0]))
    feature_dim = int(sum(int(e.feature_dim) for e in probe))
    control_factory: ExtensionFactory | None
    if control == 'random_trace':
        control_factory = (lambda s: RandomTraceControl(feature_dim, seed=s)) if feature_dim > 0 else None
    elif control is None or callable(control):
        control_factory = control
    else:
        raise ValueError("control must be a factory, None or 'random_trace'")

    arms: dict[str, list[dict[str, Any]]] = {'baseline': [], 'candidate': []}
    if control_factory is not None:
        arms['control'] = []
    for s in seeds:
        arms['baseline'].append(_arm(None, s, **kw))
        arms['candidate'].append(_arm(candidate, s, **kw))
        if control_factory is not None:
            arms['control'].append(_arm(control_factory, s, **kw))

    higher = metric == 'accuracy'
    vals = {name: [a['heldout'][metric] for a in per_seed] for name, per_seed in arms.items()}
    vs_baseline = compare(vals['candidate'], vals['baseline'], higher_is_better=higher)
    vs_control = compare(vals['candidate'], vals['control'], higher_is_better=higher) if 'control' in arms else None
    head = headline_verdict(vs_baseline, vs_control)
    receipt = base_receipt(experiment_id, 'extension_task', {
        'task': task, 'seeds': list(seeds), 'heldout_seeds': list(heldout_seeds), 'steps': steps,
        'heldout_steps': heldout_steps, 'neurons': int(neurons), 'topology': topology,
        'mechanisms': list(mechanisms), 'metric': f'heldout_{metric}', 'candidate_feature_dim': feature_dim,
        'control': None if control_factory is None else ('random_trace' if control == 'random_trace' else 'custom')})
    receipt['arms'] = {name: {'per_seed': per_seed, 'summary': _summary(per_seed)} for name, per_seed in arms.items()}
    receipt['comparisons'] = {'candidate_vs_baseline': vs_baseline}
    if vs_control is not None:
        receipt['comparisons']['candidate_vs_control'] = vs_control
    receipt['verdict'] = head['verdict']
    receipt['verdict_basis'] = head['basis']
    receipt['notes'] = str(notes)
    return seal(receipt)
