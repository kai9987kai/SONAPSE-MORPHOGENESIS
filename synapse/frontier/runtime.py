"""Live read-only observer hub and unified experiment registry for the Sonapse frontier pack.

Contract rules enforced here (see ``docs/frontier/CONTRACT.md``):

1. **Additive and default-off.** When ``spec.enabled`` is ``False`` (or ``spec`` has no observer modules),
   :meth:`FrontierObserverHub.observe` is a no-op and returns ``False``.
2. **Read-only live observers.** :meth:`FrontierObserverHub.observe` copies a deterministic unit subsample
   out of the brain activity vector (NumPy array or PyTorch tensor) before passing it to each observer,
   and never mutates the caller's array/tensor, the brain, the graft bus, the actor or global RNG state.
3. **Validate-before-mutate persistence.** :meth:`FrontierObserverHub.import_state` validates the incoming
   checkpoint on a fresh detached candidate hub before swapping state into ``self``.
4. **Verified lab receipts.** :func:`run_frontier_experiment` and :func:`run_all_frontier_experiments`
   verify every receipt with :func:`synapse.frontier.harness.verify_receipt` before returning it.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
import importlib
from typing import Any

import numpy as np

from .harness import _jsonable, receipt_digest, verify_receipt
from .spec import (EVIDENCE_TIER, LAB_SCHEMA, MODULE_IDS, MODULES, OBSERVER_MODULE_IDS,
                   FrontierSpecV1)

__all__ = [
    'OBSERVER_SUMMARY_SCHEMA', 'OBSERVER_CHECKPOINT_SCHEMA', 'LABORATORY_SCHEMA',
    'MODULE_FILES', 'OBSERVER_CLASSES', 'FrontierObserverHub', 'load_frontier_module',
    'make_observer', 'discover_experiments', 'experiment_catalog',
    'run_frontier_experiment', 'run_all_frontier_experiments', 'run_frontier_laboratory',
]

OBSERVER_SUMMARY_SCHEMA = 'FrontierObserverSummaryV1'
OBSERVER_CHECKPOINT_SCHEMA = 'FrontierObserverCheckpointV1'
LABORATORY_SCHEMA = 'FrontierLaboratoryReceiptV1'
_HUB_SUBSAMPLE_OFFSET = 50_021

MODULE_FILES: dict[str, str] = {
    'synchrony': 'synchrony',
    'continuum_memory': 'continuum_memory',
    'btsp_memory': 'btsp',
    'kuramoto_binding': 'kuramoto',
    'recursive_refiner': 'recursive_refiner',
    'dendritic_layer': 'dendritic',
    'muon_optimizer': 'muon',
    'rigl_sparse': 'rigl',
    'sparse_dictionary': 'sparse_dictionary',
    'anytime_inference': 'anytime',
    'experiment_design': 'experiment_design',
    'quality_diversity': 'quality_diversity',
    'morphogenesis': 'morphogenesis',
    'connectomics': 'connectomics',
    'causal_emergence': 'causal_emergence',
    'hh_cable': 'hh_cable',
    'nir_export': 'nir_export',
    'joint_failure': 'joint_failure',
}

OBSERVER_CLASSES: dict[str, tuple[str, str]] = {
    'synchrony': ('synchrony', 'SynchronyObserver'),
    'kuramoto_binding': ('kuramoto', 'KuramotoObserver'),
    'sparse_dictionary': ('sparse_dictionary', 'SparseDictionaryObserver'),
    'connectomics': ('connectomics', 'ConnectivityObserver'),
    'causal_emergence': ('causal_emergence', 'CausalEmergenceObserver'),
}


def load_frontier_module(module_id: str) -> Any:
    """Import ``synapse.frontier.<file>`` for a module id."""
    if module_id not in MODULE_FILES:
        raise KeyError(f'unknown frontier module: {module_id!r}')
    file_stem = MODULE_FILES[module_id]
    return importlib.import_module(f'{__package__}.{file_stem}')


def make_observer(module_id: str, n_units: int, spec: FrontierSpecV1) -> Any:
    """Instantiate a single read-only observer for ``module_id`` sized from ``spec``."""
    if not isinstance(spec, FrontierSpecV1):
        raise ValueError('spec must be a FrontierSpecV1')
    if module_id not in OBSERVER_CLASSES or module_id not in OBSERVER_MODULE_IDS:
        raise ValueError(f'{module_id!r} is not a registered observer module')
    file_stem, cls_name = OBSERVER_CLASSES[module_id]
    mod = importlib.import_module(f'{__package__}.{file_stem}')
    cls = getattr(mod, cls_name)
    if hasattr(cls, 'from_spec'):
        return cls.from_spec(n_units, spec)
    return cls(n_units, seed=int(spec.seed), history=int(spec.history))


def _extract_1d_activity(x: Any, expected_len: int) -> np.ndarray:
    """Return a validated 1-D float64 NumPy view/copy of ``x`` without mutating the input."""
    if hasattr(x, 'detach') and hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        arr = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
    else:
        arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(f'activity must be a 1-D vector of length {expected_len}')
    if not np.all(np.isfinite(arr)):
        raise ValueError('activity must be finite')
    return arr


class FrontierObserverHub:
    """Manages the active read-only frontier observers attached to a Sonapse brain."""

    def __init__(self, spec: FrontierSpecV1, n_units: int) -> None:
        if not isinstance(spec, FrontierSpecV1):
            raise ValueError('spec must be a FrontierSpecV1')
        if type(n_units) is not int or not 2 <= n_units <= 1_000_000:
            raise ValueError('n_units must be an int in [2, 1000000]')
        self.spec = spec
        self.n_units = n_units
        self.observed_unit_count = min(n_units, int(spec.max_observed_units))
        if self.observed_unit_count == n_units:
            indices = np.arange(n_units, dtype=np.int64)
        else:
            rng = np.random.default_rng(int(spec.seed) + _HUB_SUBSAMPLE_OFFSET)
            indices = np.sort(rng.choice(n_units, size=self.observed_unit_count, replace=False)).astype(np.int64)
        indices.setflags(write=False)
        self._unit_indices = indices
        self.observers: dict[str, Any] = {}
        if spec.enabled:
            for mod_id in spec.modules:
                self.observers[mod_id] = make_observer(mod_id, self.observed_unit_count, spec)
        self.agent_steps = 0
        self.observed_steps = 0

    @property
    def enabled(self) -> bool:
        return bool(self.spec.enabled)

    @property
    def unit_indices(self) -> np.ndarray:
        return self._unit_indices.copy()

    def reset(self) -> None:
        self.agent_steps = 0
        self.observed_steps = 0
        for obs in self.observers.values():
            obs.reset()

    def observe(self, activity: Any) -> bool:
        """Read a copy of ``activity`` into active observers on every ``observe_every``-th step."""
        arr = _extract_1d_activity(activity, self.n_units)
        if not self.spec.enabled or not self.observers:
            return False
        step_idx = self.agent_steps
        self.agent_steps += 1
        if step_idx % int(self.spec.observe_every) != 0:
            return False
        sample = np.array(arr[self._unit_indices], dtype=np.float64, copy=True)
        for obs in self.observers.values():
            obs.observe(sample.copy())
        self.observed_steps += 1
        return True

    def summary(self) -> dict[str, Any]:
        return _jsonable({
            'schema': OBSERVER_SUMMARY_SCHEMA,
            'schema_version': 1,
            'enabled': bool(self.spec.enabled),
            'spec_digest': self.spec.spec_digest,
            'n_units': self.n_units,
            'observed_unit_count': self.observed_unit_count,
            'observe_every': int(self.spec.observe_every),
            'agent_steps': self.agent_steps,
            'observed_steps': self.observed_steps,
            'modules': list(self.spec.modules),
            'observers': {mod_id: obs.summary() for mod_id, obs in self.observers.items()},
            'evidence_tier': EVIDENCE_TIER,
        })

    def telemetry(self, *, heavy_every: int = 32) -> dict[str, Any]:
        """Deterministic low-overhead step telemetry for ``agent.last_telemetry['frontier']``.

        Sub-millisecond observers are summarized every step; null-calibrated observers
        (``causal_emergence``) run full summaries below ``heavy_every`` and on multiples
        of ``heavy_every`` so high-step loops stay fast while preserving exact replay
        determinism across checkpoints. Call :meth:`summary` for an unconditional full view.
        """
        obs_out: dict[str, Any] = {}
        for mod_id, obs in self.observers.items():
            if mod_id == 'causal_emergence' and self.observed_steps >= heavy_every and (self.observed_steps % heavy_every) != 0:
                obs_out[mod_id] = {
                    'module': mod_id,
                    'observed_steps': self.observed_steps,
                    'deferred': True,
                    'next_full_summary_at': ((self.observed_steps // heavy_every) + 1) * heavy_every,
                }
            else:
                obs_out[mod_id] = obs.summary()
        return _jsonable({
            'schema': OBSERVER_SUMMARY_SCHEMA,
            'schema_version': 1,
            'enabled': bool(self.spec.enabled),
            'spec_digest': self.spec.spec_digest,
            'n_units': self.n_units,
            'observed_unit_count': self.observed_unit_count,
            'observe_every': int(self.spec.observe_every),
            'agent_steps': self.agent_steps,
            'observed_steps': self.observed_steps,
            'modules': list(self.spec.modules),
            'observers': obs_out,
            'evidence_tier': EVIDENCE_TIER,
        })

    def export_state(self) -> dict[str, Any]:
        return deepcopy(_jsonable({
            'schema': OBSERVER_CHECKPOINT_SCHEMA,
            'schema_version': 1,
            'spec': self.spec.to_dict(),
            'spec_digest': self.spec.spec_digest,
            'n_units': self.n_units,
            'agent_steps': self.agent_steps,
            'observed_steps': self.observed_steps,
            'unit_indices': self._unit_indices.tolist(),
            'observers': {mod_id: obs.export_state() for mod_id, obs in self.observers.items()},
        }))

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError('observer hub state must be a mapping')
        if state.get('schema') != OBSERVER_CHECKPOINT_SCHEMA or state.get('schema_version') != 1:
            raise ValueError('unsupported observer hub checkpoint schema')
        if state.get('n_units') != self.n_units:
            raise ValueError('observer hub n_units mismatch')
        if state.get('spec_digest') != self.spec.spec_digest or state.get('spec') != self.spec.to_dict():
            raise ValueError('observer hub spec_digest mismatch')
        agent_steps = state.get('agent_steps')
        observed_steps = state.get('observed_steps')
        if type(agent_steps) is not int or agent_steps < 0:
            raise ValueError('agent_steps must be a non-negative int')
        if type(observed_steps) is not int or not 0 <= observed_steps <= agent_steps:
            raise ValueError('observed_steps must be an int in [0, agent_steps]')
        expected_observed = (agent_steps + self.spec.observe_every - 1) // self.spec.observe_every \
            if (self.spec.enabled and self.observers and agent_steps > 0) else 0
        if observed_steps != expected_observed:
            raise ValueError('observed_steps is inconsistent with agent_steps and observe_every')
        if state.get('unit_indices') != self._unit_indices.tolist():
            raise ValueError('observer hub unit_indices mismatch')
        saved_obs = state.get('observers')
        if not isinstance(saved_obs, Mapping) or set(saved_obs) != set(self.observers):
            raise ValueError('observer hub modules mismatch')
        candidate = FrontierObserverHub(self.spec, self.n_units)
        for mod_id, obs in candidate.observers.items():
            obs.import_state(deepcopy(saved_obs[mod_id]))
            obs_summary = obs.summary()
            obs_count = obs_summary.get('observed_steps', obs_summary.get('observed'))
            if type(obs_count) is int and obs_count != observed_steps:
                raise ValueError(f'observer {mod_id!r} step count does not match hub observed_steps')
        candidate.agent_steps = agent_steps
        candidate.observed_steps = observed_steps
        self.observers = candidate.observers
        self.agent_steps = candidate.agent_steps
        self.observed_steps = candidate.observed_steps


# ---------------------------------------------------------------------- experiment registry & runner

def discover_experiments(*, include_unregistered: bool = False) -> dict[str, tuple[str, Callable[..., dict[str, Any]]]]:
    """Return ``{experiment_id: (module_id, runner)}`` across available frontier modules."""
    out: dict[str, tuple[str, Callable[..., dict[str, Any]]]] = {}
    ids = list(MODULE_IDS)
    if include_unregistered:
        for mid in MODULE_FILES:
            if mid not in ids:
                ids.append(mid)
    for mod_id in ids:
        try:
            mod = load_frontier_module(mod_id)
        except ModuleNotFoundError:
            continue
        exps = getattr(mod, 'EXPERIMENTS', None)
        if not isinstance(exps, Mapping):
            continue
        for exp_id, fn in exps.items():
            if callable(fn):
                out[str(exp_id)] = (mod_id, fn)
    return out


def experiment_catalog(*, include_unregistered: bool = False) -> list[dict[str, Any]]:
    """Metadata list for every available frontier experiment."""
    by_id = {m.id: m for m in MODULES}
    discovered = discover_experiments(include_unregistered=include_unregistered)
    catalog: list[dict[str, Any]] = []
    for exp_id, (mod_id, _) in discovered.items():
        info = by_id.get(mod_id)
        catalog.append({
            'experiment_id': exp_id,
            'module_id': mod_id,
            'is_primary': exp_id == mod_id,
            'title': info.title if info else mod_id,
            'kind': info.kind if info else 'addon',
            'origin': info.origin if info else 'portfolio',
            'source': info.source if info else '',
            'description': info.description if info else '',
            'observer': bool(info.observer) if info else False,
        })
    return catalog


def run_frontier_experiment(experiment_id: str, *, quick: bool = False,
                            seeds: Iterable[int] = (0, 1, 2),
                            include_unregistered: bool = True,
                            **overrides: Any) -> dict[str, Any]:
    """Run one frontier experiment by id and verify its sealed receipt."""
    discovered = discover_experiments(include_unregistered=include_unregistered)
    if experiment_id not in discovered:
        raise KeyError(f'unknown frontier experiment: {experiment_id!r}')
    _, runner = discovered[experiment_id]
    receipt = runner(quick=quick, seeds=tuple(seeds), **overrides)
    if not isinstance(receipt, dict) or receipt.get('schema') != LAB_SCHEMA or not verify_receipt(receipt):
        raise RuntimeError(f'experiment {experiment_id!r} returned an unverified receipt')
    return receipt


def run_all_frontier_experiments(*, quick: bool = True, seeds: Iterable[int] = (0, 1, 2),
                                 primary_only: bool = True,
                                 module_ids: Iterable[str] | None = None,
                                 include_unregistered: bool = False) -> list[dict[str, Any]]:
    """Run all (or selected) frontier experiments and return their verified receipts."""
    discovered = discover_experiments(include_unregistered=include_unregistered)
    allowed = None if module_ids is None else set(module_ids)
    seeds = tuple(seeds)
    receipts: list[dict[str, Any]] = []
    for exp_id, (mod_id, _) in discovered.items():
        if allowed is not None and mod_id not in allowed and exp_id not in allowed:
            continue
        if primary_only and exp_id != mod_id and (allowed is None or exp_id not in allowed):
            continue
        receipts.append(run_frontier_experiment(
            exp_id, quick=quick, seeds=seeds, include_unregistered=include_unregistered))
    return receipts


def run_frontier_laboratory(spec: FrontierSpecV1 | None = None, *,
                            hub: FrontierObserverHub | None = None,
                            experiments: bool = False,
                            quick: bool = True,
                            seeds: Iterable[int] = (0, 1),
                            module_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Build a sealed frontier-pack summary receipt for CLI, server or Max Mode."""
    active_spec = FrontierSpecV1() if spec is None else spec
    if not isinstance(active_spec, FrontierSpecV1):
        raise ValueError('spec must be a FrontierSpecV1 or None')
    exp_summary: dict[str, Any]
    if experiments:
        runs = run_all_frontier_experiments(quick=quick, seeds=seeds, primary_only=True, module_ids=module_ids)
        counts: dict[str, int] = {}
        for r in runs:
            v = str(r.get('verdict', 'unknown'))
            counts[v] = counts.get(v, 0) + 1
        exp_summary = {
            'executed': len(runs),
            'quick': bool(quick),
            'verdict_counts': counts,
            'experiments': [
                {
                    'experiment_id': r['experiment_id'],
                    'kind': r['kind'],
                    'verdict': r['verdict'],
                    'verdict_basis': r.get('verdict_basis', ''),
                    'digest': r['digest'],
                }
                for r in runs
            ],
        }
    else:
        exp_summary = {'skipped': 'experiments=False'}
    payload = _jsonable({
        'schema': LABORATORY_SCHEMA,
        'schema_version': 1,
        'section': '07 · Frontier Pack Laboratory',
        'evidence_tier': EVIDENCE_TIER,
        'spec': active_spec.to_dict(),
        'spec_digest': active_spec.spec_digest,
        'catalog_count': len(MODULES),
        'available_experiments': [c['experiment_id'] for c in experiment_catalog()],
        'live_observers': hub.summary() if hub is not None else {'skipped': 'no live observer hub attached'},
        'experiments': exp_summary,
    })
    payload['digest'] = receipt_digest(payload)
    return payload
