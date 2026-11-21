"""Workbench HTTP API for the Sonapse Frontier Pack (``/api/frontier/*``, opt-in, additive).

Follows the same dual-source contract as ``NeuroWeaveWorkbenchAPI``:

* the **live** observer hub on ``runtime.agent.frontier_hub`` (when ``SynapseConfig.frontier`` is configured
  and enabled) is read under the workbench runtime lock and is never mutated or stepped from this API;
* an isolated **lab** :class:`FrontierObserverHub` lives on the workbench, built lazily and replaced only by
  ``POST /api/frontier/configure``, so users can step observers and run experiments in the browser without
  touching the live agent.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import threading
import time
from typing import Any
import uuid

import numpy as np

from .harness import _jsonable
from .runtime import (
    FrontierObserverHub,
    discover_experiments,
    experiment_catalog,
    run_all_frontier_experiments,
    run_frontier_experiment,
)
from .spec import (
    EVIDENCE_TIER,
    GRAFT_OPTIMIZERS,
    MODULE_IDS,
    OBSERVER_MODULE_IDS,
    FrontierSpecV1,
    module_catalog,
)

__all__ = [
    'API_SCHEMA',
    'MAX_LAB_UNITS',
    'MAX_STEP_COUNT',
    'MAX_EXPERIMENT_SEEDS',
    'default_frontier_lab_spec',
    'frontier_experiment_row',
    'FrontierWorkbenchAPI',
]

API_SCHEMA = 'FrontierWorkbenchApiV1'
MAX_LAB_UNITS = 512
MAX_STEP_COUNT = 256
MAX_EXPERIMENT_SEEDS = 3
_LAB_STREAM_OFFSET = 71_023


def default_frontier_lab_spec(seed: int = 7) -> FrontierSpecV1:
    """Default workbench lab spec: all available read-only observers enabled at bounded sizes."""
    avail_obs: list[str] = []
    for mod_id in OBSERVER_MODULE_IDS:
        try:
            from .runtime import load_frontier_module
            load_frontier_module(mod_id)
            avail_obs.append(mod_id)
        except ModuleNotFoundError:
            continue
    return FrontierSpecV1(
        enabled=True,
        seed=seed,
        modules=tuple(avail_obs),
        max_observed_units=64,
        history=128,
        fc_window=64,
        sae_features=32,
        sae_k=4,
        sync_pairs=32,
    )


def _int(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{name} must be a boolean')
    return value


def _seeds(value: Any, default: tuple[int, ...], cap: int = MAX_EXPERIMENT_SEEDS) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        _int(value, 'seeds', 1, cap)
        return tuple(range(value))
    if not isinstance(value, list) or not 1 <= len(value) <= cap:
        raise ValueError(f'seeds must be a count in [1, {cap}] or a list of 1..{cap} integers')
    out = tuple(_int(s, 'seed', 0, 99) for s in value)
    if len(set(out)) != len(out):
        raise ValueError('seeds must be distinct')
    return out


def _check_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        for key in ('passed', 'ok', 'pass'):
            if isinstance(value.get(key), bool):
                return bool(value[key])
    return False


def frontier_experiment_row(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Compact table row summarising a sealed frontier lab receipt."""
    comparisons = receipt.get('comparisons') or {}
    first_cmp_name = next(iter(comparisons), None)
    first_cmp = comparisons.get(first_cmp_name) if first_cmp_name else None
    if not isinstance(first_cmp, Mapping):
        first_cmp = {}
    row: dict[str, Any] = {
        'experiment_id': receipt.get('experiment_id'),
        'kind': receipt.get('kind'),
        'verdict': receipt.get('verdict'),
        'verdict_basis': receipt.get('verdict_basis'),
        'primary_comparison': first_cmp_name,
        'mean_difference': first_cmp.get('mean'),
        'ci95': first_cmp.get('ci95'),
        'n': first_cmp.get('n'),
        'digest': receipt.get('digest'),
        'evidence_tier': receipt.get('evidence_tier'),
    }
    checks = receipt.get('checks')
    if first_cmp_name is None and isinstance(checks, Mapping) and checks:
        total = len(checks)
        passed = sum(1 for val in checks.values() if _check_passed(val))
        row['primary_comparison'] = f'{passed}/{total} checks passed'
        row['n'] = total
        row['checks_passed'] = passed
        row['checks_total'] = total
    return _jsonable(row)


class FrontierWorkbenchAPI:
    """Route handler for ``/api/frontier/*`` bound to a Sonapse ``Workbench``."""

    PREFIX = '/api/frontier/'

    def __init__(self, workbench: Any) -> None:
        self.wb = workbench
        self.lab_lock = threading.RLock()
        self.run_lock = threading.Lock()
        self._lab_hub: FrontierObserverHub | None = None
        self._lab_rng: np.random.Generator | None = None

    def _seed(self) -> int:
        seed = getattr(getattr(self.wb.runtime, 'config', None), 'seed', 7)
        return seed if type(seed) is int and 0 <= seed < 2**31 else 7

    def lab(self) -> FrontierObserverHub:
        with self.lab_lock:
            if self._lab_hub is None:
                self._set_lab(default_frontier_lab_spec(self._seed()), n_units=64)
            assert self._lab_hub is not None
            return self._lab_hub

    def _set_lab(self, spec: FrontierSpecV1, n_units: int = 64) -> None:
        hub = FrontierObserverHub(spec, n_units=n_units)
        with self.lab_lock:
            self._lab_hub = hub
            self._lab_rng = np.random.default_rng(int(spec.seed) + _LAB_STREAM_OFFSET)

    def _live_hub(self) -> FrontierObserverHub | None:
        agent = getattr(self.wb.runtime, 'agent', None)
        hub = getattr(agent, 'frontier_hub', None)
        return hub if hub is not None and getattr(hub, 'enabled', False) else None

    def catalog(self) -> dict[str, Any]:
        return _jsonable({
            'schema': API_SCHEMA,
            'modules': module_catalog(),
            'module_ids': list(MODULE_IDS),
            'observer_module_ids': list(OBSERVER_MODULE_IDS),
            'graft_optimizers': list(GRAFT_OPTIMIZERS),
            'experiments': experiment_catalog(include_unregistered=True),
            'limits': {
                'lab_units': MAX_LAB_UNITS,
                'step': MAX_STEP_COUNT,
                'experiment_seeds': MAX_EXPERIMENT_SEEDS,
            },
            'evidence_tier': EVIDENCE_TIER,
        })

    def status(self) -> dict[str, Any]:
        with self.wb.lock:
            cfg = getattr(self.wb.runtime, 'config', None)
            spec = getattr(cfg, 'frontier', None)
            live_hub = self._live_hub()
            live = {
                'configured': spec is not None,
                'enabled': live_hub is not None,
                'spec': spec.to_dict() if spec is not None else None,
            }
            if live_hub is not None:
                live['summary'] = live_hub.summary()
        with self.lab_lock:
            lab = None
            if self._lab_hub is not None:
                lab = {
                    'spec': self._lab_hub.spec.to_dict(),
                    'spec_digest': self._lab_hub.spec.spec_digest,
                    'n_units': self._lab_hub.n_units,
                    'summary': self._lab_hub.summary(),
                }
        with self.wb.job_lock:
            jobs = [
                {'id': j['id'], 'kind': j.get('kind'), 'status': j['status']}
                for j in self.wb.jobs.values()
                if str(j.get('kind', '')).startswith('frontier_')
            ]
        return _jsonable({
            'schema': API_SCHEMA,
            'live': live,
            'lab': lab,
            'lab_built': lab is not None,
            'observed_source': 'live' if live['enabled'] else 'lab',
            'run_active': self.run_lock.locked(),
            'jobs': jobs,
            'evidence_tier': EVIDENCE_TIER,
        })

    def observers(self, source: str = 'auto') -> dict[str, Any]:
        if source not in ('auto', 'live', 'lab'):
            raise ValueError("source must be 'auto', 'live' or 'lab'")
        if source in ('auto', 'live'):
            with self.wb.lock:
                live_hub = self._live_hub()
                if live_hub is not None:
                    return _jsonable({'source': 'live', 'summary': live_hub.summary()})
            if source == 'live':
                raise ValueError('the live runtime has no enabled frontier observer hub (start with --frontier)')
        with self.lab_lock:
            hub = self.lab()
            return _jsonable({'source': 'lab', 'summary': hub.summary()})

    def configure(self, data: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(data)
        n_units = _int(payload.pop('n_units', 64), 'n_units', 8, MAX_LAB_UNITS)
        payload.setdefault('enabled', True)
        spec = FrontierSpecV1.from_dict(payload)
        with self.lab_lock:
            self._set_lab(spec, n_units=n_units)
            return self.observers('lab')

    def step(self, data: Mapping[str, Any]) -> dict[str, Any]:
        extra = set(data) - {'steps', 'reset'}
        if extra:
            raise ValueError(f'unknown step fields: {sorted(extra)}')
        steps = _int(data.get('steps', 8), 'steps', 1, MAX_STEP_COUNT)
        reset = _bool(data.get('reset', False), 'reset')
        with self.lab_lock:
            hub = self.lab()
            if reset:
                hub.reset()
                self._lab_rng = np.random.default_rng(int(hub.spec.seed) + _LAB_STREAM_OFFSET)
            assert self._lab_rng is not None
            for _ in range(steps):
                vec = np.tanh(self._lab_rng.standard_normal(hub.n_units))
                hub.observe(vec)
            return _jsonable({'source': 'lab', 'stepped': steps, 'summary': hub.summary()})

    def _run(self, kind: str, fn: Any, background: bool, params: dict[str, Any]) -> dict[str, Any]:
        if not background:
            if not self.run_lock.acquire(blocking=False):
                raise ValueError('another frontier run is active; wait for it to finish')
            try:
                return _jsonable(fn())
            finally:
                self.run_lock.release()
        wb = self.wb
        with wb.job_lock:
            if any(j['status'] in ('queued', 'running') for j in wb.jobs.values()):
                raise ValueError('a background job is already active')
            if self.run_lock.locked():
                raise ValueError('another frontier run is active; wait for it to finish')
            ident = uuid.uuid4().hex
            event = threading.Event()
            wb.cancel[ident] = event
            job = {
                'id': ident,
                'kind': f'frontier_{kind}',
                'status': 'queued',
                'created': time.time(),
                'config': _jsonable(params),
                'progress': {'phase': 'queued'},
                'result': None,
                'error': None,
            }
            wb.jobs[ident] = job
            wb.persist(job)

        def worker() -> None:
            try:
                with self.run_lock:
                    if event.is_set():
                        with wb.job_lock:
                            job.update(status='cancelled')
                        return
                    with wb.job_lock:
                        job.update(status='running', progress={'phase': kind})
                        wb.persist(job)
                    result = _jsonable(fn())
                with wb.job_lock:
                    job.update(
                        status='cancelled' if event.is_set() else 'completed',
                        result=result,
                        progress={'phase': 'done'},
                    )
            except Exception as exc:
                with wb.job_lock:
                    job.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                with wb.job_lock:
                    job['finished'] = time.time()
                    wb.persist(job)

        threading.Thread(target=worker, daemon=True, name=f'frontier-{kind}-{ident[:8]}').start()
        with wb.job_lock:
            return deepcopy(job)

    def experiment(self, data: Mapping[str, Any]) -> dict[str, Any]:
        extra = set(data) - {'id', 'quick', 'seeds', 'background'}
        if extra:
            raise ValueError(f'unknown experiment fields: {sorted(extra)}')
        eid = data.get('id', 'synchrony')
        discovered = discover_experiments(include_unregistered=True)
        if eid != 'all' and eid not in discovered:
            raise ValueError(f'id must be one of {sorted(discovered)} or "all"')
        quick = _bool(data.get('quick', True), 'quick')
        background = _bool(data.get('background', False), 'background')
        if eid == 'all' and not background:
            raise ValueError("id 'all' runs only as a background job (background: true)")
        seeds = _seeds(data.get('seeds'), (0, 1) if quick else (0, 1, 2), MAX_EXPERIMENT_SEEDS)

        def run() -> dict[str, Any]:
            if eid == 'all':
                receipts = run_all_frontier_experiments(
                    quick=quick, seeds=seeds, primary_only=True, include_unregistered=True)
            else:
                receipts = [run_frontier_experiment(
                    str(eid), quick=quick, seeds=seeds, include_unregistered=True)]
            return {
                'experiment': eid,
                'quick': quick,
                'seeds': list(seeds),
                'rows': [frontier_experiment_row(r) for r in receipts],
                'receipts': receipts,
                'evidence_tier': EVIDENCE_TIER,
            }

        return self._run('experiment', run, background, {'id': eid, 'quick': quick, 'seeds': list(seeds)})

    def get(self, path: str) -> dict[str, Any]:
        if path == '/api/frontier/catalog':
            return self.catalog()
        if path == '/api/frontier/status':
            return self.status()
        if path == '/api/frontier/observers':
            return self.observers('auto')
        if path == '/api/frontier/observers/lab':
            return self.observers('lab')
        raise KeyError('unknown frontier endpoint')

    def post(self, path: str, data: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise ValueError('JSON body must be an object')
        handlers = {
            '/api/frontier/configure': self.configure,
            '/api/frontier/step': self.step,
            '/api/frontier/experiment': self.experiment,
        }
        if path not in handlers:
            raise KeyError('unknown frontier endpoint')
        return handlers[path](data)
