"""Tests for the Frontier Pack runtime hub, experiment discovery, CLI helpers, and Workbench API."""
from __future__ import annotations

from copy import deepcopy
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from synapse.frontier.cli_commands import (
    cli_frontier_spec,
    parse_observer_modules,
    parse_seeds,
    run_frontier_command,
)
from synapse.frontier.harness import verify_receipt
from synapse.frontier.runtime import (
    LABORATORY_SCHEMA,
    OBSERVER_CHECKPOINT_SCHEMA,
    OBSERVER_SUMMARY_SCHEMA,
    FrontierObserverHub,
    discover_experiments,
    experiment_catalog,
    run_all_frontier_experiments,
    run_frontier_experiment,
    run_frontier_laboratory,
)
from synapse.frontier.spec import MODULE_IDS, OBSERVER_MODULE_IDS, FrontierSpecV1
from synapse.frontier.workbench_api import FrontierWorkbenchAPI


def test_observer_hub_default_off_is_noop():
    spec = FrontierSpecV1()
    assert spec.enabled is False
    hub = FrontierObserverHub(spec, n_units=32)
    vec = np.linspace(-1.0, 1.0, 32, dtype=np.float64)
    before = vec.copy()
    assert hub.observe(vec) is False
    assert np.array_equal(vec, before)
    assert hub.agent_steps == 0 and hub.observed_steps == 0
    s = hub.summary()
    assert s['schema'] == OBSERVER_SUMMARY_SCHEMA and s['enabled'] is False and s['observers'] == {}


def test_observer_hub_all_five_observers_read_only_and_checkpoint_roundtrip():
    spec = FrontierSpecV1.max_preset(seed=11)
    hub = FrontierObserverHub(spec, n_units=48)
    assert set(hub.observers) == set(OBSERVER_MODULE_IDS)

    rng = np.random.default_rng(1234)
    t_rng_before = torch.random.get_rng_state().clone()
    np_rng_before = np.random.get_state()[1].copy()

    for _step in range(20):
        vec = torch.from_numpy(np.tanh(rng.standard_normal(48)).astype(np.float32))
        vec_copy = vec.clone()
        observed = hub.observe(vec)
        assert observed is True
        assert torch.equal(vec, vec_copy)

    assert torch.equal(torch.random.get_rng_state(), t_rng_before)
    assert np.array_equal(np.random.get_state()[1], np_rng_before)
    assert hub.agent_steps == 20 and hub.observed_steps == 20

    # JSON round-trip and exact post-restore continuation
    exported = hub.export_state()
    assert exported['schema'] == OBSERVER_CHECKPOINT_SCHEMA
    raw_json = json.loads(json.dumps(exported))
    clone = FrontierObserverHub(spec, n_units=48)
    clone.import_state(raw_json)
    assert clone.summary() == hub.summary()

    for _ in range(5):
        v = np.tanh(rng.standard_normal(48))
        hub.observe(v)
        clone.observe(v)
    assert clone.summary() == hub.summary()


def test_observer_hub_cadence_and_validate_before_mutate():
    spec = FrontierSpecV1(
        enabled=True,
        seed=5,
        modules=('synchrony', 'sparse_dictionary'),
        observe_every=3,
        max_observed_units=16,
    )
    hub = FrontierObserverHub(spec, n_units=32)
    rng = np.random.default_rng(77)
    flags = [hub.observe(np.tanh(rng.standard_normal(32))) for _ in range(7)]
    assert flags == [True, False, False, True, False, False, True]
    assert hub.agent_steps == 7 and hub.observed_steps == 3
    assert len(hub.unit_indices) == 16

    good = hub.export_state()
    before_summary = hub.summary()

    # Corrupt schema -> rejected, live hub untouched
    bad_schema = deepcopy(good)
    bad_schema['schema'] = 'Wrong'
    with pytest.raises(ValueError):
        hub.import_state(bad_schema)
    assert hub.summary() == before_summary

    bad_spec = deepcopy(good)
    bad_spec['spec']['seed'] += 1
    with pytest.raises(ValueError, match='spec_digest'):
        hub.import_state(bad_spec)
    assert hub.export_state() == good

    # Corrupt observed_steps vs agent_steps -> rejected
    bad_steps = deepcopy(good)
    bad_steps['observed_steps'] = 2
    with pytest.raises(ValueError):
        hub.import_state(bad_steps)
    assert hub.summary() == before_summary

    # Self-consistent 1-step observer state spliced into a 3-step hub -> hub step-count cross-check rejects it
    hub_one = FrontierObserverHub(spec, n_units=32)
    hub_one.observe(np.tanh(rng.standard_normal(32)))
    one_step_state = hub_one.export_state()

    bad_inner = deepcopy(good)
    bad_inner['observers']['synchrony'] = deepcopy(one_step_state['observers']['synchrony'])
    with pytest.raises(ValueError, match='step count'):
        hub.import_state(bad_inner)
    assert hub.summary() == before_summary

    bad_sae = deepcopy(good)
    bad_sae['observers']['sparse_dictionary'] = deepcopy(one_step_state['observers']['sparse_dictionary'])
    with pytest.raises(ValueError, match='step count'):
        hub.import_state(bad_sae)
    assert hub.summary() == before_summary


def test_experiment_discovery_and_single_verified_run():
    discovered = discover_experiments()
    assert 'synchrony' in discovered
    assert 'kuramoto_binding' in discovered
    assert 'anytime_inference' in discovered
    catalog = experiment_catalog()
    assert len(catalog) == len(discovered)
    assert all(row['module_id'] in MODULE_IDS for row in catalog)

    receipt = run_frontier_experiment('anytime_inference', quick=True, seeds=(0, 1))
    assert verify_receipt(receipt) is True
    assert receipt['experiment_id'] == 'anytime_inference'

    lab = run_frontier_laboratory(FrontierSpecV1.max_preset(seed=3), experiments=False)
    assert lab['schema'] == LABORATORY_SCHEMA
    assert len(lab['digest']) == 64


def test_multi_experiment_run_reuses_one_shot_seed_iterable(monkeypatch):
    from synapse.frontier import runtime

    def record_seeds(experiment_id, *, seeds, **kwargs):
        return {'experiment_id': experiment_id, 'seeds': list(seeds)}

    monkeypatch.setattr(runtime, 'run_frontier_experiment', record_seeds)
    runs = run_all_frontier_experiments(module_ids=('connectomics', 'joint_failure'),
                                        seeds=iter((3, 5)))
    assert [run['seeds'] for run in runs] == [[3, 5], [3, 5]]


def test_cli_helpers_and_commands(tmp_path):
    assert parse_observer_modules('all') == OBSERVER_MODULE_IDS
    assert parse_observer_modules('none') == ()
    assert parse_observer_modules('kuramoto_binding,synchrony') == ('synchrony', 'kuramoto_binding')
    with pytest.raises(ValueError):
        parse_observer_modules('muon_optimizer')
    assert parse_seeds('2,5') == (2, 5)
    with pytest.raises(ValueError):
        parse_seeds('1,1')

    assert cli_frontier_spec(None, None, 7) is None
    spec = cli_frontier_spec(True, None, 7, modules='synchrony', graft_optimizer='muon')
    assert spec is not None and spec.enabled is True and spec.modules == ('synchrony',) and spec.graft_optimizer == 'muon'
    with pytest.raises(ValueError):
        cli_frontier_spec(False, None, 7, modules='synchrony')

    cfg_path = tmp_path / 'frontier.json'
    cfg_path.write_text(json.dumps(spec.to_dict()), encoding='utf-8')
    loaded = cli_frontier_spec(None, cfg_path, 7)
    assert loaded == spec
    overridden = cli_frontier_spec(None, cfg_path, 7, graft_optimizer='adam')
    assert overridden.to_dict() == {**spec.to_dict(), 'graft_optimizer': 'adam'}

    cat = run_frontier_command(SimpleNamespace(frontier_action='catalog', seed=7, frontier_out=None))
    assert cat['ok'] is True and len(cat['modules']) == len(MODULE_IDS)

    out_file = tmp_path / 'observe.json'
    obs = run_frontier_command(SimpleNamespace(
        frontier_action='observe', seed=7, frontier_steps=8, frontier_units=16,
        frontier_sub_modules='synchrony,kuramoto_binding', frontier_out=out_file,
    ))
    assert obs['ok'] is True and obs['summary']['observed_steps'] == 8 and out_file.exists()


def test_workbench_api_isolated_lab_and_background_job():
    wb = SimpleNamespace(
        lock=threading.RLock(),
        job_lock=threading.RLock(),
        jobs={},
        cancel={},
        persist=lambda job: None,
        runtime=SimpleNamespace(config=SimpleNamespace(seed=7, frontier=None), agent=SimpleNamespace(frontier_hub=None)),
    )
    api = FrontierWorkbenchAPI(wb)
    cat = api.get('/api/frontier/catalog')
    assert cat['schema'] == 'FrontierWorkbenchApiV1' and len(cat['modules']) == len(MODULE_IDS)
    st = api.get('/api/frontier/status')
    assert st['live']['enabled'] is False and st['lab_built'] is False

    cfg_res = api.post('/api/frontier/configure', {'modules': ['synchrony'], 'n_units': 24, 'seed': 9})
    assert cfg_res['source'] == 'lab' and cfg_res['summary']['n_units'] == 24
    step_res = api.post('/api/frontier/step', {'steps': 6})
    assert step_res['stepped'] == 6 and step_res['summary']['observed_steps'] == 6

    sync_exp = api.post('/api/frontier/experiment', {'id': 'anytime_inference', 'quick': True, 'seeds': [0, 1]})
    assert sync_exp['experiment'] == 'anytime_inference' and len(sync_exp['rows']) == 1

    bg_job = api.post('/api/frontier/experiment', {
        'id': 'anytime_inference', 'quick': True, 'seeds': [0, 1], 'background': True,
    })
    deadline = time.time() + 15.0
    while wb.jobs[bg_job['id']]['status'] in ('queued', 'running') and time.time() < deadline:
        time.sleep(0.05)
    assert wb.jobs[bg_job['id']]['status'] == 'completed'
    assert len(wb.jobs[bg_job['id']]['result']['rows']) == 1

    with pytest.raises(ValueError):
        api.post('/api/frontier/step', {'steps': 2, 'bogus': 1})
    with pytest.raises(KeyError):
        api.get('/api/frontier/does-not-exist')
