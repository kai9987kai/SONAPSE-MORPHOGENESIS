"""End-to-end host integration tests for the Frontier Pack in Sonapse."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from synapse.agent import IntegratedAgent
from synapse.cli import main as cli_main
from synapse.config import SynapseConfig
from synapse.frontier.muon import MuonWithAuxAdam
from synapse.frontier.spec import OBSERVER_MODULE_IDS, FrontierSpecV1
from synapse.neural_graft import NeuralResidualGraft
from synapse.runtime import LabRuntime
from synapse.server import Workbench


def _base(**kw) -> SynapseConfig:
    return SynapseConfig(nodes_per_source=16, seed=11, **kw)


def test_synapse_config_frontier_defaults_coercion_and_matches_saved():
    cfg = _base()
    assert cfg.frontier is None and 'frontier' not in cfg.to_dict()
    assert cfg.matches_saved(cfg.to_dict()) is True

    enabled = replace(cfg, frontier=FrontierSpecV1.max_preset(seed=11))
    d = enabled.to_dict()
    assert d['frontier']['schema'] == 'FrontierSpecV1'
    rebuilt = SynapseConfig(**{**d, 'connectome_sources': tuple(d['connectome_sources'])})
    assert rebuilt.frontier == enabled.frontier
    assert enabled.matches_saved(d) is True
    assert enabled.matches_saved(cfg.to_dict()) is True             # additive read-only on legacy
    assert cfg.matches_saved(d) is False                            # reverse stays strict
    assert replace(cfg, frontier=FrontierSpecV1()).matches_saved(cfg.to_dict()) is False

    muon_cfg = replace(cfg, frontier=FrontierSpecV1(enabled=True, modules=('synchrony',), graft_optimizer='muon'))
    assert muon_cfg.matches_saved(cfg.to_dict()) is False           # optimizer change is NOT legacy-compatible

    with pytest.raises(ValueError):
        SynapseConfig(frontier=123)


def test_max_tandem_audit_and_exact_checkpoint_replay_with_frontier(tmp_path):
    cfg = SynapseConfig.max_tandem(total_neurons=96, seed=11, task='forage', compute_threads=1)
    audit = cfg.max_tandem_feature_audit()
    assert audit['complete'] is True
    assert 'frontier_v1' in audit['enabled_features']
    assert all(f'frontier_v1.{m}' in audit['enabled_features'] for m in OBSERVER_MODULE_IDS)
    assert 'frontier_v1' in _base().max_tandem_feature_audit()['disabled_features']

    lab = LabRuntime(cfg, output_root=tmp_path)
    assert lab.agent.frontier_hub is not None and lab.agent.frontier_hub.enabled is True
    lab.step(4, learn=True)
    assert 'frontier' in lab.agent.last_telemetry
    assert lab.agent.last_telemetry['frontier']['observed_steps'] == 4

    lab.save('frontier_max_ckpt')
    expected = lab.step(2, learn=True)
    fresh = LabRuntime(cfg, output_root=tmp_path)
    fresh.load('frontier_max_ckpt')
    actual = fresh.step(2, learn=True)
    assert actual == expected


def test_live_frontier_observers_are_strictly_trajectory_neutral():
    cfg_off = _base()
    cfg_on = replace(cfg_off, frontier=FrontierSpecV1.max_preset(seed=11))
    a_off = IntegratedAgent(cfg_off)
    a_on = IntegratedAgent(cfg_on)

    rng = np.random.default_rng(42)
    for _ in range(8):
        obs = np.clip(rng.normal(0.0, 0.3, size=155), -1.0, 1.0).astype(np.float32)
        out_off = a_off.act(obs, learn=True)
        out_on = a_on.act(obs, learn=True)
        assert np.array_equal(out_off['action'], out_on['action'])
        assert np.array_equal(out_off['features'], out_on['features'])
        assert torch.equal(a_off.brain.r, a_on.brain.r)
        a_off.feedback(0.25, False)
        a_on.feedback(0.25, False)

    assert a_off.frontier_hub is None
    assert a_on.frontier_hub is not None and a_on.frontier_hub.observed_steps == 8
    assert 'frontier' not in a_off.last_telemetry
    assert a_on.last_telemetry['frontier']['observed_steps'] == 8


def test_agent_checkpoint_roundtrip_and_additive_legacy_migration(tmp_path):
    # Legacy checkpoint saved without frontier can be loaded into a frontier-enabled agent
    legacy_cfg = _base()
    legacy_lab = LabRuntime(legacy_cfg, output_root=tmp_path)
    legacy_lab.step(3, learn=True)
    legacy_lab.save('legacy_no_frontier')

    upgraded_cfg = replace(legacy_cfg, frontier=FrontierSpecV1.max_preset(seed=11))
    upgraded_lab = LabRuntime(upgraded_cfg, output_root=tmp_path)
    upgraded_lab.step(2, learn=True)
    assert upgraded_lab.agent.frontier_hub.observed_steps == 2
    upgraded_lab.load('legacy_no_frontier')
    assert upgraded_lab.agent.frontier_hub.observed_steps == 0
    upgraded_lab.step(2, learn=True)
    assert upgraded_lab.agent.frontier_hub.observed_steps == 2

    # Reverse strictness: a frontier-enabled checkpoint cannot load into a frontier-off agent
    upgraded_lab.save('with_frontier')
    off_lab = LabRuntime(legacy_cfg, output_root=tmp_path)
    with pytest.raises(ValueError):
        off_lab.load('with_frontier')

    # Missing frontier_v1 namespace on a frontier-enabled checkpoint is rejected cleanly
    state = upgraded_lab.agent.export_state()
    assert 'frontier_v1' in state
    del state['frontier_v1']
    with pytest.raises(ValueError, match='frontier_v1'):
        upgraded_lab.agent.import_state(state)


def test_neural_graft_muon_optimizer_end_to_end_and_legacy_settings_preserved():
    # Default adam leaves settings free of optimizer_kind for legacy byte-identical checkpoints
    g_adam = NeuralResidualGraft(hidden_dim=64, seed=7)
    assert 'optimizer_kind' not in g_adam.settings
    assert isinstance(g_adam.optimizer, torch.optim.Adam)

    # Muon opt-in works in IntegratedAgent + active_training + checkpoint restore
    cfg = _base(
        neural_graft=True,
        active_training=True,
        graft_batch_size=8,
        graft_updates=1,
        frontier=FrontierSpecV1(enabled=True, seed=11, modules=('synchrony',), graft_optimizer='muon'),
    )
    agent = IntegratedAgent(cfg)
    assert agent.neural_graft is not None
    assert isinstance(agent.neural_graft.optimizer, MuonWithAuxAdam)
    assert agent.neural_graft.settings['optimizer_kind'] == 'muon'

    rng = np.random.default_rng(19)
    obs = np.clip(rng.normal(0.0, 0.2, size=155), -1.0, 1.0).astype(np.float32)
    for _ in range(16):
        agent.act(obs, learn=True)
        next_obs = np.clip(rng.normal(0.0, 0.2, size=155), -1.0, 1.0).astype(np.float32)
        agent.observe_transition(next_obs, 0.1, False)
        obs = next_obs
    assert agent.neural_graft.optimizer_updates >= 1

    exported = agent.neural_graft.export_state()
    clone = NeuralResidualGraft(seed=cfg.seed, optimizer_kind='muon')
    clone.import_state(exported)
    assert clone.optimizer_updates == agent.neural_graft.optimizer_updates

    # Mismatched optimizer_kind rejected without mutating target
    adam_target = NeuralResidualGraft(seed=cfg.seed)
    with pytest.raises(ValueError, match='optimizer mismatch'):
        adam_target.import_state(exported)


def test_cli_frontier_subcommand_and_global_flags(capsys):
    rc = cli_main(['frontier', 'catalog'])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['ok'] is True and 'synchrony' in [m['id'] for m in payload['modules']]

    rc = cli_main(['frontier', 'observe', '--steps', '6', '--units', '16', '--modules', 'synchrony'])
    assert rc == 0
    obs_payload = json.loads(capsys.readouterr().out)
    assert obs_payload['ok'] is True and obs_payload['summary']['observed_steps'] == 6

    rc = cli_main(['--nodes', '16', '--frontier', 'status'])
    assert rc == 0
    st_payload = json.loads(capsys.readouterr().out)
    assert st_payload['config']['frontier']['enabled'] is True


def test_workbench_server_frontier_routes_and_reset_preservation(tmp_path):
    cfg = _base(frontier=FrontierSpecV1.max_preset(seed=11))
    wb = Workbench(config=cfg, output_root=tmp_path)

    cat = wb.get('/api/frontier/catalog')
    assert cat['schema'] == 'FrontierWorkbenchApiV1'
    status = wb.get('/api/frontier/status')
    assert status['live']['enabled'] is True and status['observed_source'] == 'live'

    wb.post('/api/step', {'steps': 3, 'learn': False})
    live_obs = wb.get('/api/frontier/observers')
    assert live_obs['source'] == 'live' and live_obs['summary']['observed_steps'] == 3

    lab_step = wb.post('/api/frontier/step', {'steps': 4})
    assert lab_step['source'] == 'lab' and lab_step['summary']['observed_steps'] == 4
    # Stepping the isolated lab hub never mutates the live observer hub
    assert wb.get('/api/frontier/observers')['summary']['observed_steps'] == 3

    # /api/reset with a custom config dict that omits 'frontier' preserves the active frontier spec
    wb.post('/api/reset', {'config': {'nodes_per_source': 16, 'seed': 11, 'task': 'forage'}})
    assert wb.runtime.config.frontier == cfg.frontier


def test_frontier_web_ui_tab_and_loopback_http(tmp_path):
    import threading
    import urllib.request
    from synapse.config import ROOT
    from synapse.server import create_server

    html = (ROOT / 'synapse' / 'web' / 'index.html').read_text(encoding='utf-8')
    js = (ROOT / 'synapse' / 'web' / 'app.js').read_text(encoding='utf-8')
    for marker in ('data-tab="frontier"', 'id="frontier"', 'id="frl-exp-select"', 'id="frl-step-run"'):
        assert marker in html
    assert "api('frontier/catalog')" in js and "api('frontier/experiment'" in js

    server = create_server('127.0.0.1', 0, config=_base(), output_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f'http://127.0.0.1:{server.server_port}'
        with urllib.request.urlopen(f'{base_url}/api/frontier/catalog', timeout=5) as resp:
            assert resp.status == 200
            cat = json.loads(resp.read().decode('utf-8'))
            assert cat['schema'] == 'FrontierWorkbenchApiV1'
        req = urllib.request.Request(
            f'{base_url}/api/frontier/step',
            data=json.dumps({'steps': 3}).encode('utf-8'),
            headers={'Content-Type': 'application/json', 'Origin': base_url},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            stepped = json.loads(resp.read().decode('utf-8'))
            assert stepped['stepped'] == 3 and stepped['summary']['observed_steps'] == 3
    finally:
        server.shutdown()
        server.server_close()
