"""Sonapse <-> NEUROWEAVE integration: scalar-path equivalence, checkpoints, config and Max Mode audit."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.neuroweave.bridge import NeuroWeaveBusChannel, obs_to_sensory
from synapse.neuroweave.spec import MECHANISM_IDS, NeuroWeaveSpecV1
from synapse.runtime import LabRuntime

ROOT = Path(__file__).resolve().parents[1]
STEPS = 6


def _base():
    return SynapseConfig(nodes_per_source=16)


def _nw(**kw):
    base = dict(enabled=True, neurons=24, seed=3, mechanisms=MECHANISM_IDS, max_neurons=32)
    base.update(kw)
    return NeuroWeaveSpecV1(**base)


def _drive(cfg):
    agent = IntegratedAgent(cfg)
    rng = np.random.default_rng(0)
    rates, actions = [], []
    for i in range(STEPS):
        out = agent.act(rng.random(155).astype(np.float32), learn=True)
        agent.feedback(0.3 if i % 2 else -0.1)
        rates.append(agent.brain.r.detach().clone())
        actions.append(np.asarray(out['action'], dtype=np.float64).copy())
    return agent, rates, actions


@pytest.fixture(scope='module')
def legacy_run():
    return _drive(_base())


def _same(a, b):
    return (all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
            and all(np.array_equal(x, y) for x, y in zip(a[2], b[2])))


def test_scalar_pathway_bit_identical_when_neuroweave_is_neutral(legacy_run):
    disabled = _drive(replace(_base(), neuroweave=NeuroWeaveSpecV1()))
    assert disabled[0].neuroweave_channel is None
    assert _same(legacy_run, disabled)
    zero_gain = _drive(replace(_base(), neuroweave=_nw(gain=0.0)))
    assert zero_gain[0].neuroweave_channel.engine_steps == STEPS
    assert _same(legacy_run, zero_gain)
    observe = _drive(replace(_base(), neuroweave=_nw(observe_only=True, gain=0.1)))
    assert observe[0].neuroweave_channel.engine_steps == STEPS
    assert _same(legacy_run, observe)
    tel = observe[0].last_telemetry['neuroweave']
    assert tel['observe_only'] is True and 'latency_ms' not in tel and tel['raw_media_stored'] is False


def test_positive_gain_changes_rates(legacy_run):
    active = _drive(replace(_base(), neuroweave=_nw(gain=0.1, ramp_steps=2)))
    assert not all(torch.equal(x, y) for x, y in zip(legacy_run[1], active[1]))
    assert active[0].last_telemetry['neuroweave']['effective_gain'] == pytest.approx(0.1)


def test_channel_contract_disabled_ramp_and_clipping():
    r = torch.rand(40)
    off = NeuroWeaveBusChannel(NeuroWeaveSpecV1(), input_dim=40)
    assert torch.equal(off.compute_current(r), torch.zeros_like(r)) and off.engine_steps == 0
    ch = NeuroWeaveBusChannel(_nw(gain=0.1, max_gain=0.1, ramp_steps=4, mechanisms=()), input_dim=40)
    gains = []
    for _ in range(6):
        ch.compute_current(r)
        gains.append(ch.effective_gain())
    assert gains[0] == 0.0 and gains == sorted(gains) and max(gains) == pytest.approx(0.1)
    cached = ch.engine_steps
    ch.compute_current(r, advance=False)
    ch.compute_current(r, microstep=1)
    assert ch.engine_steps == cached
    channels, _ = obs_to_sensory(np.full(155, 0.5))
    assert len(channels) == 16 and channels['audio_low'] is None


def test_checkpoint_round_trip_and_legacy_rules(tmp_path):
    cfg = replace(_base(), neuroweave=_nw(gain=0.05, ramp_steps=2))
    lab = LabRuntime(cfg, output_root=tmp_path)
    lab.step(2)
    lab.save('nw')
    expected = lab.step(1)
    fresh = LabRuntime(cfg, output_root=tmp_path)
    fresh.load('nw')
    assert fresh.agent.neuroweave_channel.engine_steps == 2
    assert fresh.step(1) == expected

    # checkpoint with NEUROWEAVE state is rejected by a runtime without it
    plain = LabRuntime(_base(), output_root=tmp_path)
    with pytest.raises(ValueError):
        plain.load('nw')

    # legacy checkpoint (no namespace) loads additively into a NEUROWEAVE runtime
    legacy = LabRuntime(_base(), output_root=tmp_path)
    legacy.step(1)
    legacy.save('legacy')
    restored = LabRuntime(cfg, output_root=tmp_path)
    initial = restored.agent.neuroweave_channel.export_state()['digest']
    snapshot = restored.load('legacy')
    assert snapshot['steps'] == 1
    assert restored.agent.neuroweave_channel.export_state()['digest'] == initial
    assert restored.step(1)['steps'] == 2

    state = lab.agent.export_state()
    assert 'neuroweave_v1' in state and 'neuroweave' not in state['graft_bus']['channels']
    tampered = dict(state['neuroweave_v1'])
    tampered['input_dim'] = tampered['input_dim'] + 1
    with pytest.raises(ValueError):
        lab.agent.neuroweave_channel.import_state(tampered)


def test_config_defaults_round_trip_and_matches_saved():
    cfg = _base()
    assert cfg.neuroweave is None and 'neuroweave' not in cfg.to_dict()
    assert cfg.matches_saved(cfg.to_dict())
    enabled = replace(cfg, neuroweave=_nw())
    d = enabled.to_dict()
    assert d['neuroweave']['schema'] == 'NeuroWeaveSpecV1'
    assert SynapseConfig(**{**d, 'connectome_sources': tuple(d['connectome_sources'])}).neuroweave == enabled.neuroweave
    assert enabled.matches_saved(d) is True
    assert enabled.matches_saved(cfg.to_dict()) is True        # additive on legacy
    assert cfg.matches_saved(d) is False                       # reverse stays strict
    assert replace(cfg, neuroweave=NeuroWeaveSpecV1()).matches_saved(cfg.to_dict()) is False
    with pytest.raises(ValueError):
        SynapseConfig(neuroweave=42)


def test_max_tandem_audit_complete_with_neuroweave_entries():
    cfg = SynapseConfig.max_tandem(total_neurons=96, seed=11)
    audit = cfg.max_tandem_feature_audit()
    assert audit['complete'] is True
    assert 'neuroweave_v1' in audit['enabled_features']
    assert all(f'neuroweave_v1.{m}' in audit['enabled_features'] for m in MECHANISM_IDS)
    assert cfg.neuroweave.neurons == 48 and cfg.neuroweave.enabled is True
    assert 'neuroweave_v1' in _base().max_tandem_feature_audit()['disabled_features']


def test_compatibility_manifest_has_no_problems():
    spec = importlib.util.spec_from_file_location('nw_compat_manifest', ROOT / 'scripts' / 'neuroweave_compat_manifest.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import json
    manifest = json.loads(mod.MANIFEST.read_text(encoding='utf-8'))
    assert mod.check(manifest, mod.collect()) == []


def test_tandem_max_neuroweave_laboratory_section_is_bounded_and_honest():
    from synapse.tandem_max import run_neuroweave_laboratory
    spec = NeuroWeaveSpecV1.max_preset(seed=3, neurons=48)
    receipt = run_neuroweave_laboratory(spec, steps=4, topology_neurons=24)
    assert receipt['engine_steps'] == 4 and receipt['evidence_tier'] == 'synthetic_abstraction'
    assert len(receipt['topology_sweep']['metrics']) == 12
    assert receipt['fpga_manifest']['valid'] is True and receipt['fpga_manifest']['hardware_validated'] is False
    assert receipt['package_round_trip']['valid'] is True and receipt['package_round_trip']['inspect_valid'] is True
    assert receipt['micro_collective_receipt']['evidence_tier'] == 'synthetic_abstraction'
    with pytest.raises(ValueError):
        run_neuroweave_laboratory(spec, steps=0)


def test_tandem_max_neuroweave_laboratory_extra_work_is_bounded_and_isolates_live_agent():
    from synapse.neuroweave.experiments import verify_receipt  # noqa: F401  (digests are verified inside)
    from synapse.tandem_max import run_neuroweave_laboratory
    agent, _, _ = _drive(replace(_base(), neuroweave=_nw(gain=0.05, ramp_steps=2)))
    channel = agent.neuroweave_channel
    digest_before = channel.export_state()['digest']
    steps_before = channel.engine_steps
    rates_before = agent.brain.r.detach().clone()
    spec = NeuroWeaveSpecV1.max_preset(seed=3, neurons=48)
    receipt = run_neuroweave_laboratory(
        spec, steps=2, topology_neurons=16, experiments=1,
        experiment_ids=['meta_plasticity', 'topology_optimisation'],
        experiment_overrides={'steps': 8, 'heldout_steps': 4}, lesion=True, agent=agent, agent_steps=3)
    exps = receipt['experiments']
    assert exps['count'] == 2 and exps['all_digests_verified'] is True and exps['mode'] == 'quick'
    assert [e['experiment_id'] for e in exps['experiments']] == ['meta_plasticity', 'topology_optimisation']
    assert all(e['verdict'] in ('improved', 'no_detectable_difference', 'worse') and len(e['digest']) == 64
               for e in exps['experiments'])
    assert sum(exps['verdict_counts'].values()) == 2 and len(exps['receipts_digest']) == 64
    lesion = receipt['lesion_assay']
    assert lesion['originals_untouched'] is True and len(lesion['digest']) == 64
    assert lesion['evidence_tier'] == 'synthetic_abstraction'
    live = receipt['live_channel']
    assert live['present'] is True and live['telemetry']['schema'] == 'NeuroWeaveChannelTelemetryV1'
    assert live['observatory_summary']['neurons'] == channel.engine.n
    assert 'latency_ms' not in live['telemetry']
    eco = receipt['live_agent_ecology']
    if 'insufficient free memory' in str(eco.get('skipped')):    # honest skip on a memory-starved host
        pytest.skip(f'isolated agent copy skipped by the memory guard: {eco}')
    assert eco['steps'] == 3 and eco['actions_finite'] is True and eco['live_channel_digest_unchanged'] is True
    assert eco['copy_channel_engine_steps'] > steps_before
    for key in ('total_reward', 'food_eaten', 'collisions'):
        assert key in eco
    # the live agent was never stepped or mutated
    assert channel.export_state()['digest'] == digest_before and channel.engine_steps == steps_before
    assert torch.equal(agent.brain.r, rates_before)
    import json
    json.dumps(receipt, allow_nan=False)


def test_tandem_max_neuroweave_laboratory_extras_default_off_and_validated():
    from synapse.tandem_max import run_neuroweave_laboratory
    spec = NeuroWeaveSpecV1.max_preset(seed=3, neurons=48)
    receipt = run_neuroweave_laboratory(spec, steps=2, topology_neurons=16)
    assert 'skipped' in receipt['experiments'] and 'skipped' in receipt['lesion_assay']
    assert receipt['live_channel']['present'] is False and 'skipped' in receipt['live_agent_ecology']
    plain = IntegratedAgent(_base())
    r2 = run_neuroweave_laboratory(spec, steps=2, topology_neurons=16, agent=plain, agent_steps=0)
    assert r2['live_channel']['present'] is False and r2['live_agent_ecology'] == {'skipped': 'neuroweave_agent_steps=0'}
    for kw in (dict(experiments=2), dict(experiments=True), dict(lesion=1), dict(agent_steps=65), dict(agent_steps=-1)):
        with pytest.raises(ValueError):
            run_neuroweave_laboratory(spec, steps=2, **kw)


@pytest.mark.parametrize('option, value', [('neuroweave_experiments', 2), ('neuroweave_lesion', -1),
                                           ('neuroweave_agent_steps', 65)])
def test_tandem_max_rejects_bad_neuroweave_options_before_replacing_live_runtime(tmp_path, option, value):
    from synapse.server import Workbench
    wb = Workbench(config=_base(), output_root=tmp_path)
    live_runtime = wb.runtime
    with pytest.raises(ValueError, match=option):
        wb.post('/api/tandem/max', {option: value})
    assert wb.runtime is live_runtime
