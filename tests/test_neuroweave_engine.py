"""NeuroWeaveEngine: flag causality, all-off identity, determinism and persistence (tiny, fast)."""
import json

import numpy as np
import pytest

from synapse.neuroweave.brain import NeuroWeaveBrain
from synapse.neuroweave.engine import ENGINE_FLAGS, NeuroWeaveEngine
from synapse.neuroweave.spec import MECHANISM_IDS, NeuroWeaveSpecV1

N = 24
STEPS = 8


def _spec(mechs=(), **kw):
    base = dict(neurons=N, seed=5, sleep_interval=4, structural_interval=3, max_neurons=32)
    base.update(kw)
    return NeuroWeaveSpecV1(mechanisms=tuple(mechs), **base)


def _run(spec, steps=STEPS, *, sensory=True, actions=True):
    e = NeuroWeaveEngine(spec)
    outs = []
    for t in range(steps):
        x = np.sin(np.arange(8) * 0.7 + 0.4 * t)
        s = 0.5 * np.cos(np.arange(16) * 0.3 + 0.5 * t) if sensory else None
        kw = {}
        if actions:
            kw = dict(issued_action=[0.3, -0.2 + 0.05 * t], observed_outcome=[0.25 + 0.1 * (t % 3), -0.1])
        r = e.step(x, s, reward=(0.5 if t % 3 == 0 else -0.25), **kw)
        outs.append(r['output'].copy())
    return e, outs


def test_all_off_engine_equals_bare_brain():
    spec = _spec()
    e, b = NeuroWeaveEngine(spec), NeuroWeaveBrain(spec)
    for t in range(STEPS):
        x = np.sin(np.arange(8) + t)
        r = e.step(x, 0.3 * np.ones(16), reward=0.4 * (t % 2))
        assert np.array_equal(r['output'], b.step(x, reward=0.4 * (t % 2)))
        assert r['controls']['applied_to_brain'] is False and r['sensory']['fed_to_brain'] is False


@pytest.mark.parametrize('mech', ['multimodal_sensory', 'neuromodulator_field', 'affective_modulation',
                                  'uncertainty_gating', 'memory_ecology', 'representation_fusion',
                                  'efference_self_model'])
def test_engine_level_mechanism_changes_outputs(mech):
    # every engine-consumed flag is evaluated on top of a base that gives it something to act on
    base = {'uncertainty_gating': ('predictive_ensemble',),
            'representation_fusion': ('micro_collective',),
            'efference_self_model': ('reward_plasticity',)}.get(mech, ())
    _, off = _run(_spec(base))
    _, on = _run(_spec(base + (mech,)))
    assert any(not np.array_equal(a, b) for a, b in zip(off, on)), mech


def test_multi_region_routing_uses_modulated_route_gains():
    base = ('neuromodulator_field', 'affective_modulation')
    e_off, off = _run(_spec(base))
    e_on, on = _run(_spec(base + ('multi_region_routing',)))
    assert any(not np.array_equal(a, b) for a, b in zip(off, on))
    assert e_on.last['route_gain_applied'] is True


def test_micro_collective_trains_on_consecutive_frames_and_specialist_starts_neutral():
    e, _ = _run(_spec(('micro_collective', 'specialist_graft', 'multimodal_sensory')))
    assert e.counters['micro_train_steps'] == STEPS - 1
    assert e.last['micro']['specialist_coupling'] == 0.0
    assert e.graft is not None and e.graft.samples > 0
    receipt = e.detach_specialist()
    assert receipt['attached'] is False and e.graft is None and e.micro.specialist is None


def test_memory_provenance_and_predictive_replay():
    e, _ = _run(_spec(('memory_ecology', 'micro_collective', 'predictive_replay')), steps=9)
    assert e.counters['memory_sensory'] == 9 and e.counters['memory_predicted'] == 9
    assert e.counters['memory_confirmed'] + e.counters['memory_contradicted'] == 8
    assert e.counters['replay_cycles'] == 2 and e.counters['replayed'] > 0
    prov = e.memory.stats()['provenance']
    assert prov['predicted'] >= 1 and prov['sensory'] >= 1


def test_structural_plasticity_and_neurogenesis_stay_within_caps():
    e, _ = _run(_spec(MECHANISM_IDS), steps=10)
    assert e.n <= 32 and e.brain.synapses.size <= e.spec.max_synapses
    assert e.observatory()['neural']['ledger']['total'] >= 0


def test_determinism_and_export_import_round_trip():
    spec = _spec(MECHANISM_IDS)
    e1, o1 = _run(spec, steps=6)
    _, o2 = _run(spec, steps=6)
    assert all(np.array_equal(a, b) for a, b in zip(o1, o2))
    state = json.loads(json.dumps(e1.export_state()))
    e3 = NeuroWeaveEngine(spec)
    e3.import_state(state)
    a = e1.step(np.ones(8), np.zeros(16), reward=0.2)
    b = e3.step(np.ones(8), np.zeros(16), reward=0.2)
    assert np.array_equal(a['output'], b['output'])
    assert a['readout_probability'] == b['readout_probability']


def test_import_rejects_malformed_state_without_mutation():
    spec = _spec(('micro_collective',))
    e, _ = _run(spec, steps=3)
    before = json.dumps(e.export_state(), sort_keys=True)
    bad = e.export_state()
    bad['runtime']['authorship'] = 7.0
    with pytest.raises(ValueError):
        e.import_state(bad)
    with pytest.raises(ValueError):
        NeuroWeaveEngine(_spec()).import_state(e.export_state())   # spec mismatch
    assert json.dumps(e.export_state(), sort_keys=True) == before


def test_observatory_telemetry_package_fpga_and_ecology():
    e, _ = _run(_spec(MECHANISM_IDS), steps=4)
    e.attach_ecology('default')
    summary = e.run_ecology(3)
    assert summary['steps'] == 3 and summary['evidence_tier'] == 'synthetic_abstraction'
    ob = e.observatory()
    for section in ('neural', 'neuromodulation', 'sensory', 'learning', 'environment', 'memory'):
        assert section in ob
    assert ob['sensory']['manifest']['raw_media_stored'] is False
    json.dumps(ob)
    tel = e.telemetry()
    assert 'wallclock_ms_last' not in tel and tel['raw_media_stored'] is False
    from synapse.neuroweave.package import validate_package
    assert validate_package(e.export_package('neuromod', 'memory'))['valid'] is True
    m = e.fpga_manifest()
    assert m['independent_validation']['valid'] is True


def test_engine_flag_registry_is_known():
    assert set(ENGINE_FLAGS) <= set(MECHANISM_IDS)
