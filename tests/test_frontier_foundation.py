"""Foundation of the frontier pack: neutral spec, neutral engine hook, shared harness."""
import copy

import numpy as np
import pytest

from synapse.frontier import (EngineExtension, FrontierSpecV1, MODULE_IDS, OBSERVER_MODULE_IDS, RandomTraceControl,
                              module_catalog)
from synapse.frontier.harness import (ExtendedHead, QUICK_ARGS, VERDICTS, clone_with_extensions, compare,
                                      headline_verdict, run_extension_experiment, verify_receipt)
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.experiments import ReadoutHead, _task_arm, make_spec
from synapse.neuroweave.spec import NeuroWeaveSpecV1

RICH = ('multi_soma', 'membrane_dynamics', 'synaptic_delays', 'reward_plasticity', 'stdp', 'neuromodulator_field',
        'multimodal_sensory', 'memory_ecology', 'predictive_ensemble', 'micro_collective', 'structural_plasticity')


def _run(engine, steps=12, seed=3):
    rng = np.random.default_rng(seed)
    outs = []
    for t in range(steps):
        res = engine.step(rng.uniform(-1, 1, 8), rng.uniform(-1, 1, 16), reward=0.5 if t % 3 else -0.4)
        outs.append(res['output'].copy())
    return np.stack(outs), res


class _Shift(EngineExtension):
    id = 'shift'

    def transform_drive(self, external, ctx):
        return None if external is None else external + 0.25


# ---------------------------------------------------------------------- spec
def test_default_spec_is_neutral_and_round_trips():
    spec = FrontierSpecV1()
    assert spec.enabled is False and spec.modules == () and spec.graft_optimizer == 'adam'
    assert FrontierSpecV1.from_dict(spec.to_dict()) == spec
    preset = FrontierSpecV1.max_preset(seed=11)
    assert preset.enabled and preset.modules == OBSERVER_MODULE_IDS
    assert preset.graft_optimizer == 'adam'      # observers only
    assert FrontierSpecV1.from_dict(preset.to_dict()) == preset and preset.spec_digest != spec.spec_digest
    assert [m['id'] for m in module_catalog()] == list(MODULE_IDS)


@pytest.mark.parametrize('kwargs', [
    {'enabled': 1}, {'modules': ('nope',)}, {'modules': ('recursive_refiner',)}, {'observe_every': 0},
    {'sae_k': 999}, {'graft_optimizer': 'sgd'}, {'seed': -1}])
def test_spec_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        FrontierSpecV1(**kwargs)


def test_spec_from_dict_rejects_unknown_fields_and_schema():
    with pytest.raises(ValueError):
        FrontierSpecV1.from_dict({'surprise': 1})
    with pytest.raises(ValueError):
        FrontierSpecV1.from_dict({'schema': 'Other'})


# ---------------------------------------------------------------------- engine hook
@pytest.mark.parametrize('mechs', [(), RICH])
def test_engine_without_extensions_matches_neutral_extension_exactly(mechs):
    spec = NeuroWeaveSpecV1(seed=5, neurons=24, mechanisms=mechs, structural_interval=4, sleep_interval=6)
    plain, hooked = NeuroWeaveEngine(spec), NeuroWeaveEngine(spec)
    hooked.attach_extension(EngineExtension())
    a, res_a = _run(plain)
    b, res_b = _run(hooked)
    assert np.array_equal(a, b)
    assert plain.export_state() == hooked.export_state()
    assert 'extensions' not in res_a and res_b['extensions'] == ['extension']


def test_extension_changes_drive_and_detach_removes_it():
    spec = NeuroWeaveSpecV1(seed=5, neurons=24)
    base, shifted, detached = NeuroWeaveEngine(spec), NeuroWeaveEngine(spec), NeuroWeaveEngine(spec)
    shifted.attach_extension(_Shift())
    detached.attach_extension(_Shift())
    assert detached.detach_extension('shift').id == 'shift' and detached.detach_extension('shift') is None
    a, _ = _run(base)
    b, _ = _run(shifted)
    c, _ = _run(detached)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_engine_rejects_bad_extensions_and_bad_drives():
    engine = NeuroWeaveEngine(NeuroWeaveSpecV1(seed=1, neurons=16))
    with pytest.raises(ValueError):
        engine.attach_extension(object())
    engine.attach_extension(EngineExtension())
    with pytest.raises(ValueError):
        engine.attach_extension(EngineExtension())            # duplicate id

    class Grow(EngineExtension):
        id = 'grow'

        def transform_drive(self, external, ctx):
            return np.concatenate([external, [0.0]])

    class Nan(EngineExtension):
        id = 'nan'

        def transform_drive(self, external, ctx):
            return external * np.nan

    class Invent(EngineExtension):
        id = 'invent'

        def transform_drive(self, external, ctx):
            return np.zeros(8)

    for bad, task in ((Grow(), np.zeros(8)), (Nan(), np.zeros(8)), (Invent(), None)):
        e = NeuroWeaveEngine(NeuroWeaveSpecV1(seed=1, neurons=16))
        e.attach_extension(bad)
        before = e.export_state()
        with pytest.raises(ValueError):
            e.step(task)
        assert e.brain.export_state() == before['brain']      # the brain was not stepped


def test_import_state_keeps_attached_extensions():
    spec = NeuroWeaveSpecV1(seed=2, neurons=16)
    engine = NeuroWeaveEngine(spec)
    ext = RandomTraceControl(4, seed=1)
    engine.attach_extension(ext)
    _run(engine, steps=4)
    engine.import_state(engine.export_state())
    assert engine.extensions == [ext]


# ---------------------------------------------------------------------- control extension
def test_random_trace_control_is_deterministic_bounded_and_cloneable():
    spec = NeuroWeaveSpecV1(seed=4, neurons=20)
    feats = []
    for _ in range(2):
        engine = NeuroWeaveEngine(spec)
        ext = RandomTraceControl(6, seed=9)
        engine.attach_extension(ext)
        _run(engine, steps=8)
        feats.append(ext.features().copy())
    assert np.array_equal(feats[0], feats[1])
    assert feats[0].shape == (6,) and np.all(np.abs(feats[0]) <= 1.0) and np.any(feats[0] != 0)
    twin = copy.deepcopy(ext)
    twin.after_step(np.ones(20), {})
    assert not np.array_equal(twin.features(), ext.features())           # independent clone
    fresh = RandomTraceControl(6, seed=9)
    fresh.bind(engine)
    fresh.import_state(ext.export_state())
    assert np.array_equal(fresh.features(), ext.features())
    for mismatch in ({'seed': 10}, {'decay_range': [0.1, 0.2]}, {'trace': [1.1] * 6}):
        with pytest.raises(ValueError):
            fresh.import_state({**ext.export_state(), **mismatch})
        assert np.array_equal(fresh.features(), ext.features())
    with pytest.raises(ValueError):
        fresh.import_state({'id': 'random_trace_control', 'feature_dim': 6, 'n': 20, 'trace': [float('nan')] * 6})
    with pytest.raises(ValueError):
        RandomTraceControl(0)


# ---------------------------------------------------------------------- harness
def test_extended_head_shares_baseline_initialisation():
    base, ext = ReadoutHead(16, 3), ExtendedHead(16, 3, extra_dim=5)
    assert ext.dim == base.dim + 5 and np.array_equal(ext.w[:base.dim], base.w)
    assert ExtendedHead(16, 3).dim == base.dim


def test_clone_with_extensions_is_independent():
    engine = NeuroWeaveEngine(make_spec((), seed=1, neurons=16, topology='clustered', steps=12))
    engine.attach_extension(RandomTraceControl(3, seed=2))
    _run(engine, steps=5)
    twin = clone_with_extensions(engine)
    assert twin.extensions[0] is not engine.extensions[0]
    assert np.array_equal(twin.extensions[0].features(), engine.extensions[0].features())
    a, _ = _run(engine, steps=4, seed=8)
    b, _ = _run(twin, steps=4, seed=8)
    assert np.array_equal(a, b) and np.array_equal(twin.extensions[0].features(), engine.extensions[0].features())


def test_headline_verdict_requires_beating_the_control():
    win = compare([0.9, 0.8, 0.85], [0.5, 0.45, 0.5])
    tie = compare([0.6, 0.4, 0.5], [0.5, 0.5, 0.5])
    lose = compare([0.2, 0.25, 0.2], [0.5, 0.45, 0.5])
    assert win['verdict'] == 'improved' and tie['verdict'] == 'no_detectable_difference' and lose['verdict'] == 'worse'
    assert headline_verdict(win, None)['verdict'] == 'improved'
    assert headline_verdict(win, win)['verdict'] == 'improved'
    blocked = headline_verdict(win, tie)
    assert blocked == {'verdict': 'no_detectable_difference', 'basis': 'not_better_than_capacity_matched_control'}
    assert headline_verdict(lose, win)['verdict'] == 'worse'
    assert headline_verdict(tie, win)['verdict'] == 'no_detectable_difference'


def test_extension_experiment_receipt_is_sealed_deterministic_and_keeps_the_baseline():
    kw = dict(task='xor', **QUICK_ARGS)
    factory = lambda seed: RandomTraceControl(4, seed=seed + 50)       # noqa: E731
    first = run_extension_experiment('foundation_smoke', factory, **kw)
    second = run_extension_experiment('foundation_smoke', factory, **kw)
    assert first['digest'] == second['digest'] and verify_receipt(first)
    assert first['verdict'] in VERDICTS and set(first['arms']) == {'baseline', 'candidate', 'control'}
    assert first['config']['candidate_feature_dim'] == 4 and first['config']['control'] == 'random_trace'
    tampered = copy.deepcopy(first)
    tampered['verdict'] = 'improved' if first['verdict'] != 'improved' else 'worse'
    assert not verify_receipt(tampered)
    # the harness baseline is the NEUROWEAVE lab's own baseline arm, number for number
    lab = _task_arm((), 0, task='xor', steps=kw['steps'], heldout_seeds=kw['heldout_seeds'],
                    heldout_steps=kw['heldout_steps'], neurons=kw['neurons'], topology='clustered')
    mine = first['arms']['baseline']['per_seed'][0]
    assert mine['train']['accuracy'] == lab['train']['accuracy']
    assert mine['heldout']['accuracy'] == lab['heldout']['accuracy']
    assert mine['heldout']['log_loss'] == pytest.approx(lab['heldout']['log_loss'], abs=1e-12)
    assert mine['budget']['extension_features'] == 0
    cand = first['arms']['candidate']['per_seed'][0]
    assert cand['budget']['extension_features'] == 4 and cand['extensions'] == ['random_trace_control']


def test_extension_experiment_validates_arguments():
    f = lambda seed: RandomTraceControl(2, seed=seed)                    # noqa: E731
    for bad in ({'seeds': ()}, {'seeds': (0,), 'heldout_seeds': (0,)}, {'task': 'nope'}, {'metric': 'f1'}, {'steps': 2},
                {'control': 'magic'}, {'heldout_seeds': (100, 100)}, {'seeds': (0.5, 1)},
                {'heldout_seeds': (True,)}, {'heldout_steps': 4.5}, {'heldout_steps': 1_000_001}):
        with pytest.raises(ValueError):
            run_extension_experiment('x', f, **{**dict(task='xor', **QUICK_ARGS), **bad})
    no_feature = run_extension_experiment('x', lambda s: EngineExtension(), task='xor', **QUICK_ARGS)
    assert set(no_feature['arms']) == {'baseline', 'candidate'} and no_feature['config']['control'] is None
