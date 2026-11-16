"""Tests for NEUROWEAVE neuromodulation (neuromod.py) and the sensory hub (sensory.py)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from synapse.neuroweave.neuromod import (
    AFFECT_BASELINE, BASELINES, CONTROL_KEYS, NeuromodulatorSystem, TRANSMITTER_LABELS,
    infer_transmitters, neutral_controls,
)
from synapse.neuroweave.sensory import SensoryFrame, SensoryHub, audio_features, visual_features
from synapse.neuroweave.spec import (
    AFFECT_VARIABLES, NEUROMODULATORS, SENSORY_CHANNEL_IDS, SENSORY_DIM, NeuroWeaveSpecV1,
)

MOD = frozenset({'neuromodulator_field'})
AFF = frozenset({'affective_modulation'})
BOTH = MOD | AFF
HET = BOTH | {'receptor_heterogeneity'}


def _ring(n: int):
    src = np.arange(n)
    dst = (src + 1) % n
    return src, dst


def _system(n: int = 12, seed: int = 3) -> NeuromodulatorSystem:
    src, dst = _ring(n)
    signs = np.where(np.arange(n) % 5 == 0, -1, 1).astype(np.int8)
    return NeuromodulatorSystem(n, seed=seed, signs=signs, src=src, dst=dst)


def _assert_neutral(c: dict) -> None:
    assert set(c) == set(CONTROL_KEYS)
    for k in CONTROL_KEYS:
        if k == 'sensory_weight':
            assert c[k].shape == (16,)
            np.testing.assert_array_equal(c[k], np.ones(16))
        elif k == 'threshold_shift':
            assert c[k] == 0.0
        else:
            assert c[k] == 1.0, k


def _drive(sys, steps, flags, **kw):
    args = dict(reward=0.0, prediction_error=0.0, novelty=0.0, threat=0.0, energy=1.0)
    args.update(kw)
    for _ in range(steps):
        sys.update(args['reward'], args['prediction_error'], args['novelty'], args['threat'],
                   args['energy'], flags)


# =========================================================== neuromodulation
def test_controls_keys_and_neutral_without_flags():
    s = _system()
    _assert_neutral(s.controls(frozenset()))
    _assert_neutral(s.controls(None))
    _assert_neutral(s.controls({'receptor_heterogeneity', 'stdp'}))
    np.testing.assert_array_equal(s.per_unit_gain(frozenset()), np.ones(12))


def test_update_is_noop_without_flags():
    s = _system()
    before = json.dumps(s.export_state(), sort_keys=True)
    _drive(s, 20, frozenset(), reward=1.0, prediction_error=1.0, novelty=1.0, threat=1.0, energy=0.0)
    assert json.dumps(s.export_state(), sort_keys=True) == before
    _assert_neutral(s.controls(BOTH))  # fresh state is at baseline -> neutral even when on


def test_fresh_system_is_neutral_with_flags_on():
    s = _system()
    c = s.controls(BOTH)
    for k in CONTROL_KEYS:
        if k == 'sensory_weight':
            np.testing.assert_allclose(c[k], 1.0)
        elif k == 'threshold_shift':
            assert c[k] == pytest.approx(0.0)
        else:
            assert c[k] == pytest.approx(1.0)


def test_levels_bounded_and_decay_to_baseline():
    s = _system()
    _drive(s, 50, BOTH, reward=1.0, prediction_error=1.0, novelty=1.0, threat=1.0, energy=0.0)
    assert np.all(s.levels >= 0) and np.all(s.levels <= 1)
    assert not np.allclose(s.levels, BASELINES)
    _drive(s, 600, BOTH)  # neutral inputs
    np.testing.assert_allclose(s.levels, BASELINES, atol=0.03)


def test_dopamine_tracks_reward_and_rpe():
    s = _system()
    _drive(s, 5, MOD, reward=1.0)
    assert s.levels[0] > BASELINES[0] + 0.1
    s2 = _system()
    _drive(s2, 5, MOD, reward=-1.0)
    assert s2.levels[0] < BASELINES[0] - 0.1
    # expected reward reduces RPE-driven DA over time
    s3 = _system()
    _drive(s3, 3, MOD, reward=1.0)
    early = s3.levels[0]
    _drive(s3, 200, MOD, reward=1.0)
    assert s3.levels[0] < early + 0.2


def test_each_modulator_driver():
    names = list(NEUROMODULATORS)

    def lvl(name, **kw):
        s = _system()
        _drive(s, 60, MOD, **kw)
        return s.levels[names.index(name)]

    base = lambda name: lvl(name)
    assert lvl('acetylcholine', prediction_error=0.9) > base('acetylcholine') + 0.1
    assert lvl('acetylcholine', novelty=0.9) > base('acetylcholine') + 0.05
    assert lvl('noradrenaline', threat=0.9) > base('noradrenaline') + 0.1
    assert lvl('octopamine', energy=0.0) > base('octopamine') + 0.05
    assert lvl('serotonin', threat=0.9) < base('serotonin') - 0.05
    s = _system()
    _drive(s, 400, MOD, threat=0.9, prediction_error=0.9)
    assert s.levels[names.index('endocannabinoid')] > BASELINES[5] + 0.1


def test_affect_appraisal():
    s = _system()
    _drive(s, 80, AFF, reward=1.0)
    assert s.affect.valence > 0.5
    s = _system()
    _drive(s, 80, AFF, novelty=1.0, threat=0.0)
    cur_safe = s.affect.curiosity
    s = _system()
    _drive(s, 80, AFF, novelty=1.0, threat=1.0)
    assert cur_safe > s.affect.curiosity + 0.3
    s = _system()
    _drive(s, 80, AFF, threat=1.0, energy=0.0, prediction_error=1.0)
    assert s.affect.stress > 0.7 and s.affect.arousal > 0.5
    s = _system()
    _drive(s, 80, AFF, energy=0.1)
    assert s.affect.motivation > AFFECT_BASELINE['motivation']
    s = _system()
    s.update(1.0, 0, 0, 0, 1.0, AFF)
    assert s.affect.rpe == pytest.approx(1.0)
    assert s.affect.reward_prediction_error == s.affect.rpe
    s.update(1.0, 0, 0, 0, 1.0, AFF)
    assert s.affect.rpe < 1.0  # expectation has risen


def test_motivation_scales_with_dopamine():
    a, b = _system(), _system()
    _drive(a, 120, BOTH, energy=0.2, reward=0.8)
    _drive(b, 120, AFF, energy=0.2, reward=0.8)
    assert a.affect.motivation > b.affect.motivation


def test_endocannabinoid_buffers_stress():
    a, b = _system(), _system()
    _drive(a, 600, BOTH, threat=0.8, prediction_error=0.8)
    _drive(b, 600, AFF, threat=0.8, prediction_error=0.8)
    assert a.levels[5] > BASELINES[5] + 0.1
    assert a.affect.stress < b.affect.stress


def test_affect_bounds_under_extremes():
    s = _system()
    rng = np.random.default_rng(0)
    for _ in range(300):
        s.update(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-2, 3), rng.uniform(-2, 3),
                 rng.uniform(-2, 3), HET, unit_activity=rng.uniform(-3, 3, 12))
        for name in AFFECT_VARIABLES:
            v = getattr(s.affect, name)
            lo = -1 if name in ('valence', 'reward_prediction_error') else 0
            assert lo <= v <= 1 and np.isfinite(v)
        assert np.all((s.levels >= 0) & (s.levels <= 1))
        assert np.all(np.isfinite(s.traces)) and np.all(s.traces >= 0) and np.all(s.traces <= 1.0)
    c = s.controls(HET)
    for k in CONTROL_KEYS:
        assert np.all(np.isfinite(c[k]))


def test_rejects_non_finite_inputs():
    s = _system()
    with pytest.raises(ValueError):
        s.update(float('nan'), 0, 0, 0, 1, BOTH)
    with pytest.raises(ValueError):
        s.update(0, 0, 0, 0, float('inf'), BOTH)
    with pytest.raises(ValueError):
        s.update(0, 0, 0, 0, 1, BOTH, unit_activity=np.ones(3))


def test_control_directions():
    # curiosity -> exploration
    s = _system()
    _drive(s, 80, AFF, novelty=1.0)
    assert s.controls(AFF)['exploration'] > 1.05
    # stress -> attention & threshold_shift
    s = _system()
    _drive(s, 80, AFF, threat=1.0, energy=0.0)
    c = s.controls(AFF)
    assert c['attention'] > 1.1 and c['threshold_shift'] > 0.1
    # motivation -> persistence
    s = _system()
    _drive(s, 120, AFF, energy=0.0)
    assert s.controls(AFF)['persistence'] > 1.0
    # valence -> reward_scale
    s = _system()
    _drive(s, 80, AFF, reward=1.0)
    assert s.controls(AFF)['reward_scale'] > 1.2
    s = _system()
    _drive(s, 80, AFF, reward=-1.0)
    assert s.controls(AFF)['reward_scale'] < 0.8
    # arousal -> gain up, decision temperature down
    s = _system()
    _drive(s, 80, AFF, prediction_error=1.0, threat=1.0)
    c = s.controls(AFF)
    assert c['gain'] > 1.1 and c['decision_temperature'] < 0.9
    # ACh -> prediction precision & exteroceptive sensory weight
    s = _system()
    _drive(s, 40, MOD, prediction_error=0.8, novelty=0.8)
    c = s.controls(MOD)
    assert c['prediction_precision'] > 1.1
    sw = c['sensory_weight']
    assert sw[SENSORY_CHANNEL_IDS.index('visual_luminance')] > 1.05
    assert sw[SENSORY_CHANNEL_IDS.index('task_context')] < sw[SENSORY_CHANNEL_IDS.index('visual_luminance')]
    # DA -> learning rate / plasticity
    s = _system()
    _drive(s, 5, MOD, reward=1.0)
    c = s.controls(MOD)
    assert c['learning_rate_scale'] > 1.1 and c['plasticity_scale'] > 1.1
    s = _system()
    _drive(s, 5, MOD, reward=-1.0)
    assert s.controls(MOD)['learning_rate_scale'] < 0.9
    # 5-HT -> consolidation
    s = _system()
    _drive(s, 200, MOD, threat=0.0, prediction_error=0.0)
    calm = s.controls(MOD)['consolidation_scale']
    s = _system()
    _drive(s, 200, MOD, threat=1.0, prediction_error=1.0)
    assert calm > s.controls(MOD)['consolidation_scale'] + 0.1


def test_controls_bounds():
    s = _system()
    _drive(s, 300, HET, reward=1.0, prediction_error=1.0, novelty=1.0, threat=1.0, energy=0.0)
    c = s.controls(HET)
    for k in CONTROL_KEYS:
        v = np.asarray(c[k])
        if k == 'threshold_shift':
            assert np.all((v >= -0.5) & (v <= 0.5))
        else:
            assert np.all((v >= 0.25) & (v <= 2.0))


def test_receptor_heterogeneity_matrix():
    s = _system(n=40)
    np.testing.assert_array_equal(s.receptor_matrix(BOTH), np.ones((40, 6)))
    r = s.receptor_matrix(HET)
    assert r.shape == (40, 6)
    assert np.all((r >= 0.5) & (r <= 1.5)) and r.std() > 0.1
    # deterministic by seed
    np.testing.assert_array_equal(r, _system(n=40).receptor_matrix(HET))
    assert not np.array_equal(r, _system(n=40, seed=4).receptor_matrix(HET))


def test_per_unit_gain_heterogeneity_and_traces():
    homo, het = _system(n=20), _system(n=20)
    _drive(homo, 60, BOTH, threat=0.9, reward=0.5)
    _drive(het, 60, HET, threat=0.9, reward=0.5)
    gh = homo.per_unit_gain(BOTH)
    gx = het.per_unit_gain(HET)
    assert gh.shape == (20,) and np.ptp(gh) < 1e-12 and not np.allclose(gh, 1.0)
    assert np.ptp(gx) > 1e-3
    assert np.all((gx >= 0.5) & (gx <= 1.5))
    # traces differ per unit only under heterogeneity
    assert np.ptp(homo.traces[:, 3]) < 1e-12 and np.ptp(het.traces[:, 3]) > 1e-3
    # units with denser NE receptors see a larger NE trace deviation
    r = het.receptor_matrix(HET)[:, 3]
    dev = het.traces[:, 3] - BASELINES[3]
    assert np.corrcoef(r, dev)[0, 1] > 0.99


def test_receptor_heterogeneity_neutral_at_rest():
    # regression: a fresh / resting system must give exactly unit per-unit gain,
    # with or without receptor heterogeneity (no artefact from initial traces)
    s = _system(n=30)
    np.testing.assert_array_equal(s.per_unit_gain(HET), np.ones(30))
    mod_het = MOD | {'receptor_heterogeneity'}
    _drive(s, 50, mod_het)  # neutral inputs keep levels at baseline
    np.testing.assert_allclose(s.per_unit_gain(HET), 1.0, atol=1e-12)
    _assert_neutral_close(s.controls(mod_het))
    # toggling heterogeneity on a resting system does not change anything either
    np.testing.assert_allclose(s.per_unit_gain(BOTH), s.per_unit_gain(HET))
    # and the flag really matters once modulation moves
    a, b = _system(n=30), _system(n=30)
    _drive(a, 30, HET, threat=1.0, reward=1.0)
    _drive(b, 30, BOTH, threat=1.0, reward=1.0)
    assert not np.allclose(a.per_unit_gain(HET), b.per_unit_gain(BOTH))


def _assert_neutral_close(c: dict) -> None:
    for k in CONTROL_KEYS:
        target = 0.0 if k == 'threshold_shift' else 1.0
        np.testing.assert_allclose(c[k], target, atol=1e-12)


def test_constructor_rejects_bad_graph():
    with pytest.raises(ValueError):
        NeuromodulatorSystem(4, seed=0, src=np.array([0, 9]), dst=np.array([1, 2]))
    with pytest.raises(ValueError):
        NeuromodulatorSystem(4, seed=0, src=np.array([0]), dst=np.array([1, 2]))
    with pytest.raises(ValueError):
        NeuromodulatorSystem(4, seed=0, signs=np.ones(3))


def test_unit_activity_drives_local_ecb_traces():
    s = _system(n=6)
    act = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    for _ in range(200):
        s.update(0.0, 0.8, 0.0, 0.8, 1.0, MOD, unit_activity=act)
    assert s.traces[3:, 5].mean() > s.traces[:3, 5].mean() + 0.05
    g = s.per_unit_gain(MOD)
    assert g[3:].mean() < g[:3].mean()


def test_neuromod_determinism_and_roundtrip():
    a, b = _system(), _system()
    for s in (a, b):
        s.update(0.5, 0.2, 0.4, 0.1, 0.7, HET, unit_activity=np.linspace(0, 1, 12))
    assert a.export_state() == b.export_state()
    state = a.export_state()
    json.dumps(state)
    assert set(state) == {'schema', 'n', 'seed', 'levels', 'baselines', 'receptor_profile', 'traces',
                          'affect', 'reward_expectation', 'surprise_ema', 'sustained_activity', 'steps'}
    c = _system(seed=99)
    c.import_state(json.loads(json.dumps(state)))
    assert c.export_state()['traces'] == state['traces']
    for k in CONTROL_KEYS:
        np.testing.assert_allclose(c.controls(HET)[k], a.controls(HET)[k])
    np.testing.assert_allclose(c.per_unit_gain(HET), a.per_unit_gain(HET))


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(schema='x'),
    lambda s: s.update(n=5),
    lambda s: s['levels'].update(dopamine=2.0),
    lambda s: s['levels'].pop('serotonin'),
    lambda s: s.update(traces=[[0.0] * 6]),
    lambda s: s['receptor_profile'][0].__setitem__(0, 3.0),
    lambda s: s['affect'].update(valence=float('nan')),
    lambda s: s['affect'].update(stress=-0.1),
    lambda s: s.update(steps=-1),
    lambda s: s.update(reward_expectation='a'),
    lambda s: s.update(n=True),
    lambda s: s['traces'][0].__setitem__(0, 1.2),
])
def test_neuromod_import_rejects_and_does_not_mutate(mutate):
    s = _system()
    _drive(s, 5, BOTH, reward=0.4)
    good = s.export_state()
    bad = copy.deepcopy(good)
    mutate(bad)
    target = _system()
    before = target.export_state()
    with pytest.raises(ValueError):
        target.import_state(bad)
    assert target.export_state() == before


def test_infer_transmitters_never_modifies_sign():
    rng = np.random.default_rng(5)
    n = 30
    src = rng.integers(0, n, 120)
    dst = (src + rng.integers(1, n, 120)) % n
    w = rng.normal(0, 1, 120)
    sign = np.where(rng.random(n) < 0.2, -1, 1).astype(np.int8)
    sign_before = sign.copy()
    out = infer_transmitters(src, dst, w, sign)
    np.testing.assert_array_equal(sign, sign_before)
    assert out['labels'] == list(TRANSMITTER_LABELS) and len(TRANSMITTER_LABELS) == 6
    assert len(out['predictions']) == n and set(out['predictions']) <= set(TRANSMITTER_LABELS)
    p = out['probabilities']
    assert p.shape == (n, 6)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.all((out['confidence'] >= 1 / 6 - 1e-12) & (out['confidence'] <= 1))
    np.testing.assert_allclose(out['confidence'], p.max(axis=1))
    assert np.all((out['transmitter_gain'] >= 0.94) & (out['transmitter_gain'] <= 1.06))
    assert out['sign_consistent'].shape == (n,)
    # connectivity-only: flipping signs does not change predictions
    out2 = infer_transmitters(src, dst, w, -sign)
    assert out2['predictions'] == out['predictions']
    np.testing.assert_allclose(out2['probabilities'], p)


def test_infer_transmitters_structure_sensitive():
    # hub with high out-degree vs. sink with high in-degree
    n = 10
    src = np.array([0] * 9 + list(range(1, 10)))
    dst = np.array(list(range(1, 10)) + [5] * 9)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    out = infer_transmitters(src, dst, np.ones(src.size), np.ones(n, dtype=np.int8))
    assert out['predictions'][0] != out['predictions'][5]
    assert out['predictions'][0] in ('glutamate', 'dopamine', 'octopamine')
    assert out['predictions'][5] in ('gaba', 'serotonin')
    s = NeuromodulatorSystem(n, seed=1, signs=np.ones(n), src=src, dst=dst)
    assert s.infer_transmitters()['predictions'] == out['predictions']
    with pytest.raises(ValueError):
        infer_transmitters(np.array([0]), np.array([20]), np.ones(1), np.ones(n))


def _dense_reference(src, dst, n):
    adj = np.zeros((n, n))
    adj[src, dst] = 1.0
    outdeg = np.bincount(src, minlength=n).astype(float)
    rec = (adj * adj.T).sum(1) / np.maximum(1, outdeg)
    two = ((adj @ adj) > 0).sum(1).astype(float)
    und = np.clip(adj + adj.T, 0, 1)
    np.fill_diagonal(und, 0)
    du = und.sum(1)
    nb = (und @ du) / np.maximum(1, du)
    tri = np.diag(und @ und @ und) / 2
    poss = du * (du - 1) / 2
    cl = np.where(poss > 0, tri / np.maximum(poss, 1), 0)
    return rec, two, nb, cl


def test_structural_features_match_dense_reference():
    from synapse.neuroweave.neuromod import _structural_features
    rng = np.random.default_rng(11)
    n, m = 60, 400
    src, dst = rng.integers(0, n, m), rng.integers(0, n, m)  # includes self loops + duplicates
    f = _structural_features(src, dst, rng.normal(size=m), n)
    rec, two, nb, cl = _dense_reference(src, dst, n)
    np.testing.assert_allclose(f[:, 4], np.clip(rec, 0, 1))
    np.testing.assert_allclose(f[:, 5], two / two.max())
    np.testing.assert_allclose(f[:, 6], nb / nb.max())
    np.testing.assert_allclose(f[:, 7], cl)
    assert cl.max() > 0  # the reference graph actually has triangles


def test_infer_transmitters_scales_sparse():
    import time
    rng = np.random.default_rng(2)
    n = 2048
    m = 6 * n
    src = rng.integers(0, n, m)
    dst = (src + rng.integers(1, n, m)) % n
    t0 = time.perf_counter()
    out = infer_transmitters(src, dst, np.ones(m), np.ones(n, dtype=np.int8))
    assert time.perf_counter() - t0 < 5.0
    assert out['probabilities'].shape == (n, 6) and np.all(np.isfinite(out['probabilities']))


def test_neutral_controls_helper():
    _assert_neutral(neutral_controls())


# ================================================================= sensory
def _frame(h=48, w=64, rgb=(0, 0, 0)):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[...] = rgb
    return f


def test_visual_luminance_and_colour():
    bright = visual_features(_frame(rgb=(250, 250, 250)))
    dark = visual_features(_frame(rgb=(5, 5, 5)))
    assert bright['luminance'] > 0.9 and dark['luminance'] < 0.05
    red = visual_features(_frame(rgb=(255, 0, 0)))
    green = visual_features(_frame(rgb=(0, 255, 0)))
    blue = visual_features(_frame(rgb=(0, 0, 255)))
    assert red['red_green'] > 0.9 and green['red_green'] < -0.9
    assert blue['blue_yellow'] > 0.9
    yellow = visual_features(_frame(rgb=(255, 255, 0)))
    assert yellow['blue_yellow'] < -0.9
    # float frames in [0,1] supported
    assert visual_features(np.full((8, 8, 3), 1.0))['luminance'] == pytest.approx(1.0)
    keys = {'luminance', 'red_green', 'blue_yellow', 'edge_density', 'motion', 'change', 'centroid_x',
            'centroid_y', 'spatial_spread', 'object_regions', 'temporal_pattern'}
    assert set(bright) == keys


def test_visual_edges_centroid_regions():
    flat = visual_features(_frame(rgb=(128, 128, 128)))
    assert flat['edge_density'] == 0.0
    checker = np.zeros((64, 64, 3), dtype=np.uint8)
    checker[(np.indices((64, 64)).sum(axis=0) // 4) % 2 == 1] = 255
    assert visual_features(checker)['edge_density'] > 0.1
    right = _frame(64, 64)
    right[20:40, 48:60] = 255
    f = visual_features(right)
    assert f['centroid_x'] > 0.7 and 0.4 < f['centroid_y'] < 0.6
    assert f['object_regions'] == 1
    two = _frame(64, 64)
    two[4:16, 4:16] = 255
    two[44:60, 44:60] = 255
    assert visual_features(two)['object_regions'] == 2
    assert visual_features(_frame(rgb=(255, 255, 255)))['spatial_spread'] > f['spatial_spread']


def test_visual_motion_from_moving_square():
    a = _frame(64, 64)
    a[10:22, 10:22] = 255
    b = _frame(64, 64)
    b[10:22, 30:42] = 255
    static = visual_features(a, previous=a)
    moving = visual_features(b, previous=a)
    assert static['motion'] == 0.0 and static['temporal_pattern'] == 0.0
    assert moving['motion'] > 0.01 and moving['temporal_pattern'] > 0
    assert visual_features(b)['motion'] == 0.0  # no previous -> no motion
    hub = SensoryHub()
    hub.push_visual(a)
    f = hub.push_visual(b)
    assert f['motion'] > 0.01
    flash = visual_features(_frame(64, 64, (255, 255, 255)), previous=a)
    assert flash['change'] > 0.5


def test_visual_rejects_bad_frames():
    with pytest.raises(ValueError):
        visual_features(np.zeros((8, 8)))
    with pytest.raises(ValueError):
        visual_features(np.full((4, 4, 3), np.nan))


def _sine(freq, sr=22050, n=4096, amp=0.5):
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_audio_bands():
    low = audio_features(_sine(100), 22050)
    high = audio_features(_sine(5000), 22050)
    mid = audio_features(_sine(1000), 22050)
    assert low['low'] > 0.9 and low['low'] > low['mid'] and low['low'] > low['high']
    assert high['high'] > 0.9
    assert mid['mid'] > 0.9
    assert 0.2 < low['amplitude'] <= 1.0
    silent = audio_features(np.zeros(1024), 16000)
    assert silent['amplitude'] == 0.0 and silent['low'] == 0.0 and silent['onset'] == 0.0
    assert set(low) == {'amplitude', 'low', 'mid', 'high', 'change', 'onset', 'pattern'}


def test_audio_onset_pattern_and_change():
    burst = np.zeros(4096)
    burst[2048:] = _sine(440, n=2048)
    steady = _sine(440)
    assert audio_features(burst, 22050)['onset'] > audio_features(steady, 22050)['onset'] + 0.3
    pulses = np.zeros(4096)
    for k in range(0, 16, 4):
        pulses[k * 256:(k + 1) * 256] = _sine(440, n=256)
    assert audio_features(pulses, 22050)['pattern'] > 0.5
    hub = SensoryHub()
    hub.push_audio(_sine(200, amp=0.05), 22050)
    f = hub.push_audio(_sine(200, amp=0.5), 22050)
    assert f['change'] > 0.3
    with pytest.raises(ValueError):
        audio_features(np.array([np.nan] * 16), 1000)


def _spec(**modes):
    return NeuroWeaveSpecV1(sensory_modes=tuple(modes.items()))


def test_channel_modes():
    hub = SensoryHub(_spec(visual_luminance='disabled', audio_low='external', audio_mid='live',
                           visual_motion='live'))
    sim = hub.simulate(None, 0)
    assert sim.shape == (16,) and np.all(np.abs(sim) <= 1)
    hub.set_external('audio_low', 0.42)
    hub.push_live('audio_mid', 0.77)
    fr = hub.fuse()
    i = SENSORY_CHANNEL_IDS.index
    assert fr.value[i('visual_luminance')] == 0.0 and fr.reliability[i('visual_luminance')] == 0.0
    assert fr.provenance[i('visual_luminance')] == 'disabled'
    assert fr.value[i('audio_low')] == pytest.approx(0.42) and fr.provenance[i('audio_low')] == 'external'
    assert fr.value[i('audio_mid')] == pytest.approx(0.77) and fr.provenance[i('audio_mid')] == 'sensory'
    assert fr.value[i('olfactory')] == pytest.approx(sim[i('olfactory')])
    assert fr.provenance[i('olfactory')] == 'simulated'
    # live channel with no data yet -> missing (no cross-modal flag -> zero, reliability 0)
    assert fr.missing[i('visual_motion')] and fr.reliability[i('visual_motion')] == 0.0
    assert fr.provenance[i('visual_motion')] == 'missing'
    # simulated values never leak into live/external channels
    assert fr.value[i('visual_motion')] == 0.0


def test_live_push_routes_derived_scalars():
    hub = SensoryHub(_spec(**{c: 'live' for c in SENSORY_CHANNEL_IDS[:8]}))
    hub.push_visual(_frame(rgb=(255, 0, 0)))
    hub.push_audio(_sine(100), 22050)
    fr = hub.fuse()
    i = SENSORY_CHANNEL_IDS.index
    assert fr.value[i('visual_red_green')] > 0.9
    assert fr.value[i('audio_low')] > 0.9
    assert all(p == 'sensory' for p in fr.provenance[:8])


def test_no_raw_media_retained():
    hub = SensoryHub(_spec(visual_luminance='live'))
    frame = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    hub.visual_features(frame)
    hub.push_visual(frame)
    hub.push_visual(frame[::-1].copy())
    samples = _sine(300, n=8192)
    hub.push_audio(samples, 22050)
    hub.fuse()

    def arrays(obj, seen=None):
        seen = seen or set()
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from arrays(v, seen)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from arrays(v, seen)

    found = list(arrays(hub.__dict__))
    for arr in found:
        assert arr.shape != frame.shape and arr.shape != frame.shape[:2]
        assert arr.size < samples.size
        if arr.ndim == 2 and arr is hub._lum_map:
            assert arr.shape[0] <= 16 and arr.shape[1] <= 16
    assert hub._lum_map is not None and hub._lum_map.shape == (16, 16)
    state = hub.export_state()
    text = json.dumps(state)
    assert 'lum_map' not in text
    assert state['raw_media_stored'] is False
    assert hub.manifest()['raw_media_stored'] is False
    # visual_features (pure) does not touch stored map
    before = hub._lum_map.copy()
    hub.visual_features(frame)
    np.testing.assert_array_equal(before, hub._lum_map)


def test_staleness_decays_reliability_and_expires():
    hub = SensoryHub(_spec(audio_low='external'))
    i = SENSORY_CHANNEL_IDS.index('audio_low')
    hub.set_external('audio_low', 0.5)
    rel = []
    for _ in range(70):
        fr = hub.fuse()
        rel.append(fr.reliability[i])
    assert rel[0] > rel[5] > rel[20] > 0
    assert fr.age[i] > 60 or fr.missing[i]
    assert fr.missing[i] and fr.reliability[i] == 0.0  # beyond MAX_AGE -> missing
    hub.set_external('audio_low', 0.5)
    assert hub.fuse().reliability[i] > rel[20]


def test_reliability_from_residual_variance():
    hub = SensoryHub()
    rng = np.random.default_rng(1)
    for _ in range(200):
        raw = {c: 0.3 for c in SENSORY_CHANNEL_IDS}
        raw['audio_high'] = float(rng.uniform(-1, 1))
        fr = hub.fuse(raw)
    i = SENSORY_CHANNEL_IDS.index
    assert fr.reliability[i('audio_low')] > 0.9
    assert fr.reliability[i('audio_high')] < 0.2
    assert fr.prediction_error[i('audio_low')] < 1e-3
    assert 0 <= fr.uncertainty <= 1


def test_ar1_predictor_learns_alternating_and_constant():
    hub = SensoryHub()
    i = SENSORY_CHANNEL_IDS.index('thermal')
    errs = []
    for t in range(400):
        raw = {c: 0.0 for c in SENSORY_CHANNEL_IDS}
        raw['thermal'] = 0.6 if t % 2 else -0.6
        fr = hub.fuse(raw)
        errs.append(fr.prediction_error[i])
    assert np.mean(errs[-20:]) < 0.5 * np.mean(errs[:20])
    assert fr.predicted.shape == (16,)


def test_cross_modal_imputation_only_with_flag():
    flags = frozenset({'cross_modal_memory'})
    hub = SensoryHub()
    hub_off = SensoryHub()
    for t in range(600):
        v = hub.simulate(None, t)
        hub.fuse(None, flags)
        hub_off.simulate(None, t)
        hub_off.fuse(None)
    assert hub.assoc_updates > 0 and np.abs(hub.assoc).max() > 0
    assert hub_off.assoc_updates == 0 and np.abs(hub_off.assoc).max() == 0
    assert np.all(np.diag(hub.assoc) == 0) and np.abs(hub.assoc).max() <= 2.0
    errs_on, errs_base = [], []
    i = SENSORY_CHANNEL_IDS.index('visual_luminance')
    for t in range(600, 640):
        truth = hub.simulate(None, t)
        hub_off.simulate(None, t)
        fr = hub.fuse({'visual_luminance': None}, flags, learn=False)
        off = hub_off.fuse({'visual_luminance': None}, learn=False)
        assert fr.provenance[i] == 'predicted' and 'visual_luminance' in fr.imputations
        assert fr.missing[i] and 0 < fr.reliability[i] <= 0.5
        assert off.provenance[i] == 'missing' and off.value[i] == 0.0 and off.reliability[i] == 0.0
        errs_on.append(abs(fr.value[i] - truth[i]))
        errs_base.append(abs(hub.pred_mean[i] - truth[i]))
    assert np.mean(errs_on) < 0.6 * np.mean(errs_base)


def test_sensory_attention_weights():
    hub = SensoryHub()
    hub.simulate(None, 0)
    raw = {c: 0.5 for c in SENSORY_CHANNEL_IDS}
    a = hub.fuse(raw, learn=False)
    w = np.ones(16)
    w[0] = 2.0
    w[1] = 0.25
    b = hub.fuse(raw, sensory_weight=w, learn=False)
    assert b.weighted[0] == pytest.approx(2 * a.weighted[0])
    assert b.weighted[1] == pytest.approx(0.25 * a.weighted[1])
    np.testing.assert_allclose(a.weighted, a.value * a.reliability)
    with pytest.raises(ValueError):
        hub.fuse(raw, sensory_weight=np.ones(3))
    # neuromod controls plug in directly
    s = _system()
    _drive(s, 40, MOD, prediction_error=0.8)
    c = hub.fuse(raw, sensory_weight=s.controls(MOD)['sensory_weight'], learn=False)
    assert not np.allclose(c.weighted, a.weighted)


def test_simulate_deterministic_and_ecology_inputs():
    a, b = SensoryHub(_spec()), SensoryHub(_spec())
    np.testing.assert_array_equal(a.simulate(None, 17), b.simulate(None, 17))
    assert not np.array_equal(a.simulate(None, 17), a.simulate(None, 18))
    v = a.simulate({'energy': 0.11, 'temperature': -0.4}, 3)
    assert v[SENSORY_CHANNEL_IDS.index('interoception')] == pytest.approx(0.11)
    assert v[SENSORY_CHANNEL_IDS.index('thermal')] == pytest.approx(-0.4)
    vec = np.linspace(-1, 1, 16)
    np.testing.assert_allclose(a.simulate({'sensory': vec}, 0), vec)
    with pytest.raises(ValueError):
        a.simulate({'sensory': np.ones(3)}, 0)


def test_fuse_validation():
    hub = SensoryHub()
    with pytest.raises(ValueError):
        hub.fuse({'not_a_channel': 1.0})
    with pytest.raises(ValueError):
        hub.fuse({'thermal': float('nan')})
    with pytest.raises(ValueError):
        hub.set_external('thermal', float('inf'))
    with pytest.raises(ValueError):
        hub.set_mode('thermal', 'bogus')
    fr = hub.fuse({'thermal': 5.0})
    assert fr.value[SENSORY_CHANNEL_IDS.index('thermal')] == 1.0  # bounded
    assert isinstance(fr, SensoryFrame)
    json.dumps(fr.to_dict())


def test_non_numeric_inputs_raise_value_error():
    hub = SensoryHub(_spec(thermal='external', proprioception='live'))
    for bad in ('abc', [1.0, 2.0], {'x': 1}, True):
        with pytest.raises(ValueError):
            hub.fuse({'thermal': bad})
        with pytest.raises(ValueError):
            hub.set_external('thermal', bad)
        with pytest.raises(ValueError):
            hub.push_live('proprioception', bad)
        with pytest.raises(ValueError):
            hub.simulate({'energy': bad}, 0)
    with pytest.raises(ValueError):
        hub.simulate({'energy': float('nan')}, 0)
    with pytest.raises(ValueError):
        audio_features(_sine(100), 'fast')
    with pytest.raises(ValueError):
        audio_features(_sine(100), 0)
    # None in ecology scalars means "not provided"
    v = hub.simulate({'energy': None}, 0)
    np.testing.assert_array_equal(v, SensoryHub(_spec()).simulate(None, 0))


def test_cross_modal_lms_step_is_stable_for_large_inputs():
    # regression: plain LMS with lr * ||x||^2 > 2 overshoots; the normalised step
    # must reduce the reconstruction error of the very sample it learned from
    flags = {'cross_modal_memory'}
    hub = SensoryHub()
    hub.pred_mean[:] = -1.0
    raw = {c: 1.0 for c in SENSORY_CHANNEL_IDS}
    sub = np.full(16, 2.0)  # centred inputs, ||x||^2 = 64
    before = np.abs(sub - hub.assoc @ sub).mean()
    hub.fuse(raw, flags)
    after = np.abs(sub - hub.assoc @ sub).mean()
    assert after < 0.25 * before
    assert np.abs(hub.assoc).max() < ASSOC_BOUND_T
    # long run on large alternating inputs stays bounded and does not saturate
    hub = SensoryHub()
    for t in range(300):
        sgn = 1.0 if t % 2 else -1.0
        hub.fuse({c: sgn * (1 if k % 2 else -1) for k, c in enumerate(SENSORY_CHANNEL_IDS)}, flags)
    assert np.all(np.isfinite(hub.assoc)) and np.abs(hub.assoc).max() < ASSOC_BOUND_T


ASSOC_BOUND_T = 2.0


def test_manifest():
    hub = SensoryHub(_spec(audio_low='live'))
    m = hub.manifest()
    assert m['raw_media_stored'] is False and m['motion_map_exported'] is False
    assert len(m['channels']) == 16
    assert m['channels'][SENSORY_CHANNEL_IDS.index('audio_low')]['mode'] == 'live'
    assert m['evidence_tier'] == 'synthetic_abstraction'
    json.dumps(m)


def test_sensory_roundtrip_and_determinism():
    flags = {'cross_modal_memory'}

    def run(hub):
        for t in range(50):
            hub.simulate(None, t)
            hub.set_external('audio_low', 0.1 * (t % 3))
            hub.fuse({'visual_motion': None} if t % 7 == 0 else None, flags)

    a = SensoryHub(_spec(audio_low='external', audio_mid='live'))
    b = SensoryHub(_spec(audio_low='external', audio_mid='live'))
    run(a)
    run(b)
    a.push_audio(_sine(300), 22050)
    b.push_audio(_sine(300), 22050)
    assert a.export_state() == b.export_state()
    state = json.loads(json.dumps(a.export_state()))
    c = SensoryHub()
    c.import_state(state)
    assert c.export_state() == a.export_state()
    fa = a.fuse(None, flags)
    fc = c.fuse(None, flags)
    np.testing.assert_allclose(fa.value, fc.value)
    np.testing.assert_allclose(fa.reliability, fc.reliability)
    assert fa.provenance == fc.provenance


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(schema='nope'),
    lambda s: s.update(step=-2),
    lambda s: s['modes'].update(thermal='bogus'),
    lambda s: s['modes'].pop('thermal'),
    lambda s: s.update(pred_mean=[0.0] * 3),
    lambda s: s['assoc'][0].__setitem__(0, 0.5),
    lambda s: s['assoc'][0].__setitem__(1, 9.0),
    lambda s: s['external'].update(thermal=[float('nan'), 0]),
    lambda s: s['external'].update(bogus=[0.0, 0]),
    lambda s: s.update(has_last=[1] * 16),
    lambda s: s.update(resid_var=[-1.0] * 16),
    lambda s: s.update(last_visual={'raw_pixels': 0.1}),
    lambda s: s.update(last_audio={'amplitude': 1e9}),
])
def test_sensory_import_rejects_without_mutation(mutate):
    src = SensoryHub()
    for t in range(10):
        src.simulate(None, t)
        src.fuse(None, {'cross_modal_memory'})
    bad = copy.deepcopy(src.export_state())
    mutate(bad)
    target = SensoryHub()
    target.simulate(None, 0)
    target.fuse()
    before = target.export_state()
    with pytest.raises(ValueError):
        target.import_state(bad)
    assert target.export_state() == before
