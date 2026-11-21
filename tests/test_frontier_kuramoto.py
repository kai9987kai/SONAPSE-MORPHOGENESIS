"""Frontier module ``kuramoto_binding``: AKOrN oscillator core, engine extension, live observer, binding benchmark."""
import copy
import itertools
import json
import math

import numpy as np
import pytest

from synapse.frontier import kuramoto as K
from synapse.frontier.harness import VERDICTS, verify_receipt
from synapse.frontier.spec import EVIDENCE_TIER, FrontierSpecV1, MODULE_IDS, OBSERVER_MODULE_IDS
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.experiments import make_spec
from synapse.neuroweave.spec import REGIONS
from synapse.neuroweave.tasks import make_stream


def _unit(rng, c, n):
    v = rng.standard_normal((c, n))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _engine(seed=0, neurons=24, mechanisms=()):
    return NeuroWeaveEngine(make_spec(mechanisms, seed=seed, neurons=neurons, topology='clustered', steps=16))


def _drive(engine, steps, seed=0, learn=True):
    stream = make_stream(seed, steps)
    outs = []
    for i in range(steps):
        outs.append(engine.step(stream.inputs[i], stream.sensory[i], reward=0.2 if i % 2 else -0.1,
                                learn=learn)['output'].copy())
    return outs


# ====================================================================== registry
def test_module_is_registered_with_both_experiments():
    assert K.MODULE_ID == 'kuramoto_binding' and K.MODULE_ID in MODULE_IDS and K.MODULE_ID in OBSERVER_MODULE_IDS
    assert list(K.EXPERIMENTS) == ['kuramoto_binding', 'kuramoto_binding_benchmark']
    assert K.EXPERIMENTS['kuramoto_binding'] is K.run_experiment
    assert K.EXPERIMENTS['kuramoto_binding_benchmark'] is K.run_binding_benchmark
    assert all(k == K.MODULE_ID or k.startswith(K.MODULE_ID + '_') for k in K.EXPERIMENTS)
    assert K.KuramotoExtension.id == K.MODULE_ID and K.KuramotoObserver.module_id == K.MODULE_ID


# ====================================================================== AKOrN core
@pytest.mark.parametrize('n', [2, 4, 6])
def test_projection_is_tangent_and_the_update_keeps_unit_norm(n):
    rng = np.random.default_rng(n)
    x, y = _unit(rng, 9, n), rng.standard_normal((9, n))
    p = K.project_tangent(x, y)
    assert np.max(np.abs(np.sum(p * x, axis=1))) < 1e-12                  # tangent to the sphere at x
    assert np.allclose(K.project_tangent(x, p), p, atol=1e-12)             # idempotent
    assert np.allclose(K.project_tangent(x, x), 0.0, atol=1e-12)           # the radial direction is removed
    j = rng.standard_normal((9, 9))
    c = rng.standard_normal((9, n))
    om = K.natural_frequency_matrices(rng.standard_normal((9, n // 2)), n)
    before = (x.copy(), j.copy(), c.copy())
    out = K.akorn_step(x, j, c, om, gamma=0.3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)
    assert all(np.array_equal(a, b) for a, b in zip(before, (x, j, c), strict=True))   # inputs are not modified
    # the update is Pi(x + gamma (Omega x + Proj_x(c + J x))), written out by hand
    manual = x + 0.3 * (np.einsum('cij,cj->ci', om, x) + K.project_tangent(x, c + j @ x))
    assert np.allclose(out, manual / np.linalg.norm(manual, axis=1, keepdims=True), atol=1e-12)
    many = K.akorn_run(x, j, c, om, gamma=0.3, iterations=50)
    assert np.allclose(np.linalg.norm(many, axis=1), 1.0, atol=1e-12)


def test_block_coupling_equals_scalar_coupling_times_identity():
    rng = np.random.default_rng(5)
    js, x, c = rng.standard_normal((5, 5)), _unit(rng, 5, 4), rng.standard_normal((5, 4))
    a = K.akorn_step(x, js, c, gamma=0.2)
    b = K.akorn_step(x, np.kron(js, np.eye(4)), c, gamma=0.2)
    assert np.allclose(a, b, atol=1e-14)


def test_free_oscillators_rotate_at_their_natural_frequency():
    w = np.array([0.5, -1.0, 2.0, 0.0])
    om = K.natural_frequency_matrices(w, 2)
    assert np.allclose(om, -np.swapaxes(om, 1, 2))                         # antisymmetric
    x = _unit(np.random.default_rng(1), 4, 2)
    y = K.akorn_run(x, None, None, om, gamma=0.1, iterations=7)
    advance = np.angle(np.exp(1j * (K.phases(y) - K.phases(x))))
    assert np.allclose(advance, 7 * np.arctan(0.1 * w), atol=1e-12)       # exact discrete rotation per step
    # small steps approach the continuous flow dtheta/dt = omega (here for unit time)
    z = K.akorn_run(x, None, None, om, gamma=1e-3, iterations=1000)
    assert np.allclose(np.angle(np.exp(1j * (K.phases(z) - K.phases(x)))), w, atol=1e-5)
    # N = 4: every rotation plane advances at its own frequency
    w4 = np.array([[0.5, 2.0], [1.0, -1.5]])
    x4 = _unit(np.random.default_rng(2), 2, 4)
    y4 = K.akorn_run(x4, None, None, K.natural_frequency_matrices(w4, 4), gamma=0.1, iterations=5)
    for p in range(2):
        d = np.arctan2(y4[:, 2 * p + 1], y4[:, 2 * p]) - np.arctan2(x4[:, 2 * p + 1], x4[:, 2 * p])
        assert np.allclose(np.angle(np.exp(1j * d)), 5 * np.arctan(0.1 * w4[:, p]), atol=1e-12)


def test_strong_uniform_coupling_synchronises_and_repulsion_desynchronises():
    c = 32
    uniform = (np.ones((c, c)) - np.eye(c)) / (c - 1)
    rng = np.random.default_rng(3)
    x0 = _unit(rng, c, 2)
    same = K.natural_frequency_matrices(np.full(c, 1.0), 2)
    spread = K.natural_frequency_matrices(1.0 + 0.1 * rng.standard_normal(c), 2)
    assert K.order_parameter(x0) < 0.2
    assert K.order_parameter(K.akorn_run(x0, 2.0 * uniform, None, same, iterations=400)) > 1.0 - 1e-9
    assert K.order_parameter(K.akorn_run(x0, 2.0 * uniform, None, spread, iterations=400)) > 0.99
    assert K.order_parameter(K.akorn_run(x0, None, None, spread, iterations=400)) < 0.3
    assert K.order_parameter(K.akorn_run(x0, -2.0 * uniform, None, same, iterations=400)) < 1e-6
    x4 = _unit(rng, c, 4)
    om4 = K.natural_frequency_matrices(1.0 + 0.1 * rng.standard_normal((c, 2)), 4)
    assert K.order_parameter(K.akorn_run(x4, 2.0 * uniform, None, om4, iterations=400)) > 0.99


def test_energy_is_non_increasing_under_the_lyapunov_conditions():
    # paper Eq. 3: symmetric scalar couplings, shared Omega (here 0) and Omega c = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((12, 12))
        j = (a + a.T) / 2.0
        np.fill_diagonal(j, 0.0)
        j /= np.linalg.norm(j, 2)
        c = 0.5 * rng.standard_normal((12, 4))
        x = _unit(rng, 12, 4)
        energies = [K.kuramoto_energy(x, j, c)]
        for _ in range(200):
            x = K.akorn_step(x, j, c, gamma=0.1)
            energies.append(K.kuramoto_energy(x, j, c))
        assert np.all(np.diff(energies) <= 1e-12)
        assert energies[-1] < energies[0] - 0.1


def test_order_parameter_group_coherence_phases_and_plv():
    e = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    assert K.order_parameter(e[:2]) == 1.0 and K.order_parameter(e[1:3]) == pytest.approx(0.0, abs=1e-15)
    coh = K.group_coherence(e, [0, 0, 1, 1], n_groups=3)
    assert coh[0] == 1.0 and coh[1] == pytest.approx(math.sqrt(0.5)) and coh[2] == 0.0
    assert np.allclose(K.phases(e), [0.0, 0.0, np.pi, np.pi / 2])
    t = np.linspace(0.0, 20.0, 400)
    assert K.phase_locking_value(2.0 * t + 0.7, 2.0 * t) == pytest.approx(1.0)
    drift = np.linspace(0.0, 4 * np.pi, 400, endpoint=False)
    assert K.phase_locking_value(drift, np.zeros_like(drift)) < 1e-12
    with pytest.raises(ValueError):
        K.phase_locking_value([0.0, 1.0], [0.0])
    with pytest.raises(ValueError):
        K.group_coherence(e, [0, 0, -1, 1])


def test_phase_cluster_count_on_known_layouts():
    assert K.count_phase_clusters(np.full(20, 0.3)) == 1
    assert K.count_phase_clusters(np.r_[np.full(10, 0.3), np.full(10, 0.3 + np.pi)]) == 2
    assert K.count_phase_clusters(np.r_[np.full(10, 0.1), np.full(10, 2.2), np.full(10, 4.3)]) == 3
    assert K.count_phase_clusters(np.linspace(0.0, 2 * np.pi, 200, endpoint=False)) == 0   # incoherent
    assert K.count_phase_clusters(np.linspace(0.0, 2 * np.pi, 40, endpoint=False), bins=4) == 0   # no gap
    assert K.count_phase_clusters(np.r_[np.full(6, -0.01), np.full(6, 0.01)]) == 1   # wraps around 0
    assert K.count_phase_clusters([]) == 0
    with pytest.raises(ValueError):
        K.count_phase_clusters([0.0, np.nan])
    with pytest.raises(ValueError):
        K.count_phase_clusters([0.0], bins=1)


def test_shanahan_indices_match_the_reference_values_in_the_paper():
    half = np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=float)
    assert K.shanahan_indices(half)['chimera_index'] == pytest.approx(2.0 / 7.0)       # "perfect" chimera state
    assert K.shanahan_indices(np.vstack([half, np.ones((1, 8))]))['chimera_index'] == pytest.approx(1.0 / 7.0)
    uniform = np.linspace(0.0, 1.0, 4001)[:, None]
    assert K.shanahan_indices(uniform)['metastability_index'] == pytest.approx(1.0 / 12.0, abs=1e-4)
    still = np.tile([[0.9, 0.2, 0.5]], (6, 1))
    assert K.shanahan_indices(still) == {'metastability_index': pytest.approx(0.0, abs=1e-15),
                                         'chimera_index': pytest.approx(0.37 / 3.0)}
    assert K.shanahan_indices([[0.4, 0.6]])['metastability_index'] == 0.0           # one sample: no variance
    assert K.shanahan_indices([[0.4], [0.6]])['chimera_index'] == 0.0
    with pytest.raises(ValueError):
        K.shanahan_indices([0.1, 0.2])


# ====================================================================== clustering scores
def _ari_by_pair_counting(a, b):
    tp = fp = fn = tn = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        same_a, same_b = a[i] == a[j], b[i] == b[j]
        tp += same_a and same_b
        fn += same_a and not same_b
        fp += same_b and not same_a
        tn += not same_a and not same_b
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def test_adjusted_rand_index_on_hand_cases_and_against_pair_counting():
    assert K.adjusted_rand_index([0, 0, 1, 1, 2], [5, 5, 3, 3, 9]) == 1.0                   # relabelled copy
    assert K.adjusted_rand_index([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]) == pytest.approx(24.0 / 99.0)
    assert K.adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    assert K.adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(4.0 / 7.0)
    assert K.adjusted_rand_index([0, 0, 0], [1, 1, 1]) == 1.0 and K.adjusted_rand_index([0, 1, 2], [2, 0, 1]) == 1.0
    assert K.adjusted_rand_index([0, 0, 0, 0], [0, 1, 2, 3]) == 0.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.integers(0, 4, 30), rng.integers(0, 5, 30)
        assert K.adjusted_rand_index(a, b) == pytest.approx(_ari_by_pair_counting(a, b), abs=1e-12)
        assert K.adjusted_rand_index(a, b) == pytest.approx(K.adjusted_rand_index(b, a), abs=1e-12)
    with pytest.raises(ValueError):
        K.adjusted_rand_index([0, 1], [0, 1, 1])


def test_kmeans_recovers_separated_clusters_deterministically():
    rng = np.random.default_rng(4)
    truth = np.repeat(np.arange(3), 12)
    angles = np.array([0.0, 2.1, 4.2])[truth] + 0.05 * rng.standard_normal(truth.size)
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    lab = K.kmeans_labels(pts, 3, seed=1)
    assert K.adjusted_rand_index(truth, lab) == 1.0
    assert np.array_equal(lab, K.kmeans_labels(pts, 3, seed=1))
    assert np.array_equal(K.kmeans_labels(pts, 1, seed=1), np.zeros(truth.size, dtype=np.int64))
    with pytest.raises(ValueError):
        K.kmeans_labels(pts, 37)
    with pytest.raises(ValueError):
        K.kmeans_labels(pts[:, 0], 2)


# ====================================================================== binding task and benchmark
def test_binding_task_and_correlation_coupling():
    task = K.make_binding_task(3, groups=3, per_group=4, samples=200)
    assert task['features'].shape == (12, 200) and np.array_equal(task['labels'], np.repeat(np.arange(3), 4))
    assert task['digest'] == K.make_binding_task(3, groups=3, per_group=4, samples=200)['digest']
    assert task['digest'] != K.make_binding_task(4, groups=3, per_group=4, samples=200)['digest']
    clean = K.make_binding_task(3, groups=3, per_group=4, samples=200, noise=0.0)
    corr = np.corrcoef(clean['features'])
    assert np.allclose(corr[:4, :4], 1.0)                                    # one latent per group
    j = K.correlation_coupling(task['features'], scale=2.0)
    assert np.allclose(j, j.T) and np.all(np.diag(j) == 0.0)
    assert np.abs(j).sum(axis=1).max() == pytest.approx(2.0)
    same = task['labels'][:, None] == task['labels'][None, :]
    off = ~np.eye(12, dtype=bool)
    assert j[same & off].mean() > 0.0 > j[~same].mean()
    shuffled = K.shuffle_coupling(j, 9)
    iu = np.triu_indices(12, 1)
    assert np.allclose(shuffled, shuffled.T) and np.all(np.diag(shuffled) == 0.0)
    assert np.array_equal(np.sort(shuffled[iu]), np.sort(j[iu])) and not np.array_equal(shuffled, j)
    with pytest.raises(ValueError):
        K.shuffle_coupling(np.triu(np.ones((3, 3))), 0)
    with pytest.raises(ValueError):
        K.correlation_coupling(np.ones((1, 5)))


def test_correlation_coupled_oscillators_bind_the_latent_groups():
    receipt = K.run_binding_benchmark(seeds=(0, 1, 2), groups=3, per_group=6, samples=256)
    arms = receipt['arms']
    assert all(a['ari'] >= 0.9 for a in arms['candidate']['per_seed'])
    assert all(a['within_group_R'] > 0.9 for a in arms['candidate']['per_seed'])
    for control in ('shuffled_control', 'zero_control'):
        assert all(abs(a['ari']) < 0.35 for a in arms[control]['per_seed'])
    assert receipt['verdict'] == 'improved'
    assert receipt['verdict_basis'] == 'candidate_vs_shuffled_coupling_control'
    curve = arms['candidate']['summary']['ari_by_iteration_mean']
    assert curve[0] < 0.2 < curve[-1]                                        # binding emerges over iterations


def test_binding_benchmark_quick_receipt_is_sealed_and_deterministic():
    first = K.run_binding_benchmark(quick=True)
    second = K.run_binding_benchmark(quick=True)
    assert verify_receipt(first) and first['digest'] == second['digest']
    assert first['kind'] == 'binding_benchmark' and first['experiment_id'] == 'kuramoto_binding_benchmark'
    assert first['evidence_tier'] == EVIDENCE_TIER and first['verdict'] in VERDICTS
    assert set(first['arms']) == {'candidate', 'shuffled_control', 'zero_control', 'reference_correlation_kmeans'}
    assert set(first['comparisons']) == {'candidate_vs_shuffled_control', 'candidate_vs_zero_control',
                                         'candidate_vs_reference_correlation_kmeans'}
    assert first['config']['quick'] is True and first['config']['groups'] == 3 and first['config']['seeds'] == [0, 1, 2]
    assert first['measured_wallclock_s'] > 0.0 and first['budget']['trainable_parameters'] == 0
    for arm in ('candidate', 'shuffled_control', 'zero_control'):
        for row in first['arms'][arm]['per_seed']:
            assert -1.0 <= row['ari'] <= 1.0 and 0.0 <= row['global_R'] <= 1.0 and -1.0 <= row['max_group_cosine'] <= 1.0
    tampered = copy.deepcopy(first)
    tampered['arms']['candidate']['summary']['ari_mean'] = 1.0 - first['arms']['candidate']['summary']['ari_mean']
    assert not verify_receipt(tampered)
    assert K.run_binding_benchmark(quick=True, seeds=(3, 4))['digest'] != first['digest']


@pytest.mark.parametrize('kwargs', [{'quick': 'yes'}, {'seeds': ()}, {'seeds': (1, 1)}, {'seeds': (1.5,)},
                                    {'surprise': 1}, {'dim': 3}, {'groups': 1}, {'gamma': 0.0}, {'noise': -1.0}])
def test_binding_benchmark_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        K.run_binding_benchmark(**kwargs)


# ====================================================================== engine extension
def _assert_holds_no_engine(ext):
    for value in vars(ext).values():
        items = value if isinstance(value, tuple) else (value,)
        assert all(isinstance(v, (np.ndarray, int, float, str, bool, np.integer, np.floating)) for v in items)


def test_extension_copies_region_coupling_from_the_synapse_table():
    engine = _engine()
    ext = K.KuramotoExtension(0)
    engine.attach_extension(ext)
    _assert_holds_no_engine(ext)
    b = engine.brain
    region = np.asarray(b.cells.region)[:engine.n]
    j = np.zeros((len(REGIONS), len(REGIONS)))
    for s, d, w in zip(b.synapses.src, b.synapses.dst, b.synapses.w, strict=True):
        j[region[d], region[s]] += w
    j /= np.maximum(np.bincount(region, minlength=len(REGIONS)), 1)[:, None]
    np.fill_diagonal(j, 0.0)
    j /= np.abs(j).sum(axis=1).max()
    assert np.allclose(ext.coupling_matrix(), j, atol=1e-12)
    snapshot = ext.coupling_matrix()
    b.synapses.w[:] = 0.0                                                   # later engine changes are not tracked
    assert np.array_equal(ext.coupling_matrix(), snapshot)
    assert ext.feature_dim == len(REGIONS) + 1 + 4 == 15


def test_extension_never_changes_the_engine_trajectory():
    plain, hooked = _engine(seed=2), _engine(seed=2)
    ext = K.KuramotoExtension(1)
    hooked.attach_extension(ext)
    a, b = _drive(plain, 10), _drive(hooked, 10)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert plain.export_state() == hooked.export_state()
    feats = ext.features()
    assert feats.shape == (ext.feature_dim,) and np.all(np.isfinite(feats))
    assert np.all(np.abs(feats) <= 1.0) and 0.0 <= feats[len(REGIONS)] <= 1.0
    assert np.all(np.diff(feats[len(REGIONS) + 1:]) <= 0.0)                 # top PLVs in descending order
    assert np.allclose(np.linalg.norm(ext.states(), axis=1), 1.0, atol=1e-12)
    tel = ext.telemetry()
    assert tel['steps'] == 10 and tel['coupling_max_abs_row_sum'] == pytest.approx(1.0)
    json.dumps(tel)


def test_extension_is_deterministic_frozen_and_deep_copyable():
    engine = _engine(seed=4)
    ext = K.KuramotoExtension(3)
    engine.attach_extension(ext)
    outs = _drive(engine, 6)
    other = K.KuramotoExtension(3)
    other.bind(_engine(seed=4))
    for o in outs:
        other.after_step(o, {'learn': True})
    assert np.array_equal(other.features(), ext.features())
    learn_off = copy.deepcopy(ext)
    learn_on = copy.deepcopy(ext)
    learn_off.after_step(outs[0], {'learn': False})
    learn_on.after_step(outs[0], {'learn': True})
    assert ext.parameter_count() == 0 and np.array_equal(learn_off.features(), learn_on.features())
    assert not np.array_equal(learn_on.states(), ext.states())              # the clone moved, the original did not
    reseeded = K.KuramotoExtension(5)
    reseeded.bind(_engine(seed=4))
    for o in outs:
        reseeded.after_step(o, {})
    assert not np.array_equal(reseeded.features(), ext.features())
    ext.reset()
    fresh = K.KuramotoExtension(3)
    fresh.bind(_engine(seed=4))
    assert np.array_equal(ext.features(), fresh.features())


def test_extension_state_round_trip_and_rejection():
    engine = _engine(seed=6)
    ext = K.KuramotoExtension(2)
    engine.attach_extension(ext)
    outs = _drive(engine, 8)
    state = json.loads(json.dumps(ext.export_state()))
    clone = K.KuramotoExtension(2)
    clone.bind(_engine(seed=6))
    clone.import_state(state)
    assert np.array_equal(clone.features(), ext.features())
    ext.after_step(outs[-1], {})
    clone.after_step(outs[-1], {})
    assert np.array_equal(clone.features(), ext.features())
    before = clone.export_state()
    bad_norm = copy.deepcopy(state)
    bad_norm['x'][0] = [2.0, 0.0]
    for bad in (bad_norm, {**state, 'id': 'other'}, {**state, 'steps': -1}, {**state, 'n': state['n'] + 1},
                {**state, 'config': {**state['config'], 'gamma': 0.5}}, {k: v for k, v in state.items() if k != 'x'},
                {**state, 'plv_re': [float('nan')] * len(state['plv_re'])}):
        with pytest.raises(ValueError):
            clone.import_state(bad)
    assert clone.export_state() == before                                   # failed imports leave no trace


def test_extension_features_are_invariant_to_a_common_rotation():
    engine = _engine(seed=7)
    ext = K.KuramotoExtension(4)
    engine.attach_extension(ext)
    _drive(engine, 6)
    state = ext.export_state()
    angle = 0.7
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = K.KuramotoExtension(4)
    rotated.bind(_engine(seed=7))
    rotated.import_state({**state, 'x': (np.asarray(state['x']) @ rot.T).tolist()})
    assert not np.allclose(rotated.states(), ext.states())
    # cosines with the mean-field direction, R and phase-locking values only see relative phases
    assert np.allclose(rotated.features(), ext.features(), atol=1e-12)


def test_extension_units_mode_and_validation():
    engine = _engine(neurons=24)
    ext = K.KuramotoExtension(0, mode='units', units=12, top_pairs=3)
    engine.attach_extension(ext)
    assert ext.feature_dim == 12 + 1 + 3 and ext.coupling_matrix().shape == (12, 12)
    _drive(engine, 4)
    assert ext.features().shape == (16,) and np.all(np.abs(ext.features()) <= 1.0)
    too_many = K.KuramotoExtension(0, mode='units', units=64)
    with pytest.raises(ValueError):
        too_many.bind(_engine(neurons=24))
    with pytest.raises(ValueError):
        K.KuramotoExtension(0).after_step(np.zeros(24), {})                 # not bound
    with pytest.raises(ValueError):
        ext.after_step(np.full(24, np.nan), {})
    for bad in ({'mode': 'cells'}, {'dim': 3}, {'iterations': 0}, {'gamma': 0.0}, {'top_pairs': 46}, {'seed': -1},
                {'plv_rate': 0.0}, {'omega_spread': -1.0}):
        with pytest.raises(ValueError):
            K.KuramotoExtension(**bad)


def test_run_experiment_quick_receipt_is_sealed_and_deterministic():
    first = K.run_experiment(quick=True)
    assert verify_receipt(first) and first['verdict'] in VERDICTS and first['evidence_tier'] == EVIDENCE_TIER
    assert set(first['arms']) == {'baseline', 'candidate', 'control'}
    cfg = first['config']
    assert cfg['task'] == 'context_switch' and cfg['control'] == 'random_trace' and cfg['quick'] is True
    assert cfg['candidate_feature_dim'] == 15 and cfg['extension']['mode'] == 'region' and cfg['seeds'] == [0, 1, 2]
    cand = first['arms']['candidate']['per_seed'][0]
    assert cand['extensions'] == ['kuramoto_binding'] and cand['budget']['extension_features'] == 15
    assert cand['budget']['extension_parameters'] == 0
    assert first['arms']['control']['per_seed'][0]['extensions'] == ['random_trace_control']
    assert first['digest'] == K.run_experiment(quick=True)['digest']
    tampered = copy.deepcopy(first)
    tampered['verdict'] = 'worse' if first['verdict'] != 'worse' else 'improved'
    assert not verify_receipt(tampered)
    for bad in ({'quick': 1}, {'surprise': 1}, {'dim': 5}, {'seeds': (0, 0)}, {'heldout_seeds': (0,)}):
        with pytest.raises(ValueError):
            K.run_experiment(**{'quick': True, **bad})


# ====================================================================== live observer
def _rates(seed, n, steps):
    rng = np.random.default_rng(seed)
    return [np.tanh(rng.standard_normal(n)) for _ in range(steps)]


def test_observer_summary_is_bounded_json_safe_and_read_only():
    obs = K.KuramotoObserver(40, seed=3, history=32, oscillators=16)
    empty = obs.summary()
    assert empty['observed_steps'] == 0 and empty['R_mean'] is None and empty['chimera_index'] is None
    data = _rates(0, 40, 50)
    for v in data:
        keep = v.copy()
        obs.observe(v)
        assert np.array_equal(v, keep)                                       # the input is never written
    s = obs.summary()
    assert s['observed_steps'] == obs.observed_steps == 50 and s['history_len'] == 32
    assert 0.0 <= s['R_last'] <= 1.0 and 0.0 <= s['R_mean'] <= 1.0 and s['R_std'] >= 0.0
    assert s['metastability_index'] >= 0.0 and s['chimera_index'] >= 0.0 and 0.0 <= s['community_R_mean'] <= 1.0
    assert isinstance(s['phase_clusters'], int) and s['phase_clusters'] >= 0
    assert 0.0 <= s['hebbian_mean_abs'] <= s['hebbian_max_abs'] <= 1.0
    text = json.dumps(s)
    assert len(text) < 16_000 and 'wallclock' not in text and s['module'] == K.MODULE_ID
    data[-1][:] = 99.0                                                      # mutating old inputs changes nothing
    assert obs.summary() == s
    state = obs.export_state()
    assert len(state['r_history']) == 32 and np.array(state['coactivity']).shape == (16, 16)
    with pytest.raises(ValueError):
        obs.observe(np.zeros(39))
    with pytest.raises(ValueError):
        obs.observe(np.full(40, np.inf))
    with pytest.raises(ValueError):
        obs.observe(np.zeros((40, 1)))


def test_observer_strong_coupling_gives_one_synchronous_cluster():
    obs = K.KuramotoObserver(16, oscillators=16, omega_spread=0.0, coupling=5.0, fan_in=15, stimulus_scale=0.0,
                             hebbian_scale=0.0, history=32)
    for _ in range(100):
        obs.observe(np.zeros(16))
    s = obs.summary()
    assert s['R_last'] > 0.999 and s['phase_clusters'] == 1 and s['metastability_index'] < 1e-6


def test_observer_hebbian_estimate_recovers_the_correlation_structure():
    rng = np.random.default_rng(0)
    obs = K.KuramotoObserver(8, oscillators=8, hebbian_rate=0.01, history=16)

    def fixed_part():
        return obs.coupling_matrix() - obs.hebbian_scale * obs.hebbian() / math.sqrt(obs.m - 1)

    j0 = fixed_part()
    assert np.all(np.count_nonzero(j0, axis=1) == obs.fan_in) and np.allclose(j0.sum(axis=1), obs.coupling)
    assert np.abs(obs.hebbian()).max() == 0.0                               # identity prior: no co-activity yet
    for _ in range(3000):
        a, b = rng.standard_normal(2)
        obs.observe(np.r_[np.full(4, a), np.full(4, b)] + 0.1 * rng.standard_normal(8))
    h = obs.hebbian()
    off = ~np.eye(4, dtype=bool)
    # true correlation is 1 / (1 + 0.1^2) ~ 0.99 within a block and 0 across blocks
    assert h[:4, :4][off].min() > 0.95 and h[4:, 4:][off].min() > 0.95 and np.abs(h[:4, 4:]).max() < 0.1
    assert np.allclose(h, h.T) and np.all(np.diag(h) == 0.0) and np.all(np.abs(h) <= 1.0)
    assert np.allclose(fixed_part(), j0, atol=1e-12)                        # only the Hebbian term changed


def test_observer_round_trip_reset_and_determinism():
    data = _rates(1, 30, 40)
    obs = K.KuramotoObserver(30, seed=5, history=16, oscillators=12)
    twin = K.KuramotoObserver(30, seed=5, history=16, oscillators=12)
    for v in data[:25]:
        obs.observe(v)
        twin.observe(v)
    assert obs.summary() == twin.summary()
    state = json.loads(json.dumps(obs.export_state()))
    restored = K.KuramotoObserver(30, seed=5, history=16, oscillators=12)
    restored.import_state(state)
    assert restored.summary() == obs.summary()
    for v in data[25:]:
        obs.observe(v)
        restored.observe(v)
    assert restored.summary() == obs.summary() and restored.export_state() == obs.export_state()
    other_seed = K.KuramotoObserver(30, seed=6, history=16, oscillators=12)
    for v in data:
        other_seed.observe(v)
    assert other_seed.summary() != obs.summary()
    obs.reset()
    assert obs.summary() == K.KuramotoObserver(30, seed=5, history=16, oscillators=12).summary()


def test_observer_import_validates_before_mutating():
    obs = K.KuramotoObserver(20, seed=2, history=8, oscillators=8)
    for v in _rates(2, 20, 12):
        obs.observe(v)
    state = json.loads(json.dumps(obs.export_state()))
    before = obs.export_state()
    bad_x = copy.deepcopy(state)
    bad_x['x'][0] = [0.5, 0.0]
    bad_r = copy.deepcopy(state)
    bad_r['r_history'][0] = 1.5
    for bad in (bad_x, bad_r, {**state, 'schema': 'Other'}, {**state, 'pos': 8}, {**state, 'steps': 3},
                {**state, 'config': {**state['config'], 'seed': 3}}, {k: v for k, v in state.items() if k != 'var'},
                {**state, 'var': [-1.0] * 8}, {**state, 'cluster_history': [0.5] * 8}):
        with pytest.raises(ValueError):
            obs.import_state(bad)
    assert obs.export_state() == before
    with pytest.raises(ValueError):
        K.KuramotoObserver(20, seed=3, history=8, oscillators=8).import_state(state)


def test_observer_construction_from_spec_groups_and_validation():
    spec = FrontierSpecV1(enabled=True, seed=11, history=64, modules=('kuramoto_binding',))
    obs = K.KuramotoObserver.from_spec(100, spec, oscillators=32)
    assert obs.seed == 11 and obs.history == 64 and obs.m == 32 and obs.n_communities == 8
    with pytest.raises(ValueError):
        K.KuramotoObserver.from_spec(100, {'seed': 11})
    # FrontierSpecV1 parameters meant for other observers are accepted and ignored
    K.KuramotoObserver(10, sync_pairs=64, sae_features=64, sae_k=8, fc_window=128)
    grouped = K.KuramotoObserver(6, oscillators=6, groups=[0, 0, 1, 1, 2, 2])
    assert grouped.n_communities == 3
    for v in _rates(3, 6, 5):
        grouped.observe(v)
    assert grouped.summary()['communities'] == 3
    for bad in ({'surprise': 1}, {'dim': 3}, {'history': 1}, {'oscillators': 0}, {'gamma': 0.0},
                {'groups': [0, 1]}, {'groups': [-1] * 10}, {'hebbian_rate': 1.5}, {'seed': -1}):
        with pytest.raises(ValueError):
            K.KuramotoObserver(10, **bad)
    with pytest.raises(ValueError):
        K.KuramotoObserver(0)


# ====================================================================== validation of the core API
@pytest.mark.parametrize('call', [
    lambda: K.akorn_step(np.ones((3, 3))),                                   # odd N
    lambda: K.akorn_step(np.ones((3, 2)), np.ones((2, 2))),                  # coupling shape
    lambda: K.akorn_step(np.ones((3, 2)), stimulus=np.ones((3, 3))),         # stimulus shape
    lambda: K.akorn_step(np.ones((3, 2)), omega=np.ones((2, 2))),            # not antisymmetric
    lambda: K.akorn_step(np.full((3, 2), np.nan)),
    lambda: K.akorn_step(np.ones((3, 2)), gamma=0.0),
    lambda: K.akorn_run(np.ones((3, 2)), iterations=-1),
    lambda: K.natural_frequency_matrices([1.0, 2.0], 3),
    lambda: K.natural_frequency_matrices(np.ones((2, 2)), 2),
    lambda: K.project_tangent(np.ones(2), np.ones(3)),
    lambda: K.make_binding_task(0, groups=1),
])
def test_core_api_rejects_bad_input(call):
    with pytest.raises(ValueError):
        call()
