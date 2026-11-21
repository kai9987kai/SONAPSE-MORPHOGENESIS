"""Frontier module ``connectomics``: FC and partial correlation, dynamic FC states, ridge VAR(1) effective
connectivity with a block-permutation null, structure-function helpers, the live observer and the lab receipt."""
import copy
import json
import math
import sys
import warnings

import numpy as np
import pytest

from synapse.frontier import connectomics as C
from synapse.frontier.harness import verify_receipt
from synapse.frontier.spec import EVIDENCE_TIER, FrontierSpecV1, MODULE_IDS, OBSERVER_MODULE_IDS


def _var(seed, A, T, burn=200):
    """x_{t+1} = A x_t + N(0, I), burn-in dropped."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    noise = rng.standard_normal((T + burn, n))
    out = np.empty((T + burn, n))
    x = np.zeros(n)
    for t in range(T + burn):
        x = A @ x + noise[t]
        out[t] = x
    return out[burn:]


def _sparse_a():
    """Six units, self-coupling 0.3 and six directed edges A[target, source] (spectral radius ~0.65)."""
    A = 0.3 * np.eye(6)
    for tgt, src, w in ((1, 0, 0.45), (2, 1, -0.4), (3, 1, 0.4), (4, 3, -0.45), (5, 2, 0.4), (0, 5, 0.35)):
        A[tgt, src] = w
    return A


def _global_rng_state():
    out = {'numpy': np.random.get_state()[1].copy()}
    torch = sys.modules.get('torch')
    if torch is not None:
        out['torch'] = torch.random.get_rng_state().clone()
        out['threads'] = torch.get_num_threads()
    return out


def _same_global_state(a, b):
    if not np.array_equal(a['numpy'], b['numpy']) or set(a) != set(b):
        return False
    return 'torch' not in a or (bool((a['torch'] == b['torch']).all()) and a['threads'] == b['threads'])


def _switching(seed, *, n=6, block=150, blocks=6, r=0.8):
    """Alternating blocks of two FC states: units 0..n/2-1 correlated (r) in state 0, the others in state 1."""
    h = n // 2
    ca, cb = np.eye(n), np.eye(n)
    ca[:h, :h] = r
    cb[h:, h:] = r
    np.fill_diagonal(ca, 1.0)
    np.fill_diagonal(cb, 1.0)
    rng = np.random.default_rng(seed)
    chol = [np.linalg.cholesky(ca), np.linalg.cholesky(cb)]
    X = np.vstack([rng.standard_normal((block, n)) @ chol[b % 2].T for b in range(blocks)])
    Xs = np.random.default_rng(seed + 1).standard_normal((block * blocks, n)) @ np.linalg.cholesky(0.5 * (ca + cb)).T
    return X, Xs


# ====================================================================== registry
def test_module_is_registered_as_an_observer_with_one_experiment():
    assert C.MODULE_ID == 'connectomics' and C.MODULE_ID in MODULE_IDS and C.MODULE_ID in OBSERVER_MODULE_IDS
    assert list(C.EXPERIMENTS) == ['connectomics'] and C.EXPERIMENTS['connectomics'] is C.run_experiment
    assert C.ConnectivityObserver.module_id == C.MODULE_ID


# ====================================================================== functional connectivity
def test_partial_correlation_of_a_gaussian_chain_removes_the_indirect_link():
    a = 0.8
    # population covariance of x -> y -> z with unit-variance noise; exact partials from the SEM precision
    cov = np.array([[1.0, a, a * a], [a, a * a + 1.0, a * (a * a + 1.0)],
                    [a * a, a * (a * a + 1.0), a * a * (a * a + 1.0) + 1.0]])
    exact = C.partial_from_covariance(cov)                                  # ridge 0: exact
    assert abs(exact[0, 2]) < 1e-12
    assert exact[0, 1] == pytest.approx(a / (1.0 + a * a), abs=1e-12)
    assert exact[1, 2] == pytest.approx(a / math.sqrt(1.0 + a * a), abs=1e-12)
    omega = np.array([[1.0 + a * a, -a, 0.0], [-a, 1.0 + a * a, -a], [0.0, -a, 1.0]])
    assert np.allclose(C.precision_to_partial(omega), exact, atol=1e-12)

    T = 4000
    rng = np.random.default_rng(3)
    x = rng.standard_normal(T)
    y = a * x + rng.standard_normal(T)
    z = a * y + rng.standard_normal(T)
    X = np.column_stack([x, y, z])
    tol = 4.0 / math.sqrt(T)
    pc = C.partial_correlation(X, ridge=1e-3)
    assert abs(pc[0, 2]) < tol                                               # X-Z: ~0 given Y
    assert abs(pc[0, 1] - exact[0, 1]) < tol and abs(pc[1, 2] - exact[1, 2]) < tol
    fc = C.pearson_fc(X)
    assert fc[0, 2] == pytest.approx(a * a / math.sqrt(a * a * (a * a + 1.0) + 1.0), abs=tol) and fc[0, 2] > 0.3
    assert np.allclose(pc, pc.T) and np.allclose(np.diag(pc), 1.0) and np.all(np.abs(pc) <= 1.0)
    sp = C.spearman_fc(X)
    assert np.allclose(C.spearman_fc(np.exp(X)), sp, atol=1e-12)            # rank-based: monotone invariant
    assert sp[0, 2] > 0.3 and np.allclose(np.diag(sp), 1.0)


def test_connectivity_is_invariant_to_the_measurement_units_of_each_channel():
    X = np.random.default_rng(18).normal(size=(40, 3))
    X[:, 1] += X[:, 0]
    scale = np.array([1e-130, 1.0, 1e110])
    for estimate in (C.pearson_fc, C.partial_correlation, C.var1_fit):
        np.testing.assert_allclose(estimate(X * scale), estimate(X), atol=1e-12)
    cov = np.cov(X, rowvar=False)
    np.testing.assert_allclose(C.partial_from_covariance(cov * np.outer(scale, scale)),
                               C.partial_from_covariance(cov), atol=1e-12)
    with pytest.raises(ValueError, match='zero covariance'):
        C.partial_from_covariance([[0, 0.1], [0.1, 1]])


def test_constant_units_get_zero_rows_and_singular_inputs_need_a_ridge():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    X[:, 2] = 7.0
    fc = C.pearson_fc(X)
    assert np.all(fc[2] == 0.0) and np.all(fc[:, 2] == 0.0) and np.allclose(np.diag(fc), [1, 1, 0, 1])
    pc = C.partial_correlation(X)
    assert np.all(pc[2] == 0.0) and pc[2, 2] == 0.0 and pc[0, 0] == 1.0
    dup = np.column_stack([X[:, 0], X[:, 0], X[:, 1]])                       # perfectly collinear
    with pytest.raises(ValueError):
        C.partial_correlation(dup, ridge=0.0)
    assert np.all(np.isfinite(C.partial_correlation(dup, ridge=1e-2)))


# ====================================================================== dynamic FC
def test_dynamic_fc_recovers_two_switching_states_and_a_stationary_control_gives_one():
    X, Xs = _switching(4)
    d = C.dynamic_fc_states(X, window=40, step=8, seed=4)
    assert d['n_states'] == 2 and d['best_silhouette'] > 0.4
    starts, block = d['starts'], 150
    pure = (starts // block) == ((starts + 40 - 1) // block)
    truth = (starts // block) % 2
    acc = max(np.mean(d['labels'][pure] == truth[pure]), np.mean(d['labels'][pure] != truth[pure]))
    assert acc >= 0.95
    assert d['state_fc'].shape == (2, 6, 6) and sum(d['occupancy']) == pytest.approx(1.0)
    assert d['mean_dwell_samples'] == pytest.approx(d['mean_dwell_windows'] * 8)
    stationary = C.dynamic_fc_states(Xs, window=40, step=8, seed=4)
    assert stationary['n_states'] == 1 and stationary['transitions'] == 0
    again = C.dynamic_fc_states(X, window=40, step=8, seed=4)
    assert np.array_equal(again['labels'], d['labels']) and again['silhouette'] == d['silhouette']


def test_kmeans_silhouette_and_dwell_on_known_cases():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]])
    km = C.kmeans(pts, 2, seed=1)
    assert km['labels'].tolist() == [0, 0, 0, 1, 1, 1] and km['inertia'] < 0.1
    assert C.silhouette_score(pts, km['labels']) > 0.9
    assert C.silhouette_score(pts, np.zeros(6, dtype=int)) is None
    dw = C.dwell_statistics(np.array([0, 0, 1, 1, 1, 0]))
    assert dw['runs'] == 3 and dw['transitions'] == 2 and dw['mean_dwell'] == 2.0
    assert dw['mean_dwell_per_state'] == [1.5, 3.0] and dw['occupancy'] == [0.5, 0.5]
    # regression: repeated points used to leave a cluster empty (NaN centroid, "mean of empty slice")
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for W, k in ((3, 3), (5, 3), (6, 4)):
            out = C.kmeans(np.zeros((W, 2)), k)
            assert np.all(np.isfinite(out['centroids'])) and np.bincount(out['labels'], minlength=k).min() >= 1
        flat = C.dynamic_fc_states(np.ones((80, 4)), window=10, step=5, k_max=4)
        assert flat['n_states'] == 1


# ====================================================================== effective connectivity
def test_var1_effective_connectivity_recovers_a_sparse_stable_support():
    A = _sparse_a()
    assert np.max(np.abs(np.linalg.eigvals(A))) < 0.7
    X = _var(11, A, 2000)
    assert np.max(np.abs(C.var1_fit(X, ridge=1e-3, standardize=False) - A)) < 0.08
    ec = C.effective_connectivity(X, null_draws=49, seed=2, standardize=False)
    off = ~np.eye(6, dtype=bool)
    truth = (A != 0.0)[off]
    assert C.auroc(np.abs(ec['A'][off]), truth) > 0.9
    mask = ec['mask'][off]
    assert mask[truth].all() and int((mask & ~truth).sum()) <= 1
    assert np.all(np.sign(ec['A'][off][truth]) == np.sign(A[off][truth]))
    assert ec['A'].shape == ec['z'].shape == (6, 6) and np.all(np.isfinite(ec['z']))
    assert np.all((ec['p'] >= 0.0) & (ec['p'] <= 1.0)) and np.all((ec['p_rank'] > 0.0) & (ec['p_rank'] <= 1.0))
    assert ec['block'] == round(1999 ** (1 / 3)) and ec['draws'] == 49
    again = C.effective_connectivity(X, null_draws=49, seed=2, standardize=False)
    assert np.array_equal(again['z'], ec['z']) and np.array_equal(again['mask'], ec['mask'])
    assert not np.array_equal(C.effective_connectivity(X, null_draws=49, seed=3, standardize=False)['z'], ec['z'])
    # every knob changes the output: block length of the null, ridge shrinkage, standardisation
    assert not np.array_equal(C.effective_connectivity(X, null_draws=49, seed=2, standardize=False, block=1)['z'],
                              ec['z'])
    shrunk = C.var1_fit(X, ridge=1.0, standardize=False)
    assert np.abs(shrunk).sum() < np.abs(C.var1_fit(X, ridge=1e-3, standardize=False)).sum()
    assert not np.allclose(C.var1_fit(X, ridge=1e-3, standardize=True), C.var1_fit(X, ridge=1e-3, standardize=False))


def test_uncoupled_units_give_an_almost_empty_mask():
    for seed in (0, 1):
        ec = C.effective_connectivity(_var(seed, 0.5 * np.eye(5), 1500), null_draws=49, seed=seed)
        assert ec['significant'] <= 1


def test_block_permutation_and_benjamini_hochberg_and_auroc():
    rng = np.random.default_rng(0)
    perm = C.block_permutation(10, 3, rng)
    assert sorted(perm.tolist()) == list(range(10))
    pos = {int(v): i for i, v in enumerate(perm)}
    assert all(pos[v] == pos[v - 1] + 1 for v in range(10) if v % 3)       # blocks keep their inner order
    assert not all(pos[v] == v for v in range(10)) or C.block_permutation(10, 3, rng).tolist() != perm.tolist()
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert C.benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8
    assert C.benjamini_hochberg(np.array([0.01, 0.02, 0.03]), 0.05).all()
    assert C.auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0 and C.auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0
    assert C.auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5 and C.auroc([1, 2], [1, 1]) is None
    with pytest.raises(ValueError):
        C.benjamini_hochberg([0.5, 1.5])


# ====================================================================== structure and coupling
def test_structure_helpers_on_known_graphs():
    W = C.structural_matrix(3, [0, 0, 1, 2], [1, 1, 2, 2], [0.5, -0.25, 1.0, 9.0])
    assert W[0, 1] == 0.75 and W[1, 2] == 1.0 and np.all(np.diag(W) == 0.0)        # duplicates summed, |w|
    assert C.structural_matrix(3, [0], [1], [-2.0], absolute=False, symmetric=True)[1, 0] == -2.0
    S = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    from scipy.linalg import expm
    d = S.sum(axis=1)
    assert np.allclose(C.weighted_communicability(S), expm(S / np.sqrt(np.outer(d, d))), atol=1e-10)
    block = np.zeros((6, 6))
    block[:3, :3] = block[3:, 3:] = 1.0
    np.fill_diagonal(block, 0.0)
    assert C.modularity(block, [0, 0, 0, 1, 1, 1]) == pytest.approx(0.5)
    com = C.fc_communities(block + 0.01 * (1 - np.eye(6)))
    assert com['communities'] == 2 and com['labels'].tolist() == [0, 0, 0, 1, 1, 1] and com['modularity'] > 0.45
    coupling = C.structure_function_coupling(block, 0.9 * block + 0.05)
    assert coupling['sc_fc_spearman'] == pytest.approx(1.0) and coupling['connected_pairs'] == 6
    assert coupling['mean_fc_connected'] > coupling['mean_fc_unconnected']


# ====================================================================== live observer
def _driven(seed, n, T):
    A = 0.3 * np.eye(n)
    A[1, 0] = 0.8                                                            # one strong edge 0 -> 1
    return _var(seed, A, T)


def test_observer_is_bounded_json_safe_read_only_and_finds_the_driving_edge():
    obs = C.ConnectivityObserver(4, seed=1, history=128, window=64, ec_null_draws=19)
    empty = obs.summary()
    assert empty['ready'] is False and empty['observed_steps'] == 0 and empty['ec_top_edges'] == []
    assert empty['ec_density'] is None and empty['dfc_states'] is None
    data = _driven(5, 4, 300)
    for row in data:
        keep = row.copy()
        obs.observe(row)
        assert np.array_equal(row, keep)                                     # the input is never written
    s = obs.summary()
    assert s['ready'] and s['observed_steps'] == obs.observed_steps == 300 and s['buffered'] == 128
    assert s['computed_at'] == 288 and s['rows'] == 128 and s['recompute_every'] == 16
    top = s['ec_top_edges'][0]
    assert (top['source'], top['target']) == (0, 1) and top['significant'] and top['weight'] > 0.4
    assert len(s['ec_top_edges']) == 5 and 0.0 <= s['ec_density'] <= 1.0 and 0.0 <= s['mean_abs_fc'] <= 1.0
    assert s['dfc_windows'] == 15 and s['dfc_states'] >= 1 and s['fc_communities'] >= 1
    text = json.dumps(s)
    assert len(text) < 16_000 and 'wallclock' not in text and s['evidence_tier'] == EVIDENCE_TIER
    data[-1][:] = 99.0
    assert obs.summary() == s
    assert np.array(obs.export_state()['buffer']).shape == (128, 4)
    for bad in (np.zeros(3), np.full(4, np.nan), np.zeros((4, 1)), ['a', 'b', 'c', 'd']):
        with pytest.raises(ValueError):
            obs.observe(bad)
    assert obs.summary() == s


def test_observer_parameters_change_the_output():
    data = _driven(5, 4, 300)

    def run(**kw):
        obs = C.ConnectivityObserver(4, **{'seed': 1, 'history': 128, 'window': 64, 'ec_null_draws': 19, **kw})
        for row in data:
            obs.observe(row)
        return obs.summary()

    base = run()
    edge = lambda s: (s['ec_top_edges'][0]['source'], s['ec_top_edges'][0]['target'])   # noqa: E731
    assert edge(base) == (0, 1)
    fewer_draws, other_seed = run(ec_null_draws=9), run(seed=2)       # the null changes, the fit does not
    for other in (fewer_draws, other_seed):
        assert other['ec_top_edges'][0]['weight'] == base['ec_top_edges'][0]['weight']
        assert other['ec_top_edges'][0]['z'] != base['ec_top_edges'][0]['z'] and edge(other) == (0, 1)
    shrunk = run(ridge=2.0)                                           # stronger ridge shrinks the coefficient
    assert 0.0 < shrunk['ec_top_edges'][0]['weight'] < base['ec_top_edges'][0]['weight']
    assert run(min_silhouette=1.0)['dfc_states'] == 1                 # silhouette floor collapses dFC states
    assert run(window=32)['computed_at'] != base['computed_at']       # the window sets the recompute schedule


def test_observer_unit_subsample_from_spec_and_constructor_validation():
    obs = C.ConnectivityObserver(100, seed=3, history=32, window=16, max_units=8)
    assert obs.units.tolist() == sorted(obs.units.tolist()) and obs.units.size == 8
    assert np.array_equal(C.ConnectivityObserver(100, seed=3, history=32, window=16, max_units=8).units, obs.units)
    spec = FrontierSpecV1(enabled=True, seed=11, history=64, fc_window=128, modules=('connectomics',))
    built = C.ConnectivityObserver.from_spec(40, spec)
    assert built.seed == 11 and built.history == 64 and built.window == 64
    with pytest.raises(ValueError):
        C.ConnectivityObserver.from_spec(40, spec.to_dict())
    assert C.ConnectivityObserver(40, history=64, fc_window=128, sync_pairs=8).window == 64   # spec name, clipped
    assert C.ConnectivityObserver(40, history=64, window=32, fc_window=32).window == 32
    assert C.ConnectivityObserver(40).window == 128
    for bad in ({'history': 8}, {'window': 4}, {'window': 600}, {'max_units': 1}, {'ec_null_draws': 1},
                {'ridge': 0.0}, {'alpha': 0.9}, {'seed': -1}, {'k_max': 1}, {'fc_window': 4},
                {'window': 32, 'fc_window': 48}, {'surprise': 1}):
        with pytest.raises(ValueError):
            C.ConnectivityObserver(10, **bad)
    with pytest.raises(ValueError):
        C.ConnectivityObserver(1)


def test_observer_round_trip_reset_and_import_validates_before_mutating():
    data = _driven(6, 5, 200)
    kw = dict(seed=3, history=64, window=32, ec_null_draws=9)
    obs, twin = C.ConnectivityObserver(5, **kw), C.ConnectivityObserver(5, **kw)
    for row in data[:120]:
        obs.observe(row)
        twin.observe(row)
    assert obs.summary() == twin.summary()
    state = json.loads(json.dumps(obs.export_state()))
    restored = C.ConnectivityObserver(5, **kw)
    restored.import_state(state)
    assert restored.summary() == obs.summary() and restored.export_state() == obs.export_state()
    for row in data[120:]:
        obs.observe(row)
        restored.observe(row)
    assert restored.summary() == obs.summary() and restored.export_state() == obs.export_state()

    before = restored.export_state()
    edges = copy.deepcopy(state)
    edges['metrics']['ec_top_edges'][0]['target'] = edges['metrics']['ec_top_edges'][0]['source']
    density = copy.deepcopy(state)
    density['metrics']['ec_density'] = 1.5
    keys = copy.deepcopy(state)
    del keys['metrics']['ec_density']
    nan_buf = copy.deepcopy(state)
    nan_buf['buffer'][0][0] = float('nan')
    for bad in (edges, density, keys, nan_buf, {**state, 'schema': 'Other'}, {**state, 'seed': 4},
                {**state, 'buffer': state['buffer'][:-1]}, {**state, 'steps': 119},
                {**state, 'last': 90}, {**state, 'units': [0, 1, 2, 3, 5]}, {**state, 'metrics': None},
                {k: v for k, v in state.items() if k != 'buffer'}, 'not a state'):
        with pytest.raises(ValueError):
            restored.import_state(bad)
    assert restored.export_state() == before
    young = C.ConnectivityObserver(5, **kw)
    for row in data[:10]:
        young.observe(row)
    early = young.export_state()
    early['buffer'][20][0] = 1.0                                             # beyond the observed steps
    with pytest.raises(ValueError):
        C.ConnectivityObserver(5, **kw).import_state(early)
    obs.reset()
    assert obs.summary() == C.ConnectivityObserver(5, **kw).summary()


# ====================================================================== lab experiment
def test_run_experiment_quick_passes_every_known_answer_check():
    r = C.run_experiment(quick=True)
    assert verify_receipt(r) and r['kind'] == 'validation' and r['evidence_tier'] == EVIDENCE_TIER
    assert r['verdict'] == 'pass' and r['verdict_basis'] == 'all_known_answer_checks' and all(r['checks'].values())
    assert r['config']['quick'] is True and r['config']['seeds'] == [0, 1, 2]
    assert set(r['neuroweave']['per_topology']) == {'clustered', 'ring_lattice', 'hub_skewed'}
    for v in r['validations']['var_recovery']:
        assert v['support_auroc'] > 0.9
    assert all(s['n_states'] == 2 and s['stationary_control_states'] == 1 for s in r['validations']['dfc_switching'])
    tampered = copy.deepcopy(r)
    tampered['verdict'] = 'fail'
    assert not verify_receipt(tampered)


def test_run_experiment_is_deterministic_and_validates_arguments():
    small = dict(seeds=(0,), topologies=('ring_lattice',), neurons=16, nw_steps=80, nw_burn_in=20, nw_null_draws=9,
                 var_steps=400, ec_null_draws=19, chain_steps=400, switch_blocks=4)
    before = _global_rng_state()
    a = C.run_experiment(quick=True, **small)
    b = C.run_experiment(quick=True, **small)
    assert _same_global_state(before, _global_rng_state())        # no global NumPy/torch RNG or thread change
    assert verify_receipt(a) and a['digest'] == b['digest'] and a['verdict'] in ('pass', 'fail')
    assert C.run_experiment(quick=True, **{**small, 'seeds': (1,)})['digest'] != a['digest']
    for bad in ({'quick': 1}, {'surprise': 1}, {'seeds': ()}, {'seeds': (1, 1)}, {'topologies': ('nope',)},
                {'dfc_window': 5}, {'mechanisms': ('nope',)}, {'var_units': 2}):
        with pytest.raises(ValueError):
            C.run_experiment(**{'quick': True, **small, **bad})
