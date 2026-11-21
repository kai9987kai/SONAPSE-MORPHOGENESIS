"""Frontier modules ``hh_cable`` (Hodgkin-Huxley cable reference) and ``nir_export`` (NIR-style interchange)."""
import copy
import json
import math

import numpy as np
import pytest

from synapse.frontier import hh_cable as H
from synapse.frontier import nir_export as N
from synapse.frontier.harness import verify_receipt
from synapse.frontier.spec import EVIDENCE_TIER, MODULE_IDS, OBSERVER_MODULE_IDS
from synapse.neuroweave.brain import NeuroWeaveBrain
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.experiments import make_spec


def _jsonish(obj):
    return json.dumps(obj, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def _global_state():
    import torch
    return (np.random.get_state()[1].copy(), torch.get_rng_state().clone(), torch.get_num_threads())


def _same_global_state(a, b):
    return np.array_equal(a[0], b[0]) and bool((a[1] == b[1]).all()) and a[2] == b[2]


# ====================================================================== registry
def test_modules_are_registered_and_not_observers():
    for mod in (H, N):
        assert mod.MODULE_ID in MODULE_IDS and mod.MODULE_ID not in OBSERVER_MODULE_IDS
        assert list(mod.EXPERIMENTS) == [mod.MODULE_ID] and mod.EXPERIMENTS[mod.MODULE_ID] is mod.run_experiment


# ====================================================================== HH rate functions and rest
def test_rate_functions_match_hh1952_at_rest_and_are_continuous_at_the_removable_singularities():
    am, bm, ah, bh, an, bn = (float(x) for x in H.hh_rates(-65.0))
    # HH eqns 12, 13, 20, 21, 23, 24 at V = 0 (v = -65 mV), 6.3 C
    assert am == pytest.approx(2.5 / (math.exp(2.5) - 1.0), rel=1e-12)
    assert bm == pytest.approx(4.0) and ah == pytest.approx(0.07) and bn == pytest.approx(0.125)
    assert bh == pytest.approx(1.0 / (math.exp(3.0) + 1.0), rel=1e-12)
    assert an == pytest.approx(0.1 / (math.e - 1.0), rel=1e-12)
    # alpha_m at v = -40 and alpha_n at v = -55 are 0/0 in the raw formula: limits 1.0 and 0.1
    for v, idx, limit in ((-40.0, 0, 1.0), (-55.0, 4, 0.1)):
        vals = np.array([float(H.hh_rates(v + d)[idx]) for d in (-1e-6, 0.0, 1e-6)])
        assert np.all(np.isfinite(vals)) and np.allclose(vals, limit, atol=1e-7)
    m, h, n = (float(x) for x in H.steady_state_gates(-65.0))
    assert 0.0 < m < 0.1 and 0.5 < h < 0.7 and 0.3 < n < 0.4
    assert H.temperature_factor(6.3) == pytest.approx(1.0) and H.temperature_factor(16.3) == pytest.approx(3.0)


def test_resting_potential_is_the_documented_minus_65_mV_and_stationary():
    patch = H.HHCable(dt_ms=0.0125)
    rest = patch.resting_potential()
    assert abs(rest - H.HH_REST_MV) < 0.01           # Table 3: V_l chosen for zero ionic current at V = 0
    res = patch.run([[1.0, 5.0, 0.0]], t_stop_ms=10.0, record=[0])
    assert res['spike_times_ms'][0][0] == [] and abs(res['final_v_mv'][0, 0] - rest) < 1e-9


# ====================================================================== HH patch: spikes, toggles, convergence
def test_single_compartment_spikes_for_suprathreshold_current_and_not_for_small_current():
    patch = H.HHCable(dt_ms=0.0125)
    pulses = [[[1.0, 5.0, a]] for a in (0.0, 1.0, 2.0, 10.0, 20.0)]
    res = patch.run(pulses, t_stop_ms=15.0, record=[0], trace_site=0, trace_every=8)
    counts = [len(res['spike_times_ms'][b][0]) for b in range(5)]
    assert counts[:3] == [0, 0, 0] and counts[3] >= 1 and counts[4] >= 1
    assert res['peak_mv'][3, 0] > 0.0 and res['peak_mv'][4, 0] > 0.0 and res['peak_mv'][2, 0] < -30.0
    assert res['trace_mv'].shape == (5, 1200 // 8 + 1) and np.all(np.isfinite(res['trace_mv']))
    # spike time is after the onset and the larger current fires earlier
    assert 1.0 < res['spike_times_ms'][4][0][0] < res['spike_times_ms'][3][0][0] < 6.0


def test_mechanism_toggles_change_the_response():
    pulse = [[1.0, 5.0, 10.0]]
    base = H.HHCable(dt_ms=0.0125).run(pulse, t_stop_ms=12.0, record=[0])
    no_na = H.HHCable(dt_ms=0.0125, gNa=0.0).run(pulse, t_stop_ms=12.0, record=[0])
    warm = H.HHCable(dt_ms=0.0125, temperature_c=16.3).run(pulse, t_stop_ms=12.0, record=[0])
    euler = H.HHCable(dt_ms=0.0125, method='exp_euler').run(pulse, t_stop_ms=12.0, record=[0])
    t0 = base['spike_times_ms'][0][0][0]
    assert no_na['spike_times_ms'][0][0] == [] and no_na['peak_mv'][0, 0] < -40.0    # no sodium current, no spike
    assert warm['spike_times_ms'][0][0] and warm['spike_times_ms'][0][0][0] != t0     # Q10 = 3 changes timing
    assert euler['spike_times_ms'][0][0] and euler['spike_times_ms'][0][0][0] != t0   # different integrator
    assert abs(euler['spike_times_ms'][0][0][0] - t0) < 0.05


def test_spike_timing_converges_under_dt_halving():
    times = {}
    for method in H.METHODS:
        times[method] = [H.HHCable(dt_ms=dt, method=method).run([[0.5, 1.0, 20.0]], t_stop_ms=3.0,
                                                                  record=[0])['spike_times_ms'][0][0][0]
                         for dt in (0.01, 0.005, 0.0025)]
    rk = times['rk4']
    assert abs(rk[1] - rk[2]) < 1e-4 and abs(rk[0] - rk[2]) < 1e-3
    ee = times['exp_euler']
    e1, e2 = abs(ee[0] - ee[1]), abs(ee[1] - ee[2])
    assert 0.7 < math.log2(e1 / e2) < 1.5                                              # first order
    assert abs(ee[2] - rk[2]) < 0.02


def test_threshold_current_brackets_the_all_or_none_transition():
    patch = H.HHCable(dt_ms=0.0125)
    thr = patch.threshold_current(0.5, bracket=(0.0, 40.0), rounds=2, batch=16, window_ms=10.0)
    assert thr['resolved'] and thr['monotone'] and 0.0 < thr['bracket'][0] < thr['bracket'][1] < 40.0
    lo, hi = thr['bracket']
    res = patch.run([[[1.0, 0.5, lo]], [[1.0, 0.5, hi]]], t_stop_ms=11.0, record=[0])
    assert res['spike_times_ms'][0][0] == [] and len(res['spike_times_ms'][1][0]) == 1
    unresolved = patch.threshold_current(0.5, bracket=(0.0, 1.0), rounds=1, batch=4, window_ms=10.0)
    assert not unresolved['resolved'] and unresolved['reason'] == 'upper_silent'


def test_paired_pulses_show_refractoriness():
    patch = H.HHCable(dt_ms=0.0125)
    info = patch.refractory_period(pulse_ms=0.5, threshold=14.0, rounds=1, batch=16, max_interval_ms=15.0,
                                   tail_ms=4.0)
    assert info['fails_at_min_interval'] and info['succeeds_at_max_interval'] and info['monotone']
    r = info['refractory_ms']
    assert r is not None and 0.5 < r < 15.0
    # direct check on both sides of the estimate: second strong pulse fails just before, fires after
    amp = 4.0 * 14.0
    before = [[1.0, 0.5, 28.0], [1.0 + info['bracket_ms'][0], 0.5, amp]]
    after = [[1.0, 0.5, 28.0], [1.0 + r, 0.5, amp]]
    res = patch.run([before, after], t_stop_ms=1.0 + r + 5.0, record=[0])
    assert len(res['spike_times_ms'][0][0]) == 1 and len(res['spike_times_ms'][1][0]) == 2


def test_fi_curve_zero_at_rest_and_repetitive_firing_for_strong_steps():
    fi = H.HHCable(dt_ms=0.0125).fi_curve([0.0, 2.0, 15.0, 40.0], window_ms=60.0, transient_ms=15.0)
    assert fi['rates_hz'][0] == 0.0 and fi['spike_counts'][0] == 0 and fi['spike_counts'][1] == 0
    assert fi['rates_hz'][2] > 0.0 and fi['rates_hz'][3] > fi['rates_hz'][2]
    assert all(np.isfinite(fi['peak_mv']))


# ====================================================================== HH cable
def test_conduction_velocity_is_positive_and_increases_with_diameter():
    cable = H.HHCable(compartments=30, length_um=20000.0, diameter_um=476.0, dt_ms=0.005, temperature_c=18.5)
    rows = cable.conduction_velocity(diameters_um=(119.0, 238.0, 476.0), t_stop_ms=3.0)['rows']
    v = [r['velocity_m_s'] for r in rows]
    assert all(x is not None and x > 0.0 for x in v) and v[0] < v[1] < v[2]
    slope = np.polyfit(np.log([119.0, 238.0, 476.0]), np.log(v), 1)[0]
    assert abs(slope - 0.5) < 0.05                    # uniform cable: velocity ~ sqrt(diameter)
    assert abs(v[2] - H.HH1952_REFERENCE_AXON['computed_velocity_m_s']) / 18.8 < 0.05
    assert all(r['mid_peak_mv'] > 0.0 for r in rows)
    # the axial conductance is the HH eqn 28 coefficient a / (2 Ra dx^2) in mS/cm^2
    a_cm, dx_cm = 238e-4, (20000.0 / 30) * 1e-4
    assert cable.axial_conductance(476.0) == pytest.approx(1e3 * a_cm / (2 * 35.4 * dx_cm ** 2), rel=1e-12)


def test_cable_ends_are_sealed_so_axial_currents_conserve_charge():
    # passive cable: with sealed ends the axial terms sum to zero, so the mean voltage of N compartments follows
    # a single compartment that receives 1/N of the stimulus (exactly, up to rounding, for a linear system)
    passive = dict(gNa=0.0, gK=0.0, gL=0.3, dt_ms=0.005)
    cable = H.HHCable(compartments=6, length_um=600.0, diameter_um=10.0, **passive)
    single = H.HHCable(compartments=1, **passive)
    assert cable.resting_potential() == pytest.approx(-54.387, abs=1e-8)
    res = cable.run([[0.2, 0.5, 30.0]], t_stop_ms=2.0)
    ref = single.run([[0.2, 0.5, 5.0]], t_stop_ms=2.0)
    assert np.ptp(res['peak_mv'][0]) > 1e-3 and np.all(np.diff(res['peak_mv'][0]) < 0.0)   # axial current flowed
    assert np.mean(res['final_v_mv'][0]) == pytest.approx(ref['final_v_mv'][0, 0], abs=1e-9)
    mid = cable.run([[0.2, 0.5, 30.0]], t_stop_ms=0.5)                           # during the pulse
    ref_mid = single.run([[0.2, 0.5, 5.0]], t_stop_ms=0.5)
    assert np.ptp(mid['final_v_mv'][0]) > 1e-3
    assert np.mean(mid['final_v_mv'][0]) == pytest.approx(ref_mid['final_v_mv'][0, 0], abs=1e-9)
    # with HH channels a weak stimulus at compartment 0 spreads with a peak decreasing towards the far end
    sub = H.HHCable(compartments=6, length_um=600.0, diameter_um=10.0, dt_ms=0.005).run([[0.0, 0.5, 1.0]],
                                                                                         t_stop_ms=1.0)
    assert np.all(np.diff(sub['peak_mv'][0]) < 0.0) and not any(sub['spike_times_ms'][0])


def test_hh_validation_errors():
    for overrides, message in [({'fi_levels': (1.0, 1.02, 1.05)}, 'suprathreshold'),
                               ({'convergence_dts': (0.02, 0.01, 0.008)}, 'halve')]:
        with pytest.raises(ValueError, match=message):
            H.run_experiment(quick=True, **overrides)
    for kwargs in ({'method': 'euler'}, {'compartments': 0}, {'dt_ms': 0.0}, {'dt_ms': 0.05},
                   {'gNa': -1.0}, {'Ra': float('nan')}, {'gNa': 0.0, 'gK': 0.0, 'gL': 0.0}, {'compartments': True}):
        with pytest.raises(ValueError):
            H.HHCable(**kwargs)
    patch = H.HHCable(dt_ms=0.0125)
    for pulses in ([[1.0, 2.0]], [[-1.0, 2.0, 3.0]], [[1.0, 2.0, float('inf')]], 'x'):
        with pytest.raises(ValueError):
            patch.run(pulses, t_stop_ms=5.0)
    with pytest.raises(ValueError):
        patch.run([[1.0, 2.0, 3.0]], t_stop_ms=5.0, record=[1])
    with pytest.raises(ValueError):
        patch.conduction_velocity()                   # needs >= 4 compartments
    with pytest.raises(ValueError):
        patch.replace(diameter=3.0)
    with pytest.raises(ValueError):
        patch.threshold_current(0.5, bracket=(5.0, 1.0))
    assert patch.replace(gK=30.0).gK == 30.0 and patch.gK == 36.0


# ====================================================================== NEUROWEAVE reference helpers
def test_rank_statistics_and_adaptation_index_on_known_cases():
    assert H.average_ranks([10.0, 20.0, 20.0, 30.0]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert H.spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert H.spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert H.spearman_rho([1, 2, 3], [5, 5, 5]) is None
    assert H.adaptation_index([2.0, 2.0, 2.0]) == 0.0 and H.adaptation_index([1.0, 2.0]) == pytest.approx(1 / 3)
    assert H.adaptation_index([1.0]) is None
    cmp = H.compare_fi_curves([0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 20.0, 40.0], [0.0, 1.0, 1.5, 4.0])
    assert cmp['spearman_suprathreshold'] == pytest.approx(1.0) and cmp['hh_normalised'][-1] == 1.0
    assert cmp['hh_repetitive_onset_level'] == 1.0


def test_neuroweave_cell_fi_curve_is_monotone_and_seed_invariant():
    drives = [0.0, 0.2, 0.6, 1.5]
    a = H.neuroweave_fi_curve(drives, steps=120, transient=20, seed=0)
    b = H.neuroweave_fi_curve(drives, steps=120, transient=20, seed=5)
    assert a['spike_counts'] == b['spike_counts'] and a['rates_per_step'] == b['rates_per_step']
    assert a['spike_counts'][0] == 0 and a['spike_counts'][-1] > 0
    assert all(y >= x for x, y in zip(a['spike_counts'], a['spike_counts'][1:]))
    plain = H.neuroweave_fi_curve(drives, steps=120, transient=20, flags=())
    assert plain['spike_counts'] != a['spike_counts']      # membrane_dynamics toggled off changes the cell
    thr = H.neuroweave_threshold(steps=120, rounds=2, batch=16)
    assert thr['resolved'] and 0.0 < thr['threshold'] < 2.0
    with pytest.raises(ValueError):
        H.neuroweave_fi_curve(drives, flags=('not_a_flag',))


# ====================================================================== NIR export
def _brain(mechanisms=('membrane_dynamics', 'synaptic_delays'), seed=0, neurons=16, warmup=10):
    brain = NeuroWeaveBrain(make_spec(mechanisms, seed=seed, neurons=neurons, topology='clustered', steps=8))
    rng = np.random.default_rng(seed + 1)
    for _ in range(warmup):
        brain.step(rng.uniform(-1.0, 1.0, 8))
    return brain


def test_cell_parameters_follow_the_zero_order_hold_mapping():
    p = N.cell_parameters('CubaLIF', dt=1.0)
    assert math.exp(-1.0 / p['tau_syn']) == pytest.approx(0.58) and math.exp(-1.0 / p['tau_mem']) == pytest.approx(0.78)
    assert (1 - 0.58) * p['w_in'] == pytest.approx(1.0) and (1 - 0.78) * p['r'] == pytest.approx(0.22 * 0.84)
    assert p['v_threshold'] == pytest.approx(0.53) and p['v_reset'] == pytest.approx(0.26 * 0.53)
    q = N.cell_parameters('LIF', dt=2.0)
    assert math.exp(-2.0 / q['tau']) == pytest.approx(0.58) and math.tanh(0.84 * q['v_threshold']) == pytest.approx(0.53)
    with pytest.raises(ValueError):
        N.cell_parameters('Izhikevich')


def test_brain_export_is_a_valid_graph_and_mechanism_toggles_change_it():
    brain = _brain()
    before = _jsonish(brain.export_state())
    graph = N.brain_to_nir(brain, input_dim=8)
    assert _jsonish(brain.export_state()) == before                   # the exported brain is never mutated
    report = N.validate_graph(graph)
    assert report['valid'] and not report['errors'] and set(report['node_types']) <= set(N.NIR_NODE_TYPES)
    assert graph['nodes']['cells']['type'] == 'CubaLIF' and len(graph['metadata']['delay_classes_steps']) >= 2
    for k in graph['metadata']['delay_classes_steps']:
        assert graph['nodes'][f'delay_{k}']['params']['delay'][0] == float(k)
    assert graph['nodes']['input_projection']['params']['weight'] and len(graph['nodes']['input']['params']['shape']) == 1
    # every synapse is represented: non-zero recurrent entries = represented count <= unique (src, dst, delay)
    nnz = sum(np.count_nonzero(graph['nodes'][f'recurrent_{k}']['params']['weight'])
              for k in graph['metadata']['delay_classes_steps'])
    meta = graph['metadata']
    assert nnz == meta['represented_synapses'] <= meta['source']['unique_src_dst_delay'] <= meta['source']['synapses']
    # an engine, a brain export and an engine export give the same graph
    assert N.graphs_equal(graph, N.brain_to_nir(brain.export_state(), input_dim=8))
    engine = NeuroWeaveEngine(make_spec(('membrane_dynamics',), seed=0, neurons=12, topology='clustered', steps=8))
    assert N.graphs_equal(N.brain_to_nir(engine, input_dim=8), N.brain_to_nir(engine.export_state(), input_dim=8))
    # toggles: no membrane dynamics -> LIF; no synaptic delays -> one delay class; quantisation changes weights
    plain = N.brain_to_nir(_brain(('synaptic_delays',)), input_dim=8)
    assert plain['nodes']['cells']['type'] == 'LIF'
    nodelay = N.brain_to_nir(_brain(('membrane_dynamics',)), input_dim=8)
    assert nodelay['metadata']['delay_classes_steps'] == [1]
    quant = N.brain_to_nir(_brain(('membrane_dynamics', 'synaptic_delays', 'quantised_synapses')), input_dim=8)
    assert not N.graphs_equal(quant, graph) and 'quantised_synapses' in quant['metadata']['folded_features']


def test_lossy_features_are_listed_for_the_enabled_mechanisms():
    feats = {f['feature']: f['treatment'] for f in
             N.unrepresentable_features(_brain(('membrane_dynamics', 'synaptic_delays', 'stdp', 'sparse_code')))}
    assert feats['graded_output_transmission'] == 'approximated' and feats['refractory_period'] == 'dropped'
    assert feats['adaptive_threshold'] == 'dropped' and feats['stdp'] == 'frozen_snapshot'
    assert feats['sparse_code'] == 'dropped' and 'threshold_on_graded_output' not in feats
    plain = {f['feature'] for f in N.unrepresentable_features(_brain(()))}
    assert 'threshold_on_graded_output' in plain and 'refractory_period' not in plain
    with pytest.raises(ValueError):
        N.unrepresentable_features({'schema': 'nope'})


def _chain(weight, *, extra_nodes=None, extra_edges=None):
    W = np.asarray(weight, dtype=float)
    g = {'nodes': {'x': {'type': 'Input', 'params': {'shape': [W.shape[1]]}},
                   'lin': {'type': 'Linear', 'params': {'weight': W.tolist()}},
                   'y': {'type': 'Output', 'params': {'shape': [W.shape[0]]}}},
         'edges': [['x', 'lin'], ['lin', 'y']]}
    g['nodes'].update(extra_nodes or {})
    g['edges'] += extra_edges or []
    return g


@pytest.mark.parametrize('mutate, message', [
    (lambda g: g['nodes']['lin'].__setitem__('type', []), 'unknown node type'),
    (lambda g: g['nodes'].__setitem__('z', {'type': 'Spiky', 'params': {}}), 'unknown node type'),
    (lambda g: g['edges'].append(['lin', 'ghost']), 'endpoint does not exist'),
    (lambda g: g['edges'].append(['x', 'lin']), 'duplicate'),
    (lambda g: g['edges'].append(['lin', 'x']), 'Input node cannot have incoming'),
    (lambda g: g['nodes']['y']['params'].__setitem__('shape', [5]), 'does not match'),
    (lambda g: g['nodes']['lin']['params'].__setitem__('weight', [[float('nan'), 1.0], [0.0, 1.0], [1.0, 1.0]]),
     'non-finite'),
    (lambda g: g['nodes']['lin']['params'].__setitem__('bias', [0.0]), 'params mismatch'),
])
def test_validate_graph_rejects_broken_graphs(mutate, message):
    g = _chain(np.ones((3, 2)))
    assert N.validate_graph(g)['valid']
    mutate(g)
    report = N.validate_graph(g)
    assert not report['valid'] and any(message in e for e in report['errors'])
    with pytest.raises(ValueError):
        N.canonical_graph(g)
    assert not N.validate_graph([])['valid']


def test_simulator_known_answers_linear_delay_and_lif():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((3, 2))
    X = rng.standard_normal((5, 2))
    out = N.simulate_graph(_chain(W), X)
    assert np.allclose(out['outputs']['y'], X @ W.T, atol=1e-12) and out['finite']
    delayed = {'nodes': {'x': {'type': 'Input', 'params': {'shape': [2]}},
                         'd': {'type': 'Delay', 'params': {'delay': [3.0, 3.0]}},
                         'y': {'type': 'Output', 'params': {'shape': [2]}}},
               'edges': [['x', 'd'], ['d', 'y']]}
    y = N.simulate_graph(delayed, X)['outputs']['y']
    assert np.all(y[:3] == 0.0) and np.allclose(y[3:], X[:2])
    # constant drive into an LIF: v_t = r x (1 - a^t); first spike at the first t with v_t > threshold
    tau, r, thr, x = 4.0, 1.0, 0.5, 0.8
    lif = {'nodes': {'x': {'type': 'Input', 'params': {'shape': [1]}},
                     'n': {'type': 'LIF', 'params': {'tau': [tau], 'r': [r], 'v_leak': [0.0], 'v_threshold': [thr],
                                                     'v_reset': [0.0]}},
                     'y': {'type': 'Output', 'params': {'shape': [1]}}},
           'edges': [['x', 'n'], ['n', 'y']]}
    sim = N.simulate_graph(lif, np.full((12, 1), x), record_states=True)
    a = math.exp(-1.0 / tau)
    expected = next(t for t in range(1, 50) if r * x * (1 - a ** t) > thr)
    assert int(np.flatnonzero(sim['outputs']['y'][:, 0])[0]) == expected - 1
    assert sim['states']['n']['v'][0, 0] == pytest.approx(r * x * (1 - a))
    # a recurrent loop needs a delay
    loop = _chain(np.eye(2), extra_edges=[['lin', 'lin']])
    with pytest.raises(ValueError, match='algebraic loop'):
        N.simulate_graph(loop, X)


def test_finite_check_detects_overflow_even_when_a_spike_resets_it():
    graph = _chain(np.eye(1))
    graph['nodes']['lin'] = {'type': 'IF', 'params': {'r': [1e300], 'v_threshold': [1], 'v_reset': [0]}}
    with np.errstate(over='ignore'):
        sim = N.simulate_graph(graph, [[1e100]], record_states=True)
    assert np.isinf(sim['states']['lin']['v'][0, 0])
    assert sim['outputs']['y'][0, 0] == 1 and not sim['finite']


def test_exported_graph_simulates_finitely_with_activity():
    graph = N.brain_to_nir(_brain(), input_dim=8)
    X = np.random.default_rng(9).uniform(-1.0, 1.0, (60, 8))
    sim = N.simulate_graph(graph, X)
    assert sim['finite'] and sum(sim['spike_counts'].values()) > 0
    assert sim['outputs']['output'].shape == (60, 16) and set(sim['delay_breaks']) == {
        f'delay_{k}' for k in graph['metadata']['delay_classes_steps']}
    assert np.array_equal(sim['outputs']['output'], N.simulate_graph(graph, X)['outputs']['output'])


def test_json_fallback_round_trip(tmp_path):
    graph = N.brain_to_nir(_brain(), input_dim=8)
    path = tmp_path / 'brain.nir'
    rt = N.write_nir(graph, path, backend='json')
    assert rt['format'] == N.FALLBACK_FORMAT and rt['round_trip_equal']
    assert rt['node_count'] == rt['read_node_count'] and rt['edge_count'] == rt['read_edge_count'] == len(graph['edges'])
    doc = json.loads(path.read_text(encoding='utf-8'))
    assert doc['format'] == 'nir-json-fallback'
    back = N.read_nir(path)
    assert N.graphs_equal(graph, back) and back['metadata']['lossy_features'] == graph['metadata']['lossy_features']
    # a changed weight breaks equality; a wrong tag or garbage is rejected
    doc['graph']['nodes']['input_projection']['params']['weight'][0][0] += 1e-9
    path.write_text(json.dumps(doc), encoding='utf-8')
    assert not N.graphs_equal(graph, N.read_nir(path))
    doc['format'] = 'something-else'
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError):
        N.read_nir(path)
    path.write_bytes(b'\x00\x01garbage')
    with pytest.raises(ValueError):
        N.read_nir(path)
    with pytest.raises(ValueError):
        N.write_nir({'nodes': {}, 'edges': []}, tmp_path / 'bad.nir', backend='json')
    with pytest.raises(ValueError):
        N.write_nir(graph, tmp_path / 'x.nir', backend='onnx')
    if not N.nir_status()['available']:
        with pytest.raises(ValueError):
            N.write_nir(graph, tmp_path / 'y.nir', backend='nir')


def test_real_nir_round_trip(tmp_path):
    pytest.importorskip('nir')
    graph = N.brain_to_nir(_brain(), input_dim=8)
    rt = N.write_nir(graph, tmp_path / 'brain.nir', backend='nir')
    assert rt['format'] == N.NIR_FORMAT and rt['round_trip_equal']
    assert N.graphs_equal(graph, N.read_nir(tmp_path / 'brain.nir'))


# ====================================================================== receipts
@pytest.fixture(scope='module')
def hh_receipt():
    before = _global_state()
    receipt = H.run_experiment(quick=True)
    return receipt, _same_global_state(before, _global_state())


@pytest.fixture(scope='module')
def nir_receipt():
    before = _global_state()
    receipt = N.run_experiment(quick=True)
    return receipt, _same_global_state(before, _global_state())


def test_hh_quick_receipt_is_sealed_passes_and_is_deterministic(hh_receipt):
    receipt, untouched = hh_receipt
    assert untouched                                                   # no global RNG or thread state changed
    assert verify_receipt(receipt) and receipt['evidence_tier'] == EVIDENCE_TIER
    assert receipt['kind'] == 'validation' and receipt['verdict'] == 'pass' and all(receipt['checks'].values())
    assert receipt['config']['quick'] is True and receipt['config']['seeds'] == [0, 1, 2]
    assert receipt['neuroweave_reference']['seed_invariant']
    cmp = receipt['neuroweave_reference']['comparison']
    assert cmp['spearman_suprathreshold'] is None or -1.0 <= cmp['spearman_suprathreshold'] <= 1.0
    assert receipt['cable']['reference']['applies'] and receipt['measured_wallclock_s']['total'] > 0
    assert len(json.dumps(receipt)) < 64_000
    assert H.run_experiment(quick=True)['digest'] == receipt['digest']
    tampered = copy.deepcopy(receipt)
    tampered['verdict'] = 'fail'
    assert not verify_receipt(tampered)
    for bad in ({'quick': 1}, {'surprise': 1}, {'seeds': (0, 0)}, {'seeds': ()}, {'fi_levels': (0.0, 1.0)},
                {'cable_diameters_um': (476.0, 238.0)}, {'patch_dt_ms': 0.5}):
        with pytest.raises(ValueError):
            H.run_experiment(**{'quick': True, **bad})


def test_hh_receipt_fails_when_a_known_answer_check_is_made_impossible():
    # an impossibly tight velocity-reference tolerance must turn the verdict into 'fail' (checks are live)
    receipt = H.run_experiment(quick=True, seeds=(0,), reference_tolerance=1e-9, fi_window_ms=20.0,
                               fi_transient_ms=5.0, threshold_window_ms=10.0, cable_compartments=40,
                               fi_levels=(0.0, 0.5, 1.1, 2.0, 4.0), search_batch=16,
                               refractory_max_interval_ms=12.0, neuroweave_steps=60, neuroweave_transient=10)
    assert verify_receipt(receipt) and receipt['verdict'] == 'fail'
    assert not receipt['checks']['reference_velocity_within_tolerance_of_hh1952_computed']
    assert 'reference_velocity' in receipt['verdict_basis']


def test_nir_quick_receipt_is_sealed_passes_and_is_deterministic(nir_receipt):
    receipt, untouched = nir_receipt
    assert untouched
    assert verify_receipt(receipt) and receipt['evidence_tier'] == EVIDENCE_TIER
    assert receipt['kind'] == 'validation' and receipt['verdict'] == 'pass' and all(receipt['checks'].values())
    assert receipt['file_format'] == (N.NIR_FORMAT if N.nir_status()['available'] else N.FALLBACK_FORMAT)
    for row in receipt['per_seed']:
        assert row['validation']['valid'] and row['round_trip']['round_trip_equal'] and row['simulation']['finite']
        assert row['recurrent_parity']['applicable'] and row['recurrent_parity']['max_abs_difference'] <= 1e-9
        assert row['cell_parity']['synaptic_current_max_abs_difference'] <= 1e-9
    assert {f['feature'] for f in receipt['lossy_features']} >= {'graded_output_transmission', 'stdp'}
    assert N.run_experiment(quick=True)['digest'] == receipt['digest']
    assert N.run_experiment(quick=True, seeds=(4,))['digest'] != receipt['digest']
    tampered = copy.deepcopy(receipt)
    tampered['checks']['graph_valid'] = False
    assert not verify_receipt(tampered)
    for bad in ({'quick': 'yes'}, {'surprise': 1}, {'seeds': (1, 1)}, {'mechanisms': 'stdp'},
                {'mechanisms': ('no_such',)}, {'neurons': 2}, {'backend': 'onnx'}):
        with pytest.raises(ValueError):
            N.run_experiment(**{'quick': True, **bad})
