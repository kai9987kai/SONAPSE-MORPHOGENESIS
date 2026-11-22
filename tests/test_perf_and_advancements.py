"""Regression, equivalence, and contract tests for whole-project optimizations and innovations."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from synapse.cli import main as cli_main
from synapse.complex_world import ComplexWorld
from synapse.config import SynapseConfig
from synapse.dsag.config import DSAGSpecV1
from synapse.dsag.model import DualSomaAxodendriticGraft
from synapse.experience import ExperienceWorldModel
from synapse.frontier import connectomics as fc_mod
from synapse.frontier.hh_cable import HHCable
from synapse.frontier.runtime import FrontierObserverHub
from synapse.frontier.spec import OBSERVER_MODULE_IDS, FrontierSpecV1
from synapse.frontier.synthesis import (
    REGIME_LABELS,
    SYNTHESIS_SCHEMA,
    synthesize_frontier_observers,
    verify_synthesis_digest,
)
from synapse.neurogenesis import ConnectomeNeurogenesisEngine
from synapse.neuroweave.brain import NeuroWeaveBrain
from synapse.neuroweave.cells import CellPopulation
from synapse.neuroweave.events import EventBus, json_safe
from synapse.neuroweave.experiments import _count_numbers
from synapse.neuroweave.spec import NeuroWeaveSpecV1
from synapse.observatory import NeuralObservatory
from synapse.perf_lab import (
    PERF_RECEIPT_SCHEMA,
    PERF_TELEMETRY_SCHEMA,
    StepPerfTracker,
    run_scaling_profile,
    run_subsystem_pareto_sweep,
    verify_perf_receipt,
)
from synapse.runtime import LabRuntime
from synapse.server import Workbench
from synapse.stateio import digest
from synapse.uncertainty_calibration import LatentDynamicsEnsemble, OutDistributionDetector


# ====================================================================== Pillar 1: Optimization Equivalences
def test_neurogenesis_row_filtered_ranking_and_external_mutation_invalidation():
    engine = ConnectomeNeurogenesisEngine(num_neurons=32, seed=9)
    rng = np.random.default_rng(9)
    for _ in range(15):
        engine.record_activity(rng.normal(0.0, 0.25, 32).astype(np.float32))
    assert engine._bounds_valid
    active_rows = np.flatnonzero(engine._row_upper_bound >= 0.01)
    fast = list(engine._ranked_candidates_rows(engine._rate_cov, 32, active_rows, threshold=0.01))
    full = list(engine._ranked_candidates(engine._rate_cov, 32, threshold=0.01))
    assert set(fast) == set(full)
    flat = engine._rate_cov.ravel()
    assert np.array_equal(flat[fast], flat[full])

    # Accessing rate_cov externally invalidates upper-bound shortcut so direct edits are respected.
    engine.rate_cov.fill(0.0)
    engine.rate_cov[3, 7] = 0.5
    assert not engine._bounds_valid
    assert list(engine._ranked_candidates(engine._rate_cov, 32, threshold=0.01)) == [3 * 32 + 7]


def test_experience_world_model_matmul_matches_reference_einsum():
    rng = np.random.default_rng(17)
    model = ExperienceWorldModel(state_dim=16, action_dim=5, ensemble_size=3, seed=17, context_dim=2)
    for _ in range(18):
        s = rng.uniform(-1.0, 1.0, 16)
        a = rng.uniform(-0.8, 0.8, 5)
        ns = np.clip(s + rng.normal(0.0, 0.05, 16), -2.0, 2.0)
        r = float(rng.normal(0.0, 0.3))
        model.update(s, a, ns, r, done=False)

    state = rng.uniform(-1.0, 1.0, 16)
    actions = rng.uniform(-0.8, 0.8, (11, 5))
    phi = model._features(state, actions)
    ref_preds = np.einsum('nf,efk->enk', phi, model.weights, optimize=True)
    ref_lev = np.einsum('nf,efg,ng->en', phi, model.covariance, phi, optimize=True)
    got = model.predict(state, actions)
    np.testing.assert_allclose(got['reward'], ref_preds.mean(axis=0)[:, -1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got['delta'][:, :model.predicted_dim], ref_preds.mean(axis=0)[:, :-1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got['coverage'], np.sqrt(np.maximum(ref_lev.mean(axis=0), 0.0)), atol=1e-12, rtol=1e-12)

    member_states = rng.uniform(-1.0, 1.0, (3, 16))
    got_m = model.predict_members(member_states, actions)
    phi_m = np.stack([model._features(ms, actions) for ms in member_states])
    ref_mp = np.einsum('enf,efk->enk', phi_m, model.weights, optimize=True)
    ref_ml = np.einsum('enf,efg,eng->en', phi_m, model.covariance, phi_m, optimize=True)
    np.testing.assert_allclose(got_m['reward'], ref_mp[:, :, -1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got_m['coverage'], np.sqrt(np.maximum(ref_ml, 0.0)), atol=1e-12, rtol=1e-12)


def test_uncertainty_estimate_matches_forward_all_and_ood_spd_inverse():
    ens = LatentDynamicsEnsemble(state_dim=16, action_dim=5, ensemble_size=3, seed=5)
    z = torch.randn(4, 16)
    a = torch.randn(4, 5)
    with torch.no_grad():
        _, all_r_mu, all_r_log_var = ens.forward_all(z, a)
        ref_mean = all_r_mu.mean(dim=0).squeeze(-1)
        ref_ep = all_r_mu.var(dim=0, unbiased=False).squeeze(-1)
        ref_al = torch.exp(all_r_log_var).mean(dim=0).squeeze(-1)
    mean_r, ep_var, al_var = ens.estimate_uncertainty(z, a)
    assert torch.equal(mean_r, ref_mean)
    assert torch.equal(ep_var, ref_ep)
    assert torch.equal(al_var, ref_al)

    det = OutDistributionDetector(latent_dim=8, seed=11)
    rng = np.random.default_rng(11)
    samples = rng.normal(0.0, 1.0, (30, 8)).astype(np.float32)
    det.update_support(samples)
    assert det._fitted and det.inv_cov is not None
    expected_cov = np.cov(samples, rowvar=False) + np.eye(8) * 1e-4
    np.testing.assert_allclose(det.inv_cov, np.linalg.pinv(expected_cov), atol=1e-9, rtol=1e-9)


def test_neuroweave_bounds_events_count_numbers_and_readout_cache():
    from synapse.neuroweave.cells import _BOUNDS
    cells = CellPopulation(16, seed=3, regions=[0] * 16, signs=[1] * 16)
    cells.fast[0] = np.nan
    cells.slow[1] = np.inf
    cells.context[2] = -np.inf
    cells.output[3] = 9.5
    cells._enforce_bounds()
    assert cells.fast[0] == 0.0
    assert cells.slow[1] == _BOUNDS['slow'][1]
    assert cells.context[2] == _BOUNDS['context'][0]
    assert cells.output[3] == _BOUNDS['output'][1]

    # json_safe preserves exact diagnostic paths on error while fast-pathing valid payloads
    valid = {'a': [1, 2.5, True, None, {'b': 'ok'}]}
    copied = json_safe(valid)
    assert copied == valid and copied is not valid
    with pytest.raises(ValueError, match=r'payload\.x\[1\]: non-finite float'):
        json_safe({'x': [1.0, float('nan')]})

    bus = EventBus(strict=True)
    seen = []
    bus.subscribe('state', lambda env: seen.append(env.payload))
    bus.publish('state', {'activity_mean': 0.1, 'activity_std': 0.25}, step=1)
    assert len(seen) == 1 and seen[0]['activity_mean'] == 0.1

    nested = {'a': [1, 2.0, True, 's', None, {'b': (3, 4.5, False)}]}
    assert _count_numbers(nested) == 4

    brain = NeuroWeaveBrain(NeuroWeaveSpecV1(neurons=24, max_neurons=48, seed=5))
    f1 = brain.readout_features()
    brain.grow_neurons(2, 0, reason='test')
    f2 = brain.readout_features()
    assert f1.shape == f2.shape and np.all(np.isfinite(f2))


def test_frontier_auroc_ties_and_hh_cable_workspace_exact_match():
    assert fc_mod.auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0
    assert fc_mod.auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0
    assert fc_mod.auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5
    assert fc_mod.auroc([1.0, 2.0, 2.0, 3.0], [0, 1, 0, 1]) == pytest.approx(0.875)
    assert fc_mod.auroc([1.0, 2.0], [1, 1]) is None

    cable = HHCable(compartments=8, length_um=10000.0, dt_ms=0.01, method='rk4')
    S = np.empty((4, 2, 8))
    v0 = cable.resting_potential()
    S[0], S[1], S[2], S[3] = v0, 0.05, 0.6, 0.32
    stim0 = np.array([0.0, 15.0])
    gcol = np.full((2, 1), cable._gax)
    ws = cable._make_rk4_workspace(2, 8)
    S_ref = cable._step_rk4(S, stim0, gcol)
    S_ws = cable._step_rk4_ws(S, stim0, gcol, ws, np.empty_like(S))
    np.testing.assert_allclose(S_ws, S_ref, atol=1e-13, rtol=1e-13)


def test_dsag_bmm_and_cache_topology_invalidation():
    spec = DSAGSpecV1(
        enabled=True, gain=0.05, max_gain=0.1, seed=11,
        phase_gate_enabled=True, structural_growth_enabled=True,
        branch_segments=(0, 1), max_branches=4,
    )
    model = DualSomaAxodendriticGraft(input_dim=24, spec=spec)
    r = torch.linspace(-0.5, 0.5, 24)
    phases = torch.linspace(0.0, 3.14, 16)
    y1 = model(r, phase_angles=phases, advance=True)
    assert torch.isfinite(y1).all()
    old_key = model._route_cache_key
    model.apply_topology("add_axon_input_branch", segment_index=2)
    y2 = model(r, phase_angles=phases, advance=True)
    assert torch.isfinite(y2).all()
    assert model._route_cache_key != old_key


def test_stateio_digest_zero_copy_noncontiguous_and_scalar_arrays():
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    sliced = arr[:, ::2]
    assert not sliced.flags.c_contiguous
    assert digest(sliced) == digest(np.ascontiguousarray(sliced))
    assert digest(np.array(3.25, dtype=np.float64)) == digest(np.array(3.25, dtype=np.float64))
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert digest(t) == digest(t.clone())


def test_observatory_lazy_vectorized_positions_and_complex_world_grid_cache():
    rt = LabRuntime(SynapseConfig(nodes_per_source=16, seed=7))
    obs = NeuralObservatory(rt.agent, seed=42)
    # Accessing live activity does not trigger 3D position computation
    act_payload = obs.get_live_activity_payload()
    assert obs._node_positions_3d is None
    assert len(act_payload['node_activations']) == rt.agent.brain.n
    # Structure payload materializes positions and clusters cleanly
    struct = obs.get_structure_payload()
    assert obs._node_positions_3d is not None
    assert struct['total_nodes'] == rt.agent.brain.n
    assert len(struct['lod_clusters']) > 0

    cw = ComplexWorld(seed=7, curriculum='apex_megasite')
    g1 = cw._get_elevation_grid(resolution=11)
    g2 = cw._get_elevation_grid(resolution=11)
    assert g1 == g2
    g1[0][0] = 999.0  # caller mutation must not corrupt cached grid
    assert cw._get_elevation_grid(resolution=11) == g2


def test_runtime_step_skip_snapshot_preserves_exact_trajectory():
    rt_a = LabRuntime(SynapseConfig(nodes_per_source=16, seed=13))
    rt_b = LabRuntime(SynapseConfig(nodes_per_source=16, seed=13))
    snap_a = rt_a.step(8, learn=True, include_snapshot=True)
    ret_b = rt_b.step(8, learn=True, include_snapshot=False)
    assert ret_b is None
    snap_b = rt_b.snapshot()
    assert snap_a['steps'] == snap_b['steps'] == 8
    np.testing.assert_array_equal(rt_a.last_action, rt_b.last_action)
    assert digest(rt_a.agent.export_state()) == digest(rt_b.agent.export_state())


# ====================================================================== Pillar 2: Whole-Project Innovations
def test_step_perf_tracker_and_runtime_perf_telemetry():
    tracker = StepPerfTracker(window=16)
    empty = tracker.summary(neurons=48)
    assert empty['schema'] == PERF_TELEMETRY_SCHEMA
    assert empty['recorded_steps'] == 0
    assert empty['bottleneck_stage'] is None

    for i in range(10):
        tracker.record(10.0 + i * 0.1, 5.0, 2.0, 1.0, 1.5)
    summary = tracker.summary(neurons=48, active_subsystems=['dsag'])
    assert summary['recorded_steps'] == 10
    assert summary['bottleneck_stage'] == 'act'
    assert summary['measured_steps_per_sec'] > 0
    assert summary['measured_neuron_steps_per_sec'] > 0

    with pytest.raises(ValueError):
        StepPerfTracker(window=2)

    rt = LabRuntime(SynapseConfig(nodes_per_source=16, seed=7))
    rt.step(6, learn=True)
    perf = rt.perf_telemetry()
    assert perf['schema'] == PERF_TELEMETRY_SCHEMA
    assert perf['recorded_steps'] == 6
    assert perf['neurons'] == rt.agent.brain.n
    assert perf['total_ms']['mean'] > 0
    assert perf['bottleneck_stage'] in ('act', 'uncertainty', 'world', 'learn', 'overhead')


def test_perf_lab_scaling_and_pareto_receipts_verify_and_stay_deterministic():
    r1 = run_scaling_profile(neuron_counts=(48, 96), steps=4, learn=True, seed=7)
    r2 = run_scaling_profile(neuron_counts=(48, 96), steps=4, learn=True, seed=7)
    assert r1['schema'] == PERF_RECEIPT_SCHEMA
    assert verify_perf_receipt(r1)
    assert verify_perf_receipt(r2)
    # Wall-clock-independent SHA-256 digest must match across repeated deterministic runs!
    assert r1['digest'] == r2['digest']
    assert len(r1['profiles']) == 2
    assert r1['profiles'][0]['trajectory_digest'] == r2['profiles'][0]['trajectory_digest']

    # Tampering with deterministic trajectory digest breaks receipt verification
    tampered = dict(r1)
    tampered['profiles'] = [dict(r1['profiles'][0], trajectory_digest='0' * 64), r1['profiles'][1]]
    assert not verify_perf_receipt(tampered)

    pareto = run_subsystem_pareto_sweep(
        nodes_per_source=16, steps=4, learn=True, seed=7, arms=('baseline', 'dsag', 'frontier_observers')
    )
    assert verify_perf_receipt(pareto)
    assert len(pareto['arms']) == 3
    assert any(arm['pareto_efficient'] for arm in pareto['arms'])


def test_frontier_cross_scale_synthesis_read_only_and_verified():
    # Disabled hub -> disabled regime with null indices (no fabrication)
    off_hub = FrontierObserverHub(FrontierSpecV1(enabled=False), n_units=16)
    off_syn = off_hub.synthesis()
    assert off_syn['schema'] == SYNTHESIS_SCHEMA
    assert off_syn['regime'] == 'disabled'
    assert off_syn['indices']['multiscale_integration_index'] is None
    assert verify_synthesis_digest(off_syn)

    # Active hub across all 5 observers
    spec = FrontierSpecV1(
        enabled=True, modules=OBSERVER_MODULE_IDS, seed=7,
        history=64, fc_window=32, max_observed_units=32, sae_features=16, sae_k=4, sync_pairs=16,
    )
    hub = FrontierObserverHub(spec, n_units=32)
    rng = np.random.default_rng(7)
    for t in range(40):
        hub.observe(np.sin(0.2 * t + np.linspace(0.0, 3.0, 32)) * 0.5 + rng.normal(0.0, 0.1, 32))

    before = digest(hub.export_state())
    syn = hub.synthesis()
    after = digest(hub.export_state())
    assert before == after  # strictly read-only!
    assert syn['regime'] in REGIME_LABELS
    assert syn['active_scales'] == 3
    assert syn['indices']['multiscale_integration_index'] is not None
    assert 0.0 <= syn['indices']['multiscale_integration_index'] <= 1.0
    assert verify_synthesis_digest(syn)
    assert syn['digest'] == synthesize_frontier_observers(hub.summary())['digest']

    bad = dict(syn, regime='tampered')
    assert not verify_synthesis_digest(bad)


def test_cli_and_workbench_api_for_perf_lab_and_frontier_synthesis(tmp_path, capsys):
    # CLI: perf-lab
    rc = cli_main(['--nodes', '16', 'perf-lab', '--mode', 'scaling', '--neurons', '48', '--steps', '3', '--learn'])
    assert rc == 0
    out = capsys.readouterr().out
    assert PERF_RECEIPT_SCHEMA in out

    # CLI: frontier synthesis
    rc2 = cli_main(['frontier', 'synthesis', '--steps', '16', '--units', '16'])
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert SYNTHESIS_SCHEMA in out2

    # Workbench HTTP API (in-process)
    wb = Workbench(config=SynapseConfig(nodes_per_source=16, seed=7), output_root=tmp_path)
    wb.runtime.step(4, learn=True)
    perf = wb.get('/api/perf/telemetry')
    assert perf['schema'] == PERF_TELEMETRY_SCHEMA
    assert perf['recorded_steps'] == 4

    bench = wb.post('/api/perf/benchmark', {'mode': 'scaling', 'neurons': [48], 'steps': 3, 'learn': False})
    assert verify_perf_receipt(bench)

    syn_resp = wb.get('/api/frontier/synthesis')
    assert syn_resp['source'] == 'lab'
    assert verify_synthesis_digest(syn_resp['synthesis'])

    step_resp = wb.post('/api/frontier/step', {'steps': 12})
    assert 'synthesis' in step_resp
    assert verify_synthesis_digest(step_resp['synthesis'])
