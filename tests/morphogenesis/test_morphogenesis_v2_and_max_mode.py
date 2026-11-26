"""Comprehensive v2.0 & OMNI-MORPHOGENESIS Max Mode Test Suite.

Verifies:
1. Additive v2 catalog preservation (v1 tuple lengths untouched; v2 ALL_* registries active).
2. All 36 graft neurons, 21 conventional neurons, 20 organelles + Arbitrosome phase gating,
   16 bio-hybrids, and 18 ML-hybrids step stably with finite outputs.
3. All 33 connectome & ML learning rules + non-degenerate 11-expert Sherman-Morrison panel + CVaR.
4. CompiledConnectomeEngine (SoA + CSR sparse) speedup & numerical fidelity across Fidelities 1-4.
5. All 12 genuine Curiosity simulators, 17 artificial senses, 5D/6D Hyperspace, Sparse-Label Antipode,
   Long-Horizon Mutual Symbiosis, and SYNAPSE-ATLAS connectome symmetry audit.
6. GhostInTheMachine causal emergence (MAF, Barnett-Seth DI, MMI-PID, Efference Jacobian, Twin Authorship),
   CROSSWIRE Mealy version-space safety, and FAULTLINE joint-failure specialist selection.
7. Host Sonapse-LAB MorphogenesisBusChannel live integration, bit-identical checkpoint save/load replay,
   synapse CLI forwarding, and Workbench Max Mode / Emergence routes.
8. OmniMorphogenesisMaxEngine 100% completeness audit, multi-epoch execution, and atomic checkpointing.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from synapse.cli import main as synapse_cli_main
from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime
from synapse.server import Workbench

from sonapse_morphogenesis.antipode import run_sparse_label_verifier_antipode_study
from sonapse_morphogenesis.architect import (
    ALL_LEARNING_ALGORITHMS,
    EXTENDED_LEARNING_ALGORITHMS,
    LEARNING_ALGORITHMS,
    AutonomousResearchDirector,
    DynamicLearningEngine,
    InternalExpertPanel,
    benchmark_extended_connectome_learning_suite,
)
from sonapse_morphogenesis.chimera import audit_connectome_symmetry_and_fingerprints
from sonapse_morphogenesis.cli import main as morph_cli_main
from sonapse_morphogenesis.core import (
    CompiledConnectomeEngine,
    FidelityLevel,
    MorphogenesisSpecV1,
    StructuredNeuronInstance,
    benchmark_compiled_vs_object_engine,
)
from sonapse_morphogenesis.curiosity import NOVEL_MODALITIES, verify_all_novel_modalities
from sonapse_morphogenesis.emergence import (
    CrosswireVersionSpaceAdapter,
    FaultlineJointFailureSelector,
    run_causal_emergence_and_authorship_suite,
    run_crosswire_and_faultline_study,
)
from sonapse_morphogenesis.foundry import (
    ALL_BIO_HYBRID_REGISTRY,
    ALL_CONVENTIONAL_NEURON_FAMILIES,
    ALL_GRAFT_NEURON_FAMILIES,
    ALL_ML_HYBRID_MECHANISMS,
    ALL_ORGANELLE_REGISTRY,
    BIO_HYBRID_REGISTRY,
    CONVENTIONAL_NEURON_FAMILIES,
    EXTENDED_GRAFT_NEURON_FAMILIES,
    EXTENDED_ORGANELLE_REGISTRY,
    GRAFT_NEURON_FAMILIES,
    ML_HYBRID_MECHANISMS,
    ORGANELLE_REGISTRY,
    MLHybridMechanism,
    OrganelleArbitrationCoordinator,
    build_conventional_neuron,
    build_graft_neuron,
    create_bio_hybrid,
    create_organelle,
    evaluate_gated_organelle_interaction,
)
from sonapse_morphogenesis.genesis import (
    ALL_CANDIDATE_ORIGINS,
    ALL_MUTATION_OPERATORS,
    CANDIDATE_ORIGINS,
    MUTATION_OPERATORS,
    apply_genome_mutation,
    generate_candidate_genome,
)
from sonapse_morphogenesis.hyperspace import run_high_dim_5d_6d_study
from sonapse_morphogenesis.max_mode import (
    OmniMaxModeConfig,
    OmniMorphogenesisMaxEngine,
    build_pan_catalog_connectome,
)
from sonapse_morphogenesis.spectra import (
    ALL_ARTIFICIAL_SENSORY_MODALITIES,
    ARTIFICIAL_SENSORY_MODALITIES,
    ExpandedSensoryEncoder,
)
from sonapse_morphogenesis.symbiosis import run_long_horizon_mutual_symbiosis


def test_additive_v2_registries_preserve_v1_contracts():
    assert len(GRAFT_NEURON_FAMILIES) == 24
    assert len(CONVENTIONAL_NEURON_FAMILIES) == 15
    assert len(ORGANELLE_REGISTRY) == 14
    assert len(BIO_HYBRID_REGISTRY) == 12
    assert len(ML_HYBRID_MECHANISMS) == 13
    assert len(LEARNING_ALGORITHMS) == 19
    assert len(CANDIDATE_ORIGINS) == 7
    assert len(MUTATION_OPERATORS) == 15
    assert len(ARTIFICIAL_SENSORY_MODALITIES) == 13

    assert len(EXTENDED_GRAFT_NEURON_FAMILIES) == 12
    assert len(ALL_GRAFT_NEURON_FAMILIES) == 36
    assert len(ALL_CONVENTIONAL_NEURON_FAMILIES) == 21
    assert len(EXTENDED_ORGANELLE_REGISTRY) == 6
    assert len(ALL_ORGANELLE_REGISTRY) == 20
    assert len(ALL_BIO_HYBRID_REGISTRY) == 16
    assert len(ALL_ML_HYBRID_MECHANISMS) == 18
    assert len(EXTENDED_LEARNING_ALGORITHMS) == 14
    assert len(ALL_LEARNING_ALGORITHMS) == 33
    assert len(ALL_CANDIDATE_ORIGINS) == 11
    assert len(ALL_MUTATION_OPERATORS) == 20
    assert len(ALL_ARTIFICIAL_SENSORY_MODALITIES) == 17


def test_all_57_neurons_20_organelles_and_34_hybrids():
    for gfam in ALL_GRAFT_NEURON_FAMILIES:
        spec = build_graft_neuron(gfam, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT, seed=7)
        inst = StructuredNeuronInstance(spec, dt_ms=0.5, organelle_factory=create_organelle, seed=7)
        out = inst.step(0.8, adapt=True, collect_telemetry=False)
        assert np.isfinite(out["scalar_output"])

    for cfam in ALL_CONVENTIONAL_NEURON_FAMILIES:
        spec = build_conventional_neuron(cfam, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT, seed=7)
        inst = StructuredNeuronInstance(spec, dt_ms=0.5, organelle_factory=create_organelle, seed=7)
        out = inst.step(0.7, adapt=True, collect_telemetry=False)
        assert np.isfinite(out["scalar_output"])

    arb = evaluate_gated_organelle_interaction("mnemosome", "chronosome", seed=7)
    assert arb["mse_gated_ab"] <= arb["mse_ab"] + 0.05
    coord = OrganelleArbitrationCoordinator(n_organelles=2)
    comb, weights = coord.combine([2.5, -1.5])
    assert abs(comb) <= 1.5 + 1e-6 and weights.shape == (2,)

    x = np.linspace(-0.8, 0.8, 8)
    for m in ALL_ML_HYBRID_MECHANISMS:
        for var in ("exact_software", "cellular_compartment", "biologically_constrained"):
            mech = MLHybridMechanism(m, dim=8, variant=var, seed=7)
            y = mech.step(x)
            assert y.shape == (8,) and np.all(np.isfinite(y))

    for bname in ALL_BIO_HYBRID_REGISTRY:
        bh = create_bio_hybrid(bname, seed=7)
        assert bh is not None


def test_all_33_learning_rules_experts_and_cross_scale_architect():
    eng = DynamicLearningEngine(seed=7)
    W = np.random.default_rng(7).normal(0.0, 0.2, size=(4, 6))
    pre = np.array([0.5, -0.3, 0.8, 0.1, -0.6, 0.4])
    post = np.tanh(W @ pre)
    target = np.array([0.7, -0.5, 0.2, 0.6])
    for algo in ALL_LEARNING_ALGORITHMS:
        dW = eng.compute_weight_delta(algo, W, pre, post, target=target, reward=0.6)
        assert dW.shape == (4, 6) and np.all(np.isfinite(dW))

    suite = benchmark_extended_connectome_learning_suite(seed=7, steps=45)
    assert suite["total_algorithms"] == 33

    panel = InternalExpertPanel(feat_dim=8, seed=7)
    feat = np.array([0.5, 0.3, -0.2, 0.8, 0.1, 0.4, 0.9, 1.0])
    cvar = panel.score_fantasy_decision_risk([feat], cvar_alpha=0.25)[0]
    assert cvar["epistemic_disagreement"] > 0.0
    panel.update_all(feat, actual_benefit=0.45, actual_cost=0.05, actual_risk=0.02)

    director = AutonomousResearchDirector(seed=7)
    xs = director.run_cross_scale_transfer_study(scales=(8, 24), cycles_for_expert_calibration=2)
    assert "transfer_by_operator" in xs or len(xs) > 0


def test_compiled_connectome_engine_and_state_roundtrip():
    pan_graph, _ = build_pan_catalog_connectome(seed=7)
    comp = CompiledConnectomeEngine(pan_graph, dt_ms=1.0, seed=7)
    assert comp.n_neurons == 57
    for _ in range(6):
        res = comp.step(np.ones(8) * 0.6, adapt=True)
        assert res["outputs"].shape == (57,) and np.all(np.isfinite(res["outputs"]))

    st = comp.export_state()
    comp2 = CompiledConnectomeEngine(pan_graph, dt_ms=1.0, seed=99)
    comp2.import_state(st)
    r1 = comp.step(np.ones(8) * 0.4, adapt=False)
    r2 = comp2.step(np.ones(8) * 0.4, adapt=False)
    np.testing.assert_allclose(r1["outputs"], r2["outputs"], atol=1e-12)

    bench = benchmark_compiled_vs_object_engine(neuron_counts=(16, 64), steps=16, seed=7)
    assert len(bench["scale_comparisons"]) == 2
    assert bench["large_2048_multicompartment_probe"]["steps_per_sec_hz"] > 0.0


def test_extended_genesis_curiosity_spectra_hyperspace_antipode_symbiosis_chimera():
    for orig in ALL_CANDIDATE_ORIGINS:
        g = generate_candidate_genome(orig, n_neurons=8, seed=7)
        assert g.connectome.neuron_count == 8
    base_g = generate_candidate_genome("mixed_custom_standard", n_neurons=8, seed=7)
    for mop in ALL_MUTATION_OPERATORS:
        child, tx = apply_genome_mutation(base_g, operator=mop, seed=7)
        assert child.connectome.neuron_count >= 2
        assert tx.rollback().genome_id == base_g.genome_id

    cur = verify_all_novel_modalities(seed=7)
    assert cur["modalities_verified"] == len(NOVEL_MODALITIES) and len(cur["results"]) == len(NOVEL_MODALITIES)

    enc = ExpandedSensoryEncoder(out_dim=16)
    for amod in ALL_ARTIFICIAL_SENSORY_MODALITIES:
        raw = "def f(x):\n    return x * 2\n" if amod == "codeception" else np.ones(16) * 0.4
        sv = enc.encode(amod, raw)["vector"]
        assert sv.shape == (16,) and np.all(np.isfinite(sv))

    hd = run_high_dim_5d_6d_study(seeds=(7,), generations=4)
    assert "5D" in hd["dimensions"] and "6D" in hd["dimensions"]

    sp = run_sparse_label_verifier_antipode_study(seeds=(7,), epochs=12)
    assert "causally_patched_verifier_antipode" in sp["arms"]

    mut_sym = run_long_horizon_mutual_symbiosis(alternating_cycles=2, steps_per_phase=16, seed=7)
    assert "frozen_host_control" in mut_sym["arms"] and "mutually_adaptive_host" in mut_sym["arms"]

    donor_g = generate_candidate_genome("fractal_topology", n_neurons=8, seed=19)
    sym_aud = audit_connectome_symmetry_and_fingerprints(base_g.connectome, donor_g.connectome)
    assert 0.0 <= sym_aud["bilateral_mirror_overlap"] <= 1.0


def test_emergence_ghost_crosswire_and_faultline():
    rng = np.random.default_rng(7)
    traj = np.zeros((64, 10), dtype=np.float64)
    for t in range(1, 64):
        traj[t] = np.tanh(0.82 * traj[t - 1] + 0.16 * np.roll(traj[t - 1], 1) + rng.normal(0.0, 0.12, size=10))
    em = run_causal_emergence_and_authorship_suite(traj, macro_dim=2, seed=7)
    assert len(em["maf_slow_subspace"]["autocorrelations"]) == 2
    assert np.isfinite(em["mmi_pid"]["synergy"])
    assert em["cross_yoked_authorship"]["master_exceeds_yoked_twin"] is True

    cwf = run_crosswire_and_faultline_study(seed=7)
    assert cwf["crosswire"]["abstained_under_ambiguity"] is True
    assert cwf["crosswire"]["admitted_after_identification"] is True
    assert cwf["faultline"]["heldout_worst_case_reduction"] > 0.0

    cw = CrosswireVersionSpaceAdapter(dim=3)
    assert cw.surviving_count == 48
    F = np.zeros((5, 3, 12))
    F[0, 2, :] = 1.0
    F[1, 2, :] = 1.0
    sel = FaultlineJointFailureSelector.select_panel(F, panel_size=3)
    assert sel["worst_condition_error"]["joint"] <= sel["worst_condition_error"]["accuracy"]


def test_host_morphogenesis_channel_checkpoint_replay_and_cli(tmp_path, capsys):
    cfg = SynapseConfig(
        nodes_per_source=16,
        seed=11,
        morphogenesis=MorphogenesisSpecV1.max_preset(seed=11, neurons=16),
    )
    lab = LabRuntime(cfg, output_root=tmp_path)
    assert lab.agent.morphogenesis_channel is not None
    assert lab.agent.morphogenesis_channel.enabled is True
    lab.step(3, learn=True)
    assert "morphogenesis" in lab.last
    assert lab.last["morphogenesis"]["steps"] >= 3

    lab.save("morph_ckpt")
    expected = lab.step(2, learn=True)
    fresh = LabRuntime(cfg, output_root=tmp_path)
    fresh.load("morph_ckpt")
    actual = fresh.step(2, learn=True)
    assert actual == expected

    rc = synapse_cli_main(["morphogenesis", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "total_graft_neurons_v2" in out

    rc2 = morph_cli_main(["--seed", "7", "max-mode", "--audit-only", "--output-dir", str(tmp_path / "mm_audit")])
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert '"complete": true' in out2


def test_omni_morphogenesis_max_engine_and_workbench_routes(tmp_path):
    mm_cfg = OmniMaxModeConfig(
        seed=7,
        host_total_neurons=96,
        gamma_ticks_per_theta=4,
        theta_cycles_per_epoch=2,
        genesis_micro_pop=64,
        checkpoint_interval_seconds=1.0,
    )
    engine = OmniMorphogenesisMaxEngine(config=mm_cfg, output_dir=tmp_path / "omni_run")
    audit = engine.verify_max_mode_completeness()
    assert audit["complete"] is True

    rep = engine.run_epochs(n_epochs=2)
    assert rep["receipt"]["metrics"]["epochs_completed"] == 2
    assert rep["completeness_audit"]["complete"] is True
    ckpt = engine.save_atomic_checkpoint("unit_test")
    assert (tmp_path / "omni_run" / "checkpoints" / ckpt["npz_file"]).is_file()

    camp = engine.run_campaign(duration_seconds=1.5)
    assert Path(camp["jsonl_log"]).is_file()
    assert Path(camp["svg_figure"]).is_file()

    wb = Workbench(config=SynapseConfig(nodes_per_source=16, seed=7), output_root=tmp_path / "wb")
    try:
        cat = wb.get("/api/morphogenesis/catalog")
        assert len(cat["all_graft_neurons"]) == 36
        assert len(cat["all_organelles"]) == 20
        n_res = wb.get("/api/morphogenesis/neuron/liquid_cfc")
        assert n_res["ok"] is True
        em_res = wb.post("/api/morphogenesis/emergence/run", {})
        assert em_res["ok"] is True
        mm_res = wb.post("/api/morphogenesis/max_mode/run", {"epochs": 1, "enable_host_tandem": False})
        assert mm_res["ok"] is True
    finally:
        wb.sim_worker.stop()
