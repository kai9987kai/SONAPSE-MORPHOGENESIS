"""Comprehensive Timed Individual Modes, Subsystems & Sections Campaign Runner (v2.0).

Executes every individual SONAPSE MORPHOGENESIS v1+v2 module, optimization benchmark,
open-question resolution study, Max Mode architectural ablation, and host Sonapse-LAB
Tandem Max Sections 01–08 + PerfLab Pareto sweep prior to the 2-hour continuous Max Mode run.

Saves structured, cryptographically hashed JSON receipts and publication SVG figures under:
  - results/morphogenesis/v2_individual_sections/
  - docs/morphogenesis/figures/
  - docs/paper/figures/
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse.config import SynapseConfig
from synapse.perf_lab import run_scaling_profile, run_subsystem_pareto_sweep
from synapse.server import Workbench

from sonapse_morphogenesis import SCHEMA_VERSION, __version__
from sonapse_morphogenesis.affect import run_affective_emergence_investigation
from sonapse_morphogenesis.antipode import (
    TidalMoEInstrumentor,
    load_real_expanse_failure_corpus,
    run_antipode_six_way_study,
    run_sparse_label_verifier_antipode_study,
)
from sonapse_morphogenesis.architect import (
    AutonomousResearchDirector,
    benchmark_developmental_stage_learning_rules,
    benchmark_extended_connectome_learning_suite,
)
from sonapse_morphogenesis.chimera import (
    audit_connectome_symmetry_and_fingerprints,
    run_chimera_comparative_benchmark,
)
from sonapse_morphogenesis.core import (
    ExperimentRegistry,
    benchmark_compiled_vs_object_engine,
    content_digest,
    export_json_schemas,
    load_malecns_subgraph,
    probe_hardware,
)
from sonapse_morphogenesis.curiosity import DataQuarantinePipeline, verify_all_novel_modalities
from sonapse_morphogenesis.emergence import (
    run_cross_yoked_twin_assay,
    run_crosswire_and_faultline_study,
)
from sonapse_morphogenesis.foundry import (
    ALL_BIO_HYBRID_REGISTRY,
    EXTENDED_GRAFT_NEURON_FAMILIES,
    build_graft_neuron,
    compare_three_variants,
    create_bio_hybrid,
    evaluate_gated_organelle_interaction,
    render_neuron_svg,
    run_bio_hybrid_assays,
    run_foundry_benchmark_suite,
    run_heterogeneous_vs_homogeneous_experiment,
)
from sonapse_morphogenesis.genesis import (
    generate_candidate_genome,
    run_collective_experience_comparison,
    run_recursive_vs_evolutionary_comparison,
    screen_large_micro_population,
)
from sonapse_morphogenesis.hyperspace import (
    run_high_dim_5d_6d_study,
    run_hyperspace_developmental_study,
)
from sonapse_morphogenesis.max_mode import OmniMaxModeConfig, OmniMorphogenesisMaxEngine
from sonapse_morphogenesis.sentinel import (
    benjamini_hochberg_fdr,
    paired_bootstrap_ci,
    render_comparison_svg,
    run_matched_baseline_suite,
)
from sonapse_morphogenesis.spectra import (
    benchmark_artificial_vs_raw_encoding,
    run_internal_perception_suite,
)
from sonapse_morphogenesis.symbiosis import (
    COUPLING_MECHANISMS,
    run_long_horizon_mutual_symbiosis,
    run_symbiosis_cycle_experiment,
)


def _timed_section(name: str, fn: Callable[[], Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    print(f"[SECTION] Starting {name} ...", flush=True)
    t0 = time.perf_counter()
    c0 = time.process_time()
    payload = fn()
    wall_s = round(time.perf_counter() - t0, 3)
    cpu_s = round(time.process_time() - c0, 3)
    wrapped = {
        "section": name,
        "wallclock_seconds": wall_s,
        "cpu_seconds": cpu_s,
        "result": payload,
    }
    wrapped["sha256_digest"] = content_digest(wrapped)
    out_file = out_dir / f"{name}.json"
    out_file.write_text(json.dumps(wrapped, indent=2), encoding="utf-8")
    print(f"[SECTION] Finished {name} in {wall_s:.2f}s (CPU {cpu_s:.2f}s)", flush=True)
    return wrapped


def _run_max_mode_architecture_ablation(seeds: tuple[int, ...] = (7, 11, 19), epochs: int = 8) -> Dict[str, Any]:
    """4-Arm Controlled Ablation proving why OmniMorphogenesisMaxEngine's bespoke 3-tier architecture works:
      1. full_3tier_omni_max         : Full 3-tier Gamma/Theta/Circadian + Arbitrosome + FAULTLINE/Tether + 33-rule schedule
      2. no_faultline_uniform_panel  : Freezes uniform 1/6 specialist weights (disables FAULTLINE + Tether reliability routing)
      3. single_hebbian_rule_only    : Restricts plasticity to naive single-rule Hebbian instead of 33-rule stage schedule
      4. uncoupled_no_host_tandem    : Removes coupled Sonapse-LAB Tandem Max host bridge
    """
    arms = (
        "full_3tier_omni_max",
        "no_faultline_uniform_panel",
        "single_hebbian_rule_only",
        "uncoupled_no_host_tandem",
    )
    rewards_by_arm: Dict[str, List[float]] = {a: [] for a in arms}
    late_rewards_by_arm: Dict[str, List[float]] = {a: [] for a in arms}

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for seed in seeds:
            for arm in arms:
                cfg = OmniMaxModeConfig(
                    seed=seed,
                    host_total_neurons=96,
                    gamma_ticks_per_theta=5,
                    theta_cycles_per_epoch=4,
                    genesis_micro_pop=96,
                    enable_host_tandem=(arm != "uncoupled_no_host_tandem"),
                    enable_antipode_tidal_probe=False,
                )
                eng = OmniMorphogenesisMaxEngine(config=cfg, output_dir=td_path / f"{arm}_s{seed}")
                if arm == "single_hebbian_rule_only":
                    orig_compute = eng.learning_engine.compute_weight_delta
                    eng.learning_engine.compute_weight_delta = (  # type: ignore[method-assign]
                        lambda _algo, W, pre, post, target=None, reward=0.0, lr=0.01, _fn=orig_compute, **kw: _fn(
                            "hebbian", W, pre, post, target=target, reward=reward, lr=lr, **kw
                        )
                    )
                ep_rews: List[float] = []
                for _ in range(epochs):
                    if arm == "no_faultline_uniform_panel":
                        eng.specialist_error_history = {k: eng.specialist_error_history[k].__class__(maxlen=2) for k in eng.specialist_names}
                        eng.specialist_weights = {k: 1.0 / len(eng.specialist_names) for k in eng.specialist_names}
                    s = eng.run_single_epoch()
                    ep_rews.append(float(s["mean_epoch_reward"]))
                rewards_by_arm[arm].append(float(np.mean(ep_rews)))
                late_rewards_by_arm[arm].append(float(np.mean(ep_rews[epochs // 2 :])))

    summary = {}
    for arm in arms:
        vals = np.asarray(rewards_by_arm[arm], dtype=np.float64)
        lvals = np.asarray(late_rewards_by_arm[arm], dtype=np.float64)
        summary[arm] = {
            "mean_reward": round(float(np.mean(vals)), 5),
            "std_reward": round(float(np.std(vals)), 5),
            "late_half_mean_reward": round(float(np.mean(lvals)), 5),
            "per_seed_mean_reward": [round(float(x), 5) for x in vals],
        }
    ci_vs_uniform = paired_bootstrap_ci(rewards_by_arm["full_3tier_omni_max"], rewards_by_arm["no_faultline_uniform_panel"], seed=42)
    ci_vs_single_rule = paired_bootstrap_ci(rewards_by_arm["full_3tier_omni_max"], rewards_by_arm["single_hebbian_rule_only"], seed=42)
    return {
        "experiment": "omni_max_mode_4arm_architectural_ablation",
        "seeds": list(seeds),
        "epochs_per_seed": epochs,
        "arms": summary,
        "full_vs_uniform_bootstrap": ci_vs_uniform,
        "full_vs_single_hebbian_bootstrap": ci_vs_single_rule,
    }


def main() -> None:
    t_start = time.perf_counter()
    seeds = (7, 11, 19)
    heldout_seeds = (1009, 1013, 1019, 1021, 1031)

    out_dir = ROOT / "results" / "morphogenesis" / "v2_individual_sections"
    out_dir.mkdir(parents=True, exist_ok=True)
    morph_figs = ROOT / "docs" / "morphogenesis" / "figures"
    morph_figs.mkdir(parents=True, exist_ok=True)
    paper_figs = ROOT / "docs" / "paper" / "figures"
    paper_figs.mkdir(parents=True, exist_ok=True)
    schemas_dir = ROOT / "schemas" / "morphogenesis_v1"
    schemas_dir.mkdir(parents=True, exist_ok=True)

    for sname, schema in export_json_schemas().items():
        (schemas_dir / f"{sname}.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Render SVG morphologies for all 12 new v2 extended graft neurons
    for fam in EXTENDED_GRAFT_NEURON_FAMILIES:
        svg = render_neuron_svg(build_graft_neuron(fam))
        (morph_figs / f"neuron_{fam}.svg").write_text(svg, encoding="utf-8")

    sections: Dict[str, Any] = {}

    # 01. Performance & Compiled Engine Scaling + Host PerfLab Pareto Sweep
    sections["01_performance_and_scaling"] = _timed_section(
        "01_performance_and_scaling",
        lambda: {
            "compiled_vs_object_engine": benchmark_compiled_vs_object_engine(neuron_counts=(32, 128, 256, 512), steps=50, seed=7),
            "host_scaling_profile": run_scaling_profile(neuron_counts=(48, 192, 600, 1200), steps=16, learn=True, seed=7),
            "host_pareto_sweep_8arms": run_subsystem_pareto_sweep(nodes_per_source=16, steps=16, learn=True, seed=7),
        },
        out_dir,
    )

    # 02. Foundry v2: All 36 Graft + 21 Conventional Neurons, 20 Organelles + Arbitrosome, 16 Bio + 18 ML Hybrids
    def _run_sec02() -> Dict[str, Any]:
        cell_bench = run_foundry_benchmark_suite(seeds=seeds, steps=90, include_ablations=True, include_extended=True)
        org_gated = [
            evaluate_gated_organelle_interaction("mnemosome", "chronosome", seed=7),
            evaluate_gated_organelle_interaction("predicton", "contraston", seed=7),
            evaluate_gated_organelle_interaction("efferentosome", "crosswirosome", seed=7),
            evaluate_gated_organelle_interaction("criticalosome", "glymphatosome", seed=7),
        ]
        hetero = run_heterogeneous_vs_homogeneous_experiment(n_neurons=16, seeds=seeds, steps=90)
        bio_core = run_bio_hybrid_assays(seed=7)
        ext_bio_outputs = {}
        for b_id in ALL_BIO_HYBRID_REGISTRY:
            bh = create_bio_hybrid(b_id, seed=7)
            r_step = bh.step(np.array([0.6, -0.3, 0.8, 0.2, -0.5, 0.4]))
            ext_bio_outputs[b_id] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r_step.items()}
        ml_3v = compare_three_variants(dim=8, steps=80, seed=7, include_extended=True)
        return {
            "cellular_benchmark_all_v2": cell_bench,
            "gated_organelle_interactions": org_gated,
            "heterogeneous_vs_homogeneous": hetero,
            "bio_hybrid_core_assays": bio_core,
            "all_16_bio_hybrids_smoke": ext_bio_outputs,
            "all_18_ml_hybrids_three_variants": ml_3v,
        }

    sections["02_foundry_v2_complete"] = _timed_section("02_foundry_v2_complete", _run_sec02, out_dir)

    # 03. Genesis v2: 2048-Candidate Vectorized Funnel, SCEE & Non-Repeating Recursive Improvement
    sections["03_genesis_v2_evolution"] = _timed_section(
        "03_genesis_v2_evolution",
        lambda: {
            "micro_population_2048": screen_large_micro_population(population_size=2048, n_neurons=12, steps=64, seed=7),
            "scee_collective_curricula": run_collective_experience_comparison(generations=10, pop_size=12, seeds=seeds),
            "recursive_vs_evolutionary": run_recursive_vs_evolutionary_comparison(generations=10, n_neurons=10, seed=7),
        },
        out_dir,
    )

    # 04. Architect v2: All 33 Learning Rules, 11 Sherman-Morrison Experts, CVaR & Cross-Scale Transfer
    sections["04_architect_33_learning_rules"] = _timed_section(
        "04_architect_33_learning_rules",
        lambda: {
            "extended_33_learning_suite": benchmark_extended_connectome_learning_suite(seed=7, steps=100),
            "developmental_stage_rules": benchmark_developmental_stage_learning_rules(seed=7, steps=80),
            "cross_scale_transfer_8_to_128": AutonomousResearchDirector(seed=7).run_cross_scale_transfer_study(
                scales=(8, 24, 64, 128), cycles_for_expert_calibration=8
            ),
        },
        out_dir,
    )

    # 05. Antipode v2: Real TidalMoE Hooks, Expanse Failure Corpus, 6-Way & Sparse-Label Verifier Study
    def _run_sec05() -> Dict[str, Any]:
        inst = TidalMoEInstrumentor()
        probes = inst.probe_inputs(n_samples=28, seed=7)
        inst.close()
        corp = load_real_expanse_failure_corpus()
        return {
            "tidal_probes_total": len(probes),
            "tidal_failures_detected": sum(1 for p in probes if p.is_failure),
            "expanse_verified_failures": corp["verified_failures"],
            "expanse_verified_successes": corp["verified_successes"],
            "six_way_study": run_antipode_six_way_study(seeds=seeds, epochs=40),
            "sparse_label_verifier_study": run_sparse_label_verifier_antipode_study(seeds=seeds, epochs=35),
        }

    sections["05_antipode_v2_distillation"] = _timed_section("05_antipode_v2_distillation", _run_sec05, out_dir)

    # 06. Symbiosis v2: All 11 Couplings, Frozen Host Bit-Preservation & Long-Horizon Mutual Co-Adaptation
    sections["06_symbiosis_v2_mutual"] = _timed_section(
        "06_symbiosis_v2_mutual",
        lambda: {
            "all_11_couplings_study": run_symbiosis_cycle_experiment(couplings=COUPLING_MECHANISMS, seed=7),
            "long_horizon_mutual_vs_frozen": run_long_horizon_mutual_symbiosis(alternating_cycles=6, steps_per_phase=40, seed=7),
        },
        out_dir,
    )

    # 07. Spectra (32 Senses), Hyperspace (4D/5D/6D), Chimera (+ SYNAPSE-ATLAS), Curiosity (12 Worlds) & Affect
    def _run_sec07() -> Dict[str, Any]:
        ga = generate_candidate_genome("mixed_custom_standard", n_neurons=16, seed=7).connectome
        gb = load_malecns_subgraph(max_nodes=16, seed=7)
        return {
            "spectra_artificial_vs_raw": benchmark_artificial_vs_raw_encoding(seed=7, n_samples=160),
            "spectra_internal_perception": run_internal_perception_suite(seed=7),
            "hyperspace_4d": run_hyperspace_developmental_study(seeds=seeds, generations=14),
            "hyperspace_5d_6d_occluded": run_high_dim_5d_6d_study(seeds=seeds, generations=12),
            "chimera_5way_fusion": run_chimera_comparative_benchmark(seed=7, n_neurons=10),
            "synapse_atlas_symmetry_vs_malecns": audit_connectome_symmetry_and_fingerprints(ga, gb),
            "curiosity_all_12_modalities": verify_all_novel_modalities(seed=7),
            "affect_emergent_investigation": run_affective_emergence_investigation(seed=7),
        }

    sections["07_spectra_hyperspace_chimera_curiosity_affect"] = _timed_section(
        "07_spectra_hyperspace_chimera_curiosity_affect", _run_sec07, out_dir
    )

    # 08. Emergence (GhostInTheMachine + CROSSWIRE + FAULTLINE) & Sentinel Matched Baselines
    sections["08_emergence_and_sentinel"] = _timed_section(
        "08_emergence_and_sentinel",
        lambda: {
            "cross_yoked_twin_multi_seed": [run_cross_yoked_twin_assay(n_neurons=32, steps=160, seed=s) for s in seeds],
            "crosswire_and_faultline_multi_seed": [run_crosswire_and_faultline_study(seed=s) for s in seeds],
            "sentinel_matched_baselines_5seeds": run_matched_baseline_suite(seeds=heldout_seeds, hidden_dim=16, steps=90),
        },
        out_dir,
    )

    # 09. Host Sonapse-LAB Tandem Max Sections 01–08 + Omni Max Mode 4-Arm Architectural Ablation
    def _run_sec09() -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as td:
            wb = Workbench(config=SynapseConfig(nodes_per_source=16, seed=7), output_root=td)
            try:
                tandem = wb.post(
                    "/api/tandem/max",
                    {
                        "total_neurons": 192,
                        "task": "cybernetic_sanctuary",
                        "pretrain_steps": 32,
                        "pretrain_updates": 8,
                        "batch_size": 64,
                        "live_steps": 8,
                        "workers": 1,
                        "omnibus_steps_per_task": 4,
                        "omnibus_eval_steps": 6,
                        "six_arm_steps": 8,
                    },
                )
            finally:
                wb.sim_worker.stop()
        ablation = _run_max_mode_architecture_ablation(seeds=seeds, epochs=8)
        return {
            "sonapse_lab_tandem_max_summary": {
                "status": tandem["status"],
                "total_neurons": tandem["total_neurons"],
                "total_wallclock_seconds": tandem["total_wallclock_seconds"],
                "tandem_sha256": tandem["tandem_sha256"],
                "sections_completed": tandem["sections_completed"],
                "feature_audit_complete": tandem["section_01"]["compute"]["feature_audit"]["complete"],
                "enabled_feature_count": tandem["section_01"]["compute"]["feature_audit"]["enabled_count"],
                "section_06_neuroweave_status": tandem["section_06"].get("status", "ok"),
                "section_07_frontier_status": tandem["section_07"].get("status", "ok"),
                "section_08_morphogenesis": tandem["section_08"],
            },
            "omni_max_mode_4arm_ablation": ablation,
        }

    sections["09_host_tandem_max_and_omni_ablation"] = _timed_section(
        "09_host_tandem_max_and_omni_ablation", _run_sec09, out_dir
    )

    # Render publication comparative SVGs into docs/paper/figures/
    comp_rows = sections["01_performance_and_scaling"]["result"]["compiled_vs_object_engine"]["scale_comparisons"]
    speedup_bars = {f"N={r['neurons']} ({r['total_compartments']}c)": float(r["speedup_x"] or 1.0) for r in comp_rows if r["speedup_x"] is not None}
    (paper_figs / "fig1_compiled_engine_speedup.svg").write_text(
        render_comparison_svg("Compiled SoA+CSR vs Object Engine Speedup (x)", speedup_bars), encoding="utf-8"
    )

    learn_top = sections["04_architect_33_learning_rules"]["result"]["extended_33_learning_suite"]["top5_lowest_corrupted_mse"]
    learn_bars = {k: float(v) for k, v in learn_top}
    (paper_figs / "fig2_top_learning_rules_corrupted_mse.svg").write_text(
        render_comparison_svg("Top Learning Rules Under Channel Corruption (MSE)", learn_bars), encoding="utf-8"
    )

    abl_arms = sections["09_host_tandem_max_and_omni_ablation"]["result"]["omni_max_mode_4arm_ablation"]["arms"]
    abl_bars = {k: float(v["mean_reward"]) for k, v in abl_arms.items()}
    (paper_figs / "fig3_omni_max_mode_ablation.svg").write_text(
        render_comparison_svg("OMNI-MORPHOGENESIS 4-Arm Architectural Ablation (Reward)", abl_bars), encoding="utf-8"
    )

    total_wall_s = round(time.perf_counter() - t_start, 2)
    master_summary = {
        "version": __version__,
        "schema": SCHEMA_VERSION,
        "hardware": probe_hardware(),
        "total_wallclock_seconds": total_wall_s,
        "section_timings": {k: {"wall_s": v["wallclock_seconds"], "cpu_s": v["cpu_seconds"], "sha256": v["sha256_digest"]} for k, v in sections.items()},
    }
    (out_dir / "master_individual_sections_summary.json").write_text(json.dumps(master_summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] All 9 individual mode & section suites completed in {total_wall_s:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
