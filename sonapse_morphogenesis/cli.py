"""Command-line entry point for SONAPSE MORPHOGENESIS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from . import SCHEMA_VERSION, __version__


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SONAPSE MORPHOGENESIS — Autonomous Developmental Connectome Research Platform")
    parser.add_argument("--version", action="version", version=f"sonapse-morphogenesis {__version__} ({SCHEMA_VERSION})")
    parser.add_argument("--seed", type=int, default=7)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show hardware probe, schema version, and module catalog")
    p_f = sub.add_parser("foundry", help="Run Sonapse Foundry cellular benchmarks & organelle assays")
    p_f.add_argument("--steps", type=int, default=80)
    p_f.add_argument("--extended", action="store_true", help="Include v2 extended 36 graft + 21 conventional families & Arbitrosome")
    p_g = sub.add_parser("genesis", help="Run Sonapse Genesis 1024-candidate micro screening & SCEE comparison")
    p_g.add_argument("--pop", type=int, default=1024)
    p_ap = sub.add_parser("antipode", help="Run Supermix Antipode TidalMoE probe, Expanse failures & 6-way distillation")
    p_ap.add_argument("--sparse-labels", action="store_true", help="Also run sparse-label / verifier-only distillation study")
    p_sy = sub.add_parser("symbiosis", help="Run Sonapse Symbiosis intermittent coupling to frozen Sonapse-LAB")
    p_sy.add_argument("--mutual", action="store_true", help="Also run long-horizon bidirectional mutual co-adaptation study")
    sub.add_parser("spectra", help="Run Sonapse Spectra artificial-sense benchmark & internal perception suite")
    p_hs = sub.add_parser("hyperspace", help="Run Sonapse Hyperspace 4D developmental study & SO(4) polytope recognition")
    p_hs.add_argument("--high-dim", action="store_true", help="Also run 5D/6D navigation & occluded polytope study")
    p_ch = sub.add_parser("chimera", help="Run Sonapse Chimera connectome alignment & 5-way fusion benchmark")
    p_ch.add_argument("--symmetry-audit", action="store_true", help="Also run SYNAPSE-ATLAS automorphism symmetry audit")
    sub.add_parser("affect", help="Run Sonapse Affect emergent dynamical state & causal intervention study")
    sub.add_parser("curiosity", help="Verify 12 novel interactive training modalities")
    p_ar = sub.add_parser("architect", help="Run 14-step Sonapse Architect autonomous research cycle")
    p_ar.add_argument("--extended", action="store_true", help="Also run cross-scale transfer & all 33 learning rules suite")
    sub.add_parser("emergence", help="Run GhostInTheMachine causal emergence, CROSSWIRE safety & FAULTLINE diversity suite")
    sub.add_parser("sentinel", help="Run Sonapse Sentinel matched-budget baseline suite")
    p_mm = sub.add_parser("max-mode", help="Run OMNI-MORPHOGENESIS v2.0 Unified Max Mode (all features & capabilities simultaneously)")
    p_mm.add_argument("--duration-seconds", type=float, default=None, help="Continuous wall-clock campaign duration in seconds (e.g., 7200 for 2 hours)")
    p_mm.add_argument("--epochs", type=int, default=2, help="Number of macro-epochs when --duration-seconds is not set")
    p_mm.add_argument("--host-neurons", type=int, default=192, help="Coupled Sonapse-LAB Tandem Max total neurons")
    p_mm.add_argument("--genesis-pop", type=int, default=256, help="Per-epoch Genesis micro-population size")
    p_mm.add_argument("--checkpoint-interval", type=float, default=300.0, help="Atomic checkpoint interval in seconds")
    p_mm.add_argument("--output-dir", type=Path, default=None, help="Output directory for JSONL logs, checkpoints, and SVG charts")
    p_mm.add_argument("--audit-only", action="store_true", help="Verify 100% Max Mode completeness without running epochs")
    p_s = sub.add_parser("serve", help="Launch interactive Morphogenesis scientific workbench")
    p_s.add_argument("--host", default="127.0.0.1")
    p_s.add_argument("--port", type=int, default=8766)
    p_s.add_argument("--state-dir", type=Path, default=None)

    args = parser.parse_args(argv)
    seed = int(args.seed)

    if args.command == "status":
        from .core import probe_hardware
        from .foundry import (
            ALL_BIO_HYBRID_REGISTRY,
            ALL_CONVENTIONAL_NEURON_FAMILIES,
            ALL_GRAFT_NEURON_FAMILIES,
            ALL_ML_HYBRID_MECHANISMS,
            ALL_ORGANELLE_REGISTRY,
            CONVENTIONAL_NEURON_FAMILIES,
            GRAFT_NEURON_FAMILIES,
            ORGANELLE_REGISTRY,
        )
        from .architect import ALL_LEARNING_ALGORITHMS
        out = {
            "version": __version__,
            "schema": SCHEMA_VERSION,
            "hardware": probe_hardware(),
            "graft_neurons": len(GRAFT_NEURON_FAMILIES),
            "conventional_neurons": len(CONVENTIONAL_NEURON_FAMILIES),
            "organelles": len(ORGANELLE_REGISTRY),
            "total_graft_neurons_v2": len(ALL_GRAFT_NEURON_FAMILIES),
            "total_conventional_neurons_v2": len(ALL_CONVENTIONAL_NEURON_FAMILIES),
            "total_organelles_v2": len(ALL_ORGANELLE_REGISTRY),
            "total_bio_hybrids_v2": len(ALL_BIO_HYBRID_REGISTRY),
            "total_ml_hybrids_v2": len(ALL_ML_HYBRID_MECHANISMS),
            "total_learning_algorithms_v2": len(ALL_LEARNING_ALGORITHMS),
        }
    elif args.command == "foundry":
        from .foundry import (
            compare_three_variants,
            evaluate_gated_organelle_interaction,
            evaluate_organelle_interaction,
            run_foundry_benchmark_suite,
            run_heterogeneous_vs_homogeneous_experiment,
        )
        out = {
            "cellular_benchmark": run_foundry_benchmark_suite(seeds=(seed, seed + 12), steps=args.steps, include_extended=bool(args.extended)),
            "organelle_interaction": evaluate_organelle_interaction("mnemosome", "chronosome", seed=seed),
            "gated_organelle_arbitration": evaluate_gated_organelle_interaction("mnemosome", "chronosome", seed=seed),
            "hetero_vs_homo": run_heterogeneous_vs_homogeneous_experiment(seeds=(seed, seed + 12)),
            "ml_three_variants": compare_three_variants(seed=seed, include_extended=bool(args.extended)),
        }
    elif args.command == "genesis":
        from .genesis import screen_large_micro_population, run_collective_experience_comparison, run_recursive_vs_evolutionary_comparison
        out = {
            "large_population_funnel": screen_large_micro_population(population_size=args.pop, seed=seed),
            "scee_curricula": run_collective_experience_comparison(seeds=(seed, seed + 12)),
            "recursive_vs_evolution": run_recursive_vs_evolutionary_comparison(seed=seed),
        }
    elif args.command == "antipode":
        from .antipode import (
            TidalMoEInstrumentor,
            load_real_expanse_failure_corpus,
            run_antipode_six_way_study,
            run_sparse_label_verifier_antipode_study,
        )
        inst = TidalMoEInstrumentor()
        probes = inst.probe_inputs(n_samples=18, seed=seed)
        inst.close()
        corp = load_real_expanse_failure_corpus()
        out = {
            "tidal_probes": len(probes),
            "tidal_failures": sum(1 for p in probes if p.is_failure),
            "expanse_failures": corp["verified_failures"],
            "expanse_successes": corp["verified_successes"],
            "six_way_study": run_antipode_six_way_study(seeds=(seed, seed + 12)),
        }
        if args.sparse_labels:
            out["sparse_label_study"] = run_sparse_label_verifier_antipode_study(seeds=(seed, seed + 12))
    elif args.command == "symbiosis":
        from .symbiosis import run_long_horizon_mutual_symbiosis, run_symbiosis_cycle_experiment
        out = run_symbiosis_cycle_experiment(seed=seed)
        if args.mutual:
            out = {"frozen_host_study": out, "mutual_coadaptation_study": run_long_horizon_mutual_symbiosis(seed=seed)}
    elif args.command == "spectra":
        from .spectra import benchmark_artificial_vs_raw_encoding, run_internal_perception_suite
        out = {
            "artificial_vs_raw": benchmark_artificial_vs_raw_encoding(seed=seed),
            "internal_perception": run_internal_perception_suite(seed=seed),
        }
    elif args.command == "hyperspace":
        from .hyperspace import run_high_dim_5d_6d_study, run_hyperspace_developmental_study
        out = run_hyperspace_developmental_study(seeds=(seed, seed + 12))
        if args.high_dim:
            out = {"developmental_4d": out, "high_dim_5d_6d": run_high_dim_5d_6d_study(seeds=(seed, seed + 12))}
    elif args.command == "chimera":
        from .chimera import audit_connectome_symmetry_and_fingerprints, run_chimera_comparative_benchmark
        from .genesis import generate_candidate_genome
        out = run_chimera_comparative_benchmark(seed=seed)
        if args.symmetry_audit:
            ga = generate_candidate_genome("mixed_custom_standard", n_neurons=14, seed=seed).connectome
            gb = generate_candidate_genome("fractal_topology", n_neurons=14, seed=seed + 17).connectome
            out["symmetry_audit"] = audit_connectome_symmetry_and_fingerprints(ga, gb)
    elif args.command == "affect":
        from .affect import run_affective_emergence_investigation
        out = run_affective_emergence_investigation(seed=seed)
    elif args.command == "curiosity":
        from .curiosity import verify_all_novel_modalities
        out = verify_all_novel_modalities(seed=seed)
    elif args.command == "architect":
        from .architect import (
            AutonomousResearchDirector,
            benchmark_developmental_stage_learning_rules,
            benchmark_extended_connectome_learning_suite,
        )
        director = AutonomousResearchDirector(seed=seed)
        out = {
            "autonomous_cycle": director.run_research_cycle(),
            "stage_learning_rules": benchmark_developmental_stage_learning_rules(seed=seed),
        }
        if args.extended:
            out["cross_scale_transfer"] = director.run_cross_scale_transfer_study()
            out["extended_learning_suite"] = benchmark_extended_connectome_learning_suite(seed=seed)
    elif args.command == "emergence":
        import numpy as np
        from .emergence import (
            run_causal_emergence_and_authorship_suite,
            run_crosswire_and_faultline_study,
        )
        rng = np.random.default_rng(seed)
        t_steps = 80
        traj = np.zeros((t_steps, 12), dtype=np.float64)
        for t in range(1, t_steps):
            traj[t] = np.tanh(0.82 * traj[t - 1] + 0.18 * np.roll(traj[t - 1], 1) + rng.normal(0.0, 0.15, size=12))
        out = {
            "causal_emergence": run_causal_emergence_and_authorship_suite(traj, macro_dim=2, seed=seed),
            "crosswire_and_faultline": run_crosswire_and_faultline_study(seed=seed),
        }
    elif args.command == "sentinel":
        from .sentinel import run_matched_baseline_suite
        out = run_matched_baseline_suite()
    elif args.command == "max-mode":
        from .max_mode import OmniMaxModeConfig, OmniMorphogenesisMaxEngine
        cfg = OmniMaxModeConfig(
            seed=seed,
            host_total_neurons=int(args.host_neurons),
            genesis_micro_pop=int(args.genesis_pop),
            checkpoint_interval_seconds=float(args.checkpoint_interval),
        )
        engine = OmniMorphogenesisMaxEngine(config=cfg, output_dir=args.output_dir)
        if args.audit_only:
            out = engine.verify_max_mode_completeness()
        elif args.duration_seconds is not None:
            out = engine.run_campaign(duration_seconds=float(args.duration_seconds))
            # Keep stdout concise by omitting dense sampled_trajectory if written to disk
            out = {k: v for k, v in out.items() if k != "sampled_trajectory"}
        else:
            out = engine.run_epochs(n_epochs=int(args.epochs))
    elif args.command == "serve":
        from .workbench import serve_standalone
        serve_standalone(host=args.host, port=args.port, state_root=args.state_dir)
        return 0
    else:
        raise ValueError(args.command)

    print(json.dumps(out, indent=2))
    return 0
