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
    p_g = sub.add_parser("genesis", help="Run Sonapse Genesis 1024-candidate micro screening & SCEE comparison")
    p_g.add_argument("--pop", type=int, default=1024)
    sub.add_parser("antipode", help="Run Supermix Antipode TidalMoE probe, Expanse failures & 6-way distillation")
    sub.add_parser("symbiosis", help="Run Sonapse Symbiosis intermittent coupling to frozen Sonapse-LAB")
    sub.add_parser("spectra", help="Run Sonapse Spectra artificial-sense benchmark & internal perception suite")
    sub.add_parser("hyperspace", help="Run Sonapse Hyperspace 4D developmental study & SO(4) polytope recognition")
    sub.add_parser("chimera", help="Run Sonapse Chimera connectome alignment & 5-way fusion benchmark")
    sub.add_parser("affect", help="Run Sonapse Affect emergent dynamical state & causal intervention study")
    sub.add_parser("curiosity", help="Verify 12 novel interactive training modalities")
    sub.add_parser("architect", help="Run 14-step Sonapse Architect autonomous research cycle")
    sub.add_parser("sentinel", help="Run Sonapse Sentinel matched-budget baseline suite")
    p_s = sub.add_parser("serve", help="Launch interactive Morphogenesis scientific workbench")
    p_s.add_argument("--host", default="127.0.0.1")
    p_s.add_argument("--port", type=int, default=8766)
    p_s.add_argument("--state-dir", type=Path, default=None)

    args = parser.parse_args(argv)
    seed = int(args.seed)

    if args.command == "status":
        from .core import probe_hardware
        from .foundry import GRAFT_NEURON_FAMILIES, CONVENTIONAL_NEURON_FAMILIES, ORGANELLE_REGISTRY
        out = {
            "version": __version__,
            "schema": SCHEMA_VERSION,
            "hardware": probe_hardware(),
            "graft_neurons": len(GRAFT_NEURON_FAMILIES),
            "conventional_neurons": len(CONVENTIONAL_NEURON_FAMILIES),
            "organelles": len(ORGANELLE_REGISTRY),
        }
    elif args.command == "foundry":
        from .foundry import run_foundry_benchmark_suite, evaluate_organelle_interaction, run_heterogeneous_vs_homogeneous_experiment, compare_three_variants
        out = {
            "cellular_benchmark": run_foundry_benchmark_suite(seeds=(seed, seed + 12), steps=args.steps),
            "organelle_interaction": evaluate_organelle_interaction("mnemosome", "chronosome", seed=seed),
            "hetero_vs_homo": run_heterogeneous_vs_homogeneous_experiment(seeds=(seed, seed + 12)),
            "ml_three_variants": compare_three_variants(seed=seed),
        }
    elif args.command == "genesis":
        from .genesis import screen_large_micro_population, run_collective_experience_comparison, run_recursive_vs_evolutionary_comparison
        out = {
            "large_population_funnel": screen_large_micro_population(population_size=args.pop, seed=seed),
            "scee_curricula": run_collective_experience_comparison(seeds=(seed, seed + 12)),
            "recursive_vs_evolution": run_recursive_vs_evolutionary_comparison(seed=seed),
        }
    elif args.command == "antipode":
        from .antipode import TidalMoEInstrumentor, load_real_expanse_failure_corpus, run_antipode_six_way_study
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
    elif args.command == "symbiosis":
        from .symbiosis import run_symbiosis_cycle_experiment
        out = run_symbiosis_cycle_experiment(seed=seed)
    elif args.command == "spectra":
        from .spectra import benchmark_artificial_vs_raw_encoding, run_internal_perception_suite
        out = {
            "artificial_vs_raw": benchmark_artificial_vs_raw_encoding(seed=seed),
            "internal_perception": run_internal_perception_suite(seed=seed),
        }
    elif args.command == "hyperspace":
        from .hyperspace import run_hyperspace_developmental_study
        out = run_hyperspace_developmental_study(seeds=(seed, seed + 12))
    elif args.command == "chimera":
        from .chimera import run_chimera_comparative_benchmark
        out = run_chimera_comparative_benchmark(seed=seed)
    elif args.command == "affect":
        from .affect import run_affective_emergence_investigation
        out = run_affective_emergence_investigation(seed=seed)
    elif args.command == "curiosity":
        from .curiosity import verify_all_novel_modalities
        out = verify_all_novel_modalities(seed=seed)
    elif args.command == "architect":
        from .architect import AutonomousResearchDirector, benchmark_developmental_stage_learning_rules
        out = {
            "autonomous_cycle": AutonomousResearchDirector(seed=seed).run_research_cycle(),
            "stage_learning_rules": benchmark_developmental_stage_learning_rules(seed=seed),
        }
    elif args.command == "sentinel":
        from .sentinel import run_matched_baseline_suite
        out = run_matched_baseline_suite()
    elif args.command == "serve":
        from .workbench import serve_standalone
        serve_standalone(host=args.host, port=args.port, state_root=args.state_dir)
        return 0
    else:
        raise ValueError(args.command)

    print(json.dumps(out, indent=2))
    return 0
