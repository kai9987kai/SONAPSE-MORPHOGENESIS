"""SONAPSE FOUNDRY — Controlled Cellular Benchmark Suite & Catalogue Generator.

Evaluates every custom graft neuron family against conventional baselines
(recurrent_rate, lif_reference, adaptive_pyramidal) across controlled single-cell
tasks with ablations, resource accounting (FLOPs, metabolic energy, wall-clock ms),
and empirical strength/weakness documentation.

Never assumes complex morphologies outperform simpler baselines.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..core.compartment_engine import StructuredNeuronInstance
from ..core.interchange import NeuronMorphologySpec
from .conventional_neurons import build_conventional_neuron
from .graft_neurons import (
    GRAFT_NEURON_FAMILIES,
    build_ablation_variants,
    build_graft_neuron,
    render_neuron_svg,
)
from .organelles import create_organelle


def _make_cellular_tasks(steps: int = 100, seed: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Deterministic single-cell benchmark tasks testing distinct computational properties."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps, dtype=np.float64)

    # 1. Delayed recall (4-step echo with additive distractors)
    cue = rng.choice([-1.0, 1.0], size=steps)
    u_delay = np.stack([cue, rng.normal(0.0, 0.35, size=steps)], axis=1)
    y_delay = np.roll(cue, 4)
    y_delay[:4] = 0.0

    # 2. Temporal XOR / parity (x(t) * x(t-2))
    bits = rng.choice([-1.0, 1.0], size=steps)
    u_xor = np.stack([bits, np.roll(bits, 1)], axis=1)
    y_xor = bits * np.roll(bits, 2)
    y_xor[:2] = 0.0

    # 3. Spectral frequency discrimination (high-freq vs low-freq envelope)
    regime = (np.sin(0.09 * t) >= 0.0).astype(np.float64) * 2.0 - 1.0
    sig = np.where(regime > 0, np.sin(0.95 * t), np.sin(0.18 * t)) + rng.normal(0.0, 0.15, size=steps)
    u_freq = np.stack([sig, np.roll(sig, 1)], axis=1)
    y_freq = regime

    # 4. Noisy spike/anomaly suppression (reject corrupted channel while preserving clean channel)
    clean = np.sin(0.22 * t)
    corrupt = clean + np.where(rng.random(steps) < 0.25, rng.choice([-3.5, 3.5], size=steps), 0.0)
    u_veto = np.stack([clean, corrupt, (np.abs(corrupt - clean) > 1.0).astype(np.float64)], axis=1)
    y_veto = clean

    return {
        "delayed_recall_4": (u_delay, y_delay),
        "temporal_xor": (u_xor, y_xor),
        "spectral_discrimination": (u_freq, y_freq),
        "anomaly_suppression": (u_veto, y_veto),
    }


def evaluate_single_neuron_spec(
    spec: NeuronMorphologySpec,
    tasks: Dict[str, Tuple[np.ndarray, np.ndarray]],
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate one NeuronMorphologySpec across all tasks using a fixed-capacity linear readout."""
    per_task: Dict[str, float] = {}
    total_flops = 0
    total_energy = 0.0
    t0 = time.perf_counter()

    for task_name, (X, y) in tasks.items():
        inst = StructuredNeuronInstance(spec, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
        steps = X.shape[0]
        feats = np.zeros((steps, inst.n_comp + 1), dtype=np.float64)
        for k in range(steps):
            res = inst.step(X[k], adapt=False)
            feats[k, : inst.n_comp] = inst.state
            feats[k, inst.n_comp] = res["scalar_output"]
        split = int(0.6 * steps)
        X_tr, y_tr = feats[:split], y[:split]
        X_te, y_te = feats[split:], y[split:]
        # Ridge readout over neuron compartment states
        w = np.linalg.solve(X_tr.T @ X_tr + 0.35 * np.eye(feats.shape[1]), X_tr.T @ y_tr)
        mse = float(np.mean((X_te @ w - y_te) ** 2))
        per_task[task_name] = round(mse, 5)
        total_flops += inst.cumulative_flops
        total_energy += inst.cumulative_energy

    wall_ms = (time.perf_counter() - t0) * 1000.0
    mean_mse = float(np.mean(list(per_task.values())))
    return {
        "family": spec.family,
        "name": spec.name,
        "objective_category": spec.objective_category.value,
        "fidelity": int(spec.fidelity),
        "n_compartments": len(spec.compartments),
        "n_edges": len(spec.edges),
        "n_organelles": len(spec.organelles),
        "mean_test_mse": round(mean_mse, 5),
        "per_task_mse": per_task,
        "total_flops": int(total_flops),
        "total_energy": round(total_energy, 4),
        "wall_ms": round(wall_ms, 3),
        "efficiency_score": round(1.0 / ((mean_mse + 0.05) * (1.0 + total_energy * 0.02)), 4),
    }


def run_foundry_benchmark_suite(
    families: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (7, 19),
    steps: int = 90,
    include_ablations: bool = True,
) -> Dict[str, Any]:
    """Run controlled multi-seed cellular benchmark across graft neurons & conventional baselines."""
    target_families = list(families) if families is not None else list(GRAFT_NEURON_FAMILIES)
    baselines = ["recurrent_rate", "lif_reference", "adaptive_pyramidal"]

    agg: Dict[str, List[Dict[str, Any]]] = {}
    ablation_summary: Dict[str, Dict[str, float]] = {}

    for seed in seeds:
        tasks = _make_cellular_tasks(steps=steps, seed=seed)
        for bfam in baselines:
            spec = build_conventional_neuron(bfam, neuron_id=f"base_{bfam}")
            agg.setdefault(f"baseline:{bfam}", []).append(evaluate_single_neuron_spec(spec, tasks, seed=seed))
        for gfam in target_families:
            spec = build_graft_neuron(gfam)
            agg.setdefault(gfam, []).append(evaluate_single_neuron_spec(spec, tasks, seed=seed))
            if include_ablations and seed == seeds[0]:
                abls = build_ablation_variants(spec)
                ablation_summary[gfam] = {
                    aname: evaluate_single_neuron_spec(aspec, tasks, seed=seed)["mean_test_mse"]
                    for aname, aspec in abls.items()
                }

    summary: Dict[str, Dict[str, Any]] = {}
    base_rate_mse = float(np.mean([r["mean_test_mse"] for r in agg["baseline:recurrent_rate"]]))
    for key, runs in agg.items():
        mses = [r["mean_test_mse"] for r in runs]
        per_task_means = {
            tk: round(float(np.mean([r["per_task_mse"][tk] for r in runs])), 5)
            for tk in runs[0]["per_task_mse"]
        }
        best_task = min(per_task_means, key=per_task_means.get)  # type: ignore[arg-type]
        worst_task = max(per_task_means, key=per_task_means.get)  # type: ignore[arg-type]
        mean_m = float(np.mean(mses))
        summary[key] = {
            "name": runs[0]["name"],
            "objective_category": runs[0]["objective_category"],
            "n_compartments": runs[0]["n_compartments"],
            "n_organelles": runs[0]["n_organelles"],
            "mean_test_mse": round(mean_m, 5),
            "std_test_mse": round(float(np.std(mses)), 5),
            "delta_vs_rate_baseline": round(base_rate_mse - mean_m, 5),
            "per_task_mse": per_task_means,
            "mean_flops": int(np.mean([r["total_flops"] for r in runs])),
            "mean_energy": round(float(np.mean([r["total_energy"] for r in runs])), 4),
            "mean_wall_ms": round(float(np.mean([r["wall_ms"] for r in runs])), 3),
            "strongest_task": best_task,
            "weakest_task": worst_task,
            "ablations_seed0": ablation_summary.get(key),
        }

    ranked = sorted(summary.items(), key=lambda kv: kv[1]["mean_test_mse"])
    return {
        "experiment": "sonapse_foundry_cellular_benchmark",
        "seeds": list(seeds),
        "steps": steps,
        "caveat": "Linear readout dimensionality scales with compartment count; compare both raw MSE and energy/FLOP cost.",
        "baseline_rate_mse": round(base_rate_mse, 5),
        "ranking_top5": [(k, v["mean_test_mse"], v["mean_energy"]) for k, v in ranked[:5]],
        "results": summary,
    }
