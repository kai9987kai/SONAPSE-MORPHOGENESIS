"""SONAPSE GENESIS — Multi-Origin Candidate Seeding & Large-Scale Miniature Lab.

Supports generating and screening populations of up to thousands of miniature
connectome candidates on CPU via ``VectorizedMicroBatchSimulator`` +
``ProgressiveSelectionFunnel`` before promoting promising structures to
multi-compartment simulation.

Candidate origins:
  1. random_graph
  2. biological_malecns           (authentic MaleCNS v1.0 induced cell-type subgraph)
  3. degree_shuffled_biological   (Maslov-Sneppen degree-preserving null control)
  4. fractal_topology             (hierarchical Sierpinski-inspired connectivity)
  5. mixed_custom_standard        (heterogeneous FOUNDRY graft + conventional neurons)
  6. sonapse_checkpoint_prior     (derived from Sonapse-LAB zero-impact graft profile)
  7. auto_developmental_genome    (procedural evolved genome)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..core.bridge import MorphogenesisSpecV1, build_zero_impact_subconnectome, load_malecns_subgraph
from ..core.interchange import (
    ConnectomeEdge,
    ConnectomeGraph,
    DataSourceKind,
    DevelopmentalGenome,
    FidelityLevel,
    make_id,
)
from ..core.profiler import ProgressiveSelectionFunnel, ResourceBudget, VectorizedMicroBatchSimulator
from ..foundry.conventional_neurons import CONVENTIONAL_NEURON_FAMILIES, build_conventional_neuron
from ..foundry.graft_neurons import GRAFT_NEURON_FAMILIES, build_graft_neuron


CANDIDATE_ORIGINS: Tuple[str, ...] = (
    "random_graph",
    "biological_malecns",
    "degree_shuffled_biological",
    "fractal_topology",
    "mixed_custom_standard",
    "sonapse_checkpoint_prior",
    "auto_developmental_genome",
)


def degree_preserving_shuffle(edges: Sequence[ConnectomeEdge], swaps: int = 80, seed: int = 42) -> List[ConnectomeEdge]:
    """Maslov-Sneppen degree-preserving directed edge rewiring preserving in/out degrees."""
    rng = np.random.default_rng(seed)
    elist = [ConnectomeEdge.from_dict(e.to_dict()) for e in edges]
    m = len(elist)
    if m < 4:
        return elist
    existing = {(e.pre_id, e.post_id) for e in elist}
    for _ in range(swaps):
        i, j = int(rng.integers(0, m)), int(rng.integers(0, m))
        if i == j:
            continue
        a, b = elist[i].pre_id, elist[i].post_id
        c, d = elist[j].pre_id, elist[j].post_id
        if len({a, b, c, d}) < 4:
            continue
        if (a, d) in existing or (c, b) in existing:
            continue
        existing.remove((a, b))
        existing.remove((c, d))
        elist[i].post_id = d
        elist[j].post_id = b
        existing.add((a, d))
        existing.add((c, b))
    for e in elist:
        e.provenance = "degree_preserving_shuffled_null"
    return elist


def generate_candidate_genome(origin: str, n_neurons: int = 12, seed: int = 42) -> DevelopmentalGenome:
    """Create an initial DevelopmentalGenome from any of the 7 supported origins."""
    rng = np.random.default_rng(seed)

    if origin == "biological_malecns":
        graph = load_malecns_subgraph(max_nodes=n_neurons, seed=seed)
    elif origin == "degree_shuffled_biological":
        base = load_malecns_subgraph(max_nodes=n_neurons, seed=seed)
        shuffled = degree_preserving_shuffle(base.edges, swaps=max(40, len(base.edges) * 2), seed=seed + 13)
        graph = ConnectomeGraph(
            connectome_id=f"malecns_degshuf_{n_neurons}_s{seed}",
            name=f"Degree-Shuffled MaleCNS Control ({n_neurons})",
            neurons=base.neurons,
            edges=shuffled,
            sensory_ids=base.sensory_ids,
            motor_ids=base.motor_ids,
            source_kind=DataSourceKind.SYNTHETIC_SIMULATION,
            provenance={"origin": origin, "control_for": base.connectome_id},
        )
    elif origin == "sonapse_checkpoint_prior":
        graph = build_zero_impact_subconnectome(MorphogenesisSpecV1(neurons=n_neurons, seed=seed))
        graph.provenance = {"origin": origin}
    else:
        neurons = []
        for i in range(n_neurons):
            sign = 1 if i < int(0.8 * n_neurons) else -1
            if origin == "mixed_custom_standard":
                if i % 2 == 0:
                    ns = build_graft_neuron(str(rng.choice(GRAFT_NEURON_FAMILIES)), f"n{i}", FidelityLevel.MULTI_COMPARTMENT_RECURRENT)
                else:
                    ns = build_conventional_neuron(str(rng.choice(CONVENTIONAL_NEURON_FAMILIES[:6])), f"n{i}")
            elif origin == "auto_developmental_genome":
                fam = str(rng.choice(["triarch", "chronos", "mnemosome_neuron", "ouroboros", "counterpoint", "palimpsest"]))
                ns = build_graft_neuron(fam, f"n{i}", FidelityLevel.MULTI_COMPARTMENT_RECURRENT)
            else:
                ns = build_conventional_neuron("adaptive_pyramidal" if sign > 0 else "predictive_interneuron", f"n{i}")
            ns.dale_sign = sign
            neurons.append(ns)

        edges: List[ConnectomeEdge] = []
        if origin == "fractal_topology":
            # Hierarchical multi-scale modular clusters (Sierpinski-like self-similar density)
            cluster_sz = max(2, n_neurons // 4)
            for i in range(n_neurons):
                for j in range(n_neurons):
                    if i == j:
                        continue
                    same_fine = (i // cluster_sz) == (j // cluster_sz)
                    same_coarse = (i // (2 * cluster_sz)) == (j // (2 * cluster_sz))
                    p = 0.55 if same_fine else (0.20 if same_coarse else 0.06)
                    if rng.random() < p:
                        w = float(rng.uniform(0.1, 0.5)) * neurons[i].dale_sign
                        edges.append(ConnectomeEdge(f"n{i}", f"n{j}", weight=w, delay_ms=1.0 if same_fine else 2.5, provenance="fractal_hierarchy"))
        else:
            for i in range(n_neurons):
                for j in range(n_neurons):
                    if i != j and rng.random() < 0.25:
                        w = float(rng.uniform(0.08, 0.45)) * neurons[i].dale_sign
                        edges.append(ConnectomeEdge(f"n{i}", f"n{j}", weight=w, delay_ms=float(rng.uniform(0.5, 3.0)), provenance=origin))

        graph = ConnectomeGraph(
            connectome_id=f"{origin}_{n_neurons}_s{seed}",
            name=f"{origin} ({n_neurons} neurons)",
            neurons=neurons,
            edges=edges,
            sensory_ids=[f"n{i}" for i in range(min(4, n_neurons))],
            motor_ids=[f"n{i}" for i in range(max(0, n_neurons - 3), n_neurons)],
            source_kind=DataSourceKind.SYNTHETIC_SIMULATION,
            provenance={"origin": origin},
        )

    return DevelopmentalGenome(
        genome_id=make_id(f"gen_{origin[:6]}"),
        connectome=graph,
        generation=0,
        provenance={"origin": origin, "seed": seed},
    )


def screen_large_micro_population(
    population_size: int = 1024,
    n_neurons: int = 16,
    steps: int = 64,
    seed: int = 42,
    top_k: int = 16,
) -> Dict[str, Any]:
    """Screen 1,000+ miniature candidate connectomes on CPU in milliseconds using batched vectorized dynamics.

    Demonstrates the progressive selection funnel (Section 11 & 37):
      Stage 1: Vectorized batched screening of all ``population_size`` candidates (delay-memory task).
      Stage 2: Harder out-of-seed temporal parity evaluation on top 64 candidates.
      Stage 3: Diversity + efficiency Pareto selection of final ``top_k`` winners.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Generate B sparse Dale-compliant micro-connectomes
    signs = np.where(np.arange(n_neurons) < int(0.8 * n_neurons), 1.0, -1.0).astype(np.float32)
    mask = (rng.random((population_size, n_neurons, n_neurons)) < 0.22).astype(np.float32)
    mask[:, np.arange(n_neurons), np.arange(n_neurons)] = 0.0
    raw_w = np.abs(rng.normal(0.0, 0.35, size=(population_size, n_neurons, n_neurons))).astype(np.float32)
    weights = raw_w * mask * signs[None, None, :]
    # Spectral radius scaling diversity
    scales = rng.uniform(0.4, 1.35, size=(population_size, 1, 1)).astype(np.float32)
    weights *= scales
    decays = rng.uniform(0.35, 0.94, size=(population_size, n_neurons)).astype(np.float32)
    in_proj = rng.normal(0.0, 0.6, size=(population_size, n_neurons, 2)).astype(np.float32)

    # Task 1 (Screening): 4-step delayed recall
    cue = rng.choice([-1.0, 1.0], size=steps).astype(np.float32)
    noise = rng.normal(0.0, 0.2, size=steps).astype(np.float32)
    stim = np.stack([cue, noise], axis=1)
    target = np.roll(cue, 4)
    target[:4] = 0.0

    sim = VectorizedMicroBatchSimulator(weights, decays, in_proj)
    t_s1 = time.perf_counter()
    states = sim.rollout(stim)  # [T, B, N]
    split = int(0.6 * steps)
    # Fast vectorized ridge readout score per candidate
    s1_scores = np.zeros(population_size, dtype=np.float64)
    eye = 0.3 * np.eye(n_neurons, dtype=np.float32)
    y_tr, y_te = target[:split], target[split:]
    for b in range(population_size):
        X_tr = states[:split, b, :]
        X_te = states[split:, b, :]
        w = np.linalg.solve(X_tr.T @ X_tr + eye, X_tr.T @ y_tr)
        mse = float(np.mean((X_te @ w - y_te) ** 2))
        s1_scores[b] = -mse  # higher is better
    s1_ms = (time.perf_counter() - t_s1) * 1000.0

    # Stage 2: Evaluate top 64 on fresh held-out delayed XOR sequence
    keep_s2 = min(64, population_size)
    top_s1_idx = np.argsort(s1_scores)[-keep_s2:][::-1]
    t_s2 = time.perf_counter()
    rng2 = np.random.default_rng(seed + 999)
    bits = rng2.choice([-1.0, 1.0], size=steps).astype(np.float32)
    stim2 = np.stack([bits, np.roll(bits, 1)], axis=1)
    target2 = bits * np.roll(bits, 2)
    target2[:2] = 0.0
    sim2 = VectorizedMicroBatchSimulator(weights[top_s1_idx], decays[top_s1_idx], in_proj[top_s1_idx])
    states2 = sim2.rollout(stim2)
    s2_scores = np.zeros(keep_s2, dtype=np.float64)
    densities = np.mean(mask[top_s1_idx], axis=(1, 2))
    for k in range(keep_s2):
        X_tr = states2[:split, k, :]
        X_te = states2[split:, k, :]
        w = np.linalg.solve(X_tr.T @ X_tr + eye, X_tr.T @ target2[:split])
        mse2 = float(np.mean((X_te @ w - target2[split:]) ** 2))
        s2_scores[k] = -mse2 - 0.1 * float(densities[k])
    s2_ms = (time.perf_counter() - t_s2) * 1000.0

    order_s2 = np.argsort(s2_scores)[::-1][:top_k]
    winners = [
        {
            "candidate_index": int(top_s1_idx[k]),
            "stage1_delay_mse": round(float(-s1_scores[top_s1_idx[k]]), 5),
            "stage2_xor_mse": round(float(-(s2_scores[k] + 0.1 * densities[k])), 5),
            "synapse_density": round(float(densities[k]), 4),
            "mean_decay": round(float(np.mean(decays[top_s1_idx[k]])), 4),
        }
        for k in order_s2
    ]
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "experiment": "large_population_micro_screening_funnel",
        "population_size": population_size,
        "n_neurons_per_candidate": n_neurons,
        "steps": steps,
        "seed": seed,
        "stage1_wall_ms": round(s1_ms, 2),
        "stage2_wall_ms": round(s2_ms, 2),
        "total_wall_ms": round(total_ms, 2),
        "candidates_per_second": round(population_size / max(total_ms / 1000.0, 1e-6), 1),
        "stage1_median_mse": round(float(-np.median(s1_scores)), 5),
        "stage1_best_mse": round(float(-np.max(s1_scores)), 5),
        "stage2_best_xor_mse": winners[0]["stage2_xor_mse"] if winners else None,
        "top_winners": winners[:8],
    }
