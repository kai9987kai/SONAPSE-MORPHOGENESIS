"""SONAPSE CHIMERA — Connectome Mixing, Alignment, Grafting & Comparative Fusion Lab.

Implements all 13 connectome fusion/mixing modes (Section 13), multi-cue
functional + anatomical correspondence alignment (Hungarian linear sum assignment
+ orthogonal Procrustes), full source provenance preservation, transactional
reversibility, and 5-way comparative evaluation:
  direct_weight_averaging vs aligned_weight_merging vs ensemble vs
  anatomical_grafting vs behavioral_distillation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes

from ..core.compartment_engine import MultiFidelityConnectome
from ..core.interchange import (
    CompartmentEdge,
    ConnectomeEdge,
    ConnectomeGraph,
    DataSourceKind,
    NeuronMorphologySpec,
    content_digest,
    make_id,
)
from ..foundry.organelles import create_organelle
from ..genesis.population import generate_candidate_genome


CHIMERA_MODES: Tuple[str, ...] = (
    "whole_connectome_union",
    "anatomical_grafting",
    "partial_circuit_transplantation",
    "structural_averaging",
    "synaptic_weight_interpolation",
    "neural_crossover",
    "functional_distillation",
    "signal_level_fusion",
    "shared_chemical_fields",
    "developmental_bridge_growth",
    "compartment_transplantation",
    "heterogeneous_mosaic",
    "symbiotic_coupling",
)


@dataclass
class ChimeraMergeReceipt:
    merge_id: str
    mode: str
    ratio_a: float
    parent_a_id: str
    parent_a_digest: str
    parent_b_id: str
    parent_b_digest: str
    child_connectome_id: str
    child_digest: str
    alignment_score: float
    correspondence_pairs: List[Tuple[str, str]]
    parent_a_snapshot: Dict[str, Any]
    parent_b_snapshot: Dict[str, Any]

    def rollback_parents(self) -> Tuple[ConnectomeGraph, ConnectomeGraph]:
        return ConnectomeGraph.from_dict(self.parent_a_snapshot), ConnectomeGraph.from_dict(self.parent_b_snapshot)


def _adjacency_matrix(graph: ConnectomeGraph) -> np.ndarray:
    n = len(graph.neurons)
    id2i = {u.morphology_id: i for i, u in enumerate(graph.neurons)}
    W = np.zeros((n, n), dtype=np.float64)
    for e in graph.edges:
        if e.active and e.pre_id in id2i and e.post_id in id2i:
            W[id2i[e.post_id], id2i[e.pre_id]] += e.weight
    return W


def _shared_stimulus_responses(graph: ConnectomeGraph, steps: int = 32, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stim = rng.normal(0.0, 1.0, size=(steps, 4))
    sim = MultiFidelityConnectome(graph, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
    R = np.zeros((steps, len(sim.neuron_ids)), dtype=np.float64)
    for k in range(steps):
        R[k] = sim.step(stim[k], collect_telemetry=False)["outputs"]
    return R  # [T, N]


def estimate_correspondence(
    graph_a: ConnectomeGraph,
    graph_b: ConnectomeGraph,
    seed: int = 42,
) -> Dict[str, Any]:
    """Estimate neuron-to-neuron anatomical & functional correspondence using vectorized cues + Hungarian matching."""
    na, nb = len(graph_a.neurons), len(graph_b.neurons)
    if na == 0 or nb == 0:
        return {"pairs": [], "mean_similarity": 0.0, "procrustes_residual": 0.0, "procrustes_Q": []}

    Ra = _shared_stimulus_responses(graph_a, steps=28, seed=seed)
    Rb = _shared_stimulus_responses(graph_b, steps=28, seed=seed)
    Wa = _adjacency_matrix(graph_a)
    Wb = _adjacency_matrix(graph_b)
    deg_a = np.sum(np.abs(Wa), axis=0) + np.sum(np.abs(Wa), axis=1)
    deg_b = np.sum(np.abs(Wb), axis=0) + np.sum(np.abs(Wb), axis=1)

    sens_a, mot_a = set(graph_a.sensory_ids), set(graph_a.motor_ids)
    sens_b, mot_b = set(graph_b.sensory_ids), set(graph_b.motor_ids)

    # Vectorized similarity matrix computation [na, nb]
    pos_a = np.array([u.compartments[0].coords_3d if u.compartments else (0.0, 0.0, 0.0) for u in graph_a.neurons], dtype=np.float64)
    pos_b = np.array([v.compartments[0].coords_3d if v.compartments else (0.0, 0.0, 0.0) for v in graph_b.neurons], dtype=np.float64)
    role_a = np.array([0 if u.morphology_id in sens_a else (2 if u.morphology_id in mot_a else 1) for u in graph_a.neurons])
    role_b = np.array([0 if v.morphology_id in sens_b else (2 if v.morphology_id in mot_b else 1) for v in graph_b.neurons])
    fam_a = np.array([u.family for u in graph_a.neurons], dtype=object)
    fam_b = np.array([v.family for v in graph_b.neurons], dtype=object)
    dale_a = np.array([u.dale_sign for u in graph_a.neurons])
    dale_b = np.array([v.dale_sign for v in graph_b.neurons])

    type_sim = np.where(fam_a[:, None] == fam_b[None, :], 1.0, np.where(dale_a[:, None] == dale_b[None, :], 0.5, 0.0))
    pos_sim = np.exp(-0.2 * np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=-1))
    max_deg = np.maximum(np.maximum(deg_a[:, None], deg_b[None, :]), 1.0)
    deg_sim = np.exp(-np.abs(deg_a[:, None] - deg_b[None, :]) / max_deg)
    Ra_n = Ra / (np.linalg.norm(Ra, axis=0, keepdims=True) + 1e-8)
    Rb_n = Rb / (np.linalg.norm(Rb, axis=0, keepdims=True) + 1e-8)
    act_sim = 0.5 * (1.0 + (Ra_n.T @ Rb_n))
    role_sim = np.where(role_a[:, None] == role_b[None, :], 1.0, 0.2)
    sim_mat = 0.22 * type_sim + 0.15 * pos_sim + 0.18 * deg_sim + 0.30 * act_sim + 0.15 * role_sim

    row_ind, col_ind = linear_sum_assignment(-sim_mat)
    pairs = [(graph_a.neurons[i].morphology_id, graph_b.neurons[j].morphology_id) for i, j in zip(row_ind, col_ind)]
    mean_sim = float(np.mean(sim_mat[row_ind, col_ind]))

    # Orthogonal Procrustes representation alignment residual on matched subset
    Ra_m = Ra[:, row_ind]
    Rb_m = Rb[:, col_ind]
    Q, _ = orthogonal_procrustes(Rb_m, Ra_m)
    proc_res = float(np.linalg.norm(Rb_m @ Q - Ra_m) / (np.linalg.norm(Ra_m) + 1e-8))
    return {
        "row_ind": row_ind.tolist(),
        "col_ind": col_ind.tolist(),
        "pairs": pairs,
        "mean_similarity": round(mean_sim, 5),
        "procrustes_residual": round(proc_res, 5),
        "procrustes_Q": Q.tolist(),
    }


def merge_connectomes(
    graph_a: ConnectomeGraph,
    graph_b: ConnectomeGraph,
    mode: str = "synaptic_weight_interpolation",
    ratio_a: float = 0.5,
    seed: int = 42,
) -> Tuple[ConnectomeGraph, ChimeraMergeReceipt]:
    """Merge two parent connectomes using any of the 13 SONAPSE CHIMERA modes with reversible provenance."""
    if mode not in CHIMERA_MODES:
        raise ValueError(f"Unknown Chimera mode: {mode!r}")
    r_a = float(np.clip(ratio_a, 0.0, 1.0))
    r_b = 1.0 - r_a
    rng = np.random.default_rng(seed)

    snap_a, snap_b = graph_a.to_dict(), graph_b.to_dict()
    corr = estimate_correspondence(graph_a, graph_b, seed=seed)
    pairs = corr["pairs"]
    b_to_a = {b_id: a_id for a_id, b_id in pairs}

    child_id = make_id(f"chim_{mode[:6]}")

    if mode in ("whole_connectome_union", "anatomical_grafting", "developmental_bridge_growth", "shared_chemical_fields", "symbiotic_coupling", "signal_level_fusion"):
        # Namespace both parent graphs without losing a single neuron, then add cross-bridges
        neurons: List[NeuronMorphologySpec] = []
        edges: List[ConnectomeEdge] = []
        a_map: Dict[str, str] = {}
        b_map: Dict[str, str] = {}
        for u in graph_a.neurons:
            nu = NeuronMorphologySpec.from_dict(u.to_dict())
            nu.morphology_id = f"A_{u.morphology_id}"
            a_map[u.morphology_id] = nu.morphology_id
            neurons.append(nu)
        for v in graph_b.neurons:
            nv = NeuronMorphologySpec.from_dict(v.to_dict())
            nv.morphology_id = f"B_{v.morphology_id}"
            b_map[v.morphology_id] = nv.morphology_id
            neurons.append(nv)
        for e in graph_a.edges:
            if e.pre_id in a_map and e.post_id in a_map:
                edges.append(ConnectomeEdge(a_map[e.pre_id], a_map[e.post_id], e.weight * (0.5 + 0.5 * r_a), e.delay_ms, e.length_um, e.receptor_type, e.plasticity_rule, provenance=f"parent_a:{graph_a.connectome_id}"))
        for e in graph_b.edges:
            if e.pre_id in b_map and e.post_id in b_map:
                edges.append(ConnectomeEdge(b_map[e.pre_id], b_map[e.post_id], e.weight * (0.5 + 0.5 * r_b), e.delay_ms, e.length_um, e.receptor_type, e.plasticity_rule, provenance=f"parent_b:{graph_b.connectome_id}"))
        # Connect top aligned pairs via cross-graft bridges
        bridge_gain = 0.0 if mode == "symbiotic_coupling" else 0.22
        for a_id, b_id in pairs[: max(2, len(pairs) // 2)]:
            edges.append(ConnectomeEdge(a_map[a_id], b_map[b_id], bridge_gain, delay_ms=1.5, provenance=f"chimera_bridge:{mode}"))
            edges.append(ConnectomeEdge(b_map[b_id], a_map[a_id], bridge_gain, delay_ms=1.5, provenance=f"chimera_bridge:{mode}"))
        sens = [a_map[s] for s in graph_a.sensory_ids if s in a_map] + [b_map[s] for s in graph_b.sensory_ids if s in b_map]
        mot = [a_map[m] for m in graph_a.motor_ids if m in a_map] + [b_map[m] for m in graph_b.motor_ids if m in b_map]

    elif mode in ("structural_averaging", "synaptic_weight_interpolation", "functional_distillation"):
        # Distinct aligned merging across structural, linear weight, and Procrustes functional modes
        neurons = [NeuronMorphologySpec.from_dict(u.to_dict()) for u in graph_a.neurons]
        edge_acc: Dict[Tuple[str, str], float] = {}
        edges_a_set = {(e.pre_id, e.post_id) for e in graph_a.edges}
        edges_b_mapped = {(b_to_a[e.pre_id], b_to_a[e.post_id]) for e in graph_b.edges if e.pre_id in b_to_a and e.post_id in b_to_a}
        if mode == "functional_distillation" and corr.get("procrustes_Q"):
            Q = np.asarray(corr["procrustes_Q"], dtype=np.float64)
            Wa = _adjacency_matrix(graph_a)
            Wb = _adjacency_matrix(graph_b)
            r_ind, c_ind = corr["row_ind"], corr["col_ind"]
            Wb_aligned = Q.T @ Wb[np.ix_(c_ind, c_ind)] @ Q
            W_f = Wa.copy()
            W_f[np.ix_(r_ind, r_ind)] = r_a * Wa[np.ix_(r_ind, r_ind)] + r_b * Wb_aligned
            for i_post in range(len(neurons)):
                for j_pre in range(len(neurons)):
                    if abs(W_f[i_post, j_pre]) > 0.03:
                        edge_acc[(neurons[j_pre].morphology_id, neurons[i_post].morphology_id)] = float(W_f[i_post, j_pre])
        else:
            for e in graph_a.edges:
                edge_acc[(e.pre_id, e.post_id)] = edge_acc.get((e.pre_id, e.post_id), 0.0) + r_a * e.weight
            for e in graph_b.edges:
                if e.pre_id in b_to_a and e.post_id in b_to_a:
                    key = (b_to_a[e.pre_id], b_to_a[e.post_id])
                    if mode == "structural_averaging" and key not in edges_a_set:
                        # Structural averaging keeps shared backbone stronger than unilateral edges
                        edge_acc[key] = edge_acc.get(key, 0.0) + 0.5 * r_b * e.weight
                    else:
                        edge_acc[key] = edge_acc.get(key, 0.0) + r_b * e.weight
        id_sign = {n.morphology_id: n.dale_sign for n in neurons}
        edges = []
        for (pre, post), w in edge_acc.items():
            if pre != post and abs(w) > 1e-4:
                w_dale = abs(w) if id_sign.get(pre, 1) > 0 else -abs(w)
                edges.append(ConnectomeEdge(pre, post, float(np.clip(w_dale, -4.0, 4.0)), delay_ms=1.0, provenance=f"chimera_aligned:{mode}"))
        sens, mot = list(graph_a.sensory_ids), list(graph_a.motor_ids)

    elif mode in ("neural_crossover", "partial_circuit_transplantation", "heterogeneous_mosaic"):
        # Take subset of neurons from A and corresponding neurons from B
        neurons = []
        b_lookup = {v.morphology_id: v for v in graph_b.neurons}
        a_to_b = {a_id: b_id for a_id, b_id in pairs}
        for i, u in enumerate(graph_a.neurons):
            take_b = (rng.random() > r_a) if mode != "partial_circuit_transplantation" else (i >= len(graph_a.neurons) // 2)
            if take_b and u.morphology_id in a_to_b and a_to_b[u.morphology_id] in b_lookup:
                donor = NeuronMorphologySpec.from_dict(b_lookup[a_to_b[u.morphology_id]].to_dict())
                donor.morphology_id = u.morphology_id
                donor.metadata["chimera_origin"] = graph_b.connectome_id
                neurons.append(donor)
            else:
                nu = NeuronMorphologySpec.from_dict(u.to_dict())
                nu.metadata["chimera_origin"] = graph_a.connectome_id
                neurons.append(nu)
        edges = [ConnectomeEdge.from_dict(e.to_dict()) for e in graph_a.edges]
        for e in graph_b.edges:
            if e.pre_id in b_to_a and e.post_id in b_to_a and rng.random() < r_b:
                edges.append(ConnectomeEdge(b_to_a[e.pre_id], b_to_a[e.post_id], e.weight * 0.5, e.delay_ms, provenance=f"chimera_{mode}"))
        sens, mot = list(graph_a.sensory_ids), list(graph_a.motor_ids)

    else:  # compartment_transplantation
        neurons = [NeuronMorphologySpec.from_dict(u.to_dict()) for u in graph_a.neurons]
        b_lookup = {v.morphology_id: v for v in graph_b.neurons}
        a_to_b = {a_id: b_id for a_id, b_id in pairs}
        for nu in neurons:
            if nu.morphology_id in a_to_b and a_to_b[nu.morphology_id] in b_lookup:
                donor = b_lookup[a_to_b[nu.morphology_id]]
                if donor.organelles:
                    org = donor.organelles[0]
                    org_copy = type(org).from_dict(org.to_dict())
                    org_copy.organelle_id = f"tx_{org.organelle_id}"
                    org_copy.compartment_id = nu.compartments[0].compartment_id
                    nu.organelles.append(org_copy)
        edges = [ConnectomeEdge.from_dict(e.to_dict()) for e in graph_a.edges]
        sens, mot = list(graph_a.sensory_ids), list(graph_a.motor_ids)

    child = ConnectomeGraph(
        connectome_id=child_id,
        name=f"Chimera ({mode}) [{graph_a.name} + {graph_b.name}]",
        neurons=neurons,
        edges=edges,
        sensory_ids=sens,
        motor_ids=mot,
        source_kind=DataSourceKind.HYBRID_EXPERIMENTAL,
        provenance={
            "chimera_mode": mode,
            "ratio_a": r_a,
            "parent_a": graph_a.connectome_id,
            "parent_b": graph_b.connectome_id,
            "alignment_mean_similarity": corr["mean_similarity"],
            "procrustes_residual": corr["procrustes_residual"],
        },
    )
    receipt = ChimeraMergeReceipt(
        merge_id=make_id("mrg"),
        mode=mode,
        ratio_a=r_a,
        parent_a_id=graph_a.connectome_id,
        parent_a_digest=graph_a.digest(),
        parent_b_id=graph_b.connectome_id,
        parent_b_digest=graph_b.digest(),
        child_connectome_id=child.connectome_id,
        child_digest=child.digest(),
        alignment_score=corr["mean_similarity"],
        correspondence_pairs=pairs,
        parent_a_snapshot=snap_a,
        parent_b_snapshot=snap_b,
    )
    return child, receipt


def run_chimera_comparative_benchmark(seed: int = 42, n_neurons: int = 8) -> Dict[str, Any]:
    """Compare direct unaligned averaging vs aligned merging vs ensemble vs grafting vs distillation."""
    ga = generate_candidate_genome("mixed_custom_standard", n_neurons=n_neurons, seed=seed).connectome
    gb = generate_candidate_genome("fractal_topology", n_neurons=n_neurons, seed=seed + 17).connectome

    rng = np.random.default_rng(seed + 100)
    steps = 70
    t = np.arange(steps, dtype=np.float64)
    u = np.stack([np.sin(0.25 * t), np.cos(0.41 * t), rng.choice([-1.0, 1.0], size=steps), rng.normal(0, 0.2, size=steps)], axis=1)
    # Composite two-skill target: Skill A (delayed sine) + Skill B (parity product)
    target = 0.5 * np.roll(u[:, 0], 3) + 0.5 * (u[:, 2] * np.roll(u[:, 2], 2))
    target[:3] = 0.0
    split = int(0.6 * steps)

    def _eval_graph(graph: ConnectomeGraph) -> Tuple[float, np.ndarray, np.ndarray]:
        sim = MultiFidelityConnectome(graph, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
        S = np.zeros((steps, len(sim.neuron_ids)), dtype=np.float64)
        for k in range(steps):
            S[k] = sim.step(u[k], collect_telemetry=False)["outputs"]
        w = np.linalg.solve(S[:split].T @ S[:split] + 0.35 * np.eye(S.shape[1]), S[:split].T @ target[:split])
        preds_tr = S[:split] @ w
        preds_te = S[split:] @ w
        return round(float(np.mean((preds_te - target[split:]) ** 2)), 5), preds_te, preds_tr

    mse_a, pred_a, pred_a_tr = _eval_graph(ga)
    mse_b, pred_b, pred_b_tr = _eval_graph(gb)
    mse_ensemble = round(float(np.mean((0.5 * (pred_a + pred_b) - target[split:]) ** 2)), 5)

    # Unaligned naive index-order averaging
    ga_d = ConnectomeGraph.from_dict(ga.to_dict())
    b_edges_by_idx = {(e.pre_id.replace("n", ""), e.post_id.replace("n", "")): e.weight for e in gb.edges}
    for e in ga_d.edges:
        k = (e.pre_id.replace("n", ""), e.post_id.replace("n", ""))
        e.weight = 0.5 * e.weight + 0.5 * b_edges_by_idx.get(k, 0.0)
    mse_direct_avg, _, _ = _eval_graph(ga_d)

    # Aligned weight merging
    g_aligned, rec_al = merge_connectomes(ga, gb, mode="synaptic_weight_interpolation", ratio_a=0.5, seed=seed)
    mse_aligned, _, _ = _eval_graph(g_aligned)

    # Anatomical grafting
    g_graft, _ = merge_connectomes(ga, gb, mode="anatomical_grafting", ratio_a=0.5, seed=seed)
    mse_graft, _, _ = _eval_graph(g_graft)

    # Genuine behavioral distillation: train student readout on 50% teacher ensemble soft predictions + 50% target
    sim_s = MultiFidelityConnectome(g_aligned, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
    Ss = np.zeros((steps, len(sim_s.neuron_ids)), dtype=np.float64)
    for k in range(steps):
        Ss[k] = sim_s.step(u[k], collect_telemetry=False)["outputs"]
    teacher_soft_tr = 0.5 * (pred_a_tr + pred_b_tr)
    soft_tr = 0.5 * target[:split] + 0.5 * teacher_soft_tr
    w_dist = np.linalg.solve(Ss[:split].T @ Ss[:split] + 0.25 * np.eye(Ss.shape[1]), Ss[:split].T @ soft_tr)
    mse_distill = round(float(np.mean((Ss[split:] @ w_dist - target[split:]) ** 2)), 5)

    methods = {
        "parent_a": mse_a,
        "parent_b": mse_b,
        "direct_unaligned_averaging": mse_direct_avg,
        "aligned_weight_merging": mse_aligned,
        "ensemble_prediction_fusion": mse_ensemble,
        "anatomical_grafting": mse_graft,
        "behavioral_distillation": mse_distill,
    }
    return {
        "experiment": "sonapse_chimera_fusion_comparison",
        "seed": seed,
        "all_supported_modes": list(CHIMERA_MODES),
        "alignment_similarity": rec_al.alignment_score,
        "test_mse_by_method": methods,
        "best_method": min(methods, key=methods.get),  # type: ignore[arg-type]
        "reversible_rollback_verified": rec_al.rollback_parents()[0].digest() == ga.digest(),
    }


def audit_connectome_symmetry_and_fingerprints(graph_a: ConnectomeGraph, graph_b: ConnectomeGraph) -> Dict[str, Any]:
    """SYNAPSE-ATLAS v4.0 inspired connectome observatory metrics:
    1. Bilateral left/right mirror-wiring symmetry overlap.
    2. Partner-weight incoming/outgoing cosine wiring fingerprints.
    3. Normalized-weight Jensen-Shannon distance (JSD) between two connectomes.
    """
    Wa = np.abs(_adjacency_matrix(graph_a))
    Wb = np.abs(_adjacency_matrix(graph_b))
    n = min(Wa.shape[0], Wb.shape[0])
    if n < 2:
        return {"bilateral_mirror_overlap": 0.0, "mean_fingerprint_cosine": 0.0, "normalized_weight_jsd": 0.0}

    half = n // 2
    left_block = Wa[:half, :half]
    right_block = Wa[half : 2 * half, half : 2 * half]
    mirror_overlap = float(np.sum(np.minimum(left_block, right_block)) / (np.sum(np.maximum(left_block, right_block)) + 1e-8))

    # Partner-weight profile fingerprints [incoming || outgoing]
    prof_a = np.hstack([Wa[:n, :n], Wa[:n, :n].T])
    prof_b = np.hstack([Wb[:n, :n], Wb[:n, :n].T])
    na_nrm = np.linalg.norm(prof_a, axis=1) + 1e-8
    nb_nrm = np.linalg.norm(prof_b, axis=1) + 1e-8
    cos_diag = np.sum(prof_a * prof_b, axis=1) / (na_nrm * nb_nrm)

    # Normalized-weight Jensen-Shannon distance (raw counts vs normalized fractions separated)
    pa = Wa[:n, :n].ravel() + 1e-12
    pb = Wb[:n, :n].ravel() + 1e-12
    pa = pa / np.sum(pa)
    pb = pb / np.sum(pb)
    pm = 0.5 * (pa + pb)
    js_div = 0.5 * float(np.sum(pa * np.log(pa / pm)) + np.sum(pb * np.log(pb / pm)))
    js_dist = float(np.sqrt(max(0.0, js_div / np.log(2.0))))
    return {
        "common_scope_nodes": int(n),
        "bilateral_mirror_overlap": round(mirror_overlap, 5),
        "mean_fingerprint_cosine": round(float(np.mean(cos_diag)), 5),
        "normalized_weight_jsd": round(js_dist, 5),
        "raw_total_weight_a": round(float(np.sum(Wa)), 4),
        "raw_total_weight_b": round(float(np.sum(Wb)), 4),
    }
