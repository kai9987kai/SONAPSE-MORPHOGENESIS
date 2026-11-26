"""SONAPSE GENESIS — Developmental Genome Variation & Transactional Reversible Mutations.

Implements versioned developmental genome edits across morphology, somas,
dendrites, axons, organelles, conduction delays, physical cell dimensions,
plasticity rules, neuromodulation, cellular specialization, and circuit motifs.
Every edit executes transactionally with an archived before-snapshot for rollback.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from ..core.interchange import (
    CompartmentEdge,
    CompartmentKind,
    CompartmentNode,
    ConnectomeEdge,
    ConnectomeGraph,
    DevelopmentalGenome,
    FidelityLevel,
    OrganelleAttachment,
    make_id,
)
from ..foundry.graft_neurons import ALL_GRAFT_NEURON_FAMILIES, GRAFT_NEURON_FAMILIES, build_graft_neuron
from ..foundry.organelles import ALL_ORGANELLE_REGISTRY, ORGANELLE_REGISTRY

MUTATION_OPERATORS: Tuple[str, ...] = (
    "add_secondary_soma",
    "add_dendritic_branch",
    "redistribute_compartments",
    "add_axonal_pathway",
    "attach_organelle",
    "remove_organelle",
    "mutate_conduction_delay",
    "mutate_cell_dimensions",
    "swap_plasticity_rule",
    "mutate_neuromodulation",
    "specialize_neuron",
    "insert_circuit_motif",
    "add_synapse",
    "prune_synapse",
    "perturb_synaptic_weights",
)

EXTENDED_MUTATION_OPERATORS: Tuple[str, ...] = (
    "insert_hodge_simplex_motif",
    "rewire_rigl_synapse",
    "attach_arbitrosome_coordinator",
    "differentiate_fidelity_tier",
    "mirror_bilateral_clone",
)

ALL_MUTATION_OPERATORS: Tuple[str, ...] = MUTATION_OPERATORS + EXTENDED_MUTATION_OPERATORS

PLASTICITY_RULES: Tuple[str, ...] = (
    "three_factor", "stdp", "bcm", "hebbian", "predictive_coding",
    "eprop_eligibility", "btsp_plateau_learning", "hodge_curl_free_plasticity", "none",
)


@dataclass
class MutationTransaction:
    """Reversible developmental edit capturing before/after digests and rollback snapshot."""
    transaction_id: str
    operator: str
    parent_genome_id: str
    parent_digest: str
    child_genome_id: str
    child_digest: str
    delta_summary: Dict[str, Any]
    before_snapshot: Dict[str, Any]

    def rollback(self) -> DevelopmentalGenome:
        """Restore the exact pre-mutation DevelopmentalGenome state."""
        return DevelopmentalGenome.from_dict(self.before_snapshot)


class DevelopmentalArchive:
    """Preserves original and evolved genomes plus all transactional mutation records."""

    def __init__(self) -> None:
        self.genomes: Dict[str, Dict[str, Any]] = {}
        self.transactions: List[Dict[str, Any]] = []

    def store(self, genome: DevelopmentalGenome) -> str:
        snap = genome.to_dict()
        self.genomes[genome.genome_id] = snap
        return genome.genome_id

    def get(self, genome_id: str) -> DevelopmentalGenome:
        return DevelopmentalGenome.from_dict(self.genomes[genome_id])

    def record_tx(self, tx: MutationTransaction, accepted: bool) -> None:
        self.transactions.append({
            "transaction_id": tx.transaction_id,
            "operator": tx.operator,
            "parent_genome_id": tx.parent_genome_id,
            "child_genome_id": tx.child_genome_id,
            "parent_digest": tx.parent_digest,
            "child_digest": tx.child_digest,
            "delta_summary": tx.delta_summary,
            "accepted": bool(accepted),
        })


def apply_genome_mutation(
    genome: DevelopmentalGenome,
    operator: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DevelopmentalGenome, MutationTransaction]:
    """Apply one bounded, reversible structural/developmental mutation to a copy of ``genome``."""
    rng = np.random.default_rng(seed)
    before_snap = genome.to_dict()
    parent_digest = genome.digest()
    child = DevelopmentalGenome.from_dict(before_snap)
    child.genome_id = make_id("gen")
    child.generation = genome.generation + 1
    child.parent_ids = [genome.genome_id]

    op = operator or str(rng.choice(MUTATION_OPERATORS))
    if op not in ALL_MUTATION_OPERATORS:
        raise ValueError(f"Unknown mutation operator: {op!r}")

    neurons = child.connectome.neurons
    edges = child.connectome.edges
    max_comp = int(child.structural_constraints.get("max_compartments_per_neuron", 24))
    delta: Dict[str, Any] = {"operator": op}

    if not neurons:
        raise ValueError("Cannot mutate an empty connectome genome")

    n_idx = int(rng.integers(0, len(neurons)))
    target_n = neurons[n_idx]

    if op == "add_secondary_soma" and len(target_n.compartments) < max_comp:
        cid = f"sec_soma_{len(target_n.compartments)}"
        base_c = target_n.compartments[0]
        new_c = CompartmentNode(
            compartment_id=cid, kind=CompartmentKind.SECONDARY_SOMA,
            length_um=24.0, radius_um=8.5, tau_ms=float(rng.uniform(15.0, 45.0)),
            coords_3d=(base_c.coords_3d[0], base_c.coords_3d[1] + 1.1, base_c.coords_3d[2]),
        )
        target_n.compartments.append(new_c)
        target_n.edges.append(CompartmentEdge(base_c.compartment_id, cid, weight=float(rng.uniform(0.3, 0.8))))
        target_n.edges.append(CompartmentEdge(cid, target_n.output_compartments[0], weight=float(rng.uniform(0.3, 0.8))))
        delta.update({"neuron": target_n.morphology_id, "added_compartment": cid})

    elif op == "add_dendritic_branch" and len(target_n.compartments) < max_comp:
        cid = f"dend_b_{len(target_n.compartments)}"
        new_c = CompartmentNode(
            compartment_id=cid, kind=CompartmentKind.DENDRITE_APICAL,
            length_um=float(rng.uniform(30.0, 90.0)), radius_um=float(rng.uniform(1.0, 2.8)),
            tau_ms=float(rng.uniform(6.0, 25.0)),
            coords_3d=(-2.0, float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-1.0, 1.0))),
        )
        target_n.compartments.append(new_c)
        target_n.input_compartments.append(cid)
        target_n.edges.append(CompartmentEdge(cid, target_n.compartments[0].compartment_id, weight=float(rng.uniform(0.4, 0.9))))
        delta.update({"neuron": target_n.morphology_id, "added_dendrite": cid})

    elif op == "redistribute_compartments":
        c = target_n.compartments[int(rng.integers(0, len(target_n.compartments)))]
        old_pos = c.coords_3d
        c.coords_3d = tuple(round(float(x + rng.normal(0.0, 0.45)), 3) for x in old_pos)  # type: ignore[assignment]
        delta.update({"neuron": target_n.morphology_id, "compartment": c.compartment_id, "new_coords": list(c.coords_3d)})

    elif op == "add_axonal_pathway" and len(target_n.compartments) < max_comp:
        cid = f"axon_p_{len(target_n.compartments)}"
        new_c = CompartmentNode(
            compartment_id=cid, kind=CompartmentKind.AXON_FEEDBACK,
            length_um=float(rng.uniform(60.0, 160.0)), radius_um=1.2,
            myelin_factor=float(rng.uniform(0.1, 0.85)), tau_ms=float(rng.uniform(5.0, 20.0)),
            coords_3d=(1.5, float(rng.uniform(-1.5, 1.5)), 0.0),
        )
        target_n.compartments.append(new_c)
        target_n.edges.append(CompartmentEdge(target_n.output_compartments[0], cid, weight=0.5, coupling_type="feedback_gated", delay_ms=float(rng.uniform(1.0, 4.0))))
        target_n.edges.append(CompartmentEdge(cid, target_n.input_compartments[0], weight=0.35, coupling_type="feedback_gated", delay_ms=float(rng.uniform(1.0, 3.0))))
        delta.update({"neuron": target_n.morphology_id, "added_axon_feedback": cid})

    elif op == "attach_organelle":
        otype = str(rng.choice(sorted(ORGANELLE_REGISTRY.keys())))
        comp = target_n.compartments[int(rng.integers(0, len(target_n.compartments)))]
        oid = f"org_{otype}_{len(target_n.organelles)}"
        target_n.organelles.append(OrganelleAttachment(oid, otype, comp.compartment_id, size_scale=float(rng.uniform(0.7, 1.4)), sensitivity=float(rng.uniform(0.6, 1.4))))
        delta.update({"neuron": target_n.morphology_id, "attached": otype, "compartment": comp.compartment_id})

    elif op == "remove_organelle" and target_n.organelles:
        removed = target_n.organelles.pop(int(rng.integers(0, len(target_n.organelles))))
        delta.update({"neuron": target_n.morphology_id, "removed_organelle": removed.organelle_id})

    elif op == "mutate_conduction_delay":
        c = target_n.compartments[int(rng.integers(0, len(target_n.compartments)))]
        c.myelin_factor = float(np.clip(c.myelin_factor + rng.normal(0.0, 0.2), 0.0, 1.0))
        if edges:
            e = edges[int(rng.integers(0, len(edges)))]
            e.delay_ms = float(np.clip(e.delay_ms + rng.normal(0.0, 0.8), 0.5, 12.0))
        delta.update({"neuron": target_n.morphology_id, "new_myelin": round(c.myelin_factor, 4)})

    elif op == "mutate_cell_dimensions":
        c = target_n.compartments[int(rng.integers(0, len(target_n.compartments)))]
        c.length_um = float(np.clip(c.length_um * rng.uniform(0.75, 1.35), 5.0, 500.0))
        c.radius_um = float(np.clip(c.radius_um * rng.uniform(0.75, 1.35), 0.4, 40.0))
        delta.update({"neuron": target_n.morphology_id, "compartment": c.compartment_id, "length_um": round(c.length_um, 2), "radius_um": round(c.radius_um, 2)})

    elif op == "swap_plasticity_rule" and edges:
        e = edges[int(rng.integers(0, len(edges)))]
        new_rule = str(rng.choice(PLASTICITY_RULES))
        old_rule = e.plasticity_rule
        e.plasticity_rule = new_rule
        delta.update({"edge": f"{e.pre_id}->{e.post_id}", "old_rule": old_rule, "new_rule": new_rule})

    elif op == "mutate_neuromodulation":
        key = str(rng.choice(list(child.neuromodulation_params.keys())))
        old_v = child.neuromodulation_params[key]
        child.neuromodulation_params[key] = round(float(np.clip(old_v + rng.normal(0.0, 0.15), 0.05, 3.0)), 4)
        delta.update({"param": key, "old": old_v, "new": child.neuromodulation_params[key]})

    elif op == "specialize_neuron":
        new_fam = str(rng.choice(GRAFT_NEURON_FAMILIES))
        replacement = build_graft_neuron(new_fam, neuron_id=target_n.morphology_id, fidelity=target_n.fidelity)
        replacement.dale_sign = target_n.dale_sign
        neurons[n_idx] = replacement
        delta.update({"neuron": target_n.morphology_id, "old_family": target_n.family, "new_family": new_fam})

    elif op == "insert_circuit_motif" and len(neurons) >= 3:
        # Insert a 3-node feedforward or lateral-inhibition triangle motif
        a, b, c_i = rng.choice(len(neurons), size=3, replace=False)
        for u_i, v_i in ((a, b), (b, c_i), (a, c_i)):
            w = float(rng.uniform(0.15, 0.45)) * neurons[u_i].dale_sign
            edges.append(ConnectomeEdge(neurons[u_i].morphology_id, neurons[v_i].morphology_id, weight=w, delay_ms=1.0, provenance="motif_evolution"))
        delta.update({"motif": "feedforward_loop", "nodes": [neurons[a].morphology_id, neurons[b].morphology_id, neurons[c_i].morphology_id]})

    elif op == "add_synapse" and len(neurons) >= 2:
        u_i, v_i = rng.choice(len(neurons), size=2, replace=False)
        w = float(rng.uniform(0.08, 0.55)) * neurons[u_i].dale_sign
        edges.append(ConnectomeEdge(neurons[u_i].morphology_id, neurons[v_i].morphology_id, weight=w, delay_ms=float(rng.uniform(0.5, 4.0))))
        delta.update({"added_synapse": f"{neurons[u_i].morphology_id}->{neurons[v_i].morphology_id}", "weight": round(w, 4)})

    elif op == "prune_synapse" and len(edges) > max(2, len(neurons)):
        idx = int(rng.integers(0, len(edges)))
        rem = edges.pop(idx)
        delta.update({"pruned_synapse": f"{rem.pre_id}->{rem.post_id}"})

    elif op == "insert_hodge_simplex_motif" and len(neurons) >= 3:
        # Insert a cyclic 2-simplex (3-cycle + chord) supporting Hodge curl memory
        a, b, c_i = rng.choice(len(neurons), size=3, replace=False)
        for idx_p, (u_i, v_i) in enumerate(((a, b), (b, c_i), (c_i, a))):
            w = float(rng.uniform(0.20, 0.50)) * neurons[u_i].dale_sign
            edges.append(ConnectomeEdge(
                neurons[u_i].morphology_id, neurons[v_i].morphology_id,
                weight=w, delay_ms=1.0 + float(idx_p), plasticity_rule="hodge_curl_free_plasticity",
                provenance="hodge_2simplex_motif",
            ))
        delta.update({"motif": "hodge_cyclic_2simplex", "nodes": [neurons[a].morphology_id, neurons[b].morphology_id, neurons[c_i].morphology_id]})

    elif op == "rewire_rigl_synapse" and len(neurons) >= 2:
        if edges:
            weak_idx = int(np.argmin([abs(e.weight) for e in edges]))
            rem = edges.pop(weak_idx)
        else:
            rem = None
        # Connect a sensory neuron to a memory/motor target with delayed recurrent path
        s_id = child.connectome.sensory_ids[0] if child.connectome.sensory_ids else neurons[0].morphology_id
        m_id = child.connectome.motor_ids[-1] if child.connectome.motor_ids else neurons[-1].morphology_id
        s_sign = next((u.dale_sign for u in neurons if u.morphology_id == s_id), 1)
        w_new = float(rng.uniform(0.25, 0.65)) * s_sign
        edges.append(ConnectomeEdge(s_id, m_id, weight=w_new, delay_ms=3.0, plasticity_rule="eprop_eligibility", provenance="rigl_rewire"))
        delta.update({"dropped": f"{rem.pre_id}->{rem.post_id}" if rem else None, "grown": f"{s_id}->{m_id}"})

    elif op == "attach_arbitrosome_coordinator":
        comp = target_n.compartments[0]
        target_n.organelles.append(OrganelleAttachment(f"org_mnem_{len(target_n.organelles)}", "mnemosome", comp.compartment_id))
        target_n.organelles.append(OrganelleAttachment(f"org_chrono_{len(target_n.organelles)+1}", "chronosome", comp.compartment_id, params={"tap_delay": 3}))
        target_n.organelles.append(OrganelleAttachment(f"org_arb_{len(target_n.organelles)+2}", "arbitrosome", comp.compartment_id))
        target_n.metadata["organelle_arbitration"] = True
        delta.update({"neuron": target_n.morphology_id, "attached_bundle": ["mnemosome", "chronosome", "arbitrosome"]})

    elif op == "differentiate_fidelity_tier":
        old_f = int(target_n.fidelity)
        new_f = 2 if old_f != 2 else 3
        target_n.fidelity = FidelityLevel(new_f)
        delta.update({"neuron": target_n.morphology_id, "old_fidelity": old_f, "new_fidelity": new_f})

    elif op == "mirror_bilateral_clone":
        max_n = int(child.growth_rules.get("max_neurons", 512))
        if len(neurons) < max_n:
            clone_spec = type(target_n).from_dict(target_n.to_dict())
            clone_spec.morphology_id = f"{target_n.morphology_id}_mir{len(neurons)}"
            for cc in clone_spec.compartments:
                cc.coords_3d = (-cc.coords_3d[0], cc.coords_3d[1], cc.coords_3d[2])
            neurons.append(clone_spec)
            edges.append(ConnectomeEdge(target_n.morphology_id, clone_spec.morphology_id, weight=0.28 * target_n.dale_sign, delay_ms=1.5, provenance="bilateral_commissure"))
            edges.append(ConnectomeEdge(clone_spec.morphology_id, target_n.morphology_id, weight=0.28 * clone_spec.dale_sign, delay_ms=1.5, provenance="bilateral_commissure"))
            delta.update({"cloned_from": target_n.morphology_id, "mirror_id": clone_spec.morphology_id})

    else:
        # Default fallback: perturb synaptic weights preserving Dale's law
        enforce_dale = bool(child.structural_constraints.get("enforce_dales_law", True))
        count = 0
        for e in edges:
            if rng.random() < 0.35:
                nw = e.weight + float(rng.normal(0.0, 0.12))
                if enforce_dale:
                    nw = abs(nw) if e.weight >= 0 else -abs(nw)
                e.weight = float(np.clip(nw, -4.0, 4.0))
                count += 1
        delta.update({"perturbed_edges": count})

    child.mutation_history.append(delta)
    tx = MutationTransaction(
        transaction_id=make_id("tx"),
        operator=op,
        parent_genome_id=genome.genome_id,
        parent_digest=parent_digest,
        child_genome_id=child.genome_id,
        child_digest=child.digest(),
        delta_summary=delta,
        before_snapshot=before_snap,
    )
    return child, tx
