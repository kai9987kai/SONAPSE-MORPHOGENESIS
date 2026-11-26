"""SONAPSE FOUNDRY — 21 Specified + 3 Original Custom Synthetic Graft Neuron Families.

Every architecture provides:
- Formal mathematical description
- Machine-readable ``NeuronMorphologySpec`` (compartments, physical geometry, edges, organelles)
- Runnable simulation via ``StructuredNeuronInstance``
- Publication-grade SVG diagram renderer (``render_neuron_svg``)
- Defined scientific hypothesis & objective classification
- Automated ablation configurations (``build_ablation_variants``)

Explicit epistemic boundary: All 24 custom graft families are computational
abstractions in ``SPECULATIVE_SYNTHETIC_BIOLOGY`` / ``COMPUTATIONAL_USEFULNESS``;
structural complexity never guarantees superior task performance without
controlled baseline evidence.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

from ..core.interchange import (
    CompartmentEdge,
    CompartmentKind,
    CompartmentNode,
    FidelityLevel,
    NeuronMorphologySpec,
    ObjectiveCategory,
    OrganelleAttachment,
    ReceptorDistribution,
)


def _node(
    cid: str,
    kind: CompartmentKind,
    coords: Tuple[float, float, float],
    length: float = 22.0,
    radius: float = 6.0,
    myelin: float = 0.0,
    tau: float = 10.0,
    receptors: ReceptorDistribution | None = None,
    meta: Dict[str, Any] | None = None,
) -> CompartmentNode:
    return CompartmentNode(
        compartment_id=cid,
        kind=kind,
        length_um=length,
        radius_um=radius,
        myelin_factor=myelin,
        tau_ms=tau,
        coords_3d=coords,
        receptors=receptors or ReceptorDistribution(),
        metadata=meta or {},
    )


def _edge(src: str, dst: str, w: float = 0.7, ctype: str = "electrical_axial",
          delay: float = 0.5, gated_by: str | None = None, plast: str = "three_factor") -> CompartmentEdge:
    return CompartmentEdge(src, dst, weight=w, coupling_type=ctype, delay_ms=delay, gated_by=gated_by, plasticity_rule=plast)


# ==============================================================================
# 21 SPECIFIED GRAFT NEURON FAMILIES + 3 AUTONOMOUS ORIGINAL FAMILIES
# ==============================================================================

def _build_triarch(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="triarch", name="Triarch Neuron",
        objective_category=ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend_in", CompartmentKind.DENDRITE_BASAL, (-2.5, 0.0, 0.0), 40.0, 2.2, tau=6.0),
            _node("soma_immediate", CompartmentKind.SOMA, (0.0, 1.6, 0.0), 20.0, 8.0, tau=4.0, meta={"role": "immediate"}),
            _node("soma_context", CompartmentKind.SECONDARY_SOMA, (0.0, 0.0, 0.0), 26.0, 9.5, tau=28.0, meta={"role": "contextual_memory"}),
            _node("soma_predict", CompartmentKind.SECONDARY_SOMA, (0.0, -1.6, 0.0), 22.0, 8.5, tau=12.0, meta={"role": "prediction"}),
            _node("disagreement_hub", CompartmentKind.INTEGRATION_CENTER, (1.8, 0.0, 0.0), 18.0, 6.5, tau=6.0),
            _node("axon_out", CompartmentKind.AXON_SEGMENT, (3.4, 0.0, 0.0), 90.0, 1.4, myelin=0.65, tau=4.0),
        ],
        edges=[
            _edge("dend_in", "soma_immediate", 0.95),
            _edge("dend_in", "soma_context", 0.55),
            _edge("soma_immediate", "soma_context", 0.45),
            _edge("soma_context", "soma_predict", 0.70),
            _edge("soma_immediate", "disagreement_hub", 0.85),
            _edge("soma_predict", "disagreement_hub", -0.75, "chemical_synapse"),
            _edge("soma_context", "disagreement_hub", 0.45),
            _edge("disagreement_hub", "axon_out", 0.90),
        ],
        organelles=[
            OrganelleAttachment("tri_pred", "predicton", "soma_predict"),
            OrganelleAttachment("tri_cont", "contraston", "disagreement_hub"),
            OrganelleAttachment("tri_plast", "plasticity_nucleus", "soma_context"),
        ],
        input_compartments=["dend_in"], output_compartments=["axon_out"],
        formal_math=r"s_1=\phi_\text{fast}(x),\; s_2=\phi_\text{slow}(s_2, s_1),\; s_3=\hat{s}_1(s_2),\; e_{\text{local}}=s_1-s_3,\; y=\phi(s_1, s_2, e_{\text{local}})",
        hypothesis="Disagreement between immediate, contextual, and predictive somas supplies a local self-supervised error signal.",
        description="Three coupled somas (immediate, contextual memory, prediction) integrated via an error-sensitive disagreement hub.",
    )


def _build_ouroboros(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="ouroboros", name="Ouroboros Neuron",
        objective_category=ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend_primary", CompartmentKind.DENDRITE_BASAL, (-2.4, 0.8, 0.0), 45.0, 2.0, tau=7.0),
            _node("dend_return", CompartmentKind.DENDRITE_APICAL, (-2.4, -0.8, 0.0), 55.0, 1.8, tau=18.0),
            _node("soma", CompartmentKind.SOMA, (0.0, 0.0, 0.0), 24.0, 9.0, tau=9.0),
            _node("axon_seg1", CompartmentKind.AXON_SEGMENT, (1.8, 0.0, 0.0), 100.0, 1.3, myelin=0.4, tau=5.0),
            _node("fb_fast", CompartmentKind.AXON_FEEDBACK, (0.6, 1.8, 0.0), 80.0, 1.1, myelin=0.5, tau=6.0),
            _node("fb_slow", CompartmentKind.AXON_FEEDBACK, (0.6, -1.8, 0.0), 180.0, 0.9, myelin=0.1, tau=24.0),
        ],
        edges=[
            _edge("dend_primary", "soma", 0.9),
            _edge("dend_return", "soma", 0.65),
            _edge("soma", "axon_seg1", 0.95),
            _edge("axon_seg1", "fb_fast", 0.75, "feedback_gated", delay=1.5, gated_by="soma"),
            _edge("axon_seg1", "fb_slow", 0.65, "feedback_gated", delay=4.5, gated_by="dend_return"),
            _edge("fb_fast", "dend_primary", 0.45, "feedback_gated", delay=1.0, gated_by="dend_primary"),
            _edge("fb_slow", "dend_return", 0.60, "feedback_gated", delay=3.0, gated_by="soma"),
        ],
        organelles=[OrganelleAttachment("ouro_chrono", "chronosome", "fb_slow", params={"tap_delay": 5})],
        input_compartments=["dend_primary"], output_compartments=["axon_seg1"],
        formal_math=r"d_{\text{ret}}(t) = \phi(g_1 a(t-\tau_1) + g_2 a(t-\tau_2)),\; a(t)=\phi(W[d_{\text{in}}(t), d_{\text{ret}}(t)])",
        hypothesis="Dual gated axonal-to-dendritic feedback loops with distinct physical delays sustain multi-scale temporal echoes.",
        description="Self-referential neuron with fast and slow gated axonal feedback loops returning to apical and basal dendrites.",
    )


def _build_hydra(nid: str) -> NeuronMorphologySpec:
    comps = [_node(f"head_{i}", CompartmentKind.DENDRITE_APICAL, (-2.5, 1.8 - 1.2 * i, 0.0), 50.0 + 10.0 * i, 1.8, tau=6.0 + 4.0 * i) for i in range(4)]
    comps += [
        _node("gate_hub", CompartmentKind.INTEGRATION_CENTER, (-0.3, 0.0, 0.0), 20.0, 7.0, tau=6.0),
        _node("soma", CompartmentKind.SOMA, (1.5, 0.0, 0.0), 24.0, 9.0, tau=8.0),
        _node("axon", CompartmentKind.AXON_SEGMENT, (3.2, 0.0, 0.0), 90.0, 1.5, myelin=0.7, tau=4.0),
    ]
    edges = [_edge(f"head_{i}", "gate_hub", 0.65 + 0.05 * i, "feedback_gated", gated_by=f"head_{i}") for i in range(4)]
    edges += [_edge("gate_hub", "soma", 0.9), _edge("soma", "axon", 0.95)]
    orgs = [OrganelleAttachment(f"hydra_att_{i}", "attentosome", f"head_{i}") for i in range(4)] + [OrganelleAttachment("hydra_route", "routosome", "gate_hub")]
    return NeuronMorphologySpec(
        morphology_id=nid, family="hydra", name="Hydra Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=comps, edges=edges, organelles=orgs,
        input_compartments=[f"head_{i}" for i in range(4)], output_compartments=["axon"],
        formal_math=r"y = \phi\left(\sum_{k=1}^4 \alpha_k(x_k)\, f_k(x_k)\right),\; \alpha_k = \text{softmax}(q^\top h_k)",
        hypothesis="Independently adaptive dendritic branches implement cellular mixture-of-experts and branch-level attention.",
        description="Four specialized adaptive dendritic heads with Attentosomes converging on a routed integration hub.",
    )


def _build_mnemosome_neuron(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="mnemosome_neuron", name="Mnemosome Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.2, 0.0, 0.0), 40.0, 2.0, tau=8.0),
            _node("mem_core", CompartmentKind.MEMORY_COMPARTMENT, (-0.2, 1.3, 0.0), 30.0, 10.0, tau=35.0),
            _node("soma", CompartmentKind.SOMA, (0.4, -0.3, 0.0), 22.0, 8.5, tau=9.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.6, 0.0, 0.0), 90.0, 1.4, myelin=0.6, tau=4.0),
        ],
        edges=[
            _edge("dend", "soma", 0.85), _edge("dend", "mem_core", 0.75),
            _edge("mem_core", "soma", 0.80), _edge("soma", "mem_core", 0.35),
            _edge("soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("mnem_main", "mnemosome", "mem_core", params={"slots": 12}),
            OrganelleAttachment("mnem_ckpt", "checkpoint_vesicle", "mem_core"),
        ],
        input_compartments=["dend"], output_compartments=["axon"],
        formal_math=r"m_t = \sum_j w_j(q_t) (0.55 v_j^{\text{fast}} + 0.45 v_j^{\text{slow}}),\; s_t = \phi(x_t + m_t)",
        hypothesis="Intracellular associative memory organelle extends single-cell retention beyond exponential decay limits.",
        description="Neuron equipped with a dedicated associative memory compartment and Mnemosome + CheckpointVesicle organelles.",
    )


def _build_chameleon(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="chameleon", name="Chameleon Neuron",
        objective_category=ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend_exc", CompartmentKind.DENDRITE_BASAL, (-2.2, 0.9, 0.0), 35.0, 2.0, tau=6.0, receptors=ReceptorDistribution(ampa=1.8, dopamine=1.2)),
            _node("dend_mod", CompartmentKind.DENDRITE_APICAL, (-2.2, -0.9, 0.0), 50.0, 1.8, tau=20.0, receptors=ReceptorDistribution(acetylcholine=1.5, serotonin=1.2)),
            _node("soma", CompartmentKind.SOMA, (0.0, 0.0, 0.0), 22.0, 8.0, tau=11.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.4, 0.0, 0.0), 85.0, 1.4, myelin=0.5, tau=5.0),
        ],
        edges=[
            _edge("dend_exc", "soma", 0.8, "feedback_gated", gated_by="dend_mod"),
            _edge("dend_mod", "soma", 0.55),
            _edge("soma", "axon", 0.9),
        ],
        organelles=[
            OrganelleAttachment("cham_dev", "developmental_genome", "soma", params={"maturation_horizon": 50}),
            OrganelleAttachment("cham_plast", "plasticity_nucleus", "dend_exc"),
        ],
        input_compartments=["dend_exc", "dend_mod"], output_compartments=["axon"],
        formal_math=r"\theta_{\text{pheno}}(t) = \text{Maturation}(\text{age}_t, \text{mod}_t);\; y_t = F_{\theta_{\text{pheno}}(t)}(x_t)",
        hypothesis="Developmentally shifting computational phenotype adapts excitability from exploration to exploitation.",
        description="Phenotype-shifting neuron whose membrane dynamics and receptor gating mature over developmental stages.",
    )


def _build_janus(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="janus", name="Janus Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.4, 0.0, 0.0), 40.0, 2.2, tau=8.0),
            _node("soma", CompartmentKind.SOMA, (-0.3, 0.0, 0.0), 24.0, 9.0, tau=10.0),
            _node("aiz_predict", CompartmentKind.AXON_HILLOCK, (1.5, 1.2, 0.0), 16.0, 3.0, tau=6.0, meta={"pathway": "predictive"}),
            _node("aiz_behave", CompartmentKind.AXON_HILLOCK, (1.5, -1.2, 0.0), 16.0, 3.0, tau=5.0, meta={"pathway": "behavioral"}),
            _node("axon_pred", CompartmentKind.AXON_SEGMENT, (3.2, 1.2, 0.0), 90.0, 1.2, myelin=0.6, tau=4.0),
            _node("axon_beh", CompartmentKind.AXON_SEGMENT, (3.2, -1.2, 0.0), 110.0, 1.6, myelin=0.8, tau=3.5),
        ],
        edges=[
            _edge("dend", "soma", 0.9),
            _edge("soma", "aiz_predict", 0.85, "feedback_gated", gated_by="soma"),
            _edge("soma", "aiz_behave", 0.85, "feedback_gated", gated_by="dend"),
            _edge("aiz_predict", "axon_pred", 0.95),
            _edge("aiz_behave", "axon_beh", 0.95),
        ],
        organelles=[
            OrganelleAttachment("janus_pred", "predicton", "aiz_predict"),
            OrganelleAttachment("janus_route", "routosome", "aiz_behave"),
        ],
        input_compartments=["dend"], output_compartments=["axon_pred", "axon_beh"],
        formal_math=r"y_{\text{pred}} = \sigma(g_p) \hat{s}_t,\; y_{\text{beh}} = \sigma(g_b) s_t",
        hypothesis="Dual independently gated axonal initiation zones decouple predictive reporting from overt behavioral firing.",
        description="Dual-hillock neuron with separate gated predictive and behavioral axonal output pathways.",
    )


def _build_chronos(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="chronos", name="Chronos Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.6, 0.0, 0.0), 35.0, 2.0, tau=5.0),
            _node("tau_fast", CompartmentKind.INTEGRATION_CENTER, (-0.5, 1.4, 0.0), 20.0, 5.0, tau=3.0),
            _node("tau_mid", CompartmentKind.INTEGRATION_CENTER, (-0.5, 0.0, 0.0), 35.0, 6.0, tau=14.0),
            _node("tau_slow", CompartmentKind.INTEGRATION_CENTER, (-0.5, -1.4, 0.0), 50.0, 7.0, tau=48.0),
            _node("soma", CompartmentKind.SOMA, (1.4, 0.0, 0.0), 24.0, 9.0, tau=8.0),
            _node("axon_delay", CompartmentKind.AXON_SEGMENT, (3.2, 0.0, 0.0), 220.0, 1.1, myelin=0.25, tau=5.0),
        ],
        edges=[
            _edge("dend", "tau_fast", 0.9, delay=0.5),
            _edge("dend", "tau_mid", 0.8, delay=2.0),
            _edge("dend", "tau_slow", 0.7, delay=5.0),
            _edge("tau_fast", "soma", 0.6), _edge("tau_mid", "soma", 0.7), _edge("tau_slow", "soma", 0.8),
            _edge("soma", "axon_delay", 0.95, delay=2.0),
        ],
        organelles=[
            OrganelleAttachment("chr_1", "chronosome", "tau_mid", params={"tap_delay": 3}),
            OrganelleAttachment("chr_2", "chronosome", "tau_slow", params={"tap_delay": 7}),
        ],
        input_compartments=["dend"], output_compartments=["axon_delay"],
        formal_math=r"h_k(t+1) = (1-\Delta t/\tau_k) h_k(t) + (\Delta t/\tau_k)\tanh(x(t-\delta_k)),\; \tau_k \in \{3,14,48\}\text{ ms}",
        hypothesis="Multi-timescale compartments paired with adjustable delay taps resolve temporal credit & rhythm tasks.",
        description="Multi-horizon temporal integration neuron with fast/mid/slow compartments and dual Chronosomes.",
    )


def _build_nexus(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="nexus", name="Nexus Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("mod_vis", CompartmentKind.DENDRITE_BASAL, (-2.5, 1.5, 0.0), 40.0, 2.0, tau=7.0, receptors=ReceptorDistribution(nmda=1.0)),
            _node("mod_aud", CompartmentKind.DENDRITE_BASAL, (-2.5, 0.0, 0.0), 40.0, 2.0, tau=7.0, receptors=ReceptorDistribution(nmda=1.0)),
            _node("mod_sym", CompartmentKind.DENDRITE_APICAL, (-2.5, -1.5, 0.0), 50.0, 2.0, tau=12.0, receptors=ReceptorDistribution(nmda=1.2)),
            _node("binding_soma", CompartmentKind.SOMA, (0.2, 0.0, 0.0), 26.0, 10.0, tau=9.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.6, 0.0, 0.0), 90.0, 1.5, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("mod_vis", "binding_soma", 0.75), _edge("mod_aud", "binding_soma", 0.75),
            _edge("mod_sym", "binding_soma", 0.75),
            _edge("mod_vis", "mod_aud", 0.35, "gap_junction"),
            _edge("mod_aud", "mod_sym", 0.35, "gap_junction"),
            _edge("binding_soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("nex_trans", "translator_membrane", "binding_soma"),
            OrganelleAttachment("nex_att", "attentosome", "binding_soma"),
        ],
        input_compartments=["mod_vis", "mod_aud", "mod_sym"], output_compartments=["axon"],
        formal_math=r"z_{\text{bind}} = \phi\left(\sum_m W_m x_m + \sum_{m<n} \gamma_{mn} (x_m \odot x_n)\right)",
        hypothesis="Cross-coupled modality dendrites learn multiplicative cross-modal coincidences.",
        description="Multimodal integration neuron with lateral gap-junction binding across sensory dendritic trees.",
    )


def _build_paracrine_broadcast(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="paracrine_broadcast", name="Paracrine Broadcast Neuron",
        objective_category=ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.0, 0.0, 0.0), 35.0, 2.0, tau=10.0),
            _node("soma", CompartmentKind.SOMA, (0.0, 0.0, 0.0), 24.0, 9.0, tau=14.0),
            _node("secretory_hub", CompartmentKind.METABOLIC_REGULATOR, (2.0, 0.0, 0.0), 22.0, 8.0, tau=25.0, receptors=ReceptorDistribution(paracrine=2.5)),
        ],
        edges=[_edge("dend", "soma", 0.85), _edge("soma", "secretory_hub", 0.90, "paracrine")],
        organelles=[OrganelleAttachment("para_res", "resource_regulator", "secretory_hub")],
        input_compartments=["dend"], output_compartments=["secretory_hub"],
        formal_math=r"\dot{C}_{\text{para}} = -\gamma C_{\text{para}} + \sum_i \beta_i s_{\text{sec},i}",
        hypothesis="Diffusible volume transmission coordinates neighboring cells independently of point-to-point synapses.",
        description="Diffusible-signal broadcaster modulating nearby connectome cells via simulated paracrine fields.",
    )


def _build_mosaic(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="mosaic", name="Mosaic Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.5, 0.0, 0.0), 35.0, 2.0, tau=8.0),
            _node("dock_a", CompartmentKind.ORGANELLE_DOCK, (-0.5, 1.1, 0.0), 20.0, 6.0, tau=10.0),
            _node("dock_b", CompartmentKind.ORGANELLE_DOCK, (-0.5, -1.1, 0.0), 20.0, 6.0, tau=10.0),
            _node("soma", CompartmentKind.SOMA, (1.2, 0.0, 0.0), 22.0, 8.5, tau=9.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.9, 0.0, 0.0), 85.0, 1.4, myelin=0.6, tau=4.0),
        ],
        edges=[
            _edge("dend", "dock_a", 0.8), _edge("dend", "dock_b", 0.8),
            _edge("dock_a", "soma", 0.7), _edge("dock_b", "soma", 0.7),
            _edge("soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("mos_a", "mnemosome", "dock_a"),
            OrganelleAttachment("mos_b", "phase_organelle", "dock_b"),
        ],
        input_compartments=["dend"], output_compartments=["axon"],
        formal_math=r"y = \phi(W_a G_a(x) + W_b G_b(x))",
        hypothesis="Hot-swappable internal computational graft docks allow targeted evolution without cell replacement.",
        description="Modular neuron exposing swappable organelle/graft docks preserving cell boundary interfaces.",
    )


def _build_sentinel_neuron(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="sentinel_neuron", name="Sentinel Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.3, 0.0, 0.0), 35.0, 2.0, tau=6.0),
            _node("monitor_comp", CompartmentKind.INTEGRATION_CENTER, (-0.2, 1.3, 0.0), 20.0, 6.5, tau=5.0),
            _node("soma", CompartmentKind.SOMA, (0.2, -0.4, 0.0), 22.0, 8.0, tau=8.0),
            _node("alarm_axon", CompartmentKind.AXON_SEGMENT, (2.5, 0.0, 0.0), 90.0, 1.5, myelin=0.85, tau=3.0),
        ],
        edges=[
            _edge("dend", "monitor_comp", 0.9), _edge("dend", "soma", 0.8),
            _edge("monitor_comp", "soma", 0.85), _edge("soma", "alarm_axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("sent_cont", "contraston", "monitor_comp"),
            OrganelleAttachment("sent_unc", "uncertaintosome", "monitor_comp"),
        ],
        input_compartments=["dend"], output_compartments=["alarm_axon"],
        formal_math=r"a_{\text{err}} = |x_t - \hat{x}_t| + \lambda \hat{\sigma}_t",
        hypothesis="Dedicated error-detection compartments flag anomalous or contradictory local neural patterns rapidly.",
        description="Anomaly-monitoring neuron combining Contraston and Uncertaintosome compartments with fast myelinated alarm output.",
    )


def _build_frontier_neuron(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="frontier_neuron", name="Frontier Neuron",
        objective_category=ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.0, 0.0, 0.0), 35.0, 2.0, tau=10.0),
            _node("soma", CompartmentKind.SOMA, (0.0, 0.0, 0.0), 20.0, 7.5, tau=10.0),
            _node("growth_cone", CompartmentKind.AXON_SEGMENT, (2.2, 0.5, 0.0), 65.0, 1.0, myelin=0.0, tau=12.0, meta={"growth_cone": True}),
        ],
        edges=[_edge("dend", "soma", 0.85), _edge("soma", "growth_cone", 0.85, plast="three_factor")],
        organelles=[
            OrganelleAttachment("front_dev", "developmental_genome", "growth_cone"),
            OrganelleAttachment("front_plast", "plasticity_nucleus", "growth_cone"),
        ],
        input_compartments=["dend"], output_compartments=["growth_cone"],
        formal_math=r"\Delta p_{\text{cone}} \propto \nabla \text{coherence}(s_{\text{cone}}, s_{\text{target}})",
        hypothesis="Simulated growth-cone compartment explores candidate targets based on correlation and novelty gradients.",
        description="Developmental exploration neuron with active unmyelinated growth-cone compartment.",
    )


def _build_bifrost(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="bifrost", name="Bifrost Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("bio_dend", CompartmentKind.DENDRITE_BASAL, (-2.4, 0.9, 0.0), 40.0, 2.0, tau=8.0),
            _node("ml_port", CompartmentKind.ML_ADAPTER, (-2.4, -0.9, 0.0), 20.0, 6.0, tau=5.0),
            _node("soma", CompartmentKind.SOMA, (0.0, 0.0, 0.0), 24.0, 9.0, tau=9.0),
            _node("bio_axon", CompartmentKind.AXON_SEGMENT, (2.4, 0.9, 0.0), 90.0, 1.4, myelin=0.65, tau=4.0),
            _node("ml_out", CompartmentKind.ML_ADAPTER, (2.4, -0.9, 0.0), 20.0, 6.0, tau=4.0),
        ],
        edges=[
            _edge("bio_dend", "soma", 0.8), _edge("ml_port", "soma", 0.6, "ml_bridge"),
            _edge("soma", "bio_axon", 0.9), _edge("soma", "ml_out", 0.85, "ml_bridge"),
        ],
        organelles=[OrganelleAttachment("bif_trans", "translator_membrane", "ml_port")],
        input_compartments=["bio_dend", "ml_port"], output_compartments=["bio_axon", "ml_out"],
        formal_math=r"y_{\text{bio}} = \phi(W_b s),\; y_{\text{ML}} = W_{\text{adapt}} s",
        hypothesis="Dual native-connectome and ML-adapter interfaces enable clean cross-paradigm coupling with Supermix models.",
        description="Hybrid bridge neuron uniting biological-style dendrites/axons with explicit ML adapter ports.",
    )


def _build_resonator(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="resonator", name="Resonator Neuron",
        objective_category=ObjectiveCategory.BIOLOGICAL_REALISM,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.2, 0.0, 0.0), 35.0, 2.0, tau=6.0),
            _node("osc_a", CompartmentKind.INTEGRATION_CENTER, (0.0, 1.0, 0.0), 20.0, 7.0, tau=8.0),
            _node("osc_b", CompartmentKind.INTEGRATION_CENTER, (0.0, -1.0, 0.0), 20.0, 7.0, tau=8.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.4, 0.0, 0.0), 90.0, 1.5, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("dend", "osc_a", 0.85),
            _edge("osc_a", "osc_b", 0.9, "electrical_axial"),
            _edge("osc_b", "osc_a", -0.9, "chemical_synapse"),
            _edge("osc_a", "axon", 0.9),
        ],
        organelles=[
            OrganelleAttachment("res_ph1", "phase_organelle", "osc_a", params={"omega": 0.35}),
            OrganelleAttachment("res_ph2", "phase_organelle", "osc_b", params={"omega": 0.70}),
        ],
        input_compartments=["dend"], output_compartments=["axon"],
        formal_math=r"\dot{x} = \omega y + I,\; \dot{y} = -\omega x - \gamma y",
        hypothesis="Coupled oscillatory compartments selectively amplify resonant frequencies and encode phase synchrony.",
        description="Intrinsic oscillatory neuron with quadrature excitatory-inhibitory loop and dual PhaseOrganelles.",
    )


def _build_palimpsest(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="palimpsest", name="Palimpsest Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.5, 0.0, 0.0), 35.0, 2.0, tau=6.0),
            _node("trace_recent", CompartmentKind.MEMORY_COMPARTMENT, (-0.3, 1.2, 0.0), 22.0, 7.0, tau=6.0),
            _node("trace_archive", CompartmentKind.MEMORY_COMPARTMENT, (-0.3, -1.2, 0.0), 34.0, 10.0, tau=95.0),
            _node("soma", CompartmentKind.SOMA, (1.4, 0.0, 0.0), 24.0, 8.5, tau=9.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (3.0, 0.0, 0.0), 90.0, 1.4, myelin=0.6, tau=4.0),
        ],
        edges=[
            _edge("dend", "trace_recent", 0.9),
            _edge("trace_recent", "trace_archive", 0.25, "feedback_gated", gated_by="soma"),
            _edge("trace_recent", "soma", 0.65), _edge("trace_archive", "soma", 0.65),
            _edge("soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("pal_mnem", "mnemosome", "trace_archive", params={"slow_decay": 0.996}),
            OrganelleAttachment("pal_ckpt", "checkpoint_vesicle", "trace_archive"),
        ],
        input_compartments=["dend"], output_compartments=["axon"],
        formal_math=r"m_{\text{fast}}(t) \text{ overwrites freely},\; m_{\text{slow}}(t) = (1-\epsilon g_t) m_{\text{slow}}(t-1) + \epsilon g_t m_{\text{fast}}(t)",
        hypothesis="Gated separation of fast labile vs slow consolidated compartments mitigates catastrophic forgetting.",
        description="Multi-timescale layered memory neuron protecting consolidated traces from recent overwriting.",
    )


def _build_counterpoint(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="counterpoint", name="Counterpoint Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend_in", CompartmentKind.DENDRITE_BASAL, (-2.5, 0.0, 0.0), 35.0, 2.0, tau=6.0),
            _node("path_exc", CompartmentKind.INTEGRATION_CENTER, (-0.3, 1.2, 0.0), 22.0, 6.5, tau=7.0),
            _node("path_inh", CompartmentKind.LOCAL_INHIBITORY, (-0.3, -1.2, 0.0), 22.0, 6.5, tau=9.0),
            _node("comparator", CompartmentKind.SOMA, (1.5, 0.0, 0.0), 24.0, 8.5, tau=6.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (3.1, 0.0, 0.0), 90.0, 1.4, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("dend_in", "path_exc", 0.95), _edge("dend_in", "path_inh", 0.85),
            _edge("path_exc", "path_inh", -0.35, "chemical_synapse"),
            _edge("path_inh", "path_exc", -0.35, "chemical_synapse"),
            _edge("path_exc", "comparator", 1.0), _edge("path_inh", "comparator", -0.95, "chemical_synapse"),
            _edge("comparator", "axon", 0.95),
        ],
        input_compartments=["dend_in"], output_compartments=["axon"],
        formal_math=r"d = \tanh(e_+ - e_-),\; e_+ \dashv\vdash e_-",
        hypothesis="Competing excitatory and inhibitory internal pathways sharpen local discrimination and temporal edge detection.",
        description="Internal push-pull opponent-process neuron with mutually inhibiting pathways feeding a comparator soma.",
    )


def _build_compass(nid: str) -> NeuronMorphologySpec:
    # 6-node cyclic ring attractor inside one neuron
    ring_n = 6
    comps = []
    for k in range(ring_n):
        ang = 2.0 * math.pi * k / ring_n
        comps.append(_node(f"ring_{k}", CompartmentKind.INTEGRATION_CENTER, (1.4 * math.cos(ang), 1.4 * math.sin(ang), 0.0), 20.0, 5.0, tau=10.0))
    comps.append(_node("soma_out", CompartmentKind.SOMA, (3.2, 0.0, 0.0), 22.0, 8.0, tau=7.0))
    edges = []
    for k in range(ring_n):
        edges.append(_edge(f"ring_{k}", f"ring_{(k + 1) % ring_n}", 0.55, "electrical_axial"))
        edges.append(_edge(f"ring_{k}", f"ring_{(k - 1) % ring_n}", 0.55, "electrical_axial"))
        edges.append(_edge(f"ring_{k}", f"ring_{(k + ring_n // 2) % ring_n}", -0.45, "chemical_synapse"))
        edges.append(_edge(f"ring_{k}", "soma_out", 0.4 * math.cos(2.0 * math.pi * k / ring_n)))
    return NeuronMorphologySpec(
        morphology_id=nid, family="compass", name="Compass Neuron",
        objective_category=ObjectiveCategory.BIOLOGICAL_REALISM,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=comps, edges=edges,
        input_compartments=[f"ring_{k}" for k in range(ring_n)], output_compartments=["soma_out"],
        formal_math=r"\dot{r}_k = -r_k + \phi\left(\sum_j W_{\text{ring}}(k-j) r_j + I_k\right)",
        hypothesis="Cyclic ring-attractor internal compartments maintain directional/heading phase representations.",
        description="Ellipsoid-body-inspired neuron containing a 6-compartment local excitation / opposite inhibition cyclic ring.",
    )


def _build_prism(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="prism", name="Prism Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("band_high", CompartmentKind.DENDRITE_BASAL, (-2.4, 1.4, 0.0), 20.0, 3.0, tau=2.5),
            _node("band_mid", CompartmentKind.DENDRITE_PROXIMAL, (-2.4, 0.0, 0.0), 50.0, 2.0, tau=12.0),
            _node("band_low", CompartmentKind.DENDRITE_DISTAL, (-2.4, -1.4, 0.0), 110.0, 1.2, tau=45.0),
            _node("soma", CompartmentKind.SOMA, (0.5, 0.0, 0.0), 24.0, 9.0, tau=7.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.6, 0.0, 0.0), 90.0, 1.4, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("band_high", "soma", 0.8), _edge("band_mid", "soma", 0.8), _edge("band_low", "soma", 0.8),
            _edge("soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("pr_hi", "phase_organelle", "band_high", params={"omega": 0.9}),
            OrganelleAttachment("pr_mid", "phase_organelle", "band_mid", params={"omega": 0.35}),
            OrganelleAttachment("pr_lo", "phase_organelle", "band_low", params={"omega": 0.1}),
        ],
        input_compartments=["band_high", "band_mid", "band_low"], output_compartments=["axon"],
        formal_math=r"d_b(t) = H_{\tau_b, \omega_b}[x(t)],\; y = \phi(\sum_b w_b d_b)",
        hypothesis="Cable-geometry-tuned dendrites decompose mixed temporal inputs into distinct frequency bands.",
        description="Spectral-decomposition neuron using fast/medium/slow dendritic cables and matched PhaseOrganelles.",
    )


def _build_consensus(nid: str) -> NeuronMorphologySpec:
    comps = [_node(f"ev_{i}", CompartmentKind.DENDRITE_BASAL, (-2.4, 1.5 - 1.5 * i, 0.0), 38.0, 2.0, tau=8.0 + 2.0 * i) for i in range(3)]
    comps += [
        _node("agree_hub", CompartmentKind.INTEGRATION_CENTER, (0.0, 0.0, 0.0), 22.0, 8.0, tau=7.0),
        _node("axon", CompartmentKind.AXON_SEGMENT, (2.4, 0.0, 0.0), 90.0, 1.4, myelin=0.65, tau=4.0),
    ]
    edges = [
        _edge("ev_0", "agree_hub", 0.7, "feedback_gated", gated_by="ev_1"),
        _edge("ev_1", "agree_hub", 0.7, "feedback_gated", gated_by="ev_2"),
        _edge("ev_2", "agree_hub", 0.7, "feedback_gated", gated_by="ev_0"),
        _edge("agree_hub", "axon", 0.95),
    ]
    return NeuronMorphologySpec(
        morphology_id=nid, family="consensus", name="Consensus Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=comps, edges=edges,
        organelles=[OrganelleAttachment("cons_unc", "uncertaintosome", "agree_hub")],
        input_compartments=["ev_0", "ev_1", "ev_2"], output_compartments=["axon"],
        formal_math=r"y = \phi\left(\sum_{i=1}^3 \sigma(3 e_{(i+1)\bmod 3})\, e_i\right)",
        hypothesis="Cross-gated evidence dendrites fire strongly only under multi-branch agreement, rejecting single-channel noise.",
        description="Three-branch cross-gated agreement neuron suppressing isolated single-branch corruption.",
    )


def _build_veto(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="veto", name="Veto Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend_drive", CompartmentKind.DENDRITE_BASAL, (-2.3, 0.8, 0.0), 35.0, 2.2, tau=7.0),
            _node("veto_guard", CompartmentKind.LOCAL_INHIBITORY, (-2.3, -0.8, 0.0), 28.0, 2.5, tau=4.0),
            _node("soma", CompartmentKind.SOMA, (0.0, 0.0, 0.0), 24.0, 9.0, tau=8.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.4, 0.0, 0.0), 90.0, 1.5, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("dend_drive", "soma", 0.95),
            _edge("veto_guard", "soma", -1.8, "chemical_synapse"),
            _edge("soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("veto_unc", "uncertaintosome", "veto_guard"),
            OrganelleAttachment("veto_res", "resource_regulator", "soma"),
        ],
        input_compartments=["dend_drive", "veto_guard"], output_compartments=["axon"],
        formal_math=r"y = \phi(I_{\text{drive}} - \beta_{\text{veto}} I_{\text{veto}}^+)",
        hypothesis="Fast shunting veto compartment suppresses unstable or contextually inappropriate outputs before axonal launch.",
        description="Inhibitory-guard neuron with strong shunting veto compartment preventing unsafe firing.",
    )


def _build_tessera(nid: str) -> NeuronMorphologySpec:
    return NeuronMorphologySpec(
        morphology_id=nid, family="tessera", name="Tessera Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.5, 0.0, 0.0), 38.0, 2.2, tau=8.0),
            _node("soma", CompartmentKind.SOMA, (-0.5, 0.0, 0.0), 24.0, 9.0, tau=9.0),
            _node("branch_fast", CompartmentKind.AXON_SEGMENT, (1.8, 1.4, 0.0), 80.0, 1.8, myelin=0.85, tau=3.5),
            _node("branch_mem", CompartmentKind.AXON_SEGMENT, (1.8, 0.0, 0.0), 140.0, 1.2, myelin=0.35, tau=12.0),
            _node("branch_mod", CompartmentKind.AXON_SEGMENT, (1.8, -1.4, 0.0), 110.0, 1.1, myelin=0.20, tau=20.0),
        ],
        edges=[
            _edge("dend", "soma", 0.9),
            _edge("soma", "branch_fast", 0.9, "feedback_gated", gated_by="dend"),
            _edge("soma", "branch_mem", 0.85, "feedback_gated", delay=2.0, gated_by="soma"),
            _edge("soma", "branch_mod", 0.75, "paracrine", delay=3.0),
        ],
        organelles=[OrganelleAttachment("tess_route", "routosome", "soma")],
        input_compartments=["dend"], output_compartments=["branch_fast", "branch_mem", "branch_mod"],
        formal_math=r"y_k = g_k(c_t)\, \phi(s(t - \tau_k)),\; k \in \{\text{fast}, \text{mem}, \text{mod}\}",
        hypothesis="Independently gated output branches multiplex fast, delayed, and modulatory projections from one soma.",
        description="Multi-branch demultiplexing neuron routing distinct activity facets to separate target populations.",
    )


# --- 3 AUTONOMOUS ORIGINAL ARCHITECTURES ---

def _build_symbiote_echo(nid: str) -> NeuronMorphologySpec:
    """Original architecture #1: Dual-soma Echo + CheckpointVesicle for cross-connectome alignment."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="symbiote_echo", name="SymbioteEcho Neuron (Original)",
        objective_category=ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("self_dend", CompartmentKind.DENDRITE_BASAL, (-2.4, 0.9, 0.0), 38.0, 2.0, tau=8.0),
            _node("peer_dend", CompartmentKind.ML_ADAPTER, (-2.4, -0.9, 0.0), 38.0, 2.0, tau=12.0),
            _node("self_soma", CompartmentKind.SOMA, (0.0, 0.7, 0.0), 22.0, 8.5, tau=9.0),
            _node("mirror_soma", CompartmentKind.SECONDARY_SOMA, (0.0, -0.7, 0.0), 22.0, 8.5, tau=16.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.5, 0.0, 0.0), 90.0, 1.4, myelin=0.65, tau=4.0),
        ],
        edges=[
            _edge("self_dend", "self_soma", 0.9),
            _edge("peer_dend", "mirror_soma", 0.85, "ml_bridge"),
            _edge("mirror_soma", "self_soma", 0.45, "feedback_gated", gated_by="self_soma"),
            _edge("self_soma", "mirror_soma", -0.35, "chemical_synapse"),
            _edge("self_soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("sym_cont", "contraston", "mirror_soma"),
            OrganelleAttachment("sym_ckpt", "checkpoint_vesicle", "self_soma"),
        ],
        input_compartments=["self_dend", "peer_dend"], output_compartments=["axon"],
        formal_math=r"m_t = \phi(x_{\text{peer}} - \alpha s_t),\; s_t = \phi(x_{\text{self}} + \sigma(s_t) m_t)",
        hypothesis="Dedicated mirror soma tracks peer-connectome discrepancy while CheckpointVesicle protects native identity.",
        description="Original dual-soma architecture designed for intermittent cross-connectome symbiosis without identity erasure.",
    )


def _build_bifurcation_gate(nid: str) -> NeuronMorphologySpec:
    """Original architecture #2: Bistable edge-of-chaos switching neuron."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="bifurcation_gate", name="BifurcationGate Neuron (Original)",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend", CompartmentKind.DENDRITE_BASAL, (-2.3, 0.0, 0.0), 35.0, 2.0, tau=6.0),
            _node("latch_a", CompartmentKind.SOMA, (-0.1, 1.0, 0.0), 22.0, 8.0, tau=14.0),
            _node("latch_b", CompartmentKind.SECONDARY_SOMA, (-0.1, -1.0, 0.0), 22.0, 8.0, tau=14.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.4, 0.0, 0.0), 90.0, 1.4, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("dend", "latch_a", 0.85), _edge("dend", "latch_b", -0.85),
            _edge("latch_a", "latch_a", 0.95), _edge("latch_b", "latch_b", 0.95),
            _edge("latch_a", "latch_b", -0.75, "chemical_synapse"),
            _edge("latch_b", "latch_a", -0.75, "chemical_synapse"),
            _edge("latch_a", "axon", 0.7), _edge("latch_b", "axon", -0.7),
        ],
        input_compartments=["dend"], output_compartments=["axon"],
        formal_math=r"\dot{a} = -a + \tanh(0.95 a - 0.75 b + x),\; \dot{b} = -b + \tanh(0.95 b - 0.75 a - x)",
        hypothesis="Symmetric pitchfork-bistable coupled somas act as a single-neuron noise-tolerant 1-bit working memory latch.",
        description="Original mutually-inhibiting self-exciting dual-soma latch maintaining persistent sign memory across long delays.",
    )


def _build_holographic_dendrite(nid: str) -> NeuronMorphologySpec:
    """Original architecture #3: Phase-coded multi-delay interference dendrite."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="holographic_dendrite", name="HolographicDendrite Neuron (Original)",
        objective_category=ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("d_sin", CompartmentKind.DENDRITE_APICAL, (-2.4, 1.0, 0.0), 60.0, 1.6, tau=9.0),
            _node("d_cos", CompartmentKind.DENDRITE_BASAL, (-2.4, -1.0, 0.0), 60.0, 1.6, tau=9.0),
            _node("interfere", CompartmentKind.INTEGRATION_CENTER, (0.0, 0.0, 0.0), 24.0, 8.5, tau=12.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.4, 0.0, 0.0), 90.0, 1.4, myelin=0.6, tau=4.0),
        ],
        edges=[
            _edge("d_sin", "interfere", 0.85), _edge("d_cos", "interfere", 0.85),
            _edge("interfere", "d_sin", 0.3, "feedback_gated", delay=2.0, gated_by="d_cos"),
            _edge("interfere", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("holo_p1", "phase_organelle", "d_sin", params={"omega": 0.4, "init_phase": 0.0}),
            OrganelleAttachment("holo_p2", "phase_organelle", "d_cos", params={"omega": 0.4, "init_phase": 1.5708}),
            OrganelleAttachment("holo_mnem", "mnemosome", "interfere"),
        ],
        input_compartments=["d_sin", "d_cos"], output_compartments=["axon"],
        formal_math=r"z_t = \phi(d_{\sin}\cos\omega t + d_{\cos}\sin\omega t + m_t)",
        hypothesis="Quadrature phase-modulated dendrites superpose temporal sequences via phase interference inside one cell.",
        description="Original quadrature-oscillator + Mnemosome neuron exploring phase-superposition memory.",
    )


# --- 12 EXTENDED V2 RESEARCH & ECOSYSTEM GRAFT ARCHITECTURES (+12 -> 36 TOTAL) ---

def _build_liquid_cfc(nid: str) -> NeuronMorphologySpec:
    """25. Closed-Form Continuous-Time Liquid Time-Constant Neuron (Hasani et al.)."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="liquid_cfc", name="LiquidCfC Dendritic Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend_drive", CompartmentKind.DENDRITE_BASAL, (-2.5, 0.9, 0.0), 36.0, 2.2, tau=5.0),
            _node("dend_tau", CompartmentKind.DENDRITE_APICAL, (-2.5, -0.9, 0.0), 50.0, 1.8, tau=18.0),
            _node("cfc_fast", CompartmentKind.INTEGRATION_CENTER, (-0.3, 0.8, 0.0), 20.0, 6.5, tau=4.0),
            _node("cfc_slow", CompartmentKind.MEMORY_COMPARTMENT, (-0.3, -0.8, 0.0), 28.0, 8.0, tau=32.0),
            _node("soma", CompartmentKind.SOMA, (1.5, 0.0, 0.0), 24.0, 9.0, tau=7.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (3.2, 0.0, 0.0), 90.0, 1.5, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("dend_drive", "cfc_fast", 0.90),
            _edge("dend_drive", "cfc_slow", 0.65),
            _edge("dend_tau", "cfc_fast", 0.55, "feedback_gated", gated_by="dend_tau"),
            _edge("dend_tau", "cfc_slow", -0.45, "feedback_gated", gated_by="dend_tau"),
            _edge("cfc_fast", "soma", 0.80, "feedback_gated", gated_by="dend_tau"),
            _edge("cfc_slow", "soma", 0.75, "feedback_gated", gated_by="cfc_fast"),
            _edge("soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("lcfc_route", "routosome", "soma"),
            OrganelleAttachment("lcfc_crit", "criticalosome", "cfc_slow"),
        ],
        input_compartments=["dend_drive", "dend_tau"], output_compartments=["axon"],
        formal_math=r"h(t) = \sigma(-f(x,I)t)\, g(x,I) + (1 - \sigma(-f(x,I)t))\, u(x,I)",
        hypothesis="Closed-form input-modulated liquid time-constant branches interpolate fast transient and slow persistent attractors without stiff ODE overhead.",
        description="Liquid continuous-time neuron interpolating fast and slow gated trajectories via input-dependent effective time constants.",
    )


def _build_kan_synapse(nid: str) -> NeuronMorphologySpec:
    """26. Kolmogorov-Arnold Spline Dendrite Neuron (Liu et al. KAN)."""
    comps = [
        _node(f"spline_b{k}", CompartmentKind.DENDRITE_BASAL, (-2.4, 1.5 - 1.0 * k, 0.0), 32.0 + 6.0 * k, 2.0, tau=6.0 + 2.0 * k)
        for k in range(4)
    ] + [
        _node("kan_sum_soma", CompartmentKind.SOMA, (0.4, 0.0, 0.0), 24.0, 9.0, tau=7.0),
        _node("axon", CompartmentKind.AXON_SEGMENT, (2.7, 0.0, 0.0), 85.0, 1.4, myelin=0.65, tau=4.0),
    ]
    edges = [_edge(f"spline_b{k}", "kan_sum_soma", 0.75) for k in range(4)] + [_edge("kan_sum_soma", "axon", 0.95)]
    orgs = [
        OrganelleAttachment(f"kan_sp_{k}", "splinosome", f"spline_b{k}", params={"shift": -0.9 + 0.6 * k})
        for k in range(4)
    ]
    return NeuronMorphologySpec(
        morphology_id=nid, family="kan_synapse", name="KAN-Spline Dendritic Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=comps, edges=edges, organelles=orgs,
        input_compartments=[f"spline_b{k}" for k in range(4)], output_compartments=["axon"],
        formal_math=r"y = \Phi_{\text{soma}}\!\left(\sum_{p=1}^4 \sum_{m=1}^M c_{p,m}\, B_m(x_p)\right)",
        hypothesis="Placing learnable univariate radial-basis spline organelles on dendritic branches approximates smooth multivariate functions with fewer compartments.",
        description="Kolmogorov-Arnold inspired neuron equipped with Splinosome basis nonlinearities on four dendritic branches.",
    )


def _build_burst_ccn(nid: str) -> NeuronMorphologySpec:
    """27. Burst-Multiplexed Apical Credit Neuron (Payeur, Guerguiev, Zenke, Richards & Naud)."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="burst_ccn", name="BurstCCN Multiplexed Neuron",
        objective_category=ObjectiveCategory.BIOLOGICAL_REALISM,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("basal_event", CompartmentKind.DENDRITE_BASAL, (-2.5, -1.0, 0.0), 42.0, 2.4, tau=6.0, receptors=ReceptorDistribution(ampa=1.6)),
            _node("apical_tuft", CompartmentKind.DENDRITE_APICAL, (-2.5, 1.2, 0.0), 95.0, 1.6, tau=18.0, receptors=ReceptorDistribution(nmda=1.6, acetylcholine=1.1)),
            _node("ca_spike_zone", CompartmentKind.INTEGRATION_CENTER, (-0.2, 0.9, 0.0), 18.0, 6.0, tau=11.0),
            _node("soma_event", CompartmentKind.SOMA, (0.2, -0.5, 0.0), 24.0, 9.0, tau=7.0),
            _node("axon_event", CompartmentKind.AXON_SEGMENT, (2.8, -0.6, 0.0), 95.0, 1.5, myelin=0.75, tau=3.5),
            _node("axon_burst", CompartmentKind.AXON_SEGMENT, (2.8, 0.8, 0.0), 95.0, 1.3, myelin=0.55, tau=6.0),
        ],
        edges=[
            _edge("basal_event", "soma_event", 0.92),
            _edge("apical_tuft", "ca_spike_zone", 0.88),
            _edge("soma_event", "ca_spike_zone", 0.55, "feedback_gated", gated_by="apical_tuft"),
            _edge("ca_spike_zone", "soma_event", 0.48, "feedback_gated", gated_by="soma_event"),
            _edge("soma_event", "axon_event", 0.95),
            _edge("ca_spike_zone", "axon_burst", 0.90, "feedback_gated", gated_by="soma_event"),
        ],
        organelles=[
            OrganelleAttachment("bccn_plast", "plasticity_nucleus", "ca_spike_zone"),
            OrganelleAttachment("bccn_pred", "predicton", "apical_tuft"),
        ],
        input_compartments=["basal_event", "apical_tuft"], output_compartments=["axon_event", "axon_burst"],
        formal_math=r"e_t = \sigma(I_{\text{basal}}),\; p_{\text{burst},t} = \sigma(I_{\text{apical}}),\; b_t = e_t \cdot p_{\text{burst},t},\; \Delta w \propto (p_{\text{burst},t} - \bar{p}) e_{\text{pre}}",
        hypothesis="Segregating basal event rates from apical burst-probability modulation enables single-phase credit assignment without freezing forward inference.",
        description="Cortical L5-inspired burst-multiplexing neuron routing separate event and burst-credit axonal channels.",
    )


def _build_btsp_plateau(nid: str) -> NeuronMorphologySpec:
    """28. Behavioral Time-Scale Synaptic Plasticity CA1 Neuron (Bittner & Magee)."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="btsp_plateau", name="BTSP Plateau-Potential Neuron",
        objective_category=ObjectiveCategory.BIOLOGICAL_REALISM,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("ca3_proximal", CompartmentKind.DENDRITE_PROXIMAL, (-2.5, -0.8, 0.0), 48.0, 2.2, tau=14.0),
            _node("ec3_distal_plateau", CompartmentKind.DENDRITE_DISTAL, (-2.5, 1.1, 0.0), 110.0, 1.4, tau=36.0, receptors=ReceptorDistribution(nmda=1.8, dopamine=1.2)),
            _node("elig_bank", CompartmentKind.MEMORY_COMPARTMENT, (-0.3, -0.4, 0.0), 28.0, 8.5, tau=45.0),
            _node("soma", CompartmentKind.SOMA, (1.2, 0.0, 0.0), 24.0, 9.0, tau=9.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (3.0, 0.0, 0.0), 95.0, 1.4, myelin=0.65, tau=4.0),
        ],
        edges=[
            _edge("ca3_proximal", "elig_bank", 0.85),
            _edge("ca3_proximal", "soma", 0.75),
            _edge("ec3_distal_plateau", "elig_bank", 0.90, "feedback_gated", gated_by="ec3_distal_plateau"),
            _edge("elig_bank", "soma", 0.85, "feedback_gated", gated_by="ec3_distal_plateau"),
            _edge("soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("btsp_chrono", "chronosome", "elig_bank", params={"max_delay": 16, "tap_delay": 6}),
            OrganelleAttachment("btsp_mnem", "mnemosome", "elig_bank", params={"slots": 10, "slow_decay": 0.995}),
            OrganelleAttachment("btsp_arb", "arbitrosome", "elig_bank"),
        ],
        input_compartments=["ca3_proximal", "ec3_distal_plateau"], output_compartments=["axon"],
        formal_math=r"\dot{E}_i = -E_i/\tau_E + x_i(t),\; \Delta w_i = \eta_+ (w_{\max}-w_i) f_+(E_i P_t) - \eta_- w_i f_-(E_i P_t)",
        hypothesis="Distal instructional plateau potentials gate multi-step eligibility traces for rapid few-shot associative binding.",
        description="Hippocampal CA1-inspired neuron pairing long-horizon eligibility traces with distal plateau gating.",
    )


def _build_mamba_selective(nid: str) -> NeuronMorphologySpec:
    """29. Selective State-Space Scan Dendritic Neuron (Gu & Dao S6)."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="mamba_selective", name="Mamba-S6 Selective Scan Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend_x", CompartmentKind.DENDRITE_BASAL, (-2.5, 0.8, 0.0), 35.0, 2.1, tau=6.0),
            _node("delta_gate", CompartmentKind.INTEGRATION_CENTER, (-0.8, 1.2, 0.0), 18.0, 6.0, tau=5.0),
            _node("ssm_state", CompartmentKind.MEMORY_COMPARTMENT, (-0.4, -0.6, 0.0), 32.0, 9.5, tau=42.0),
            _node("soma_proj", CompartmentKind.SOMA, (1.4, 0.0, 0.0), 22.0, 8.5, tau=7.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (3.1, 0.0, 0.0), 90.0, 1.5, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("dend_x", "delta_gate", 0.90),
            _edge("dend_x", "ssm_state", 0.85, "feedback_gated", gated_by="delta_gate"),
            _edge("ssm_state", "ssm_state", 0.82, "electrical_axial"),
            _edge("ssm_state", "soma_proj", 0.90, "feedback_gated", gated_by="dend_x"),
            _edge("dend_x", "soma_proj", 0.35),
            _edge("soma_proj", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("mamba_att", "attentosome", "delta_gate"),
            OrganelleAttachment("mamba_route", "routosome", "soma_proj"),
        ],
        input_compartments=["dend_x"], output_compartments=["axon"],
        formal_math=r"\Delta_t = \text{softplus}(W_\Delta x_t),\; h_t = \exp(\Delta_t A) h_{t-1} + \Delta_t B(x_t) x_t,\; y_t = C(x_t) h_t",
        hypothesis="Content-selective step-size gating lets a single state compartment filter irrelevant noise while latching salient tokens indefinitely.",
        description="Selective state-space cellular architecture implementing input-dependent discretization and retention.",
    )


def _build_hodge_simplicial(nid: str) -> NeuronMorphologySpec:
    """30. Topological 2-Simplex Hodge-Laplacian Curl/Harmonic Neuron."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="hodge_simplicial", name="Hodge-Simplicial Cycle Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("v0", CompartmentKind.DENDRITE_BASAL, (-2.2, 1.4, 0.0), 35.0, 2.0, tau=8.0),
            _node("v1", CompartmentKind.DENDRITE_BASAL, (-2.2, 0.0, 0.0), 35.0, 2.0, tau=8.0),
            _node("v2", CompartmentKind.DENDRITE_BASAL, (-2.2, -1.4, 0.0), 35.0, 2.0, tau=8.0),
            _node("curl_face", CompartmentKind.INTEGRATION_CENTER, (-0.2, 0.8, 0.0), 22.0, 7.5, tau=12.0),
            _node("div_node", CompartmentKind.INTEGRATION_CENTER, (-0.2, -0.8, 0.0), 22.0, 7.5, tau=7.0),
            _node("soma", CompartmentKind.SOMA, (1.5, 0.0, 0.0), 24.0, 9.0, tau=8.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (3.1, 0.0, 0.0), 90.0, 1.4, myelin=0.65, tau=4.0),
        ],
        edges=[
            _edge("v0", "v1", 0.55, "gap_junction"),
            _edge("v1", "v2", 0.55, "gap_junction"),
            _edge("v2", "v0", 0.55, "gap_junction"),
            _edge("v0", "curl_face", 0.70), _edge("v1", "curl_face", -0.70, "chemical_synapse"), _edge("v2", "curl_face", 0.70),
            _edge("v0", "div_node", 0.55), _edge("v1", "div_node", 0.55), _edge("v2", "div_node", 0.55),
            _edge("curl_face", "soma", 0.80), _edge("div_node", "soma", 0.65),
            _edge("soma", "axon", 0.95),
        ],
        organelles=[OrganelleAttachment("hodge_ph", "phase_organelle", "curl_face", params={"omega": 0.42})],
        input_compartments=["v0", "v1", "v2"], output_compartments=["axon"],
        formal_math=r"\Delta_1 = B_1^\top B_1 + B_2 B_2^\top,\; y = \phi(w_{\text{div}} B_1 f + w_{\text{curl}} B_2^\top f)",
        hypothesis="Explicit separation of irrotational divergence from cyclic curl across a 2-simplex dendritic triangle detects higher-order loop motifs.",
        description="Topological deep-learning neuron decomposing 3-way inputs into boundary divergence and 2-simplex cyclic curl.",
    )


def _build_efference_twin(nid: str) -> NeuronMorphologySpec:
    """31. Efference-Copy Authorship Comparator Neuron (inspired by GhostInTheMachine v8)."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="efference_twin", name="EfferenceTwin Comparator Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("reafferent_in", CompartmentKind.DENDRITE_BASAL, (-2.4, 1.0, 0.0), 40.0, 2.1, tau=7.0),
            _node("motor_ soma", CompartmentKind.SOMA, (-0.4, -0.9, 0.0), 24.0, 9.0, tau=8.0),
            _node("efference_pred", CompartmentKind.SECONDARY_SOMA, (-0.2, 0.9, 0.0), 22.0, 8.0, tau=10.0),
            _node("comparator", CompartmentKind.INTEGRATION_CENTER, (1.4, 0.4, 0.0), 20.0, 7.0, tau=6.0),
            _node("axon_act", CompartmentKind.AXON_SEGMENT, (3.0, -0.8, 0.0), 90.0, 1.5, myelin=0.75, tau=4.0),
            _node("axon_exog", CompartmentKind.AXON_SEGMENT, (3.0, 0.8, 0.0), 90.0, 1.3, myelin=0.65, tau=4.0),
        ],
        edges=[
            _edge("reafferent_in", "motor_ soma", 0.80),
            _edge("motor_ soma", "efference_pred", 0.85, delay=1.0),
            _edge("reafferent_in", "comparator", 0.95),
            _edge("efference_pred", "comparator", -0.90, "chemical_synapse"),
            _edge("comparator", "motor_ soma", -0.30, "feedback_gated", gated_by="efference_pred"),
            _edge("motor_ soma", "axon_act", 0.95),
            _edge("comparator", "axon_exog", 0.95),
        ],
        organelles=[
            OrganelleAttachment("eff_org", "efferentosome", "comparator"),
            OrganelleAttachment("eff_pred", "predicton", "efference_pred"),
        ],
        input_compartments=["reafferent_in"], output_compartments=["axon_act", "axon_exog"],
        formal_math=r"\hat{o}_t = F(s_{t-1}, e_{t-1}),\; m_t = o_t - \hat{o}_t,\; W_{\text{closed}} = gW + \gamma_{\text{cmp}} W_{\text{cmp}} B",
        hypothesis="Subtracting forward efference predictions from sensory reafference isolates exogenous disturbances and stabilizes closed-loop eigenvalues.",
        description="GhostInTheMachine-inspired neuron canceling self-caused reafference while stabilizing closed-loop dynamics.",
    )


def _build_crosswire_transducer(nid: str) -> NeuronMorphologySpec:
    """32. Version-Space Mealy Safety-Contract Neuron (inspired by CROSSWIRE)."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="crosswire_transducer", name="Crosswire Mealy Safety Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("iface_in", CompartmentKind.ML_ADAPTER, (-2.5, 0.0, 0.0), 25.0, 6.0, tau=6.0),
            _node("map_pos", CompartmentKind.INTEGRATION_CENTER, (-0.5, 1.1, 0.0), 20.0, 6.5, tau=8.0),
            _node("map_inv", CompartmentKind.INTEGRATION_CENTER, (-0.5, -1.1, 0.0), 20.0, 6.5, tau=8.0),
            _node("contract_hillock", CompartmentKind.AXON_HILLOCK, (1.3, 0.0, 0.0), 18.0, 5.5, tau=5.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (3.0, 0.0, 0.0), 90.0, 1.4, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("iface_in", "map_pos", 0.90, "ml_bridge"),
            _edge("iface_in", "map_inv", -0.90, "ml_bridge"),
            _edge("map_pos", "contract_hillock", 0.80, "feedback_gated", gated_by="iface_in"),
            _edge("map_inv", "contract_hillock", 0.50, "feedback_gated", gated_by="map_pos"),
            _edge("contract_hillock", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("cw_org", "crosswirosome", "contract_hillock"),
            OrganelleAttachment("cw_unc", "uncertaintosome", "contract_hillock"),
        ],
        input_compartments=["iface_in"], output_compartments=["axon"],
        formal_math=r"V(D) = \{h \in \mathcal{H} : h \models D\},\; y = \mathbb{I}[\forall h \in V(D),\, \mathcal{C}_{\text{safe}}(h(x))]\, \bar{h}(x)",
        hypothesis="Maintaining surviving interface interpretations and gating axonal output on unanimous safety-contract satisfaction prevents catastrophic mis-wiring.",
        description="CROSSWIRE-inspired transducer neuron preserving multiple candidate adapter hypotheses with abstention safety.",
    )


def _build_tether_reliability(nid: str) -> NeuronMorphologySpec:
    """33. Dark-Knowledge Specialist Reliability Neuron (inspired by Supermix-Tether v0.2)."""
    comps = [_node(f"donor_{k}", CompartmentKind.DENDRITE_BASAL, (-2.5, 1.4 - 1.4 * k, 0.0), 38.0, 2.0, tau=7.0) for k in range(3)]
    comps += [
        _node("tail_entropy_hub", CompartmentKind.INTEGRATION_CENTER, (-0.2, 0.0, 0.0), 22.0, 8.0, tau=8.0),
        _node("soma", CompartmentKind.SOMA, (1.5, 0.0, 0.0), 24.0, 9.0, tau=8.0),
        _node("axon", CompartmentKind.AXON_SEGMENT, (3.1, 0.0, 0.0), 90.0, 1.5, myelin=0.7, tau=4.0),
    ]
    edges = [
        _edge("donor_0", "tail_entropy_hub", 0.75),
        _edge("donor_1", "tail_entropy_hub", 0.75),
        _edge("donor_2", "tail_entropy_hub", 0.75),
        _edge("donor_0", "soma", 0.65, "feedback_gated", gated_by="tail_entropy_hub"),
        _edge("donor_1", "soma", 0.65, "feedback_gated", gated_by="tail_entropy_hub"),
        _edge("donor_2", "soma", 0.65, "feedback_gated", gated_by="tail_entropy_hub"),
        _edge("tail_entropy_hub", "soma", 0.45),
        _edge("soma", "axon", 0.95),
    ]
    orgs = [
        OrganelleAttachment("teth_unc", "uncertaintosome", "tail_entropy_hub"),
        OrganelleAttachment("teth_arb", "arbitrosome", "soma"),
    ]
    return NeuronMorphologySpec(
        morphology_id=nid, family="tether_reliability", name="Tether Dark-Knowledge Reliability Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=comps, edges=edges, organelles=orgs,
        input_compartments=["donor_0", "donor_1", "donor_2"], output_compartments=["axon"],
        formal_math=r"r_k = 1 - H(p_{-1}^{(k)}) / \log(K-1),\; y = \phi\!\left(\sum_k \frac{r_k}{\sum_j r_j} d_k\right)",
        hypothesis="Intrinsic specialist dispersion/reliability weighting survives majority donor corruption where naive consensus collapses.",
        description="Supermix-Tether-inspired receiver neuron weighting specialist dendrites by intrinsic reliability rather than majority vote.",
    )


def _build_hyperbolic_poincare(nid: str) -> NeuronMorphologySpec:
    """34. Poincare Disk Geodesic Hierarchy Neuron."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="hyperbolic_poincare", name="Hyperbolic Poincare Hierarchy Neuron",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("radial_depth", CompartmentKind.DENDRITE_APICAL, (-2.4, 1.0, 0.0), 65.0, 1.7, tau=16.0),
            _node("angular_sector", CompartmentKind.DENDRITE_BASAL, (-2.4, -1.0, 0.0), 40.0, 2.2, tau=7.0),
            _node("mobius_soma", CompartmentKind.SOMA, (0.2, 0.0, 0.0), 25.0, 9.5, tau=9.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.7, 0.0, 0.0), 90.0, 1.4, myelin=0.65, tau=4.0),
        ],
        edges=[
            _edge("radial_depth", "mobius_soma", 0.85),
            _edge("angular_sector", "mobius_soma", 0.85, "feedback_gated", gated_by="radial_depth"),
            _edge("mobius_soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("hyp_sp", "splinosome", "radial_depth"),
            OrganelleAttachment("hyp_ph", "phase_organelle", "angular_sector", params={"omega": 0.3}),
        ],
        input_compartments=["radial_depth", "angular_sector"], output_compartments=["axon"],
        formal_math=r"d_{\mathbb{B}}(u,v) = \operatorname{arcosh}\!\left(1 + \frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)",
        hypothesis="Separating hyperbolic radial depth (abstraction hierarchy) from angular sector embeds exponentially branching trees compactly.",
        description="Hyperbolic geometry-inspired neuron coupling radial depth scaling with angular phase representation.",
    )


def _build_astrocytic_tripartite(nid: str) -> NeuronMorphologySpec:
    """35. Astrocytic Tripartite Calcium-Wave & Glymphatic Homeostatic Neuron."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="astrocytic_tripartite", name="Astrocytic Tripartite Neuron",
        objective_category=ObjectiveCategory.BIOLOGICAL_REALISM,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("presyn_bouton", CompartmentKind.DENDRITE_PROXIMAL, (-2.4, 0.6, 0.0), 28.0, 2.0, tau=6.0),
            _node("astrocyte_ca", CompartmentKind.METABOLIC_REGULATOR, (-0.3, 1.4, 0.0), 35.0, 9.0, tau=55.0, receptors=ReceptorDistribution(paracrine=2.2, noradrenaline=1.2)),
            _node("postsyn_soma", CompartmentKind.SOMA, (0.4, -0.4, 0.0), 24.0, 8.5, tau=10.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (2.8, -0.2, 0.0), 90.0, 1.4, myelin=0.65, tau=4.0),
        ],
        edges=[
            _edge("presyn_bouton", "postsyn_soma", 0.88),
            _edge("presyn_bouton", "astrocyte_ca", 0.55, "paracrine"),
            _edge("postsyn_soma", "astrocyte_ca", 0.45, "paracrine"),
            _edge("astrocyte_ca", "postsyn_soma", -0.38, "feedback_gated", gated_by="astrocyte_ca"),
            _edge("postsyn_soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("astro_glym", "glymphatosome", "astrocyte_ca"),
            OrganelleAttachment("astro_res", "resource_regulator", "postsyn_soma"),
        ],
        input_compartments=["presyn_bouton"], output_compartments=["axon"],
        formal_math=r"\tau_{\text{Ca}}\dot{c} = -c + \alpha(|x|+|s|),\; y = \phi((1 - \gamma\sigma(c-\theta)) Wx)",
        hypothesis="Slow tripartite astrocytic calcium integration and glymphatic downscaling stabilize long-horizon recurrent excitability.",
        description="Tripartite neuron-glia unit pairing fast synaptic transmission with slow astrocytic Ca2+ clamping and waste clearance.",
    )


def _build_soliton_arbiter(nid: str) -> NeuronMorphologySpec:
    """36. SolitonArbiter Neuron (Original v2): Resolves Mnemosome+Chronosome interference via Arbitrosome."""
    return NeuronMorphologySpec(
        morphology_id=nid, family="soliton_arbiter", name="SolitonArbiter Multi-Organelle Neuron (Original v2)",
        objective_category=ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY,
        fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
        compartments=[
            _node("dend_in", CompartmentKind.DENDRITE_BASAL, (-2.6, 0.0, 0.0), 38.0, 2.2, tau=6.0),
            _node("mnem_lane", CompartmentKind.MEMORY_COMPARTMENT, (-0.4, 1.2, 0.0), 28.0, 8.5, tau=26.0),
            _node("chrono_lane", CompartmentKind.MEMORY_COMPARTMENT, (-0.4, -1.2, 0.0), 28.0, 8.5, tau=14.0),
            _node("arbiter_soma", CompartmentKind.SOMA, (1.5, 0.0, 0.0), 25.0, 9.5, tau=8.0),
            _node("axon", CompartmentKind.AXON_SEGMENT, (3.2, 0.0, 0.0), 95.0, 1.5, myelin=0.7, tau=4.0),
        ],
        edges=[
            _edge("dend_in", "mnem_lane", 0.85),
            _edge("dend_in", "chrono_lane", 0.85),
            _edge("dend_in", "arbiter_soma", 0.55),
            _edge("mnem_lane", "arbiter_soma", 0.70, "feedback_gated", gated_by="arbiter_soma"),
            _edge("chrono_lane", "arbiter_soma", 0.70, "feedback_gated", gated_by="mnem_lane"),
            _edge("arbiter_soma", "axon", 0.95),
        ],
        organelles=[
            OrganelleAttachment("sol_mnem", "mnemosome", "mnem_lane", params={"slots": 10}),
            OrganelleAttachment("sol_chrono", "chronosome", "chrono_lane", params={"tap_delay": 4}),
            OrganelleAttachment("sol_arb", "arbitrosome", "arbiter_soma"),
        ],
        input_compartments=["dend_in"], output_compartments=["axon"],
        formal_math=r"s_t = \phi\!\left(w_0 x_t + \alpha_1(\pi) m_{\text{mnem}}(t) + \alpha_2(\pi) c_{\text{chrono}}^\perp(t)\right)",
        hypothesis="Isolating Mnemosome and Chronosome into parallel memory lanes arbitrated by an Arbitrosome converts destructive saturation into constructive temporal synergy.",
        description="Original v2 architecture designed specifically to resolve multi-organelle memory interference through compartment isolation and precision arbitration.",
        metadata={"organelle_arbitration": True},
    )


GRAFT_NEURON_BUILDERS: Dict[str, Callable[[str], NeuronMorphologySpec]] = {
    "triarch": _build_triarch,
    "ouroboros": _build_ouroboros,
    "hydra": _build_hydra,
    "mnemosome_neuron": _build_mnemosome_neuron,
    "chameleon": _build_chameleon,
    "janus": _build_janus,
    "chronos": _build_chronos,
    "nexus": _build_nexus,
    "paracrine_broadcast": _build_paracrine_broadcast,
    "mosaic": _build_mosaic,
    "sentinel_neuron": _build_sentinel_neuron,
    "frontier_neuron": _build_frontier_neuron,
    "bifrost": _build_bifrost,
    "resonator": _build_resonator,
    "palimpsest": _build_palimpsest,
    "counterpoint": _build_counterpoint,
    "compass": _build_compass,
    "prism": _build_prism,
    "consensus": _build_consensus,
    "veto": _build_veto,
    "tessera": _build_tessera,
    # 3 original v1 autonomous designs
    "symbiote_echo": _build_symbiote_echo,
    "bifurcation_gate": _build_bifurcation_gate,
    "holographic_dendrite": _build_holographic_dendrite,
}

EXTENDED_GRAFT_NEURON_BUILDERS: Dict[str, Callable[[str], NeuronMorphologySpec]] = {
    "liquid_cfc": _build_liquid_cfc,
    "kan_synapse": _build_kan_synapse,
    "burst_ccn": _build_burst_ccn,
    "btsp_plateau": _build_btsp_plateau,
    "mamba_selective": _build_mamba_selective,
    "hodge_simplicial": _build_hodge_simplicial,
    "efference_twin": _build_efference_twin,
    "crosswire_transducer": _build_crosswire_transducer,
    "tether_reliability": _build_tether_reliability,
    "hyperbolic_poincare": _build_hyperbolic_poincare,
    "astrocytic_tripartite": _build_astrocytic_tripartite,
    "soliton_arbiter": _build_soliton_arbiter,
}

ALL_GRAFT_NEURON_BUILDERS: Dict[str, Callable[[str], NeuronMorphologySpec]] = {
    **GRAFT_NEURON_BUILDERS,
    **EXTENDED_GRAFT_NEURON_BUILDERS,
}

GRAFT_NEURON_FAMILIES: Tuple[str, ...] = tuple(GRAFT_NEURON_BUILDERS.keys())
EXTENDED_GRAFT_NEURON_FAMILIES: Tuple[str, ...] = tuple(EXTENDED_GRAFT_NEURON_BUILDERS.keys())
ALL_GRAFT_NEURON_FAMILIES: Tuple[str, ...] = tuple(ALL_GRAFT_NEURON_BUILDERS.keys())


def build_graft_neuron(
    family: str,
    neuron_id: str | None = None,
    fidelity: FidelityLevel | None = None,
    seed: int | None = None,
) -> NeuronMorphologySpec:
    if family not in ALL_GRAFT_NEURON_BUILDERS:
        raise KeyError(f"Unknown graft neuron family {family!r}. Valid: {sorted(ALL_GRAFT_NEURON_BUILDERS)}")
    spec = ALL_GRAFT_NEURON_BUILDERS[family](neuron_id or f"{family}_0")
    if fidelity is not None:
        spec.fidelity = fidelity
    if seed is not None:
        spec.metadata["seed"] = int(seed)
    return spec


def build_ablation_variants(spec: NeuronMorphologySpec) -> Dict[str, NeuronMorphologySpec]:
    """Generate matched ablation specs (no_organelles, no_feedback_edges, single_soma_collapse)."""
    base_d = spec.to_dict()
    no_org = NeuronMorphologySpec.from_dict(base_d)
    no_org.morphology_id = f"{spec.morphology_id}_ablate_no_organelles"
    no_org.organelles = []

    no_fb = NeuronMorphologySpec.from_dict(base_d)
    no_fb.morphology_id = f"{spec.morphology_id}_ablate_no_feedback"
    no_fb.edges = [e for e in no_fb.edges if e.coupling_type != "feedback_gated"]

    single_soma = NeuronMorphologySpec(
        morphology_id=f"{spec.morphology_id}_ablate_point_control",
        family="point_control",
        name=f"{spec.name} (Point Control)",
        objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
        fidelity=FidelityLevel.ABSTRACT_RATE,
        compartments=[CompartmentNode("soma", CompartmentKind.SOMA, tau_ms=10.0)],
        edges=[CompartmentEdge("soma", "soma", weight=0.45)],
        input_compartments=["soma"],
        output_compartments=["soma"],
    )
    return {
        "no_organelles": no_org,
        "no_feedback": no_fb,
        "point_control": single_soma,
    }


_KIND_COLORS: Dict[str, str] = {
    "soma": "#38bdf8",
    "secondary_soma": "#818cf8",
    "dendrite_basal": "#34d399",
    "dendrite_apical": "#10b981",
    "dendrite_proximal": "#2dd4bf",
    "dendrite_distal": "#059669",
    "integration_center": "#f59e0b",
    "axon_hillock": "#f97316",
    "axon_segment": "#fb7185",
    "axon_feedback": "#e879f9",
    "local_inhibitory": "#ef4444",
    "memory_compartment": "#a78bfa",
    "metabolic_regulator": "#facc15",
    "ml_adapter": "#22d3ee",
    "organelle_dock": "#c084fc",
}


def render_neuron_svg(spec: NeuronMorphologySpec, width: int = 440, height: int = 240) -> str:
    """Render a clean, deterministic, self-contained SVG visualization of a neuron morphology."""
    xs = [c.coords_3d[0] for c in spec.compartments] or [0.0]
    ys = [c.coords_3d[1] for c in spec.compartments] or [0.0]
    min_x, max_x = min(xs) - 1.0, max(xs) + 1.0
    min_y, max_y = min(ys) - 1.0, max(ys) + 1.0
    pad_x, pad_y = 55.0, 44.0

    def _px(x: float) -> float:
        return pad_x + (x - min_x) / max(max_x - min_x, 1e-3) * (width - 2 * pad_x)

    def _py(y: float) -> float:
        return height - (pad_y + (y - min_y) / max(max_y - min_y, 1e-3) * (height - 2 * pad_y))

    pos = {c.compartment_id: (_px(c.coords_3d[0]), _py(c.coords_3d[1])) for c in spec.compartments}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="10" fill="#0f172a"/>',
        f'<text x="14" y="20" fill="#e2e8f0" font-family="monospace" font-size="13" font-weight="bold">{spec.name} [{spec.family}] · F{int(spec.fidelity)}</text>',
        f'<text x="14" y="{height - 10}" fill="#94a3b8" font-family="monospace" font-size="10">Compartments: {len(spec.compartments)} | Edges: {len(spec.edges)} | Organelles: {len(spec.organelles)} | Cost: {spec.total_metabolic_cost:.2f}</text>',
    ]
    for e in spec.edges:
        x1, y1 = pos[e.source_id]
        x2, y2 = pos[e.target_id]
        stroke = "#e879f9" if e.coupling_type == "feedback_gated" else ("#ef4444" if e.weight < 0 else "#64748b")
        dash = ' stroke-dasharray="4 3"' if e.coupling_type in ("feedback_gated", "paracrine", "ml_bridge") else ""
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="2"{dash}/>')

    org_by_comp: Dict[str, List[str]] = {}
    for o in spec.organelles:
        org_by_comp.setdefault(o.compartment_id, []).append(o.organelle_type)

    for c in spec.compartments:
        cx, cy = pos[c.compartment_id]
        r = max(7.0, min(16.0, 5.5 + c.radius_um * 0.7))
        fill = _KIND_COLORS.get(c.kind.value, "#94a3b8")
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="#f8fafc" stroke-width="1.3"/>')
        lines.append(f'<text x="{cx:.1f}" y="{cy - r - 4:.1f}" text-anchor="middle" fill="#cbd5e1" font-family="monospace" font-size="9">{c.compartment_id}</text>')
        if c.compartment_id in org_by_comp:
            tag = ",".join(org_by_comp[c.compartment_id])[:18]
            lines.append(f'<text x="{cx:.1f}" y="{cy + r + 10:.1f}" text-anchor="middle" fill="#fde047" font-family="monospace" font-size="8">[{tag}]</text>')

    lines.append("</svg>")
    return "\n".join(lines)
