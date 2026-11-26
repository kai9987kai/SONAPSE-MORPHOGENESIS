"""SONAPSE FOUNDRY — 15 Improved Conventional Neuron Families & Population Heterogeneity Assay.

Implements standard & improved conventional neurons independently of exotic grafts:
1.  lif_reference            - Leaky Integrate-and-Fire reference neuron (Fidelity 3)
2.  recurrent_rate           - Conventional recurrent rate neuron (Fidelity 1)
3.  izhikevich               - Adaptive bursting/spiking Izhikevich-style neuron (Fidelity 3)
4.  conductance_hh           - Hodgkin-Huxley conductance-based model (Fidelity 4)
5.  adaptive_pyramidal       - Dual-dendrite apical/basal adaptive pyramidal neuron (Fidelity 2)
6.  predictive_interneuron   - Inhibitory predictive-error interneuron (Fidelity 2, Dale -1)
7.  dynamic_threshold        - Activity-dependent sliding threshold neuron (Fidelity 3)
8.  homeostatic_regulated    - Turrigiano-inspired gain-regulated neuron (Fidelity 2)
9.  metaplastic_bcm          - BCM sliding-threshold metaplastic neuron (Fidelity 2)
10. sparse_burst             - High-threshold sparse burst coding neuron (Fidelity 3)
11. context_motor            - Context-gated descending motor neuron (Fidelity 2)
12. adaptive_conduction      - Myelination-tunable adaptive conduction neuron (Fidelity 2)
13. developmental_precursor  - Maturing developmental precursor cell (Fidelity 2)
14. recovery_compensatory    - Repairosome-equipped lesion-compensating neuron (Fidelity 2)
15. reliability_sensitive    - Uncertainty-weighted reliability-sensitive neuron (Fidelity 2)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple
import numpy as np

from ..core.compartment_engine import MultiFidelityConnectome, StructuredNeuronInstance
from ..core.interchange import (
    CompartmentEdge,
    CompartmentKind,
    CompartmentNode,
    ConnectomeEdge,
    ConnectomeGraph,
    FidelityLevel,
    NeuronMorphologySpec,
    ObjectiveCategory,
    OrganelleAttachment,
    ReceptorDistribution,
)
from .organelles import create_organelle


def _c(cid: str, kind: CompartmentKind, length: float = 20.0, radius: float = 6.0, myelin: float = 0.0,
       tau: float = 10.0, coords=(0.0, 0.0, 0.0), thresh: float = -50.0, receptors: ReceptorDistribution | None = None) -> CompartmentNode:
    return CompartmentNode(
        compartment_id=cid, kind=kind, length_um=length, radius_um=radius,
        myelin_factor=myelin, tau_ms=tau, coords_3d=coords, threshold_mV=thresh,
        receptors=receptors or ReceptorDistribution(),
    )


def build_conventional_neuron(
    family: str,
    neuron_id: str = "conv_0",
    fidelity: FidelityLevel | None = None,
    seed: int | None = None,
) -> NeuronMorphologySpec:
    """Factory constructing any of the 21 conventional neuron families."""
    builders: Dict[str, Callable[[str], NeuronMorphologySpec]] = {
        "lif_reference": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="lif_reference", name="LIF Reference Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.SPIKING,
            compartments=[_c("soma", CompartmentKind.SOMA, 20.0, 8.0, tau=12.0, thresh=-52.0)],
            edges=[], input_compartments=["soma"], output_compartments=["soma"],
            formal_math=r"\tau_m \dot{V} = (E_{\text{rest}} - V) + R I_{\text{syn}}; V \ge V_{\text{th}} \Rightarrow \text{spike}, V \leftarrow E_{\text{reset}}",
            hypothesis="Baseline spiking neuron benchmark for matching spike-rate efficiency.",
        ),
        "recurrent_rate": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="recurrent_rate", name="Conventional Recurrent Rate Neuron",
            objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS, fidelity=FidelityLevel.ABSTRACT_RATE,
            compartments=[_c("soma", CompartmentKind.SOMA, 15.0, 6.0, tau=8.0)],
            edges=[CompartmentEdge("soma", "soma", weight=0.45, coupling_type="chemical_synapse")],
            input_compartments=["soma"], output_compartments=["soma"],
            formal_math=r"\tau \dot{r} = -r + \tanh(w_{\text{rec}} r + I_{\text{ext}})",
            hypothesis="Lowest-cost continuous rate control baseline.",
        ),
        "izhikevich": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="izhikevich", name="Izhikevich Adaptive Spiking Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.SPIKING,
            compartments=[
                _c("dend", CompartmentKind.DENDRITE_BASAL, 30.0, 2.0, tau=8.0, coords=(0.0, 1.0, 0.0)),
                _c("soma", CompartmentKind.SOMA, 22.0, 8.5, tau=6.0, thresh=-48.0, coords=(0.0, 0.0, 0.0)),
            ],
            edges=[CompartmentEdge("dend", "soma", weight=0.85, coupling_type="electrical_axial")],
            input_compartments=["dend"], output_compartments=["soma"],
            formal_math=r"\dot{v} = 0.04v^2 + 5v + 140 - u + I,\; \dot{u} = a(bv - u)",
            hypothesis="Supports tonic spiking, adaptation, and bursting at low CPU cost.",
        ),
        "conductance_hh": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="conductance_hh", name="Hodgkin-Huxley Conductance Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.CONDUCTANCE,
            compartments=[
                _c("dend", CompartmentKind.DENDRITE_PROXIMAL, 40.0, 2.5, tau=10.0, coords=(-1.0, 0.0, 0.0)),
                _c("soma", CompartmentKind.SOMA, 25.0, 10.0, tau=10.0, coords=(0.0, 0.0, 0.0)),
                _c("axon", CompartmentKind.AXON_SEGMENT, 100.0, 1.5, myelin=0.6, tau=5.0, coords=(2.0, 0.0, 0.0)),
            ],
            edges=[
                CompartmentEdge("dend", "soma", weight=0.9, coupling_type="electrical_axial"),
                CompartmentEdge("soma", "axon", weight=1.0, coupling_type="electrical_axial"),
            ],
            input_compartments=["dend"], output_compartments=["axon"],
            formal_math=r"C_m \dot{V} = I_{\text{stim}} + g_{\text{ax}}\Delta V - \bar{g}_{\text{Na}} m^3 h(V-E_{\text{Na}}) - \bar{g}_K n^4(V-E_K) - g_L(V-E_L)",
            hypothesis="Biophysical conductance reference quantifying ion-channel nonlinearity vs compute cost.",
        ),
        "adaptive_pyramidal": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="adaptive_pyramidal", name="Adaptive Pyramidal Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[
                _c("apical", CompartmentKind.DENDRITE_APICAL, 90.0, 1.8, tau=18.0, coords=(0.0, 2.5, 0.0), receptors=ReceptorDistribution(nmda=1.2)),
                _c("basal", CompartmentKind.DENDRITE_BASAL, 45.0, 2.2, tau=8.0, coords=(0.0, -1.2, 0.0)),
                _c("soma", CompartmentKind.SOMA, 24.0, 9.0, tau=10.0, coords=(0.0, 0.0, 0.0)),
                _c("axon", CompartmentKind.AXON_SEGMENT, 120.0, 1.2, myelin=0.7, tau=4.0, coords=(2.0, 0.0, 0.0)),
            ],
            edges=[
                CompartmentEdge("apical", "soma", weight=0.75, coupling_type="electrical_axial"),
                CompartmentEdge("basal", "soma", weight=0.85, coupling_type="electrical_axial"),
                CompartmentEdge("soma", "axon", weight=0.95, coupling_type="electrical_axial"),
            ],
            input_compartments=["basal", "apical"], output_compartments=["axon"],
            formal_math=r"s_{\text{soma}} = \phi(g_a s_{\text{apical}} + g_b s_{\text{basal}} + g_{\text{NMDA}} s_{\text{apical}}^+ s_{\text{basal}}^+)",
            hypothesis="Apical-basal NMDA coincidence detects top-down context paired with bottom-up input.",
        ),
        "predictive_interneuron": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="predictive_interneuron", name="Predictive Inhibitory Interneuron",
            objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            dale_sign=-1,
            compartments=[
                _c("dend", CompartmentKind.DENDRITE_PROXIMAL, 25.0, 2.0, tau=5.0, coords=(-1.0, 0.0, 0.0)),
                _c("soma", CompartmentKind.LOCAL_INHIBITORY, 16.0, 6.0, tau=5.0, coords=(0.0, 0.0, 0.0)),
            ],
            edges=[CompartmentEdge("dend", "soma", weight=0.9, coupling_type="electrical_axial")],
            organelles=[OrganelleAttachment("pred_i", "predicton", "soma")],
            input_compartments=["dend"], output_compartments=["soma"],
            formal_math=r"r_{\text{inh}} = -\tanh(\hat{x}_t + \alpha(x_t - \hat{x}_t))",
            hypothesis="Fast inhibitory cancellation of predictable signals improves surprise contrast.",
        ),
        "dynamic_threshold": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="dynamic_threshold", name="Dynamic-Threshold Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.SPIKING,
            compartments=[_c("soma", CompartmentKind.SOMA, 20.0, 7.5, tau=9.0, thresh=-51.0)],
            edges=[], organelles=[OrganelleAttachment("rep_th", "repairosome", "soma")],
            input_compartments=["soma"], output_compartments=["soma"],
            formal_math=r"V_{\text{th}}(t+1) = V_{\text{th},0} + \beta_{\text{adapt}} u_t",
            hypothesis="Dynamic spike thresholds prevent saturation across wide stimulus amplitudes.",
        ),
        "homeostatic_regulated": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="homeostatic_regulated", name="Homeostatically Regulated Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[_c("soma", CompartmentKind.SOMA, 20.0, 8.0, tau=12.0)],
            edges=[],
            organelles=[
                OrganelleAttachment("res_reg", "resource_regulator", "soma"),
                OrganelleAttachment("rep_reg", "repairosome", "soma"),
            ],
            input_compartments=["soma"], output_compartments=["soma"],
            formal_math=r"g_{\text{homeo}}(t+1) = g_{\text{homeo}}(t) \cdot (r^* / \bar{r}_t)^\gamma",
            hypothesis="Maintains target firing band under drifting input statistics.",
        ),
        "metaplastic_bcm": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="metaplastic_bcm", name="Metaplastic BCM Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[
                _c("dend", CompartmentKind.DENDRITE_BASAL, 35.0, 2.0, tau=10.0),
                _c("soma", CompartmentKind.SOMA, 20.0, 7.5, tau=12.0),
            ],
            edges=[CompartmentEdge("dend", "soma", weight=0.7, plasticity_rule="bcm")],
            organelles=[OrganelleAttachment("plast_nuc", "plasticity_nucleus", "soma")],
            input_compartments=["dend"], output_compartments=["soma"],
            formal_math=r"\Delta w = \eta r_{\text{post}}(r_{\text{post}} - \theta_M) r_{\text{pre}},\; \dot{\theta}_M \propto r_{\text{post}}^2 - \theta_M",
            hypothesis="Sliding modification threshold stabilizes Hebbian selectivity.",
        ),
        "sparse_burst": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="sparse_burst", name="Sparse Burst Neuron",
            objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS, fidelity=FidelityLevel.SPIKING,
            compartments=[
                _c("dend", CompartmentKind.DENDRITE_APICAL, 60.0, 1.5, tau=15.0, receptors=ReceptorDistribution(nmda=1.5)),
                _c("soma", CompartmentKind.SOMA, 22.0, 8.0, tau=7.0, thresh=-46.0),
            ],
            edges=[CompartmentEdge("dend", "soma", weight=1.1, coupling_type="electrical_axial")],
            input_compartments=["dend"], output_compartments=["soma"],
            formal_math=r"b_t = \mathbb{I}[V_t > V_{\text{high}}] \cdot (1 + \kappa s_{\text{apical}})",
            hypothesis="Sparse high-confidence bursts reduce metabolic cost per transmitted bit.",
        ),
        "context_motor": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="context_motor", name="Context-Sensitive Motor Neuron",
            objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[
                _c("cmd", CompartmentKind.DENDRITE_PROXIMAL, 30.0, 2.5, tau=6.0),
                _c("ctx", CompartmentKind.DENDRITE_APICAL, 45.0, 2.0, tau=20.0),
                _c("soma", CompartmentKind.SOMA, 28.0, 11.0, tau=6.0),
                _c("axon", CompartmentKind.AXON_SEGMENT, 200.0, 3.0, myelin=0.9, tau=3.0),
            ],
            edges=[
                CompartmentEdge("cmd", "soma", weight=0.9),
                CompartmentEdge("ctx", "soma", weight=0.5, coupling_type="feedback_gated", gated_by="ctx"),
                CompartmentEdge("soma", "axon", weight=1.0),
            ],
            organelles=[OrganelleAttachment("route_m", "routosome", "soma")],
            input_compartments=["cmd", "ctx"], output_compartments=["axon"],
            formal_math=r"u_{\text{motor}} = \sigma(s_{\text{ctx}}) \cdot \tanh(W_{\text{cmd}} x_{\text{cmd}})",
            hypothesis="Separates motor drive from contextual execution permission.",
        ),
        "adaptive_conduction": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="adaptive_conduction", name="Adaptive Conduction Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[
                _c("soma", CompartmentKind.SOMA, 20.0, 7.0, tau=8.0),
                _c("axon1", CompartmentKind.AXON_SEGMENT, 140.0, 1.2, myelin=0.35, tau=5.0),
                _c("axon2", CompartmentKind.AXON_SEGMENT, 140.0, 1.5, myelin=0.75, tau=4.0),
            ],
            edges=[
                CompartmentEdge("soma", "axon1", weight=0.9, delay_ms=2.0),
                CompartmentEdge("axon1", "axon2", weight=0.9, delay_ms=1.0),
            ],
            organelles=[OrganelleAttachment("chrono_ac", "chronosome", "axon2", params={"tap_delay": 3})],
            input_compartments=["soma"], output_compartments=["axon2"],
            formal_math=r"\tau_{\text{cond}} = L / (v_0 \sqrt{r}(1 + 6 m_{\text{myelin}}))",
            hypothesis="Myelin-modulated conduction delays synchronize arrival times across distant pathways.",
        ),
        "developmental_precursor": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="developmental_precursor", name="Developmental Precursor Neuron",
            objective_category=ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[_c("soma", CompartmentKind.SOMA, 14.0, 5.0, tau=12.0)],
            edges=[],
            organelles=[OrganelleAttachment("dev_gen", "developmental_genome", "soma", params={"maturation_horizon": 40})],
            input_compartments=["soma"], output_compartments=["soma"],
            formal_math=r"\theta_i(t) = f_{\text{dev}}(\text{age}_t, \text{activity}_t)",
            hypothesis="Undifferentiated progenitor cells specialize dynamically during early exposure.",
        ),
        "recovery_compensatory": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="recovery_compensatory", name="Recovery & Compensatory Plasticity Neuron",
            objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[
                _c("dend_a", CompartmentKind.DENDRITE_BASAL, 35.0, 2.0, tau=10.0),
                _c("dend_b", CompartmentKind.DENDRITE_BASAL, 35.0, 2.0, tau=10.0),
                _c("soma", CompartmentKind.SOMA, 22.0, 8.0, tau=10.0),
            ],
            edges=[
                CompartmentEdge("dend_a", "soma", weight=0.7, plasticity_rule="three_factor"),
                CompartmentEdge("dend_b", "soma", weight=0.7, plasticity_rule="three_factor"),
            ],
            organelles=[
                OrganelleAttachment("rep_rec", "repairosome", "soma"),
                OrganelleAttachment("ckpt_rec", "checkpoint_vesicle", "soma"),
            ],
            input_compartments=["dend_a", "dend_b"], output_compartments=["soma"],
            formal_math=r"g_{\text{comp}} = \text{clip}(g_0 + \alpha_{\text{rep}}(r^* - \bar{r}), g_{\min}, g_{\max})",
            hypothesis="Redundant basal dendrites + Repairosome preserve signal throughput after single-branch lesions.",
        ),
        "reliability_sensitive": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="reliability_sensitive", name="Reliability-Sensitive Neuron",
            objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[_c("soma", CompartmentKind.SOMA, 20.0, 7.5, tau=10.0)],
            edges=[],
            organelles=[
                OrganelleAttachment("unc_rel", "uncertaintosome", "soma"),
                OrganelleAttachment("att_rel", "attentosome", "soma"),
            ],
            input_compartments=["soma"], output_compartments=["soma"],
            formal_math=r"y_t = \tanh(x_t) / (1 + \lambda \hat{\sigma}_t)",
            hypothesis="Down-weights volatile noisy inputs via local variance tracking.",
        ),
        # --- 6 Extended V2 Biophysical & Conventional Families (+6 -> 21 Total) ---
        "adex_bursting": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="adex_bursting", name="Brette-Gerstner AdEx Bursting Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.SPIKING,
            compartments=[
                _c("dend", CompartmentKind.DENDRITE_BASAL, 38.0, 2.2, tau=9.0, coords=(-1.2, 0.0, 0.0)),
                _c("soma", CompartmentKind.SOMA, 23.0, 8.5, tau=7.0, thresh=-49.0, coords=(0.0, 0.0, 0.0)),
                _c("axon", CompartmentKind.AXON_SEGMENT, 95.0, 1.4, myelin=0.65, tau=4.0, coords=(1.8, 0.0, 0.0)),
            ],
            edges=[
                CompartmentEdge("dend", "soma", weight=0.9, coupling_type="electrical_axial"),
                CompartmentEdge("soma", "axon", weight=0.95, coupling_type="electrical_axial"),
            ],
            input_compartments=["dend"], output_compartments=["axon"],
            formal_math=r"C\dot{V} = -g_L(V-E_L) + g_L\Delta_T e^{(V-V_T)/\Delta_T} - w + I,\; \tau_w\dot{w} = a(V-E_L) - w",
            hypothesis="Exponential spike-initiation slope paired with subthreshold adaptation reproduces regular, adapting, and phasic bursting.",
        ),
        "theta_phase_precession": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="theta_phase_precession", name="Hippocampal Theta Phase-Precession Cell",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[
                _c("dend_place", CompartmentKind.DENDRITE_APICAL, 65.0, 1.9, tau=12.0, coords=(-1.5, 0.8, 0.0)),
                _c("soma_theta", CompartmentKind.SOMA, 22.0, 8.5, tau=9.0, coords=(0.0, 0.0, 0.0)),
            ],
            edges=[CompartmentEdge("dend_place", "soma_theta", weight=0.88, coupling_type="electrical_axial")],
            organelles=[OrganelleAttachment("th_ph", "phase_organelle", "soma_theta", params={"omega": 0.52})],
            input_compartments=["dend_place"], output_compartments=["soma_theta"],
            formal_math=r"r(t) = \phi(I_{\text{place}}(t) + \cos(\omega_\theta t + \kappa I_{\text{place}}(t)))",
            hypothesis="Dual-oscillator interference encodes sequential spatial/temporal order within individual theta cycles.",
        ),
        "chandelier_axonic_inhibitor": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="chandelier_axonic_inhibitor", name="Chandelier Axo-Axonic Inhibitory Interneuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.SPIKING,
            dale_sign=-1,
            compartments=[
                _c("dend", CompartmentKind.DENDRITE_PROXIMAL, 28.0, 2.0, tau=4.5, coords=(-1.2, 0.0, 0.0)),
                _c("soma_fs", CompartmentKind.LOCAL_INHIBITORY, 18.0, 6.5, tau=4.0, thresh=-51.0, coords=(0.0, 0.0, 0.0)),
                _c("cartridge_axon", CompartmentKind.AXON_SEGMENT, 70.0, 1.5, myelin=0.8, tau=2.8, coords=(1.6, 0.0, 0.0)),
            ],
            edges=[
                CompartmentEdge("dend", "soma_fs", weight=0.95, coupling_type="electrical_axial"),
                CompartmentEdge("soma_fs", "cartridge_axon", weight=1.0, coupling_type="electrical_axial"),
            ],
            input_compartments=["dend"], output_compartments=["cartridge_axon"],
            formal_math=r"I_{\text{AIS}} = -g_{\text{GABA,A}}(V_{\text{AIS}} - E_{\text{Cl}})",
            hypothesis="Ultra-fast perisomatic/axon-initial-segment shunting synchronizes population spike windows and prevents runaway excitation.",
        ),
        "thalamocortical_relay": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="thalamocortical_relay", name="Thalamocortical Tonic/Burst Relay Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.CONDUCTANCE,
            compartments=[
                _c("sensory_dend", CompartmentKind.DENDRITE_PROXIMAL, 36.0, 2.4, tau=8.0, coords=(-1.6, 0.6, 0.0)),
                _c("cortical_fb", CompartmentKind.DENDRITE_DISTAL, 55.0, 1.6, tau=18.0, coords=(-1.6, -0.6, 0.0)),
                _c("relay_soma", CompartmentKind.SOMA, 24.0, 9.5, tau=9.0, coords=(0.0, 0.0, 0.0)),
                _c("tc_axon", CompartmentKind.AXON_SEGMENT, 120.0, 1.6, myelin=0.75, tau=3.5, coords=(2.0, 0.0, 0.0)),
            ],
            edges=[
                CompartmentEdge("sensory_dend", "relay_soma", weight=0.92, coupling_type="electrical_axial"),
                CompartmentEdge("cortical_fb", "relay_soma", weight=0.55, coupling_type="feedback_gated", gated_by="cortical_fb"),
                CompartmentEdge("relay_soma", "tc_axon", weight=0.98, coupling_type="electrical_axial"),
            ],
            organelles=[OrganelleAttachment("tc_route", "routosome", "relay_soma")],
            input_compartments=["sensory_dend", "cortical_fb"], output_compartments=["tc_axon"],
            formal_math=r"y_{\text{TC}} = g_{\text{tonic}}(V_m) I_{\text{sens}} + (1 - g_{\text{tonic}}(V_m)) B_{\text{rebound}}(I_{\text{cort}})",
            hypothesis="Corticothalamic membrane polarization switches thalamic relay cells between linear sensory transmission and wake-up burst detection.",
        ),
        "cerebellar_purkinje_arbor": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="cerebellar_purkinje_arbor", name="Cerebellar Purkinje Planar Dendritic Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            dale_sign=-1,
            compartments=[
                _c("pf_branch_0", CompartmentKind.DENDRITE_DISTAL, 70.0, 1.5, tau=12.0, coords=(-2.0, 1.2, 0.0)),
                _c("pf_branch_1", CompartmentKind.DENDRITE_DISTAL, 70.0, 1.5, tau=12.0, coords=(-2.0, 0.0, 0.0)),
                _c("pf_branch_2", CompartmentKind.DENDRITE_DISTAL, 70.0, 1.5, tau=12.0, coords=(-2.0, -1.2, 0.0)),
                _c("climbing_error", CompartmentKind.DENDRITE_PROXIMAL, 30.0, 2.8, tau=6.0, coords=(-0.8, 1.5, 0.0)),
                _c("purkinje_soma", CompartmentKind.SOMA, 28.0, 11.0, tau=7.0, coords=(0.4, 0.0, 0.0)),
                _c("dcn_axon", CompartmentKind.AXON_SEGMENT, 110.0, 1.6, myelin=0.75, tau=3.5, coords=(2.4, 0.0, 0.0)),
            ],
            edges=[
                CompartmentEdge("pf_branch_0", "purkinje_soma", weight=0.65, plasticity_rule="local_error"),
                CompartmentEdge("pf_branch_1", "purkinje_soma", weight=0.65, plasticity_rule="local_error"),
                CompartmentEdge("pf_branch_2", "purkinje_soma", weight=0.65, plasticity_rule="local_error"),
                CompartmentEdge("climbing_error", "purkinje_soma", weight=1.20, coupling_type="chemical_synapse"),
                CompartmentEdge("purkinje_soma", "dcn_axon", weight=0.95),
            ],
            organelles=[OrganelleAttachment("pkj_cont", "contraston", "purkinje_soma")],
            input_compartments=["pf_branch_0", "pf_branch_1", "pf_branch_2", "climbing_error"], output_compartments=["dcn_axon"],
            formal_math=r"\Delta w_{\text{PF},i} = -\eta\, I_{\text{CF}}(t)\, r_{\text{PF},i}(t-\delta)",
            hypothesis="High-fan-in parallel-fiber branches supervised by climbing-fiber error pulses learn millisecond forward motor timing.",
        ),
        "neuromodulatory_pacemaker": lambda nid: NeuronMorphologySpec(
            morphology_id=nid, family="neuromodulatory_pacemaker", name="Tonic-Phasic Neuromodulatory Pacemaker Neuron",
            objective_category=ObjectiveCategory.BIOLOGICAL_REALISM, fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
            compartments=[
                _c("rpe_dend", CompartmentKind.DENDRITE_BASAL, 40.0, 2.0, tau=10.0, coords=(-1.6, 0.0, 0.0)),
                _c("pacemaker_soma", CompartmentKind.METABOLIC_REGULATOR, 24.0, 9.0, tau=18.0, coords=(0.0, 0.0, 0.0), receptors=ReceptorDistribution(dopamine=1.8, paracrine=2.0)),
                _c("diffuse_axon", CompartmentKind.AXON_SEGMENT, 150.0, 1.2, myelin=0.2, tau=14.0, coords=(2.2, 0.0, 0.0)),
            ],
            edges=[
                CompartmentEdge("rpe_dend", "pacemaker_soma", weight=0.85),
                CompartmentEdge("pacemaker_soma", "diffuse_axon", weight=0.90, coupling_type="paracrine"),
            ],
            organelles=[
                OrganelleAttachment("nmp_ph", "phase_organelle", "pacemaker_soma", params={"omega": 0.22}),
                OrganelleAttachment("nmp_plast", "plasticity_nucleus", "pacemaker_soma"),
            ],
            input_compartments=["rpe_dend"], output_compartments=["diffuse_axon"],
            formal_math=r"m(t) = m_{\text{tonic}} + \phi_{\text{phasic}}(r_t - \hat{V}_t)",
            hypothesis="Combines autonomous tonic baseline firing with phasic surprise bursts for global three-factor plasticity gating.",
        ),
    }
    if family not in builders:
        raise KeyError(f"Unknown conventional neuron family {family!r}. Valid: {sorted(builders)}")
    spec = builders[family](neuron_id)
    if fidelity is not None:
        spec.fidelity = fidelity
    if seed is not None:
        spec.metadata["seed"] = int(seed)
    return spec


CONVENTIONAL_NEURON_FAMILIES: Tuple[str, ...] = (
    "lif_reference",
    "recurrent_rate",
    "izhikevich",
    "conductance_hh",
    "adaptive_pyramidal",
    "predictive_interneuron",
    "dynamic_threshold",
    "homeostatic_regulated",
    "metaplastic_bcm",
    "sparse_burst",
    "context_motor",
    "adaptive_conduction",
    "developmental_precursor",
    "recovery_compensatory",
    "reliability_sensitive",
)

EXTENDED_CONVENTIONAL_NEURON_FAMILIES: Tuple[str, ...] = (
    "adex_bursting",
    "theta_phase_precession",
    "chandelier_axonic_inhibitor",
    "thalamocortical_relay",
    "cerebellar_purkinje_arbor",
    "neuromodulatory_pacemaker",
)

ALL_CONVENTIONAL_NEURON_FAMILIES: Tuple[str, ...] = CONVENTIONAL_NEURON_FAMILIES + EXTENDED_CONVENTIONAL_NEURON_FAMILIES


def run_heterogeneous_vs_homogeneous_experiment(
    n_neurons: int = 12,
    seeds: Sequence[int] = (7, 11, 19),
    steps: int = 90,
) -> Dict[str, Any]:
    """Compare heterogeneous conventional populations vs homogeneous rate/LIF under matched neuron count."""
    results_by_arm: Dict[str, List[float]] = {"homogeneous_rate": [], "homogeneous_lif": [], "heterogeneous_mixed": []}
    energy_by_arm: Dict[str, List[float]] = {"homogeneous_rate": [], "homogeneous_lif": [], "heterogeneous_mixed": []}

    hetero_cycle = [
        "adaptive_pyramidal", "predictive_interneuron", "metaplastic_bcm",
        "homeostatic_regulated", "adaptive_conduction", "reliability_sensitive",
    ]

    for seed in seeds:
        rng = np.random.default_rng(seed)
        t = np.arange(steps, dtype=np.float64)
        # Task: nonlinear combination of 3-step delayed cue + current harmonic
        u1 = np.sign(np.sin(0.21 * t + rng.uniform(0, 3.0)))
        u2 = np.cos(0.37 * t)
        target = np.tanh(0.8 * np.roll(u1, 3) * u2 + 0.4 * np.roll(u2, 6))
        target[:6] = 0.0

        # Shared sparse topology across all 3 arms for matched connectivity
        adj = []
        for i in range(n_neurons):
            for _ in range(3):
                j = int(rng.integers(0, n_neurons))
                if j != i:
                    adj.append((i, j, float(rng.uniform(0.1, 0.45))))

        for arm in ("homogeneous_rate", "homogeneous_lif", "heterogeneous_mixed"):
            neurons = []
            for i in range(n_neurons):
                if arm == "homogeneous_rate":
                    fam = "recurrent_rate"
                elif arm == "homogeneous_lif":
                    fam = "lif_reference"
                else:
                    fam = hetero_cycle[i % len(hetero_cycle)]
                neurons.append(build_conventional_neuron(fam, neuron_id=f"n{i}"))
            edges = [
                ConnectomeEdge(f"n{i}", f"n{j}", weight=w * neurons[i].dale_sign, delay_ms=1.0)
                for i, j, w in adj
            ]
            graph = ConnectomeGraph(
                connectome_id=f"{arm}_s{seed}", name=arm, neurons=neurons, edges=edges,
                sensory_ids=["n0", "n1", "n2", "n3"], motor_ids=[f"n{n_neurons - 1}"],
            )
            sim = MultiFidelityConnectome(graph, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
            states = np.zeros((steps, n_neurons), dtype=np.float64)
            for k in range(steps):
                ext = {"n0": float(u1[k]), "n1": float(u2[k]), "n2": float(u1[k] * u2[k]), "n3": float(-u1[k])}
                out = sim.step(ext)
                states[k] = out["outputs"]
            split = int(0.65 * steps)
            X_tr, y_tr = states[:split], target[:split]
            X_te, y_te = states[split:], target[split:]
            w = np.linalg.solve(X_tr.T @ X_tr + 0.25 * np.eye(n_neurons), X_tr.T @ y_tr)
            mse = float(np.mean((X_te @ w - y_te) ** 2))
            results_by_arm[arm].append(mse)
            energy_by_arm[arm].append(sim.total_energy)

    summary = {}
    for arm in results_by_arm:
        vals = np.asarray(results_by_arm[arm])
        eng = np.asarray(energy_by_arm[arm])
        summary[arm] = {
            "mean_test_mse": round(float(np.mean(vals)), 6),
            "std_test_mse": round(float(np.std(vals)), 6),
            "mean_energy": round(float(np.mean(eng)), 4),
            "per_seed_mse": [round(float(v), 6) for v in vals],
        }
    best_arm = min(summary, key=lambda k: summary[k]["mean_test_mse"])
    return {
        "experiment": "heterogeneous_vs_homogeneous_conventional_populations",
        "n_neurons": n_neurons,
        "seeds": list(seeds),
        "steps": steps,
        "arms": summary,
        "lowest_mse_arm": best_arm,
    }
