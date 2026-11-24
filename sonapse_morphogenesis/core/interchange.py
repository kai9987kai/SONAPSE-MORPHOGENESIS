"""Common Interchange Format v1 (`morphogenesis-schema-v1`).

Provides strictly validated, JSON-serializable, SHA-256 content-digested schemas
shared across all 11 SONAPSE MORPHOGENESIS research modules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
import hashlib
import json
import math
import time
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0.0"
SCHEMA_NAME = "morphogenesis-schema-v1"


class ObjectiveCategory(str, Enum):
    """Explicit scientific distinction across research objectives."""
    BIOLOGICAL_REALISM = "biological_realism"
    COMPUTATIONAL_USEFULNESS = "computational_usefulness"
    SPECULATIVE_SYNTHETIC_BIOLOGY = "speculative_synthetic_biology"


class FidelityLevel(IntEnum):
    """Five-level multi-fidelity neural simulation hierarchy."""
    ABSTRACT_RATE = 1
    MULTI_COMPARTMENT_RECURRENT = 2
    SPIKING = 3
    CONDUCTANCE = 4
    REFERENCE_CABLE = 5


FIDELITY_APPROXIMATION_LABELS: Dict[int, str] = {
    1: "Abstract continuous rate dynamics (no spikes, no explicit ionic channels).",
    2: "Multi-compartment synthetic continuous recurrent dynamics coupled by physical cable geometry.",
    3: "Phenomenological multi-compartment spiking dynamics (LIF/Izhikevich threshold-reset).",
    4: "Conductance-based compartment model with Hodgkin-Huxley Na/K/Leak channels (Rush-Larsen Exp-Euler).",
    5: "Sub-discretized RK4 Hodgkin-Huxley reference cable with explicit linear stability verification.",
}


class CompartmentKind(str, Enum):
    SOMA = "soma"
    SECONDARY_SOMA = "secondary_soma"
    DENDRITE_PROXIMAL = "dendrite_proximal"
    DENDRITE_DISTAL = "dendrite_distal"
    DENDRITE_APICAL = "dendrite_apical"
    DENDRITE_BASAL = "dendrite_basal"
    INTEGRATION_CENTER = "integration_center"
    AXON_HILLOCK = "axon_hillock"
    AXON_SEGMENT = "axon_segment"
    AXON_FEEDBACK = "axon_feedback"
    LOCAL_INHIBITORY = "local_inhibitory"
    MEMORY_COMPARTMENT = "memory_compartment"
    METABOLIC_REGULATOR = "metabolic_regulator"
    ML_ADAPTER = "ml_adapter"
    ORGANELLE_DOCK = "organelle_dock"


class DataSourceKind(str, Enum):
    """Explicit separation between biological measurements, model telemetry, and synthetic data."""
    BIOLOGICAL_RECORDING = "biological_recording"
    ARTIFICIAL_MODEL_ACTIVITY = "artificial_model_activity"
    SYNTHETIC_SIMULATION = "synthetic_simulation"
    HYBRID_EXPERIMENTAL = "hybrid_experimental"


class EvidenceState(str, Enum):
    """Five-stage evidence progression for SONAPSE SENTINEL discovery governance."""
    UNVERIFIED_OBSERVATION = "unverified_observation"
    CANDIDATE_DISCOVERY = "candidate_discovery"
    INDEPENDENTLY_REPLICATED = "independently_replicated"
    CAUSALLY_SUPPORTED = "causally_supported"
    LARGER_SCALE_VALIDATED = "larger_scale_validated"


def _check_finite(name: str, val: float, lo: float = -1e9, hi: float = 1e9) -> float:
    v = float(val)
    if not math.isfinite(v) or v < lo or v > hi:
        raise ValueError(f"{name} must be finite in [{lo}, {hi}], got {val!r}")
    return v


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass
class ReceptorDistribution:
    """Adjustable receptor sensitivities per compartment (normalized 0..5)."""
    ampa: float = 1.0
    nmda: float = 0.25
    gaba_a: float = 0.8
    gaba_b: float = 0.2
    dopamine: float = 0.3
    serotonin: float = 0.2
    acetylcholine: float = 0.3
    noradrenaline: float = 0.2
    paracrine: float = 0.2

    def __post_init__(self) -> None:
        for f in ("ampa", "nmda", "gaba_a", "gaba_b", "dopamine", "serotonin", "acetylcholine", "noradrenaline", "paracrine"):
            setattr(self, f, _check_finite(f, getattr(self, f), 0.0, 10.0))

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReceptorDistribution":
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class CompartmentNode:
    """Single physical-computational compartment inside a structured neuron.

    Physical dimensions directly govern computational dynamics:
    - Surface area A = 2 * pi * r * L -> Membrane capacitance C = Cm * A
    - Radius r & length L -> Axial resistance Ra_eff = (ra_ohm_cm * L) / (pi * r^2)
    - Myelination & radius -> Conduction speed v = v0 * sqrt(r) * (1 + 4.5 * myelin)
    - Complexity & surface area -> Resting + active metabolic ATP cost
    """
    compartment_id: str
    kind: CompartmentKind = CompartmentKind.SOMA
    length_um: float = 25.0
    radius_um: float = 8.0
    myelin_factor: float = 0.0
    coords_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cm_uF_cm2: float = 1.0
    ra_ohm_cm: float = 100.0
    g_leak_mS_cm2: float = 0.3
    e_rest_mV: float = -65.0
    threshold_mV: float = -50.0
    tau_ms: float = 10.0
    receptors: ReceptorDistribution = field(default_factory=ReceptorDistribution)
    metabolic_weight: float = 1.0
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = CompartmentKind(self.kind)
        if isinstance(self.receptors, Mapping):
            self.receptors = ReceptorDistribution.from_dict(self.receptors)
        self.length_um = _check_finite("length_um", self.length_um, 0.5, 50000.0)
        self.radius_um = _check_finite("radius_um", self.radius_um, 0.05, 500.0)
        self.myelin_factor = _check_finite("myelin_factor", self.myelin_factor, 0.0, 1.0)
        self.cm_uF_cm2 = _check_finite("cm_uF_cm2", self.cm_uF_cm2, 0.1, 20.0)
        self.ra_ohm_cm = _check_finite("ra_ohm_cm", self.ra_ohm_cm, 1.0, 5000.0)
        self.g_leak_mS_cm2 = _check_finite("g_leak_mS_cm2", self.g_leak_mS_cm2, 1e-4, 50.0)
        self.e_rest_mV = _check_finite("e_rest_mV", self.e_rest_mV, -120.0, 20.0)
        self.threshold_mV = _check_finite("threshold_mV", self.threshold_mV, -90.0, 50.0)
        self.tau_ms = _check_finite("tau_ms", self.tau_ms, 0.2, 5000.0)
        self.coords_3d = tuple(_check_finite("coord", c, -1e5, 1e5) for c in self.coords_3d)  # type: ignore[assignment]
        if len(self.coords_3d) != 3:
            raise ValueError("coords_3d must have length 3")

    @property
    def surface_area_cm2(self) -> float:
        """Cylindrical lateral membrane surface area in cm^2."""
        r_cm = self.radius_um * 1e-4
        l_cm = self.length_um * 1e-4
        return 2.0 * math.pi * r_cm * l_cm

    @property
    def effective_cm_uF_cm2(self) -> float:
        """Myelination reduces effective membrane capacitance."""
        return self.cm_uF_cm2 / (1.0 + 3.0 * self.myelin_factor)

    @property
    def capacitance_nF(self) -> float:
        """Total compartment membrane capacitance in nF (1 uF = 1000 nF)."""
        return self.effective_cm_uF_cm2 * self.surface_area_cm2 * 1000.0

    @property
    def axial_resistance_MOhm(self) -> float:
        """Longitudinal axial resistance R_a = rho_a * L / (pi * r^2) in MegaOhms."""
        r_cm = self.radius_um * 1e-4
        l_cm = self.length_um * 1e-4
        r_ohm = self.ra_ohm_cm * l_cm / (math.pi * r_cm * r_cm)
        return r_ohm * 1e-6

    @property
    def conduction_velocity_m_s(self) -> float:
        """Radius- and myelination-dependent propagation velocity (m/s)."""
        unmyelinated = 0.5 * math.sqrt(max(self.radius_um, 0.05))
        myelinated_boost = 1.0 + 6.0 * self.myelin_factor * math.sqrt(max(self.radius_um, 0.1))
        return unmyelinated * myelinated_boost

    @property
    def transmission_delay_ms(self) -> float:
        """Physical propagation delay tau = L / v in milliseconds."""
        length_m = self.length_um * 1e-6
        return max(0.05, (length_m / max(self.conduction_velocity_m_s, 1e-4)) * 1000.0)

    @property
    def effective_time_constant_ms(self) -> float:
        """Geometry-modulated membrane time constant (ms)."""
        area_scale = math.pow(max(self.surface_area_cm2 / 1.2566e-5, 0.1), 0.25)
        myelin_scale = 1.0 / (1.0 + 1.5 * self.myelin_factor)
        return float(max(0.5, min(2000.0, self.tau_ms * area_scale * myelin_scale)))

    @property
    def baseline_metabolic_cost(self) -> float:
        """Dimensionless metabolic upkeep per simulation step proportional to area & myelination."""
        area_ratio = self.surface_area_cm2 / 1.2566e-5
        myelin_saving = 1.0 - 0.35 * self.myelin_factor
        return float(max(0.05, self.metabolic_weight * (0.25 + 0.75 * math.sqrt(area_ratio)) * myelin_saving))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compartment_id": self.compartment_id,
            "kind": self.kind.value,
            "length_um": self.length_um,
            "radius_um": self.radius_um,
            "myelin_factor": self.myelin_factor,
            "coords_3d": list(self.coords_3d),
            "cm_uF_cm2": self.cm_uF_cm2,
            "ra_ohm_cm": self.ra_ohm_cm,
            "g_leak_mS_cm2": self.g_leak_mS_cm2,
            "e_rest_mV": self.e_rest_mV,
            "threshold_mV": self.threshold_mV,
            "tau_ms": self.tau_ms,
            "receptors": self.receptors.to_dict(),
            "metabolic_weight": self.metabolic_weight,
            "active": self.active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CompartmentNode":
        return cls(
            compartment_id=str(d["compartment_id"]),
            kind=CompartmentKind(d.get("kind", "soma")),
            length_um=float(d.get("length_um", 25.0)),
            radius_um=float(d.get("radius_um", 8.0)),
            myelin_factor=float(d.get("myelin_factor", 0.0)),
            coords_3d=tuple(float(x) for x in d.get("coords_3d", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            cm_uF_cm2=float(d.get("cm_uF_cm2", 1.0)),
            ra_ohm_cm=float(d.get("ra_ohm_cm", 100.0)),
            g_leak_mS_cm2=float(d.get("g_leak_mS_cm2", 0.3)),
            e_rest_mV=float(d.get("e_rest_mV", -65.0)),
            threshold_mV=float(d.get("threshold_mV", -50.0)),
            tau_ms=float(d.get("tau_ms", 10.0)),
            receptors=ReceptorDistribution.from_dict(d.get("receptors", {})),
            metabolic_weight=float(d.get("metabolic_weight", 1.0)),
            active=bool(d.get("active", True)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class CompartmentEdge:
    """Internal connection between two compartments of a neuron."""
    source_id: str
    target_id: str
    weight: float = 0.5
    coupling_type: str = "electrical_axial"  # electrical_axial | chemical_synapse | gap_junction | feedback_gated | paracrine | ml_bridge
    delay_ms: float = 0.5
    gated_by: Optional[str] = None
    plasticity_rule: str = "none"

    def __post_init__(self) -> None:
        self.weight = _check_finite("weight", self.weight, -20.0, 20.0)
        self.delay_ms = _check_finite("delay_ms", self.delay_ms, 0.0, 500.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CompartmentEdge":
        return cls(
            source_id=str(d["source_id"]),
            target_id=str(d["target_id"]),
            weight=float(d.get("weight", 0.5)),
            coupling_type=str(d.get("coupling_type", "electrical_axial")),
            delay_ms=float(d.get("delay_ms", 0.5)),
            gated_by=d.get("gated_by"),
            plasticity_rule=str(d.get("plasticity_rule", "none")),
        )


@dataclass
class OrganelleAttachment:
    """Attachment specification for a synthetic organelle inside a neuron compartment."""
    organelle_id: str
    organelle_type: str
    compartment_id: str
    params: Dict[str, Any] = field(default_factory=dict)
    size_scale: float = 1.0
    sensitivity: float = 1.0
    active: bool = True

    def __post_init__(self) -> None:
        self.size_scale = _check_finite("size_scale", self.size_scale, 0.1, 10.0)
        self.sensitivity = _check_finite("sensitivity", self.sensitivity, 0.0, 10.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OrganelleAttachment":
        return cls(
            organelle_id=str(d["organelle_id"]),
            organelle_type=str(d["organelle_type"]),
            compartment_id=str(d["compartment_id"]),
            params=dict(d.get("params", {})),
            size_scale=float(d.get("size_scale", 1.0)),
            sensitivity=float(d.get("sensitivity", 1.0)),
            active=bool(d.get("active", True)),
        )


@dataclass
class NeuronMorphologySpec:
    """Complete machine-readable morphology and scientific specification of a neuron."""
    morphology_id: str
    family: str
    name: str
    objective_category: ObjectiveCategory = ObjectiveCategory.COMPUTATIONAL_USEFULNESS
    fidelity: FidelityLevel = FidelityLevel.MULTI_COMPARTMENT_RECURRENT
    compartments: List[CompartmentNode] = field(default_factory=list)
    edges: List[CompartmentEdge] = field(default_factory=list)
    organelles: List[OrganelleAttachment] = field(default_factory=list)
    input_compartments: List[str] = field(default_factory=list)
    output_compartments: List[str] = field(default_factory=list)
    formal_math: str = ""
    hypothesis: str = ""
    description: str = ""
    dale_sign: int = 1
    schema_version: str = SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.objective_category, str):
            self.objective_category = ObjectiveCategory(self.objective_category)
        if isinstance(self.fidelity, int) and not isinstance(self.fidelity, FidelityLevel):
            self.fidelity = FidelityLevel(self.fidelity)
        if self.dale_sign not in (-1, 1):
            raise ValueError("dale_sign must be +1 or -1")
        ids = {c.compartment_id for c in self.compartments}
        for e in self.edges:
            if e.source_id not in ids or e.target_id not in ids:
                raise ValueError(f"Edge {e.source_id}->{e.target_id} references unknown compartment")
        for o in self.organelles:
            if o.compartment_id not in ids:
                raise ValueError(f"Organelle {o.organelle_id} references unknown compartment {o.compartment_id}")
        if not self.input_compartments and self.compartments:
            self.input_compartments = [self.compartments[0].compartment_id]
        if not self.output_compartments and self.compartments:
            self.output_compartments = [self.compartments[-1].compartment_id]

    @property
    def approximation_label(self) -> str:
        return FIDELITY_APPROXIMATION_LABELS[int(self.fidelity)]

    @property
    def total_metabolic_cost(self) -> float:
        comp_cost = sum(c.baseline_metabolic_cost for c in self.compartments if c.active)
        org_cost = sum(0.35 * o.size_scale for o in self.organelles if o.active)
        return round(comp_cost + org_cost, 6)

    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "morphology_id": self.morphology_id,
            "family": self.family,
            "name": self.name,
            "objective_category": self.objective_category.value,
            "fidelity": int(self.fidelity),
            "approximation_label": self.approximation_label,
            "dale_sign": self.dale_sign,
            "compartments": [c.to_dict() for c in self.compartments],
            "edges": [e.to_dict() for e in self.edges],
            "organelles": [o.to_dict() for o in self.organelles],
            "input_compartments": list(self.input_compartments),
            "output_compartments": list(self.output_compartments),
            "formal_math": self.formal_math,
            "hypothesis": self.hypothesis,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NeuronMorphologySpec":
        return cls(
            morphology_id=str(d["morphology_id"]),
            family=str(d["family"]),
            name=str(d["name"]),
            objective_category=ObjectiveCategory(d.get("objective_category", "computational_usefulness")),
            fidelity=FidelityLevel(int(d.get("fidelity", 2))),
            compartments=[CompartmentNode.from_dict(c) for c in d.get("compartments", [])],
            edges=[CompartmentEdge.from_dict(e) for e in d.get("edges", [])],
            organelles=[OrganelleAttachment.from_dict(o) for o in d.get("organelles", [])],
            input_compartments=list(d.get("input_compartments", [])),
            output_compartments=list(d.get("output_compartments", [])),
            formal_math=str(d.get("formal_math", "")),
            hypothesis=str(d.get("hypothesis", "")),
            description=str(d.get("description", "")),
            dale_sign=int(d.get("dale_sign", 1)),
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class ConnectomeEdge:
    """Directed synapse between two neurons in a ConnectomeGraph."""
    pre_id: str
    post_id: str
    weight: float
    delay_ms: float = 1.0
    length_um: float = 100.0
    receptor_type: str = "ampa"
    plasticity_rule: str = "three_factor"
    source_compartment: Optional[str] = None
    target_compartment: Optional[str] = None
    active: bool = True
    provenance: str = "synthetic"

    def __post_init__(self) -> None:
        self.weight = _check_finite("weight", self.weight, -50.0, 50.0)
        self.delay_ms = _check_finite("delay_ms", self.delay_ms, 0.0, 1000.0)
        self.length_um = _check_finite("length_um", self.length_um, 1.0, 1e6)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConnectomeEdge":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ConnectomeGraph:
    """Unified interchange format for an entire heterogeneous connectome."""
    connectome_id: str
    name: str
    neurons: List[NeuronMorphologySpec] = field(default_factory=list)
    edges: List[ConnectomeEdge] = field(default_factory=list)
    regions: Dict[str, List[str]] = field(default_factory=dict)
    sensory_ids: List[str] = field(default_factory=list)
    motor_ids: List[str] = field(default_factory=list)
    source_kind: DataSourceKind = DataSourceKind.SYNTHETIC_SIMULATION
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.source_kind, str):
            self.source_kind = DataSourceKind(self.source_kind)

    @property
    def neuron_count(self) -> int:
        return len(self.neurons)

    @property
    def synapse_count(self) -> int:
        return sum(1 for e in self.edges if e.active)

    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "connectome_id": self.connectome_id,
            "name": self.name,
            "source_kind": self.source_kind.value,
            "neurons": [n.to_dict() for n in self.neurons],
            "edges": [e.to_dict() for e in self.edges],
            "regions": {k: list(v) for k, v in self.regions.items()},
            "sensory_ids": list(self.sensory_ids),
            "motor_ids": list(self.motor_ids),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConnectomeGraph":
        return cls(
            connectome_id=str(d["connectome_id"]),
            name=str(d["name"]),
            neurons=[NeuronMorphologySpec.from_dict(n) for n in d.get("neurons", [])],
            edges=[ConnectomeEdge.from_dict(e) for e in d.get("edges", [])],
            regions={k: list(v) for k, v in d.get("regions", {}).items()},
            sensory_ids=list(d.get("sensory_ids", [])),
            motor_ids=list(d.get("motor_ids", [])),
            source_kind=DataSourceKind(d.get("source_kind", "synthetic_simulation")),
            provenance=dict(d.get("provenance", {})),
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class DevelopmentalGenome:
    """Versioned developmental genome defining cells, connectome topology, plasticity, and growth."""
    genome_id: str
    connectome: ConnectomeGraph
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    growth_rules: Dict[str, Any] = field(default_factory=lambda: {"activity_threshold": 0.45, "max_new_synapses_per_epoch": 4, "max_neurons": 512})
    pruning_rules: Dict[str, Any] = field(default_factory=lambda: {"min_abs_weight": 0.02, "inactivity_steps": 64})
    differentiation_rules: Dict[str, Any] = field(default_factory=lambda: {"allow_phenotype_shift": True, "plasticity_families": ["three_factor", "stdp", "bcm"]})
    neuromodulation_params: Dict[str, float] = field(default_factory=lambda: {"da_gain": 1.0, "serotonin_gain": 0.6, "ach_gain": 0.8, "ne_gain": 0.5, "paracrine_decay": 0.85})
    metabolic_budget: float = 250.0
    structural_constraints: Dict[str, Any] = field(default_factory=lambda: {"enforce_dales_law": True, "max_fan_in_l1": 4.0, "max_compartments_per_neuron": 24})
    external_interfaces: Dict[str, Any] = field(default_factory=dict)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "connectome": self.connectome.to_dict(),
            "growth_rules": self.growth_rules,
            "pruning_rules": self.pruning_rules,
            "differentiation_rules": self.differentiation_rules,
            "neuromodulation_params": self.neuromodulation_params,
            "metabolic_budget": self.metabolic_budget,
            "structural_constraints": self.structural_constraints,
            "external_interfaces": self.external_interfaces,
            "mutation_history": self.mutation_history,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DevelopmentalGenome":
        return cls(
            genome_id=str(d["genome_id"]),
            connectome=ConnectomeGraph.from_dict(d["connectome"]),
            generation=int(d.get("generation", 0)),
            parent_ids=list(d.get("parent_ids", [])),
            growth_rules=dict(d.get("growth_rules", {})),
            pruning_rules=dict(d.get("pruning_rules", {})),
            differentiation_rules=dict(d.get("differentiation_rules", {})),
            neuromodulation_params=dict(d.get("neuromodulation_params", {})),
            metabolic_budget=float(d.get("metabolic_budget", 250.0)),
            structural_constraints=dict(d.get("structural_constraints", {})),
            external_interfaces=dict(d.get("external_interfaces", {})),
            mutation_history=list(d.get("mutation_history", [])),
            provenance=dict(d.get("provenance", {})),
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class ActivationStream:
    """Common interchange format for neural activation streams & recordings."""
    stream_id: str
    source_kind: DataSourceKind
    node_ids: List[str]
    timestamps_ms: List[float]
    values: List[List[float]]  # [T, N]
    spikes: Optional[List[List[int]]] = None
    chemical_fields: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stream_id": self.stream_id,
            "source_kind": self.source_kind.value,
            "node_ids": self.node_ids,
            "timestamps_ms": self.timestamps_ms,
            "values": self.values,
            "spikes": self.spikes,
            "chemical_fields": self.chemical_fields,
            "metadata": self.metadata,
        }


@dataclass
class SensoryExperience:
    """Structured sensory-motor experience packet."""
    experience_id: str
    modality: str
    encoding: str  # rate | spiking | event | temporal | frequency | latent
    source_kind: DataSourceKind
    signal: List[float]
    timestamp_ms: float = 0.0
    action: Optional[List[float]] = None
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_kind"] = self.source_kind.value
        return d


@dataclass
class TrainingExample:
    """Verified positive, negative, or corrective training example with full provenance."""
    example_id: str
    task_domain: str
    inputs: Any
    target_output: Any
    observed_output: Optional[Any] = None
    is_failure: bool = False
    verified: bool = False
    failure_category: Optional[str] = None
    source_model: str = "synthetic"
    source_kind: DataSourceKind = DataSourceKind.SYNTHETIC_SIMULATION
    internal_activations: Optional[List[float]] = None
    confidence: Optional[float] = None
    license: str = "synthetic-local"
    quarantine_passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_kind"] = self.source_kind.value
        return d


@dataclass
class ExperimentReceipt:
    """Cryptographically bound scientific experiment receipt."""
    experiment_id: str
    title: str
    module: str
    hypothesis: str
    objective_category: ObjectiveCategory
    evidence_state: EvidenceState = EvidenceState.UNVERIFIED_OBSERVATION
    created_at_utc: float = field(default_factory=time.time)
    seeds: List[int] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_comparisons: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    failed_experiments: List[Dict[str, Any]] = field(default_factory=list)
    parent_checkpoint_id: Optional[str] = None
    descendant_checkpoint_id: Optional[str] = None
    code_version: str = "feat/sonapse-morphogenesis"
    notes: str = ""
    schema_version: str = SCHEMA_VERSION
    sha256_digest: str = ""

    def seal(self) -> "ExperimentReceipt":
        payload = self.to_dict(include_digest=False)
        self.sha256_digest = content_digest(payload)
        return self

    def verify(self) -> bool:
        if not self.sha256_digest:
            return False
        return self.sha256_digest == content_digest(self.to_dict(include_digest=False))

    def to_dict(self, include_digest: bool = True) -> Dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "title": self.title,
            "module": self.module,
            "hypothesis": self.hypothesis,
            "objective_category": self.objective_category.value,
            "evidence_state": self.evidence_state.value,
            "created_at_utc": self.created_at_utc,
            "seeds": list(self.seeds),
            "config": self.config,
            "metrics": self.metrics,
            "baseline_comparisons": self.baseline_comparisons,
            "uncertainty": self.uncertainty,
            "resource_usage": self.resource_usage,
            "failed_experiments": self.failed_experiments,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "descendant_checkpoint_id": self.descendant_checkpoint_id,
            "code_version": self.code_version,
            "notes": self.notes,
        }
        if include_digest:
            d["sha256_digest"] = self.sha256_digest
        return d


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def export_json_schemas() -> Dict[str, Dict[str, Any]]:
    """Export machine-readable JSON Schemas (Draft 2020-12) for the interchange types."""
    base = "https://sonapse.morphogenesis/schemas/v1"
    return {
        "neuron_morphology": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"{base}/neuron_morphology.json",
            "title": "NeuronMorphologySpec",
            "type": "object",
            "required": ["schema_version", "morphology_id", "family", "name", "fidelity", "compartments", "edges"],
            "properties": {
                "schema_version": {"type": "string"},
                "morphology_id": {"type": "string"},
                "family": {"type": "string"},
                "name": {"type": "string"},
                "objective_category": {"enum": [e.value for e in ObjectiveCategory]},
                "fidelity": {"type": "integer", "minimum": 1, "maximum": 5},
                "dale_sign": {"enum": [-1, 1]},
                "compartments": {"type": "array", "items": {"type": "object"}},
                "edges": {"type": "array", "items": {"type": "object"}},
                "organelles": {"type": "array", "items": {"type": "object"}},
            },
        },
        "developmental_genome": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"{base}/developmental_genome.json",
            "title": "DevelopmentalGenome",
            "type": "object",
            "required": ["schema_version", "genome_id", "generation", "connectome"],
            "properties": {
                "schema_version": {"type": "string"},
                "genome_id": {"type": "string"},
                "generation": {"type": "integer", "minimum": 0},
                "parent_ids": {"type": "array", "items": {"type": "string"}},
                "connectome": {"type": "object"},
                "growth_rules": {"type": "object"},
                "pruning_rules": {"type": "object"},
                "differentiation_rules": {"type": "object"},
                "metabolic_budget": {"type": "number", "minimum": 0.0},
            },
        },
        "training_example": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"{base}/training_example.json",
            "title": "TrainingExample",
            "type": "object",
            "required": ["schema_version", "example_id", "task_domain", "inputs", "target_output", "source_kind"],
            "properties": {
                "example_id": {"type": "string"},
                "task_domain": {"type": "string"},
                "is_failure": {"type": "boolean"},
                "verified": {"type": "boolean"},
                "source_kind": {"enum": [e.value for e in DataSourceKind]},
                "quarantine_passed": {"type": "boolean"},
            },
        },
        "experiment_receipt": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"{base}/experiment_receipt.json",
            "title": "ExperimentReceipt",
            "type": "object",
            "required": ["schema_version", "experiment_id", "title", "module", "hypothesis", "objective_category", "evidence_state", "sha256_digest"],
            "properties": {
                "experiment_id": {"type": "string"},
                "objective_category": {"enum": [e.value for e in ObjectiveCategory]},
                "evidence_state": {"enum": [e.value for e in EvidenceState]},
                "sha256_digest": {"type": "string"},
            },
        },
    }
