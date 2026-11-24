"""SONAPSE MORPHOGENESIS — Autonomous Developmental Connectome Intelligence Platform.

Unified, CPU-first, backward-compatible research ecosystem extending Sonapse-LAB
into physically structured multi-compartment neurons, synthetic organelles,
automated connectome evolution, multimodal signal mixing, higher-dimensional
environments, emergent affect-like organization, failure-based Supermix distillation,
cross-connectome symbiosis, autonomous scientific direction, and independent oversight.
"""
from __future__ import annotations

__version__ = "1.0.0"
SCHEMA_VERSION = "morphogenesis-schema-v1"

from .core.interchange import (
    ActivationStream,
    CompartmentEdge,
    CompartmentKind,
    CompartmentNode,
    ConnectomeEdge,
    ConnectomeGraph,
    DataSourceKind,
    DevelopmentalGenome,
    EvidenceState,
    ExperimentReceipt,
    FidelityLevel,
    NeuronMorphologySpec,
    ObjectiveCategory,
    OrganelleAttachment,
    ReceptorDistribution,
    SensoryExperience,
    TrainingExample,
    export_json_schemas,
)

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "ObjectiveCategory",
    "FidelityLevel",
    "CompartmentKind",
    "DataSourceKind",
    "EvidenceState",
    "ReceptorDistribution",
    "CompartmentNode",
    "CompartmentEdge",
    "OrganelleAttachment",
    "NeuronMorphologySpec",
    "DevelopmentalGenome",
    "ConnectomeEdge",
    "ConnectomeGraph",
    "ActivationStream",
    "SensoryExperience",
    "TrainingExample",
    "ExperimentReceipt",
    "export_json_schemas",
]
