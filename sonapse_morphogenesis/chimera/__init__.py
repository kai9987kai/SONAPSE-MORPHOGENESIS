"""SONAPSE CHIMERA — Connectome Mixing, Alignment & Grafting Laboratory."""
from .fusion_lab import (
    CHIMERA_MODES,
    ChimeraMergeReceipt,
    audit_connectome_symmetry_and_fingerprints,
    estimate_correspondence,
    merge_connectomes,
    run_chimera_comparative_benchmark,
)

__all__ = [
    "CHIMERA_MODES",
    "ChimeraMergeReceipt",
    "estimate_correspondence",
    "merge_connectomes",
    "run_chimera_comparative_benchmark",
    "audit_connectome_symmetry_and_fingerprints",
]
