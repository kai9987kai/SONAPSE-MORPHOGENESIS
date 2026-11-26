"""SONAPSE EMERGENCE — Causal Emergence (GhostInTheMachine), CROSSWIRE Safety & FAULTLINE Ensembles."""
from .causal_emergence import (
    analyze_closed_loop_efference_spectrum,
    compute_dynamical_independence,
    compute_gaussian_mmi_pid,
    extract_maf_slow_subspace,
    gaussian_mi,
    run_causal_emergence_and_authorship_suite,
    run_cross_yoked_twin_assay,
)
from .crosswire_safety import (
    CrosswireVersionSpaceAdapter,
    FaultlineJointFailureSelector,
    run_crosswire_and_faultline_study,
)

__all__ = [
    "gaussian_mi",
    "extract_maf_slow_subspace",
    "compute_dynamical_independence",
    "compute_gaussian_mmi_pid",
    "analyze_closed_loop_efference_spectrum",
    "run_cross_yoked_twin_assay",
    "run_causal_emergence_and_authorship_suite",
    "CrosswireVersionSpaceAdapter",
    "FaultlineJointFailureSelector",
    "run_crosswire_and_faultline_study",
]
