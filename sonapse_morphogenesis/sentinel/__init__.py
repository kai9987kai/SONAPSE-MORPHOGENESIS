"""SONAPSE SENTINEL — Independent Evaluation, Discovery Engine & Safety Governance."""
from .evaluation import (
    FORBIDDEN_AUTONOMOUS_ACTIONS,
    GovernanceController,
    benjamini_hochberg_fdr,
    classify_evidence_state,
    generate_discovery_report,
    paired_bootstrap_ci,
    render_comparison_svg,
    run_matched_baseline_suite,
)

__all__ = [
    "FORBIDDEN_AUTONOMOUS_ACTIONS",
    "GovernanceController",
    "paired_bootstrap_ci",
    "benjamini_hochberg_fdr",
    "run_matched_baseline_suite",
    "classify_evidence_state",
    "render_comparison_svg",
    "generate_discovery_report",
]
