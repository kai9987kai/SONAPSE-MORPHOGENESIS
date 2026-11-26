"""OMNI-MORPHOGENESIS Max Mode Architecture Subpackage.

Provides the purpose-built 3-tier nested oscillatory architecture (`OmniMorphogenesisMaxEngine`)
that activates and coordinates 100% of SONAPSE MORPHOGENESIS v1 + v2 and host Sonapse-LAB
capabilities simultaneously without destructive interference or memory drift.
"""
from .omni_architecture import (
    MAX_MODE_SCHEMA,
    HyperionServiceScheduler,
    OmniMaxModeConfig,
    OmniMorphogenesisMaxEngine,
    build_pan_catalog_connectome,
    render_max_mode_campaign_svg,
)

__all__ = [
    "MAX_MODE_SCHEMA",
    "OmniMaxModeConfig",
    "HyperionServiceScheduler",
    "build_pan_catalog_connectome",
    "OmniMorphogenesisMaxEngine",
    "render_max_mode_campaign_svg",
]
