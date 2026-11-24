"""SUPERMIX ANTIPODE — Failure-Based Neural Distillation & External AI-Assisted Training."""
from .external_advisor import AdvisorRecommendation, AdvisorSessionConfig, ExternalAIAssistedTrainer
from .instrumentation import (
    FAILURE_CATEGORIES,
    TidalMoEInstrumentor,
    classify_response_outcome,
    load_real_expanse_failure_corpus,
)
from .negative_distillation import (
    NEGATIVE_TRAINING_OBJECTIVES,
    CrossArchitectureAdapter,
    compute_negative_objective,
    localize_causal_failure_channels,
    run_antipode_six_way_study,
)

__all__ = [
    "FAILURE_CATEGORIES",
    "classify_response_outcome",
    "TidalMoEInstrumentor",
    "load_real_expanse_failure_corpus",
    "NEGATIVE_TRAINING_OBJECTIVES",
    "CrossArchitectureAdapter",
    "localize_causal_failure_channels",
    "compute_negative_objective",
    "run_antipode_six_way_study",
    "AdvisorSessionConfig",
    "AdvisorRecommendation",
    "ExternalAIAssistedTrainer",
]
