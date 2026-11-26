"""SONAPSE ARCHITECT — Independent Autonomous Research Director, 33 Learning Rules & 11 Experts."""
from .director import AutonomousResearchDirector, CandidateTrainingPack, construct_training_pack
from .experts import (
    EXPERT_ROLES,
    ExpertMetaAuditor,
    ExpertPrediction,
    InternalExpertPanel,
    LearnedResearchExpert,
)
from .learning_rules import (
    ALL_LEARNING_ALGORITHMS,
    EXTENDED_LEARNING_ALGORITHMS,
    LEARNING_ALGORITHMS,
    DynamicLearningEngine,
    LearningScheduleVersion,
    benchmark_developmental_stage_learning_rules,
    benchmark_extended_connectome_learning_suite,
)

__all__ = [
    "LEARNING_ALGORITHMS",
    "EXTENDED_LEARNING_ALGORITHMS",
    "ALL_LEARNING_ALGORITHMS",
    "LearningScheduleVersion",
    "DynamicLearningEngine",
    "benchmark_developmental_stage_learning_rules",
    "benchmark_extended_connectome_learning_suite",
    "EXPERT_ROLES",
    "ExpertPrediction",
    "LearnedResearchExpert",
    "ExpertMetaAuditor",
    "InternalExpertPanel",
    "CandidateTrainingPack",
    "construct_training_pack",
    "AutonomousResearchDirector",
]
