"""SONAPSE ARCHITECT — Independent Autonomous Research Director, 19 Learning Rules & 11 Experts."""
from .director import AutonomousResearchDirector, CandidateTrainingPack, construct_training_pack
from .experts import (
    EXPERT_ROLES,
    ExpertMetaAuditor,
    ExpertPrediction,
    InternalExpertPanel,
    LearnedResearchExpert,
)
from .learning_rules import (
    LEARNING_ALGORITHMS,
    DynamicLearningEngine,
    LearningScheduleVersion,
    benchmark_developmental_stage_learning_rules,
)

__all__ = [
    "LEARNING_ALGORITHMS",
    "LearningScheduleVersion",
    "DynamicLearningEngine",
    "benchmark_developmental_stage_learning_rules",
    "EXPERT_ROLES",
    "ExpertPrediction",
    "LearnedResearchExpert",
    "ExpertMetaAuditor",
    "InternalExpertPanel",
    "CandidateTrainingPack",
    "construct_training_pack",
    "AutonomousResearchDirector",
]
