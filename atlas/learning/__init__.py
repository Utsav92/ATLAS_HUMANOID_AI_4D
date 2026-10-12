"""Atlas Learning Package: Hybrid Imitation, Reinforcement Learning, Normalization, and Evaluation."""

from atlas.learning.normalization import ObservationActionNormalizer
from atlas.learning.metrics import MotionAccuracyEvaluator, MotionAccuracyReport
from atlas.learning.dagger_trainer import DAggerTrainer, DAggerIterationSummary
from atlas.learning.bc_ppo_pipeline import (
    EnhancedBCPlusPPOPipeline,
    BCPlusPPOPipeline,
    TemporalActorCriticPolicy,
    ActorCriticPolicy,
    CombinedMotionAccuracyLoss,
    LossBreakdown,
    EnhancedTrainingSummary
)

# Backwards compatibility alias
BCPlusPPOPipeline = EnhancedBCPlusPPOPipeline
ActorCriticPolicy = TemporalActorCriticPolicy

__all__ = [
    "ObservationActionNormalizer",
    "MotionAccuracyEvaluator",
    "MotionAccuracyReport",
    "DAggerTrainer",
    "DAggerIterationSummary",
    "EnhancedBCPlusPPOPipeline",
    "BCPlusPPOPipeline",
    "TemporalActorCriticPolicy",
    "ActorCriticPolicy",
    "CombinedMotionAccuracyLoss",
    "LossBreakdown",
    "EnhancedTrainingSummary",
]
