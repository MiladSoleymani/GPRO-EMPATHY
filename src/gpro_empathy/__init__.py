"""
GPRO Empathy: Group Relative Policy Optimization for Empathy Training
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .training.grpo_trainer import GPROEmpathyTrainer
from .utils.inference import EmpathyInference
from .data.dataset_loader import load_wassa_empathy, get_system_prompt
from .models.reward_functions import (
    semantic_sts_reward,
    empathy_model_reward,
    SemanticSimilarityReward,
    EmpathyModelReward,
)

__all__ = [
    "GPROEmpathyTrainer",
    "EmpathyInference",
    "load_wassa_empathy",
    "get_system_prompt",
    "semantic_sts_reward",
    "empathy_model_reward",
    "SemanticSimilarityReward",
    "EmpathyModelReward",
]