"""Reusable reinforcement-learning utilities."""

from .advantages import generalized_advantage_estimation, td_error
from .alignment import align_state_values_for_actions, rescore_actions_from_logits
from .policy_gradient import Reduction, reinforce_loss
from .ppo import categorical_entropy, masked_mean, ppo_policy_loss, ppo_value_loss
from .returns import discounted_returns
from .rollouts import RolloutBatch, select_rollout_sequences

__all__ = [
    "Reduction",
    "RolloutBatch",
    "align_state_values_for_actions",
    "categorical_entropy",
    "discounted_returns",
    "generalized_advantage_estimation",
    "masked_mean",
    "ppo_policy_loss",
    "ppo_value_loss",
    "reinforce_loss",
    "rescore_actions_from_logits",
    "select_rollout_sequences",
    "td_error",
]
