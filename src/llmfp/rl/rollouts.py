"""Frozen PPO rollout records and aligned sequence minibatch selection."""

from dataclasses import dataclass

import torch

from .alignment import _validate_action_positions


@dataclass(frozen=True)
class RolloutBatch:
    """Store collection facts and fixed learning targets, not current predictions.

    token_ids: (B, L). Action metadata, log-probabilities, rewards, advantages,
    value targets and boundary masks: (B, T). rollout_values: (B, T+1).
    Action positions must be contiguous and action masks right padded.

    The dataclass prevents reassignment of fields; Tensor contents remain
    mutable, so callers must not modify snapshot data during PPO updates.
    """

    token_ids: torch.Tensor
    action_token_positions: torch.Tensor
    action_mask: torch.Tensor
    old_log_probabilities: torch.Tensor
    rewards: torch.Tensor
    rollout_values: torch.Tensor
    bootstrap_mask: torch.Tensor
    continuation_mask: torch.Tensor
    advantages: torch.Tensor
    value_targets: torch.Tensor

    def __post_init__(self) -> None:
        """Validate collection-time tensor shapes, masks and detachment."""
        if self.token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (B, L).")

        if self.token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must contain integer indices.")

        if self.action_token_positions.ndim != 2:
            raise ValueError("action_token_positions must have shape (B, T).")

        batch_size, sequence_length = self.token_ids.shape
        num_actions = self.action_token_positions.shape[1]

        _validate_action_positions(
            self.action_token_positions,
            self.action_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=self.token_ids.device,
        )

        for name in (
            "old_log_probabilities", "rewards", "bootstrap_mask",
            "continuation_mask", "advantages", "value_targets",
        ):
            tensor: torch.Tensor = getattr(self, name)
            if tensor.shape != (batch_size, num_actions):
                raise ValueError(f"{name} must have shape (B, T).")
            if tensor.device != self.token_ids.device:
                raise ValueError(f"{name} must match the token_ids device.")

        if self.rollout_values.shape != (batch_size, num_actions + 1):
            raise ValueError("rollout_values must have shape (B, T+1).")
        if self.rollout_values.device != self.token_ids.device:
            raise ValueError("rollout_values must match the token_ids device.")

        for name in (
            "old_log_probabilities", "rewards", "rollout_values",
            "advantages", "value_targets",
        ):
            tensor = getattr(self, name)
            if not torch.is_floating_point(tensor):
                raise ValueError(f"{name} must be floating point.")
            if tensor.requires_grad:
                raise ValueError(f"{name} must be detached from autograd.")

        for name in ("bootstrap_mask", "continuation_mask"):
            mask: torch.Tensor = getattr(self, name)
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError(f"{name} must contain only 0 or 1.")
            if bool((mask[~self.action_mask] != 0).any()):
                raise ValueError(f"{name} must be zero in padded action slots.")

        if bool(((self.continuation_mask == 1) & (self.bootstrap_mask == 0)).any()):
            raise ValueError("Continuation requires bootstrapping to be enabled.")


def select_rollout_sequences(
    batch: RolloutBatch, sequence_indices: torch.Tensor
) -> RolloutBatch:
    """Select whole sequences with identical indices for every tensor field."""
    if sequence_indices.ndim != 1 or sequence_indices.numel() == 0:
        raise ValueError("sequence_indices must be a nonempty 1D tensor.")
    if sequence_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("sequence_indices must contain integer indices.")
    if sequence_indices.device != batch.token_ids.device:
        raise ValueError("sequence_indices must match the rollout device.")
    if bool(((sequence_indices < 0) | (sequence_indices >= batch.token_ids.shape[0])).any()):
        raise ValueError("sequence_indices are out of range.")

    return RolloutBatch(
        token_ids=batch.token_ids[sequence_indices],
        action_token_positions=batch.action_token_positions[sequence_indices],
        action_mask=batch.action_mask[sequence_indices],
        old_log_probabilities=batch.old_log_probabilities[sequence_indices],
        rewards=batch.rewards[sequence_indices],
        rollout_values=batch.rollout_values[sequence_indices],
        bootstrap_mask=batch.bootstrap_mask[sequence_indices],
        continuation_mask=batch.continuation_mask[sequence_indices],
        advantages=batch.advantages[sequence_indices],
        value_targets=batch.value_targets[sequence_indices],
    )


__all__ = ["RolloutBatch", "select_rollout_sequences"]
