"""Causal token log-probability rescoring and Critic state-value alignment.

Both helpers expect complete sequences, contiguous generated tokens and
right-padded action slots; the state-value helper returns (B, T+1).
"""

import torch


def _validate_action_positions(
    action_token_positions: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> None:
    """Validate right-padded contiguous generated-action token positions."""
    if action_token_positions.ndim != 2:
        raise ValueError("action_token_positions must have shape (B, T).")

    if action_token_positions.shape[0] != batch_size:
        raise ValueError("Action positions must match the batch size.")

    if action_mask.shape != action_token_positions.shape:
        raise ValueError("action_mask must match action_token_positions.")

    if action_mask.dtype != torch.bool:
        raise ValueError("action_mask must have boolean dtype.")

    if action_token_positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Action positions must have integer dtype.")

    if action_token_positions.device != device or action_mask.device != device:
        raise ValueError("Action positions and mask must match model device.")

    if batch_size == 0 or action_token_positions.shape[1] == 0:
        raise ValueError("Each batch must contain sequences and action slots.")

    if sequence_length < 2:
        raise ValueError("A predictor token must precede each generated action.")

    if not bool(action_mask[:, 0].all()):
        raise ValueError("Each sequence must have at least one valid action.")

    if bool((action_mask[:, 1:] & ~action_mask[:, :-1]).any()):
        raise ValueError("Action masks must have right-side padding only.")

    valid_positions = action_token_positions[action_mask]
    if bool(((valid_positions < 1) | (valid_positions >= sequence_length)).any()):
        raise ValueError("Valid action positions must lie between 1 and L - 1.")

    expected = action_token_positions[:, :1] + torch.arange(
        action_token_positions.shape[1], device=device
    ).unsqueeze(0)
    if not torch.equal(valid_positions, expected[action_mask]):
        raise ValueError("Generated action token positions must be consecutive.")


def rescore_actions_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    action_token_positions: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract current-policy log-probabilities for stored sampled actions."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, L, V).")

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (B, L).")

    if action_token_positions.ndim != 2:
        raise ValueError("action_token_positions must have shape (B, T).")

    if action_mask.shape != action_token_positions.shape:
        raise ValueError("action_mask must match action_token_positions.")

    if logits.shape[:2] != token_ids.shape:
        raise ValueError(
            "The batch and sequence dimensions of logits and token_ids must match."
        )

    if action_mask.dtype != torch.bool:
        raise ValueError("action_mask must have boolean dtype.")

    valid_positions = action_token_positions[action_mask]

    if torch.any(
        (valid_positions < 1) | (valid_positions >= token_ids.shape[1])
    ):
        raise ValueError("Valid action positions must lie between 1 and L - 1.")

    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must contain integer indices.")

    if logits.device != token_ids.device:
        raise ValueError("Logits and token_ids must be on the same device.")

    _validate_action_positions(
        action_token_positions,
        action_mask,
        batch_size=token_ids.shape[0],
        sequence_length=token_ids.shape[1],
        device=token_ids.device,
    )

    safe_action_token_positions = torch.where(
        action_mask,
        action_token_positions,
        torch.ones_like(action_token_positions),
    )

    policy_positions = safe_action_token_positions - 1

    all_log_probabilities = torch.log_softmax(logits, dim=-1)

    batch_indices = torch.arange(
        token_ids.shape[0], device=token_ids.device
    ).unsqueeze(1)

    action_token_ids = token_ids[batch_indices, safe_action_token_positions]

    if bool(((action_token_ids < 0) | (action_token_ids >= logits.shape[-1])).any()):
        raise ValueError("Selected token IDs must lie within the vocabulary.")

    selected_log_probabilities = all_log_probabilities[
        batch_indices, policy_positions, action_token_ids
    ]

    return torch.where(
        action_mask,
        selected_log_probabilities,
        torch.zeros_like(selected_log_probabilities),
    )


def align_state_values_for_actions(
    values_by_token: torch.Tensor,
    action_token_positions: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Align critic values with contiguous generated actions and their next states."""

    if values_by_token.ndim != 2:
        raise ValueError("values_by_token must have shape (B, L).")

    if not torch.is_floating_point(values_by_token):
        raise ValueError("Critic state values must be floating point.")

    _validate_action_positions(
        action_token_positions,
        action_mask,
        batch_size=values_by_token.shape[0],
        sequence_length=values_by_token.shape[1],
        device=values_by_token.device,
    )

    # V(s_0) is predicted before the first generated action.
    first_state_positions = action_token_positions[:, :1] - 1

    # V(s_0), V(s_1), ..., V(s_T).
    value_positions = torch.cat(
        [first_state_positions, action_token_positions], dim=1
    )

    value_mask = torch.cat([action_mask[:, :1], action_mask], dim=1)

    safe_positions = torch.where(
        value_mask, value_positions, torch.zeros_like(value_positions)
    )

    batch_indices = torch.arange(
        values_by_token.shape[0], device=values_by_token.device
    ).unsqueeze(1)

    aligned_values = values_by_token[batch_indices, safe_positions]

    return torch.where(
        value_mask,
        aligned_values,
        torch.zeros_like(aligned_values),
    )


__all__ = ["align_state_values_for_actions", "rescore_actions_from_logits"]
