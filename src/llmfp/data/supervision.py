"""Target alignment and right padding derived in Lesson 22.

These functions consume token IDs, not text or role names. Tokenization, chat
formatting, selection of supervised spans, and stopping rules belong to the
caller. No model or tokenizer configuration is changed here.
"""

from collections.abc import Sequence

import torch


def shift_supervised_tokens(
    token_ids: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align one full token sequence and its supervision mask for causal LM.

    Args:
        token_ids: (L,) nonnegative int64 IDs, including prompt and response.
        target_mask: (L,) Boolean selection of tokens to predict, aligned to
            the FULL sequence rather than predictor positions.
        ignore_index: Negative label sentinel used only by the loss.

    Returns:
        Independent (T=L-1,) input and label tensors on the original device.
        Input position j predicts token_ids[j+1]. A label is retained exactly
        when target_mask[j+1] is true. The first full-sequence token has no
        predictor, so target_mask[0] cannot supply supervision.

    Raises:
        ValueError: If shapes, dtypes, IDs, devices, or supervision are invalid.
    """
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool) or ignore_index >= 0:
        raise ValueError("ignore_index must be a negative integer.")
    if token_ids.ndim != 1 or token_ids.numel() < 2:
        raise ValueError("token_ids must have shape (L,) with L >= 2.")
    if token_ids.dtype != torch.long or (token_ids < 0).any().item():
        raise ValueError("token_ids must contain nonnegative int64 IDs.")
    if target_mask.shape != token_ids.shape or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be Boolean with the full token_ids shape.")
    if target_mask.device != token_ids.device:
        raise ValueError("token_ids and target_mask must share a device.")
    if not target_mask[1:].any().item():
        raise ValueError("At least one predictable target must be supervised.")

    # (T=L-1,): predictor inputs; keep inputs independent of the caller's data.
    inputs = token_ids[:-1].clone()
    # (T=L-1,): next-token labels, with the mask shifted by the same amount.
    labels = token_ids[1:].masked_fill(~target_mask[1:], ignore_index)
    return inputs, labels


def pad_supervised_batch(
    examples: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_token_id: int,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad already shifted input/label pairs without shifting again.

    Args:
        examples: Nonempty sequence of (inputs, labels) pairs. Each pair has
            shape (T_i,), dtype int64, and at least one supervised target.
            All tensors must be on one device. Labels already contain the
            chosen ignore_index at unsupervised positions.
        pad_token_id: Nonnegative placeholder ID in the MODEL's vocabulary.
            It may also occur in real text; padding is identified by length,
            never by token value. Vocabulary upper bounds are the caller's
            responsibility because this function does not know the model.
        ignore_index: Negative sentinel used for prompt/padding labels.

    Returns:
        (inputs, labels), each (B, T_max), with right padding. Inputs use
        pad_token_id; added labels use ignore_index. No attention mask is
        returned. Without a padding attention mask, this contract is suitable
        for independent, strictly causal sequences, not left padding or packing.

    Raises:
        ValueError: If the batch, pairs, IDs, devices, or supervision are invalid.
    """
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool) or ignore_index >= 0:
        raise ValueError("ignore_index must be a negative integer.")
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer.")
    if not examples:
        raise ValueError("examples must not be empty.")

    device = examples[0][0].device
    for inputs, labels in examples:
        if inputs.ndim != 1 or inputs.numel() == 0 or labels.shape != inputs.shape:
            raise ValueError("Each input/label pair must have matching nonempty (T_i,) shapes.")
        if inputs.dtype != torch.long or labels.dtype != torch.long:
            raise ValueError("Inputs and labels must use int64.")
        if inputs.device != device or labels.device != device:
            raise ValueError("All inputs and labels must share a device.")
        if (inputs < 0).any().item():
            raise ValueError("Input IDs must be nonnegative; ignore_index belongs only in labels.")
        if ((labels < 0) & (labels != ignore_index)).any().item():
            raise ValueError("Labels must be nonnegative IDs or ignore_index.")
        if not labels.ne(ignore_index).any().item():
            raise ValueError("Every example needs at least one supervised target.")

    batch_size = len(examples)
    max_length = max(inputs.numel() for inputs, _ in examples)
    # (B, T_max): sequences and already aligned predictor-input positions.
    batch_inputs = torch.full(
        (batch_size, max_length), pad_token_id, dtype=torch.long, device=device
    )
    # (B, T_max): next-token targets; all unfilled positions are ignored.
    batch_labels = torch.full_like(batch_inputs, ignore_index)
    for row, (inputs, labels) in enumerate(examples):
        length = inputs.numel()
        batch_inputs[row, :length] = inputs
        batch_labels[row, :length] = labels
    return batch_inputs, batch_labels


__all__ = ["shift_supervised_tokens", "pad_supervised_batch"]
