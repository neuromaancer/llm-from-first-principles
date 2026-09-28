"""Autoregressive generation helpers."""

import torch
import torch.nn as nn

from .sampling import sample_next_token


@torch.no_grad()
def generate_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate tokens greedily without a cache.

    This reference implementation recomputes the complete sequence at every
    decoding step.

    Args:
        model: Decoder-only language model returning `(B, T, V)` logits.
        input_ids: Prompt token IDs with shape `(B, T)`.
        max_new_tokens: Number of tokens to append.

    Returns:
        Prompt followed by generated token IDs.
    """
    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be non-negative."
        )

    generated: torch.Tensor = input_ids

    was_training: bool = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            logits = model(generated)

            if not isinstance(
                logits,
                torch.Tensor,
            ):
                raise RuntimeError(
                    "Expected model to return logits without a cache."
                )

            next_token_logits: torch.Tensor = (
                logits[:, -1, :]
            )

            next_token: torch.Tensor = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )
    finally:
        if was_training:
            model.train()

    return generated


@torch.no_grad()
def generate_greedy_cached(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate greedily using a model's autoregressive cache.

    The model is expected to follow the project GPT cache interface:
    `model(ids, past_key_values=..., use_cache=True)`.

    Args:
        model: Decoder-only model supporting the project cache interface.
        input_ids: Prompt token IDs with shape `(B, T)`.
        max_new_tokens: Number of tokens to append.

    Returns:
        Prompt followed by generated token IDs.
    """
    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be non-negative."
        )

    if max_new_tokens == 0:
        return input_ids

    generated: torch.Tensor = input_ids

    was_training: bool = model.training
    model.eval()

    try:
        prefill_result = model(
            input_ids,
            use_cache=True,
        )

        if not isinstance(
            prefill_result,
            tuple,
        ):
            raise RuntimeError(
                "Expected cached model output."
            )

        logits, cache = prefill_result

        for step in range(max_new_tokens):
            next_token_logits: torch.Tensor = (
                logits[:, -1, :]
            )

            next_token: torch.Tensor = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            if step == max_new_tokens - 1:
                break

            decode_result = model(
                next_token,
                past_key_values=cache,
                use_cache=True,
            )

            if not isinstance(
                decode_result,
                tuple,
            ):
                raise RuntimeError(
                    "Expected cached model output."
                )

            logits, cache = decode_result
    finally:
        if was_training:
            model.train()

    return generated


@torch.no_grad()
def generate_sampled_cached(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate sampled tokens using an autoregressive cache.

    Args:
        model: Decoder-only model supporting the project cache interface.
        input_ids: Prompt token IDs with shape `(B, T)`.
        max_new_tokens: Number of tokens to append.
        temperature: Positive sampling temperature.
        top_k: Optional top-k candidate count.
        top_p: Optional nucleus probability threshold.
        generator: Optional random generator for reproducible sampling.

    Returns:
        Prompt followed by generated token IDs.
    """
    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be non-negative."
        )

    if max_new_tokens == 0:
        return input_ids

    generated: torch.Tensor = input_ids

    was_training: bool = model.training
    model.eval()

    try:
        prefill_result = model(
            input_ids,
            use_cache=True,
        )

        if not isinstance(
            prefill_result,
            tuple,
        ):
            raise RuntimeError(
                "Expected cached model output."
            )

        logits, cache = prefill_result

        for step in range(max_new_tokens):
            next_token_logits: torch.Tensor = (
                logits[:, -1, :]
            )

            next_token: torch.Tensor = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            if step == max_new_tokens - 1:
                break

            decode_result = model(
                next_token,
                past_key_values=cache,
                use_cache=True,
            )

            if not isinstance(
                decode_result,
                tuple,
            ):
                raise RuntimeError(
                    "Expected cached model output."
                )

            logits, cache = decode_result
    finally:
        if was_training:
            model.train()

    return generated


__all__ = [
    "generate_greedy",
    "generate_greedy_cached",
    "generate_sampled_cached",
]
