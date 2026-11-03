"""Checkpoint-free composition: tokenizer -> tiny GPT -> supervised fitting -> DPO.

Run from the repository root: uv run python examples/tiny_sft_dpo.py
These three synthetic examples illustrate wiring, not generalization.
"""

import argparse
import copy
import json

import torch

from llmfp.data import CharacterTokenizer, pad_supervised_batch, shift_supervised_tokens
from llmfp.generation import generate_greedy_cached
from llmfp.models import GPT, GPTConfig
from llmfp.training import completion_sequence_log_probs, dpo_loss, per_example_sft_nll

PAIRS = (
    ("Say yes", "Yes", "No"),
    ("Say no", "No", "Yes"),
    ("Say hello", "Hello", "Farewell"),
)


def encode_candidates(
    tokenizer: CharacterTokenizer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (6, T) inputs/labels, chosen then rejected for each pair."""
    examples = []
    for instruction, chosen, rejected in PAIRS:
        prompt = f"User: {instruction}\nAssistant: "
        for response in (chosen, rejected):
            ids = torch.tensor(
                tokenizer.encode(prompt + response + "\n"),
                dtype=torch.long,
                device=device,
            )
            targets = torch.zeros_like(ids, dtype=torch.bool)
            targets[len(tokenizer.encode(prompt)):] = True
            examples.append(shift_supervised_tokens(ids, targets))
    return pad_supervised_batch(examples, pad_token_id=tokenizer.token_to_id[" "])


@torch.no_grad()
def responses(model: GPT, tokenizer: CharacterTokenizer) -> dict[str, str]:
    """Read the first generated line; flag budget cutoffs rather than hide them."""
    device = next(model.parameters()).device
    result = {}
    for instruction, _, _ in PAIRS:
        ids = torch.tensor(
            [tokenizer.encode(f"User: {instruction}\nAssistant: ")],
            dtype=torch.long,
            device=device,
        )
        generated = generate_greedy_cached(model, ids, max_new_tokens=12)
        text = tokenizer.decode(generated[0, ids.shape[1]:].cpu().tolist())
        result[instruction] = text.split("\n", 1)[0]
        if "\n" not in text:
            result[instruction] += " [budget exhausted]"
    return result


def run_demo(
    *, sft_steps: int = 80, dpo_steps: int = 20, device: str = "cpu", seed: int = 7
) -> dict:
    """Fit a synthetic batch and optimize its preference pairs; download nothing."""
    if sft_steps < 1 or dpo_steps < 1:
        raise ValueError("Both training step counts must be positive.")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    torch.manual_seed(seed)
    corpus = "".join(
        f"User: {prompt}\nAssistant: {answer}\n"
        for prompt, chosen, rejected in PAIRS
        for answer in (chosen, rejected)
    )
    tokenizer = CharacterTokenizer.from_text(corpus)
    inputs, labels = encode_candidates(tokenizer, target_device)
    model = GPT(GPTConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=32,
        num_query_heads=4,
        num_kv_heads=2,
        num_layers=1,
        hidden_dim=64,
    )).to(target_device)

    # 1. Preferred-completion supervision from random weights. This illustrates
    # SFT-style plumbing; it is NOT fine-tuning a pretrained language model.
    sft_inputs, sft_labels = inputs[0::2], labels[0::2]
    with torch.no_grad():
        initial_nll = per_example_sft_nll(model(sft_inputs), sft_labels).mean().item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    for _ in range(sft_steps):
        loss = per_example_sft_nll(model(sft_inputs), sft_labels).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_nll = per_example_sft_nll(model(sft_inputs), sft_labels).mean().item()
    sft_answers = responses(model, tokenizer)

    # 2. Separate frozen reference and one-time sequence-score cache.
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    reference_weights = {n: v.clone() for n, v in reference.state_dict().items()}
    with torch.no_grad():
        ref_logps = completion_sequence_log_probs(reference(inputs), labels)

    # 3. Direct preference optimization: only actor sequence logP receives grad.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    initial_dpo = None
    for _ in range(dpo_steps):
        logps = completion_sequence_log_probs(model(inputs), labels)
        loss = dpo_loss(
            logps[0::2], logps[1::2], ref_logps[0::2], ref_logps[1::2], beta=0.5
        )
        if initial_dpo is None:
            initial_dpo = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logps = completion_sequence_log_probs(model(inputs), labels)
        final_dpo = dpo_loss(
            logps[0::2], logps[1::2], ref_logps[0::2], ref_logps[1::2], beta=0.5
        ).item()
        change = (
            (logps[0::2] - logps[1::2]) - (ref_logps[0::2] - ref_logps[1::2])
        ).cpu().tolist()
    return {
        "device": str(target_device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "candidate_shape": list(inputs.shape),
        "sft_mean_nll": {"before": initial_nll, "after": final_nll},
        "dpo_loss": {"before": initial_dpo, "after": final_dpo},
        "relative_margin_change": dict(zip((p[0] for p in PAIRS), change)),
        "sft_responses": sft_answers,
        "dpo_responses": responses(model, tokenizer),
        "reference_unchanged": all(
            torch.equal(v, reference_weights[n])
            for n, v in reference.state_dict().items()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-steps", type=int, default=80)
    parser.add_argument("--dpo-steps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        result = run_demo(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    print("Toy training examples only: lower loss is not evidence of generalization.")


if __name__ == "__main__":
    main()
