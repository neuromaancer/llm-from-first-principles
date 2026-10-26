# Lesson 22 — Extraction Audit

Date: 2026-10-26. Notebook source: [Lesson 22 at the user's closure commit](https://github.com/neuromaancer/llm-from-first-principles/blob/2a4935748b597e786e126554f1991379eb1c615c/lessons/22_supervised_fine_tuning.ipynb).

This audit applies the README's understand → implement → verify → extract → reuse rule. It supersedes the earlier blanket deferral of SFT extraction; it does not turn the toy experiment into a framework. The notebook and its saved outputs are unchanged by this maintenance.

## Accepted primitives

| Notebook mechanism | Reusable API | Contract and reason |
|---|---|---|
| 22.01–22.03: next-token shift and assistant-target selection | `llmfp.data.shift_supervised_tokens` | Accept already serialized full-sequence IDs `(L,)` and a Boolean target-selection mask `(L,)`; shift both once, returning independent `(L-1,)` inputs/labels. No role IDs, text template, or tokenizer assumptions. |
| 22.04 and 22.05.2: right-padded input/label batches | `llmfp.data.pad_supervised_batch` | Accept already shifted `(T_i,)` pairs; right-pad inputs with a caller-provided vocabulary ID and labels with a negative ignore sentinel. No second shift or token-value-based padding detection. |
| 22.05.3: per-example assistant-token NLL | `llmfp.training.per_example_sft_nll` | Consume aligned `(B,T,V)` logits and `(B,T)` labels; retain autograd; reject empty supervised rows; return `(B,)` means. This same scoring contract can be reused when comparing candidate responses. |

These are generalized tensor boundaries around mechanisms already exercised in the notebook, not new training algorithms. The additional input validation makes misuse explicit. All three APIs use `ignore_index=-100` by default and support an explicit negative sentinel.

## Deliberately retained in the notebook

- `encode_sft_pair`: teaching-specific symbolic role IDs and EOS convention.
- `encode_character_sft_pair`: the literal `User: ...\nAssistant: ` format and a character-tokenizer boundary calculation. Do not assume separately tokenized prompt/completion pieces compose identically for arbitrary subword tokenizers.
- Four training examples, three probe instructions, 40-step training configuration, and comparison runners: experimental choices rather than reusable mechanisms.
- `generate_sft_response`: a newline-based first-line evaluator using a fixed character budget, not a general EOS-stopping interface. Reuse `generate_greedy_cached` from the existing package.
- Plotly figure construction and checkpoint experiment metadata: teaching/reporting content.

No new `SFTTrainer`, model wrapper, tokenizer, generation engine, or checkpoint schema is introduced. The existing `language_model_training_step` already calls `F.cross_entropy` with the default `ignore_index=-100`; it can consume correctly prepared aligned labels. Callers must still ensure nonempty supervision.

## Tensor and objective boundaries

`shift_supervised_tokens` takes a mask aligned to the **full** sequence. Token `j` is supervised at predictor position `j-1`; full-sequence position zero cannot be predicted. EOS inclusion is the caller's explicit choice. Returned inputs are independent of caller storage.

`pad_supervised_batch` takes **already shifted** inputs/labels. Padding ID may also appear in real text; original lengths define the boundary. All tensor pairs must be nonempty int64 tensors on one device, with at least one selected target per row. The caller must ensure input IDs, including the padding placeholder, are inside the model vocabulary. No attention mask is returned: the current model's independent strictly causal right-padded training contract is supported, not left padding, packed samples, or noncausal processing.

`per_example_sft_nll` does not shift or detach. With row means `loss_i` and supervised counts `n_i`, `(loss_i * n_i).sum() / n_i.sum()` reproduces the unweighted supervised-token batch objective. `loss_i.mean()` is a different, example-weighted objective when lengths differ. No class weighting or label smoothing is applied. Ignored labels are never passed to embedding lookup.

## Minimal reuse example

```python
import torch
import torch.nn.functional as F

from llmfp.data import pad_supervised_batch, shift_supervised_tokens
from llmfp.training import per_example_sft_nll

# (L=6,): one serialized prompt/response with explicit target selection.
tokens = torch.tensor([0, 1, 2, 3, 4, 5])
target_mask = torch.tensor([False, False, False, False, True, True])
example = shift_supervised_tokens(tokens, target_mask)

# (B=1, T=5): already aligned predictor inputs and masked labels.
inputs, labels = pad_supervised_batch([example], pad_token_id=0)

# (B=1, T=5, V=6): stand-in predictions for checking the loss contract.
logits = torch.zeros(1, 5, 6, requires_grad=True)

# (B=1,): one differentiable mean per sequence.
row_nll = per_example_sft_nll(logits, labels)
counts = labels.ne(-100).sum(dim=1)
loss = (row_nll * counts).sum() / counts.sum()
reference = F.cross_entropy(logits.reshape(-1, 6), labels.reshape(-1))
torch.testing.assert_close(loss, reference)
loss.backward()
```

## Focused verification

`tests/test_supervised_primitives.py` contains 12 checks for alignment and storage independence, template-independent/discontiguous selection, no double shift, right padding, placeholder IDs appearing in real text, invalid/empty supervision, per-example normalization, Boolean/ignore-index loss and gradient equivalence, ignored-position invariance, and a real parameter update.

The 12 tests passed on CPU with Python 3.13.5 / PyTorch 2.10.0. They were executed against the extracted leaf modules in isolation using namespace-package paths; full package initialization and the existing PPO suite were not rerun in that check. The test file itself uses normal package imports and Python `unittest`, so normal repository discovery exercises it with the existing package:

```bash
uv run python -m unittest discover -s tests -v
```

The project requests `torch>=2.14.0`; that environment and CUDA were not available for this extraction. The user's local pretrained/SFT checkpoints were not available either. Saved notebook results support the recorded experiment, but are not a newly executed full-notebook Run All.

## Checkpoint boundary

The notebook's `step_00040.pt` stores a plain configuration dictionary, ordered tokenizer vocabulary, model weights, and selected experiment metadata. Its saved output reports a successful parameter-restoration check. It contains no optimizer/RNG state, so it is a model-restoration artifact, not a claim of exact training resumption. Keep the weights out of Git.

