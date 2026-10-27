# Lesson 23 — Extraction Audit

Date: 2026-10-27. Teaching source: [Lesson 23 at the user's last evaluation commit](https://github.com/neuromaancer/llm-from-first-principles/blob/02f332f73c57680d46db99f8fbcc54d8260bd889/lessons/23_preference_data_and_reward_modeling.ipynb).

The README's extraction rule is **understand → implement explicitly → verify → selectively extract → reuse**. This audit covers the notebook's already demonstrated math primitives. The notebook, its saved outputs, and pretrained/SFT checkpoint files were not edited.

## Evidence at chapter closure

The notebook uses three synthetic prompt-conditioned training preference pairs and three manually written held-out probes. The original 65-character SFT GPT provides fixed hidden states to a zero-initialized, trainable linear scalar reward head; 80 AdamW steps fit the three training preference comparisons. The BT loss decreased from 0.693147 to 0.019112, with a reward-head weight-update norm of approximately 4.19385.

| Split | Frozen-backbone reward head | SFT policy sequence-logP ranking |
|---|---:|---:|
| Three training preference pairs | 3/3 | 2/3 |
| Three held-out preference probes | 1/3 | 0/3 |

The reward margins on the held-out probes were `Say welcome` −1.290, `Say please` +2.590 and `Say goodbye` −2.829. The hidden-state recovery check and reward/head consistency checks passed in the saved notebook. Plots are stored as Plotly outputs. These values are *saved user experiment outputs*, not newly run results from this maintenance session.

The trained reward head does **not** show reliable generalization in these three held-out comparisons. Dataset labels are pedagogical judgments, not representative human preference data. BT probabilities do not establish calibrated human agreement. The reward margins and complete-response policy log-probability margins are in different units and cannot be compared directly in magnitude. The SFT model's response score sums autoregressive log-probabilities of **all selected completion characters, including the final newline**.

## Accepted reusable primitives

| Mechanism learned in notebook | Public source API | Why it is reusable |
|---|---|---|
| 23.02–23.03: log-softmax, gather, prompt/padding mask and token sum | `llmfp.training.completion_sequence_log_probs(logits, labels)` | Scores any already aligned `(B,T,V)` causal logits and `(B,T)` completion-only targets into differentiable `(B,)` **sequence sums**. No template, tokenizer, reward model, or label-length normalization is assumed. |
| 23.04–23.05: BT sigmoid preference model and stable softplus loss | `llmfp.training.bradley_terry_loss(chosen_rewards, rejected_rewards)` | Scores two `(B,)` reward vectors with stable `softplus(rejected-chosen)`, preserving gradients and supporting `none`, `mean` or `sum` reductions. |

Both functions are in `src/llmfp/training/losses.py`, exported by `llmfp.training`. The earlier `per_example_sft_nll` remains unchanged. Output is only as meaningful as the caller's alignment and preference labels.

### Exact tensor contracts

**Completion log-probability:** Every prompt and padding target must contain the negative `ignore_index` sentinel, default `-100`; target indices are shifted and aligned **before** this function is called. A completion must have at least one selected target per row, and valid targets must be in `[0,V)`. Before `gather`, ignored labels are replaced with a safe vocabulary index. The corresponding token log-probabilities are then masked back to zero and summed. This operation never introduces a second shift or averages by length. The model and inputs remain the caller's responsibility, including any padding/causal attention constraints.

**Bradley–Terry:** The chosen and rejected rewards are finite, floating-point, nonempty one-dimensional tensors of equal shape, dtype and device. With a margin `Δr=r_chosen-r_rejected`, loss per pair is `softplus(-Δr)=-log(sigmoid(Δr))`. The default mean averages preference pairs equally. Adding the same scalar to both rewards leaves the loss unchanged; rescaling a reward difference generally changes it. BT is a model assumption rather than an intrinsic property of human judgments.

### Minimal reuse example

```python
import torch
from llmfp.training import bradley_terry_loss, completion_sequence_log_probs

# (B_candidates=2, T=2, V=3): aligned logits for two candidate completions
logits = torch.zeros(2, 2, 3, requires_grad=True)
# (B_candidates=2, T=2): ignored prompt and selected completion targets
labels = torch.tensor([[-100, 1], [-100, 2]])

# (B_candidates=2,): sum of completion-token log-probabilities
log_p = completion_sequence_log_probs(logits, labels)

# (B_pairs=1,): separately predicted reward scores
chosen_rewards = torch.tensor([1.0], requires_grad=True)
rejected_rewards = torch.tensor([0.0], requires_grad=True)
loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
loss.backward()
```

The reward vectors above are illustrative and are **not** derived from the policy log-probabilities.

## Selective non-extractions

- **FrozenGPTRewardModel:** The notebook's copy of the SFT embedding, blocks, final norm and scalar head demonstrates real reward learning, but it duplicates the present `GPT.forward` internal feature path, hardcodes freezing and an end-token pooling choice, and lacks a cross-chapter head/checkpoint contract. A reusable reward-model API should be considered when Lesson 24 establishes actual needs; extracting a brittle architecture now would violate the README's abstraction discipline.
- **PreferenceExample and serialization:** The toy `User: ...\nAssistant: ...` template, chosen/rejected response strings, fixed character vocabulary and newline terminator remain teaching/data choices. Arbitrary subword tokenizers require explicit boundary alignment.
- **Training/evaluation loop and Plotly:** 80-step frozen feature probing, synthetic labels, exact ranking checks, score plots and held-out probes remain the notebook's experimental record. Do not add a generic trainer or evaluation framework solely for this toy study.
- **Reward-model checkpoint:** This chapter's notebook saves no independent reward-model checkpoint, and the reusable math functions do not depend on one. Future reward-model reloading is not assumed.

## Focused verification and limitations

The new `tests/test_preference_primitives.py` exercises 12 cases: notebook-equivalent masked gather, equality to negative mean NLL times supervised length, gradient agreement with cross-entropy, ignored-position and padding invariance, custom ignore sentinel, invalid supervision, BT equivalence to negative log-sigmoid, loss reductions, analytic gradients, shift invariance, ties, extreme margins, invalid rewards and a real reward-head parameter update. The update test deliberately avoids two exactly opposing feature-difference vectors, which would cancel the gradient despite a correct loss formula.

All 12 new tests passed in **local leaf-module isolation on CPU with Python 3.13.5 / PyTorch 2.10.0**. This is not a run of the full upstream repository, the previous 12 SFT/16 PPO tests, the user's `torch>=2.14.0` installation, or GPU. The user's actual SFT checkpoint and private local training state were not available to the assistant. The full notebook was not re-executed during documentation/source extraction.

Run all repository regression tests locally with:

```bash
uv run python -m unittest discover -s tests -v
```

