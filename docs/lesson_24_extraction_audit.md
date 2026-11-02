# Lesson 24 — RLHF with PPO and KL: Extraction Audit

Date: 2026-11-02. Source: [Lesson 24 at the final user evaluation commit](https://github.com/neuromaancer/llm-from-first-principles/blob/6851b9872d4b848284c2dc7c32c2895cc57eaffa/lessons/24_rlhf_with_ppo_and_kl.ipynb).

This record documents scientific observations and the narrow source-extraction decisions. The learning notebook and its saved outputs are not changed by this maintenance commit. Experimental results below are *saved user-notebook outputs*, not experiments rerun during the extraction.

## Scope and objective

The notebook connects the fixed SFT reference, a separately trainable GPT actor, a frozen GPT-feature + scalar reward head, an independent trainable critic, on-policy autoregressive collection, token-level sampled KL shaping, GAE and PPO. The reference policy is distinct from the rollout-time old policy used in PPO importance ratios.

The frozen reward head was **retrained** from three synthetic chosen/rejected examples: its prior notebook did not provide a separately reloadable reward-model checkpoint. All rewards are proxy scores learned from these three pairs, not independent human satisfaction scores. The tokenizer has a 65-character vocabulary and uses an ordinary newline as an experimental stopping marker, not a dedicated EOS token. The full training program is intentionally kept in the teaching notebook.

## Results supported by saved outputs

| Stage | Recorded observation | Interpretation |
|---|---|---|
| 24.01–24.02 | Two-response regularized optimum P(B)=0.450853; exact forward KL=0.407677 and MC estimate=0.400038 | Probability math and the sampling-distribution distinction were checked; a single signed log-ratio is not an exact KL |
| 24.03 | Eight sampled responses all newline-terminated, with identical responses across two samples per prompt | Actor/reference sampling and teacher-forced action logP agreed; exploration was very limited |
| 24.04 | Reward-sum identity, GAE/return equivalence with initial zero critic, and termination versus truncation checks passed | The project rollout and GAE primitives have correct behavior for the checked cases |
| 24.05 | One PPO step: Actor LM-head ΔW norm ≈0.002533; value loss 7.5556 → 7.5219; PPO approx KL 0.02295, clip fraction 0.0476; importance ratios 0.3097–1.0002 | Parameters really changed; PPO clipping does **not** hard-bound the post-step ratios |
| 24.06 | Four collect/update cycles, eight PPO updates; observed exact matches stayed 6/12 per cycle | The complete pipeline executed, without observed accuracy improvement |
| 24.07 | Greedy exact match 2/6 for each model; fresh sampled exact match 8/24 for each; all 48 samples terminated | No observed behavioral improvement in this small evaluation |

During cycles 1–4, average sampled reward scores were **4.032, 3.850, 3.850, 3.850** (reported before their cycle's PPO update). Their measured post-update conditional reference KL at old-rollout states was approximately **0.0007, 0.0009, 0.0004, 0.0000**. These state averages and PPO approximate KL are not the exact whole-response forward KL of the updated policy.

Although observed behavior did not change, the trained Actor slightly increased teacher-forced logP of expected completions: about **+0.0135** for "Say no", **+0.0455** for "Say welcome", **+0.0373** for "Say please", and **+0.0176** for "Say goodbye". Mean exact first-token conditional KL on the six evaluation prompts was about **0.0001447**; one individual printed estimate was slightly negative at ~10^-7 due to finite-precision arithmetic.

For "Say goodbye", the **incorrect** sampled response "Lady\n" received **+9.0721** from the reward model. This is evidence of a reward/target mismatch in this toy probe. The runs do **not** establish reward hacking, a causal effect of PPO on that response, or reliable human alignment. Two probes were held out from the PPO updates but already appeared in previous lessons, so the evaluation is **PPO-held-out**, not a previously untouched benchmark. The comparison used four fresh samples per prompt/model with shared seeds for a paired descriptive comparison; there is no statistical power for general claims.

## Reused source components

The following are **already reusable** and were called without new wrappers:

- \`llmfp.rl.RolloutBatch\` and \`select_rollout_sequences\`: fixed collection-time facts and aligned sampling.
- \`rescore_actions_from_logits\` and \`align_state_values_for_actions\`: correct causal next-token probabilities and critic state positions.
- \`generalized_advantage_estimation\`, \`ppo_policy_loss\`, \`ppo_value_loss\`, \`masked_mean\`: action-level advantages, fixed targets, clipped updates and masked reductions.
- \`llmfp.training.completion_sequence_log_probs\` and \`bradley_terry_loss\`: already extracted from Lesson 23.

Do not duplicate the previous math or introduce a second generic PPO trainer.

## New narrowly reusable primitives

| Notebook operation | Extracted API | Exact contract |
|---|---|---|
| 24.05–24.07: exact next-token KL over a complete categorical vocabulary | \`llmfp.rl.categorical_kl_from_logits(current_logits, reference_logits)\` | Accept identical finite floating logits of shape \`(..., V)\` and return \`(...,)\` conditional **forward** KL \`D_KL(current(.|s) || reference(.|s))\`. It preserves autograd and does not aggregate over states, sample trajectories, or estimate whole-sequence KL. Callers decide which states to average; the reference network is frozen separately. |
| 24.03–24.06: signed token KL shaping and terminal score placement | \`llmfp.rl.kl_shaped_action_rewards(old_logp, ref_logp, mask, terminal_scores, terminated, beta=...)\` | Accept old and reference log-probs of the SAME collected actions \`(B,T)\`, a nonempty right-padded Boolean mask, scalar per-sequence scores \`(B,)\`, and Boolean termination flags \`(B,)\`. Return detached \`(B,T)\` rewards: \`-beta*(old-ref)\` on valid actions; attach one sequence score to the last valid action only for naturally terminated episodes. Truncated trajectories have no terminal score and may require Critic bootstrapping, handled elsewhere. |

\`kl_shaped_action_rewards\` intentionally does not calculate GAE or handle PPO parameter updates. A negative sampled old-minus-reference log-ratio produces a positive KL-shaped token reward; it must not be clipped to zero. On-policy expectations of these signed ratios are KL-related, but individual samples need not be nonnegative.

### Minimal usage

\`\`\`python
import torch
from llmfp.rl import categorical_kl_from_logits, kl_shaped_action_rewards

# (B=1, T=2): log-probabilities for two sampled actions.
old = torch.tensor([[-0.2, -0.8]])
reference = torch.tensor([[-0.4, -0.5]])
mask = torch.tensor([[True, True]])
scores = torch.tensor([1.0])
terminated = torch.tensor([True])

# (B=1, T=2): frozen per-action RLHF rewards.
rewards = kl_shaped_action_rewards(
    old, reference, mask, scores, terminated, beta=0.1
)

# (B=1, T=2, V=3): full vocabulary logits at the sampled states.
actor_logits = torch.zeros(1, 2, 3)
reference_logits = torch.ones_like(actor_logits)

# (B=1, T=2): exact categorical KL, without averaging states.
conditional_kl = categorical_kl_from_logits(actor_logits, reference_logits)
\`\`\`

## Not extracted and why

- \`collect_one_response\`, \`collect_rlhf_training_batch\` and \`update_rlhf_ppo_batch\` capture an experiment-specific character tokenizer, newline stopping, frozen GPT reward feature path, implicit model/optimizer state, four cycles and fixed sampling hyperparameters. Generalizing them now would conceal assumptions. Existing generation and PPO primitives are the correct reusable boundaries.
- \`RLHFValueCritic\` copies GPT internals and chooses a particular initialization/pooling/value-head architecture; a generalized critic wrapper would require broader model contracts not yet established.
- The three-pair synthetic Reward Model and instruction-exact-match evaluator are experiment definitions, not reusable rewards or general evaluation benchmarks.
- Plotly training/diagnostic figures, manually selected prompts, and checkpoint paths remain notebook and analysis details. The notebook does not save a trained RLHF Actor or a stand-alone Reward Model checkpoint.
- A separate advantage-normalization library helper is deferred: the notebook's action masking and zero-variance fallback are clear already, and introducing another abstraction is unnecessary for later sections.

## Verification boundary

\`tests/test_rlhf_kl_primitives.py\` has **12 focused regression tests** for exact categorical KL and its analytic gradient, zero KL on identical logits, shape/dtype/finite constraints, signed KL rewards, last valid terminal action, truncation, padding independence, no unexpected gradient tracking, invalid boundary masks and beta.

Those 12 new tests **passed on CPU with Python 3.13.5 / PyTorch 2.10.0 in isolated module testing**. The test file uses normal package imports for repository runs. The assistant has **not** rerun the user's checkpoint-backed notebook, the pre-existing 16 PPO, 12 supervised and 12 preference regression tests, or the entire repository's declared PyTorch 2.14/CUDA environment.

To verify the complete repository locally:

\`\`\`bash
uv run python -m unittest discover -s tests -v
\`\`\`

The source notebook's saved outputs provide evidence of the prior experiments, but a complete fresh Restart Kernel → Run All has not been separately verified in this maintenance.
