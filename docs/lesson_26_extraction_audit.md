# Lesson 26 — Modern LLM RL & Preference Optimization: Extraction Audit

Source: [Lesson 26 at user commit c220feb](https://github.com/neuromaancer/llm-from-first-principles/blob/c220feb654ff1493fab1e8c7b38ac5d51227a215/lessons/26_modern_llm_rl_and_preference_optimization.ipynb). Results here are copied from the notebook's **saved executed outputs**, not from an independent re-run of the learner's notebook. The extraction does not modify that notebook.

## Established from first principles

- **Critic-free credit assignment:** GRPO centers/scales rewards among completions sampled for the same prompt; RLOO subtracts the mean of *other* completions. On the exactly enumerated Bernoulli toy with p=0.3 and group size 2, true score-function policy gradient was 0.210, group-mean centered was 0.105, and RLOO was 0.210. This is a result for the stated sampling/baseline setup, not a proof that full GRPO is always exactly half the desired gradient.
- **Aggregation semantics:** Original per-response mean produced sampled token-logP gradient coefficients −0.2500 (short length 2) and +0.0833 (long length 6); global token mean produced −0.1250 and +0.1250; unstandardized fixed-budget Dr. GRPO produced −0.0417 and +0.0417. These are gradients with respect to *selected log-probabilities*, not guaranteed Transformer parameter updates or real-world sequence-length changes.
- **Actual critic-free update:** In a two-token autoregressive policy, a 16-response group contained 6 rewarded completions; a single clipped GRPO policy step raised the exact probability of response 11 from 0.2500 to 0.2924. This case used equal-length answers and deliberately omitted reference KL.
- **Reference KL:** For the updated two-token policy, exact sequence KL was 0.005257. A hypothetical further policy-only step raised it to 0.007579; adding an old-sample k3 penalty gave 0.007552. This shows a direct sampled KL penalty can change the gradient, not that KL must decrease or that old-sample KL estimates are on-policy exact.
- **Reward variation, not just text variation:** With group size 16, the uniform policy had 12/12 informative groups and final P(11)=0.703184 after 12 sampled updates. A policy with initial P(11)≈0.000394 had 0/12 informative groups despite response diversity in all 12 groups, and its task-success probability did not improve. For independent Bernoulli rewards, P(informative)=1−p^G−(1−p)^G.
- **Clip-Higher:** At action probability ratio 1.25 and positive advantage, the standard PPO [0.8,1.2] clipped surrogate had zero derivative with respect to the sampled log-ratio; the asymmetric [0.8,1.28] surrogate produced −1.25. This is a local gradient example, not proof of improved exploration.

The notebook **did not train a GPT using full GRPO or DAPO**, nor compare algorithm performance under equivalent model/data compute budgets. Do not treat any of these controlled numerical examples as general benchmark evidence.

## What was extracted

Only two reusable primitives, in `src/llmfp/rl/group_relative.py`:

| Function | Contract and rationale |
|---|---|
| `group_relative_advantages(rewards, normalize=True, epsilon=1e-8)` | Takes finite rewards `(P,G)` for P prompts and G≥2 sampled responses each. Returns detached mean-centered, optionally population-std-normalized `(P,G)` advantages. Never mixes different prompts; same-reward groups have zero advantage. `normalize=False` permits the centered advantages studied for Dr. GRPO. |
| `grpo_clipped_policy_loss(current_log_probs, old_log_probs, response_advantages, action_mask, clip_epsilon=0.2, clip_epsilon_high=None)` | Expects selected action log-probs `(P,G,T)`, group advantages `(P,G)`, and right-padded token mask `(P,G,T)`. Uses old-policy ratios, PPO clipping, and **mean across valid tokens per response before averaging responses/prompts**, preserving original GRPO's distinct length weighting. Optional upper clip distance allows the Clip-Higher diagnostic. Old log-probs and advantages detach. Returns only the policy surrogate; **no reference KL penalty** is included. |

These APIs complement but do not duplicate the existing `ppo_policy_loss`, whose masked mean weights all valid tokens equally, not all responses equally. Both functions require valid response grouping by prompt; the caller must provide correct sampled action log-probs and any desired reference regularization separately.

### Minimal use

```python
import torch
from llmfp.rl import group_relative_advantages, grpo_clipped_policy_loss

# (P=1, G=3): outcome rewards from one prompt's sampled responses.
rewards = torch.tensor([[1.0, 0.0, 0.0]])
advantages = group_relative_advantages(rewards)

# (P=1, G=3, T=2): selected token log-probabilities.
old_logp = torch.full((1, 3, 2), -0.7)
current_logp = old_logp.clone().requires_grad_(True)
mask = torch.ones((1, 3, 2), dtype=torch.bool)

loss = grpo_clipped_policy_loss(current_logp, old_logp, advantages, mask)
loss.backward()
```

## Deliberately not extracted

- RLOO's `R_i-(\sum_jR_j-R_i)/(G-1)` remains a short, transparent notebook formula and does not need another package helper yet.
- The two-token policy, exact P(11), sampling RNG, dynamic-group demonstration and training loops are tied to a fixed toy task. No generic GRPO Trainer was created.
- Fixed-budget Dr. GRPO, token-mean DAPO and dynamic sampling are algorithm choices with different weighting/data-distribution semantics. They remain explicit in the notebook instead of being hidden behind an expansive multipurpose configuration API.
- The exact conditional KL computation is already reusable as `llmfp.rl.categorical_kl_from_logits`. No new KL abstraction was created.

## Verification and documentation

The **6 focused tests** in `tests/test_group_relative.py` passed in an isolated CPU module check with Python 3.13.5 / PyTorch 2.10.0. They cover normalized and centered advantages, zero-reward-variance groups, per-response length weighting and padding, asymmetric clipping, detached old-policy/advantage tensors, and basic invalid input contracts. The entire repository test suite and checkpoint/GPU notebook have not been rerun by the assistant.

Newly indexed Torch APIs: `Tensor.eq` and `torch.unique` (both sample outputs checked on the same CPU runtime), raising the tracker from 99 to 101 entries. The repository's specified `torch>=2.14.0` runtime has not been independently tested. Public README changes are limited to the course link and file tree.
