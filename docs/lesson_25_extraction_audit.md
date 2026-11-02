# Lesson 25 — DPO Extraction Audit

Source notebook: [Lesson 25, user evaluation commit](https://github.com/neuromaancer/llm-from-first-principles/blob/499ad61c493786deb95613f51b1dac805acaa205/lessons/25_direct_preference_optimization.ipynb). Saved notebook outputs are observations from the user's run, not results rerun during extraction.

## What the experiment established

- From KL-regularized policy optimization and the Bradley–Terry model, the prompt-specific partition term cancels between chosen/rejected responses, yielding the DPO objective on **reference-corrected sequence log-probability differences**.
- With Actor = Reference, the initial DPO loss is log(2) ≈ 0.693147 and the gradient with respect to chosen/rejected sequence logP is nonzero and opposite-signed (for beta=0.5: −0.25, +0.25 before batch reduction). Backpropagation reached the GPT LM head.
- Training on just three synthetic pairs for 40 steps reduced training DPO loss from ≈0.6931 to ≈0.01499. Reference-relative margins improved by +12.189 (Say yes), +11.871 (Say no), and +6.393 (Say hello). The **raw** chosen/rejected margin for Say no remained negative (≈−6.752).
- Greedy exact-match accuracy over six prompts stayed **2/6** for both SFT Reference and DPO Actor. The Actor changed Say no from Yes to Thanks and Say please from Farewell to Hello; neither was the requested response.
- All three DPO-held-out preference margins **decreased**: Say welcome −0.029; Say please −1.212; Say goodbye −3.604. These pairs were excluded from the DPO optimizer steps, but were previously used in Lesson 23 diagnostics; this is not an untouched benchmark. The tiny experiment supports neither a generalization claim nor a causal account of why the margins deteriorated.

## Source extraction

One reusable primitive was added to `src/llmfp/training/losses.py`:

`dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, *, beta=0.1, reduction='mean')`

- Input tensors are paired `(B,)` **sums** of completion token log-probabilities, not mean token scores or isolated vocabulary logits. Each chosen/rejected pair must share the same prompt.
- Computes `-logsigmoid(beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)))` with stable PyTorch math; supports `none`, `mean`, and `sum` reductions.
- Gradients flow to Actor sequence log-probs only; reference inputs are detached internally, while the caller should also freeze the reference model to avoid building an unnecessary network graph.
- Designed to compose directly with the already extracted `completion_sequence_log_probs` and data supervision helpers. No Reward Model, Critic, rollout collector, or PPO trainer is needed for DPO.

Five focused tests in `tests/test_dpo_loss.py` cover the initial analytical gradient, the Lesson 25 numeric example and reductions, frozen reference gradients, extreme margins, and invalid inputs. These tests passed locally on CPU Python 3.13.5 / PyTorch 2.10.0 in an isolated copy of the new function. Full repository suite and the user's CUDA/checkpoint-backed notebook were **not** rerun in this extraction.

## Why nothing else was extracted

The three-pair task data, prompt serialization, training loop, full GPT state, greedy-generation evaluator and held-out probes belong in the notebook: their contracts are experiment-specific. Existing generic model, generation, and training primitives already cover the reusable boundaries. The notebook did not save a portable trained DPO Actor checkpoint.

The Torch API reference remains at **99 entries**: this lesson reused already indexed PyTorch operations and introduced no newly needed API entries. README is updated only as a public course index/file listing; no progress diary is added.
