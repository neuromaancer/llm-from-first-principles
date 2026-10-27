# LLM from First Principles

Building a small language model from scratch to understand every tensor, operation, and gradient.

This repository is my public learning log for understanding modern language models from the bottom up.

The goal is **not** to reproduce a large model, hide complexity behind high-level frameworks, or reach a benchmark as quickly as possible.

The goal is to reach the point where I can look at a Transformer and explain:

- what every tensor represents,
- why it has that shape,
- what every operation does,
- how gradients flow through it,
- and what changes when the architecture is modified.

The project therefore starts below the Transformer itself: tensor operations, numerical stability, autodiff, tokenization, batching, losses, optimizers, and training loops — before building attention and a complete decoder-only language model. From there, it continues into inference, reinforcement-learning foundations, supervised fine-tuning, preference learning, RLHF, and modern LLM post-training.

> **Learning rule:** every implementation in this repository is typed and worked through manually.
>
> AI code completion is disabled. AI may be used as a tutor for explanations, derivations, questions, discussion, and code review, but not as an autocomplete or bulk code-generation workflow.
>
> **If I cannot explain a line of code, it does not belong here yet.**

---

## What this repository is — and is not

This is primarily a **learning repository**, not a production LLM framework.

Educational implementations intentionally favor clarity over efficiency. Intermediate tensors may stay visible, algorithms may first be implemented explicitly, and related mechanisms may initially be written separately when that makes their differences easier to understand.

Once a mechanism has been implemented, derived, and verified, later lessons are free to use mature PyTorch primitives instead of repeatedly rebuilding the same idea.

The point is not to reinvent PyTorch forever. The point is to understand what PyTorch is doing before relying on it.

---

## Lessons

The notebooks are intended to be read in order. The roadmap is organized into five stages: first understand the language model itself, then pretrain a tiny GPT end to end, then study sparse conditional computation with Mixture of Experts, then build the reinforcement-learning foundations needed for post-training, and finally study modern LLM post-training objectives.

### Part I — Language Models from First Principles

| Lesson | Topic | Status |
|---|---|---|
| [00](lessons/00_tensor_basics.ipynb) | Tensor Basics | ✅ Complete |
| [01](lessons/01_autograd_from_scratch.ipynb) | Autograd from Scratch | ✅ Complete |
| [02](lessons/02_text_and_tokenization.ipynb) | Text and Tokenization | ✅ Complete |
| [03](lessons/03_sequences_and_batching.ipynb) | Sequences and Batching | ✅ Complete |
| [04](lessons/04_embeddings_and_linear_layers.ipynb) | Embeddings and Linear Layers | ✅ Complete |
| [05](lessons/05_language_modeling_objectives.ipynb) | Language Modeling Objectives | ✅ Complete |
| [06](lessons/06_optimizers_from_scratch.ipynb) | Optimizers from Scratch | ✅ Complete |
| [07](lessons/07_training_loop_from_scratch.ipynb) | Training Loop from Scratch | ✅ Complete |
| [08](lessons/08_causal_self_attention.ipynb) | Causal Self-Attention | ✅ Complete |
| [09](lessons/09_multi_head_attention.ipynb) | Multi-Head Attention | ✅ Complete |
| [10](lessons/10_normalization_and_feed_forward.ipynb) | Normalization and Feed-Forward Networks | ✅ Complete |
| [11](lessons/11_positional_information.ipynb) | Positional Information | ✅ Complete |
| [12](lessons/12_transformer_blocks_and_gpt.ipynb) | Transformer Blocks and GPT | ✅ Complete |
| [13](lessons/13_generation_and_kv_cache.ipynb) | Generation and KV Cache | ✅ Complete |
| [14](lessons/14_multi_query_and_grouped_query_attention.ipynb) | Multi-Query and Grouped-Query Attention | ✅ Complete |
| [15](lessons/15_linear_attention.ipynb) | Linear Attention | ✅ Complete |

### Part II — Pretraining a Tiny GPT

| Lesson | Topic | Status |
|---|---|---|
| [16](lessons/16_pretraining_a_tiny_gpt.ipynb) | Pretraining a Tiny GPT End to End | ✅ Complete |

This lesson is the integration point for Part I. It connects raw text, tokenization, batching, the decoder-only model, cross-entropy, AdamW, learning-rate scheduling, validation, checkpoints, perplexity, and generation in one real training run.

### Part III — Sparse Conditional Computation

| Lesson | Topic | Status |
|---|---|---|
| [17](lessons/17_mixture_of_experts.ipynb) | Mixture of Experts (MoE) from First Principles | ✅ Complete |

This lesson starts from the dense SwiGLU feed-forward network already used by the GPT baseline and asks whether every token must use the same feed-forward parameters. It introduces expert networks, learned routers, top-k routing, token dispatch/gather, expert utilization, load balancing, capacity constraints, and the distinction between total and active parameters.

### Part IV — Reinforcement Learning from First Principles

| Lesson | Topic | Status |
|---|---|---|
| [18](lessons/18_rl_foundations.ipynb) | RL Foundations: Trajectories, Return, Value, Q, Advantage, Bellman Equations | ✅ Complete |
| [19](lessons/19_policy_gradients_and_reinforce.ipynb) | Policy Gradients and REINFORCE from Scratch | ✅ Complete |
| [20](lessons/20_actor_critic_td_and_gae.ipynb) | Actor-Critic, TD Learning, and GAE | ✅ Complete |
| [21A](lessons/21a_ppo_from_scratch.ipynb) | PPO from Scratch: clipping, GAE, rollout reuse, and a complete toy training loop | ✅ Complete |
| [21B](lessons/21b_ppo_with_tiny_gpt.ipynb) | PPO with Tiny GPT: Transformer Actor/Critic, rollout collection, PPO updates, and evaluation | ✅ Complete |

**Lessons 21A and 21B are complete.** Lesson 21A derives PPO and validates the basic toy rollout; its reusable PPO losses, GAE, token alignment, and frozen rollout contracts live in `llmfp.rl` and are covered by `tests/test_ppo_primitives.py`. Lesson 21B integrates those contracts with a GPT Actor and an independent Transformer Critic, tests autoregressive sampling versus teacher-forced rescoring, and completes PPO training and exact policy evaluation. Toy environments, comparison runners, and the teaching-only Critic stay in the notebook instead of being prematurely extracted into the reusable library.

The RL section is intentionally focused. It does not try to reproduce a complete general-purpose RL curriculum; it develops the concepts needed to understand LLM post-training objectives from first principles.

The implementation goal is a **minimal RL-for-LLM stack**, not a generic RL framework. Each lesson should add one verified piece to a small end-to-end system rather than leaving the mathematics isolated in notebooks.

The design is organized around five responsibilities:

```text
Models
├── Actor / token policy
└── Critic / value model

Collection
└── sample token trajectories under the current policy

Data
└── store rollout facts and learning targets with explicit token alignment

RL math
├── returns
├── TD errors
├── advantages / GAE
└── policy / value / PPO objectives

Experiment loop
└── collect → construct targets → update → evaluate → checkpoint
```

The first version deliberately keeps these responsibilities explicit. Trainable predictors use `nn.Module`; trajectory data uses typed tensor containers; core RL mathematics stays in small tensor functions; and the experiment loop remains thin and readable.

A central design rule is to distinguish three kinds of quantities:

```text
facts recorded at collection time
        ≠
fixed learning targets for an update
        ≠
current differentiable model predictions
```

For example, PPO will distinguish rollout-time log-probabilities from current log-probabilities, while advantages and value targets are treated as fixed credit signals during an update.

The LLM version stores each prompt-plus-completion sequence once rather than materializing a separate full prefix for every state. Per-action quantities such as rewards, log-probabilities, values, advantages, and masks are aligned to generated token positions.

The framework will grow only after mechanisms are derived and verified in the lessons. Empty abstractions are intentionally avoided.

### Part V — LLM Post-Training

| Lesson | Topic | Status |
|---|---|---|
| [22](lessons/22_supervised_fine_tuning.ipynb) | Supervised Fine-Tuning (SFT) | ✅ Complete |
| [23](lessons/23_preference_data_and_reward_modeling.ipynb) | Preference Data, Sequence Log-Probabilities, and Reward Modeling | ✅ Complete |
| 24 | RLHF with PPO and KL Regularization | Planned |
| 25 | Direct Preference Optimization (DPO) | Planned |
| 26 | Modern LLM RL and Preference Optimization | Planned |

The post-training section will connect token-level language modeling to sequence-level optimization. It will make explicit the roles of completion log-probabilities, reference policies, KL penalties, pairwise preferences, learned rewards, and policy optimization before using higher-level training frameworks.

**Lesson 22 is complete.** It connects assistant-target alignment, masking and right-padded batches to full-parameter fine-tuning of the Lesson 16 pretrained GPT, followed by teacher-forced diagnostics, greedy generation, and a Plotly evaluation heatmap. Three verified token/tensor primitives now live in `llmfp.data.supervision` and `llmfp.training.losses`; text templates, toy data, experiment loops, and plotting remain in the notebook. See the [Lesson 22 extraction audit](docs/lesson_22_extraction_audit.md) for the precise contracts, tests, and exclusions.

---

**Lesson 23 is complete.** It links teacher-forced completion log-probabilities to pairwise Bradley–Terry preference learning and a frozen-backbone GPT reward-head experiment. Two reusable math primitives have been extracted to `llmfp.training`: completion sequence log-probability and Bradley–Terry loss. The notebook retains the synthetic pairs, reward-head prototype, train/evaluation runners, and Plotly figures. See the [Lesson 23 extraction audit](docs/lesson_23_extraction_audit.md).

---

## Learning philosophy

This repository follows a few rules.

### 1. Build from simple operations upward

Understand `reshape`, `transpose`, broadcasting, matrix multiplication, reductions, and softmax before using them inside attention.

### 2. Predict tensor shapes before running the code

Shapes should become something that can be reasoned about, not something discovered through trial and error.

### 3. Implement new mechanisms explicitly

When a mechanism is the object of study, implement it in a transparent way first.

Examples include:

- numerically stable softmax,
- scalar reverse-mode autodiff,
- BPE,
- cross-entropy from logits,
- SGD, Momentum, Adam, and AdamW,
- causal masking,
- attention,
- normalization,
- residual connections,
- and gated feed-forward networks.

### 4. Reuse a mechanism once it is understood

A later notebook does not need to reimplement a mechanism that has already been derived and verified.

For example, once cross-entropy has been implemented manually, later training code can use `torch.nn.functional.cross_entropy`.

The purpose is understanding, not ritual duplication.

### 5. Prefer explicit tensor semantics

The central question is always:

> What happens to the tensor?

For attention, that means reasoning through transformations such as:

```text
(B, T, C)
    ↓
Q, K, V
    ↓
(B, H, T, D)
    ↓
QKᵀ
    ↓
(B, H, T, T)
    ↓
softmax
    ↓
weighted sum of V
    ↓
(B, T, C)
```

### 6. Refactor only after understanding the duplication

MHA, MQA, and GQA may initially be implemented separately before extracting their common structure.

Abstraction should follow understanding, not hide what has not yet been understood.

### 7. Keep lessons separate from reusable implementations

`lessons/` contains verbose educational notebooks with derivations, intermediate tensors, sanity checks, and reference comparisons.

The `src/llmfp/` package contains cleaner reusable implementations extracted only after the underlying mechanisms have been derived and verified in the lessons.

The extraction rule is:

```text
understand
    ↓
implement explicitly
    ↓
verify
    ↓
extract into src/
    ↓
reuse in later lessons
```

This keeps later notebooks focused on the new idea rather than repeatedly copying already-understood mechanisms.

Extraction is selective: move a verified, template-independent operation with an explicit tensor contract and a concrete reuse role, not the whole teaching experiment. Conversely, avoiding a generic framework is not a reason to leave every reusable primitive in a notebook. Keep the original derivation and experiments available as the reference implementation; test the extracted code against them.

### 8. Use AI as a teacher, not as an autocomplete engine

The coding environment used for this project disables AI code completion.

AI can help challenge an explanation, discuss a derivation, review an implementation, or suggest questions to investigate. The actual learning code is typed and reasoned through manually.


---

## What is being built

The long-term target is broader than a small GPT implementation. The repository aims to trace the path from tensors to a pretrained language model and then to a modern post-trained assistant while keeping every major abstraction explainable.

The path is deliberately incremental:

```text
Part I — Language model foundations
├── tensors and scalar autodiff
├── tokenization and batching
├── objectives and optimizers
├── causal self-attention
├── multi-head attention
├── normalization and SwiGLU
├── positional information and RoPE
├── decoder-only GPT
├── generation and sampling
├── KV caching
├── MQA / GQA
└── linear attention

Part II — End-to-end pretraining
├── raw text → training tokens
├── train / validation split
├── sequence windows and batches
├── GPT forward pass
├── cross-entropy and backpropagation
├── AdamW and learning-rate scheduling
├── gradient clipping
├── checkpoints
├── validation loss and perplexity
└── generation from the trained model

Part III — Sparse conditional computation
├── dense FFN → multiple experts
├── router logits and routing probabilities
├── top-k expert selection
├── token dispatch and gather
├── expert utilization
├── load balancing
├── capacity constraints
└── total vs active parameters

Part IV — Reinforcement-learning foundations
├── states, actions, trajectories, and rewards
├── returns and discounting
├── V(s), Q(s,a), and advantage
├── Bellman equations
├── Monte Carlo estimation
├── policy gradients / REINFORCE
├── baselines and variance reduction
├── actor-critic methods
├── temporal-difference learning
├── generalized advantage estimation
├── PPO
└── minimal RL-for-LLM integration
    ├── token-policy rollout
    ├── rollout-time log-probabilities
    ├── critic values
    ├── action masks and trajectory boundaries
    ├── advantages and value targets
    └── policy / value updates

Part V — LLM post-training
├── supervised fine-tuning
├── chat formatting and assistant-only loss
├── token-level and sequence-level log-probabilities
├── preference datasets
├── pairwise reward modeling
├── reference policies and KL regularization
├── RLHF with PPO
├── direct preference optimization
└── modern LLM RL / preference optimization

Experiments
├── parameter count
├── training and validation loss
├── perplexity
├── tokens / second
├── GPU memory
├── inference latency
├── KV-cache memory
├── reward / preference accuracy
├── policy KL
└── pretraining and post-training behavior comparisons
```

The intention is not to implement every architecture or every reinforcement-learning algorithm. The main path focuses on the concepts required to understand how a decoder-only language model is built, pretrained, and post-trained from first principles.

---

## How to use this repository

The notebooks are not a collection of final answers. They are a record of successive stages of understanding.

A useful way to follow the project is:

1. read the explanation before the code,
2. predict tensor shapes before running a cell,
3. derive the operation on paper,
4. implement it yourself,
5. test edge cases,
6. compare it against a trusted PyTorch implementation when appropriate,
7. only then move to the next abstraction.

If you are following along, typing the code yourself is strongly recommended.

---

## Development environment

Primary development environment:

- WSL2 / Linux
- Python 3.13
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- PyTorch
- Jupyter
- VS Code

Install the project environment with:

```bash
uv sync
```

The lessons are Jupyter notebooks and can be opened directly in VS Code or through Jupyter.

Run the reusable PPO and supervised-learning regression checks with:

```bash
uv run python -m unittest discover -s tests -v
```

The PPO helpers assume generated-action positions are consecutive with right-side action padding; the three masks (`action_mask`, `bootstrap_mask`, `continuation_mask`) retain distinct meanings.

The project interpreter should point to:

```text
.venv/bin/python
```

The local `.venv/` directory is not committed to Git.

---

## VS Code learning environment

This repository includes a small shared VS Code configuration under
[`.vscode/`](.vscode/).

The workspace intentionally disables AI-assisted code completion.

Recommended extensions are also included, covering Python, Pylance,
Ruff, Jupyter, debugging, and WSL.

The intention is deliberate:

> AI may help explain, question, and review the code, but the learning
> implementation itself should be typed and reasoned through manually.

---

## Repository structure

The repository grows gradually rather than being generated all at once.

```text
llm-from-first-principles/
├── lessons/
│   ├── 00_tensor_basics.ipynb
│   ├── 01_autograd_from_scratch.ipynb
│   ├── 02_text_and_tokenization.ipynb
│   ├── 03_sequences_and_batching.ipynb
│   ├── 04_embeddings_and_linear_layers.ipynb
│   ├── 05_language_modeling_objectives.ipynb
│   ├── 06_optimizers_from_scratch.ipynb
│   ├── 07_training_loop_from_scratch.ipynb
│   ├── 08_causal_self_attention.ipynb
│   ├── 09_multi_head_attention.ipynb
│   ├── 10_normalization_and_feed_forward.ipynb
│   ├── 11_positional_information.ipynb
│   ├── 12_transformer_blocks_and_gpt.ipynb
│   ├── 13_generation_and_kv_cache.ipynb
│   ├── 14_multi_query_and_grouped_query_attention.ipynb
│   ├── 15_linear_attention.ipynb
│   ├── 16_pretraining_a_tiny_gpt.ipynb
│   ├── 17_mixture_of_experts.ipynb
│   ├── 18_rl_foundations.ipynb
│   ├── 19_policy_gradients_and_reinforce.ipynb
│   ├── 20_actor_critic_td_and_gae.ipynb
│   ├── 21a_ppo_from_scratch.ipynb
│   ├── 21b_ppo_with_tiny_gpt.ipynb
│   ├── 22_supervised_fine_tuning.ipynb
│   └── 23_preference_data_and_reward_modeling.ipynb
│
├── src/
│   └── llmfp/
│       ├── nn/
│       │   ├── attention.py
│       │   ├── fast_weights.py
│       │   ├── linear_attention.py
│       │   ├── mlp.py
│       │   ├── moe.py
│       │   ├── rope.py
│       │   └── transformer.py
│       ├── data/
│       │   ├── batching.py
│       │   ├── supervision.py
│       │   └── tokenization.py
│       ├── models/
│       │   └── gpt.py
│       ├── rl/
│       │   ├── __init__.py
│       │   ├── advantages.py
│       │   ├── alignment.py
│       │   ├── policy_gradient.py
│       │   ├── ppo.py
│       │   ├── returns.py
│       │   └── rollouts.py
│       ├── generation/
│       │   ├── cache.py
│       │   ├── generate.py
│       │   └── sampling.py
│       ├── training/
│       │   ├── checkpoint.py
│       │   ├── evaluation.py
│       │   ├── losses.py
│       │   ├── schedules.py
│       │   └── steps.py
│       └── utils/
│           ├── inspection.py
│           └── metrics.py
│
├── tests/
│   ├── test_ppo_primitives.py
│   ├── test_supervised_primitives.py
│   └── test_preference_primitives.py
├── docs/
│   ├── lesson_22_extraction_audit.md
│   └── lesson_23_extraction_audit.md
├── data/                # large/generated datasets are not committed
├── checkpoints/         # ignored
├── TORCH_FUNCTIONS.md   # PyTorch API learning tracker
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Data and generated artifacts

Large or generated files are intentionally kept out of Git.

Examples include:

```text
data/raw/
data/processed/
checkpoints/
runs/
outputs/
logs/
```

Small teaching datasets may still be committed when they are useful for explaining an implementation.

---

## Git workflow

Development uses small, meaningful commits following the [Conventional Commits](https://www.conventionalcommits.org/) format.

Examples:

```text
feat(autograd): implement scalar reverse-mode autodiff
feat(tokenization): implement character and byte-level BPE concepts
feat(attention): implement causal self-attention
feat(attention): implement multi-head causal attention
feat(transformer): implement normalization and gated feed-forward networks
docs(readme): update learning philosophy and project progress
```

The Git history is part of the learning record: small commits make it possible to see how the model was built one concept at a time.


---

## Current status

**Completed:** Lessons 00–23, including supervised fine-tuning, pairwise preference learning, and the Lesson 21 PPO sequence.

**Next lesson:** 24 — RLHF with PPO and KL Regularization.

### Lesson 21 closure — from PPO equations to GPT updates

- **21A — PPO from Scratch:** derive the clipped policy objective, distinguish frozen rollout facts from current predictions, implement GAE and action/state alignment, and verify rollout reuse with a Bigram Actor-Critic.
- **21B — PPO with Tiny GPT:** replace the Bigram policy with a randomly initialized causal Transformer Actor, implement an independent Transformer Critic, collect autoregressive completions, compute fixed GAE targets, execute minibatch PPO updates, and evaluate a complete collect → update → collect loop.
- **Verification:** The saved 21B notebook contains **48 executed code cells** with no saved errors (Restart Kernel → Run All, October 2026). Existing reusable RL primitives are covered by **16 regression tests** in `tests/test_ppo_primitives.py`.

### What the toy experiments established

| Setting | Outcome |
|---|---|
| Simple two-token GPT PPO task | Exact response success improved from **2.44% to 65.40%** over 12 collection-update cycles. |
| Context task, Bigram PPO | **47.84%** exact success after 1,536 trajectories and 192 updates; its **50% upper bound** follows mathematically from identical final prompt tokens. |
| Context task, Tiny GPT PPO | **45.24%** exact success with the same trajectory and update counts; this single run did **not** exceed the Bigram bound. |
| Context task, independently initialized GPT with supervised targets | **99.61%** exact success after 160 teacher-forced updates, demonstrating the architecture can represent the two mappings. |

The context PPO run was highly asymmetric across prompts: **88.37%** success for `[0,1,2] → [3,4]` versus **2.12%** for `[0,3,2] → [4,3]`. This demonstrates an optimization/exploration failure in the observed run, **not** an architectural impossibility. These are small, single-seed toy experiments; the Bigram and GPT use different learning rates and do not have equal computation costs. They do not support a universal sample-efficiency ranking, and the GPT used here was not pretrained.

### Reusable-code boundary

The general-purpose pieces already exist under `src/llmfp/rl/`:

```text
src/llmfp/rl/
├── advantages.py        # TD errors and generalized advantage estimation
├── alignment.py         # causal action log-probs and Critic state alignment
├── ppo.py               # PPO policy/value losses, entropy, masked means
├── rollouts.py          # validated frozen rollout records and sequence slicing
├── policy_gradient.py   # REINFORCE primitives
└── returns.py           # discounted returns
```

The chapter's `TinyGPTCritic`, `collect_gpt_toy_rollout`, `update_gpt_ppo`, exact toy evaluator, and task-specific Bigram comparisons remain in `lessons/21b_ppo_with_tiny_gpt.ipynb`. Extracting those pedagogical implementations into a generic training framework would obscure assumptions (two actions, fixed prompt/task types, specific evaluation and model-mode contracts) before later post-training lessons establish reusable needs.

Verify the existing RL regression suite with:

```bash
uv run python -m unittest discover -s tests -v
```

### Lesson 22 closure — supervised fitting versus generalization

The [SFT notebook](lessons/22_supervised_fine_tuning.ipynb) restores the Lesson 16 `step_02000.pt` checkpoint: a 3,050,240-parameter GPT with a 65-character vocabulary. Forty AdamW updates fit four instruction-response pairs containing 26 supervised character targets, with the original pretrained model preserved for comparison.

| Measurement | Pretrained | After SFT |
|---|---:|---:|
| Training assistant-token mean NLL | 3.0573 | 0.000189 |
| Greedy first-line exact matches on training instructions | 0/4 | 4/4 |
| Greedy first-line exact matches on three held-out probes | 0/3 | 0/3 |
| Teacher-forced NLL on `Say welcome` → `Welcome` | 2.3556 | 3.4502 |

The LM-head weight-change norm was approximately 0.2926. On the held-out prompts `Say welcome`, `Say no`, and `Say please`, the fine-tuned model produced `Thanks`, `Yes`, and `Farewell`, respectively. These observations establish successful fitting of the four supervised examples, **not** general instruction-following or a causal account of how the model represented the task. There is only one training configuration and a tiny, manually constructed probe set.

Exact match requires the expected first response line plus an observed newline within a 24-character decoding budget. The decoder generates a fixed number of characters; the notebook extracts the first line afterward. This is not a native EOS token or an early-stopping implementation.

**Reusable pieces:** `shift_supervised_tokens` aligns full-sequence token/mask inputs once; `pad_supervised_batch` right-pads already shifted examples; `per_example_sft_nll` computes differentiable per-sequence supervised means. The [extraction audit](docs/lesson_22_extraction_audit.md) links their contracts to the notebook and explains why the task-specific encoders, generation wrapper, training loop, and Plotly code stay there. The existing `language_model_training_step` already calls cross-entropy with its default `ignore_index=-100`; a second SFT trainer was not added.

**Checkpoint:** The notebook saves and reloads `checkpoints/tiny_shakespeare_sft/step_00040.pt` locally, including a plain configuration dictionary, ordered tokenizer vocabulary, and model weights. It does not save optimizer or RNG state, so it supports model restoration rather than exact training resumption. Checkpoint files are not committed.

**Verification boundary:** The experiment numbers and checkpoint-restoration message above are saved outputs in the user's notebook at commit `2a4935748b597e786e126554f1991379eb1c615c`; the assistant did not rerun that pretrained-checkpoint experiment. The 12 new supervised primitive tests were run in isolation on CPU with Python 3.13.5 / PyTorch 2.10.0. This is not a verification of the declared `torch>=2.14.0` environment, CUDA execution, or a newly completed full-notebook Run All. The existing 16 PPO tests were not rerun during this extraction; the normal repository test-discovery command above includes both suites.

The next conceptual step is **Lesson 23**: complete-response log-probabilities, preference pairs, and reward modeling. It connects supervised token prediction to the sequence-level objectives used later in RLHF and DPO.

### Lesson 23 closure — preference signals versus policy likelihood

The [preference learning notebook](lessons/23_preference_data_and_reward_modeling.ipynb) builds synthetic prompt-conditioned chosen/rejected pairs, scores whole completions by summing correctly aligned next-token log-probabilities, demonstrates length and masking effects, derives the Bradley–Terry (BT) loss and gradients, and trains a **frozen-backbone** GPT scalar reward head. It finishes by comparing reward ranking with the original SFT policy's teacher-forced likelihood ranking.

The reward head uses the final real-token hidden state (not a padding position) from the 256-dimensional GPT backbone. Only the linear reward head was optimized for 80 steps; the BT training loss fell from **0.693147 to 0.019112**, and the reward-head weight-change norm was about **4.19385**. The backbone is not fine-tuned in this experiment.

| Evaluation split | Learned reward ranking | SFT completion-logP ranking |
|---|---:|---:|
| Three synthetic **training** pairs | **3/3** | **2/3** |
| Three manually designed **held-out** probes | **1/3** | **0/3** |

On the held-out probes, the reward margins were **−1.290** for `Say welcome`, **+2.590** for `Say please`, and **−2.829** for `Say goodbye`. The reward model corrected the SFT likelihood ranking of the training pair `Say no` but did not reliably transfer to the three unseen prompts. Scores are not directly comparable in magnitude across the two models: policy log-probability is not a learned preference reward. BT probabilities are not demonstrated to be calibrated to real human preferences.

**Modeling and evaluation boundaries:** Preference labels and probes were manually written, not sampled from a representative human-judgment distribution. The reward head used a frozen SFT Transformer; this is not full reward-model fine-tuning. Results are from one saved notebook run with three training and three held-out pairs, not a generalization benchmark, repeated-seed study, or causal analysis of failure. Each policy completion score includes the terminating newline character in the character tokenizer. The reward model's source-level wrapper is not yet reusable across arbitrary backbones, and no portable reward-model checkpoint was saved.

**Reusable math:** `completion_sequence_log_probs(logits, labels)` applies stable log-softmax, gathers already aligned completion targets, masks ignored prompt/padding labels, and **sums** (never silently averages) into one differentiable log-probability per candidate. `bradley_terry_loss(chosen_rewards, rejected_rewards)` uses `softplus(rejected - chosen)` with selectable loss reduction. Both live in [`src/llmfp/training/losses.py`](src/llmfp/training/losses.py), alongside the Lesson 22 mean-NLL helper. The [Lesson 23 extraction audit](docs/lesson_23_extraction_audit.md) documents their exact contracts, 12 focused tests and rejected abstraction candidates.

**Verification boundary:** The reported numeric outcomes and Plotly visualizations are saved outputs in the learner's notebook at commit `02f332f73c57680d46db99f8fbcc54d8260bd889`. The new tests were exercised separately on CPU with Python 3.13.5 and PyTorch 2.10.0, rather than on the user's PyTorch 2.14/CUDA environment. The complete upstream repository test suite, checkpoint-backed notebook and full Restart Kernel → Run All were not rerun during this extraction. Notebook cells and saved outputs were not modified.

The next conceptual step is **Lesson 24 — RLHF with PPO and KL Regularization**, which brings together SFT policy scores, preference rewards, reference policies and policy optimization.
