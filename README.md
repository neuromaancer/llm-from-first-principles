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
| 18 | RL Foundations: Trajectories, Return, Value, Q, Advantage, Bellman Equations | 🚧 In Progress |
| 19 | Policy Gradients and REINFORCE from Scratch | Planned |
| 20 | Actor-Critic, TD Learning, and GAE | Planned |
| 21 | PPO from Scratch | Planned |

The RL section is intentionally focused. It does not try to reproduce a complete general-purpose RL curriculum; it develops the concepts needed to understand LLM post-training objectives from first principles.

### Part V — LLM Post-Training

| Lesson | Topic | Status |
|---|---|---|
| 22 | Supervised Fine-Tuning (SFT) | Planned |
| 23 | Preference Data, Sequence Log-Probabilities, and Reward Modeling | Planned |
| 24 | RLHF with PPO and KL Regularization | Planned |
| 25 | Direct Preference Optimization (DPO) | Planned |
| 26 | Modern LLM RL and Preference Optimization | Planned |

The post-training section will connect token-level language modeling to sequence-level optimization. It will make explicit the roles of completion log-probabilities, reference policies, KL penalties, pairwise preferences, learned rewards, and policy optimization before using higher-level training frameworks.

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
└── PPO

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
│   └── 17_mixture_of_experts.ipynb
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
│       │   └── tokenization.py
│       ├── models/
│       │   └── gpt.py
│       ├── generation/
│       │   ├── cache.py
│       │   ├── generate.py
│       │   └── sampling.py
│       ├── training/
│       │   ├── checkpoint.py
│       │   ├── evaluation.py
│       │   ├── schedules.py
│       │   └── steps.py
│       └── utils/
│           ├── inspection.py
│           └── metrics.py
│
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

Lessons **00–17** are complete.

Currently working on:

```text
18_rl_foundations.ipynb
```

Lesson 17 introduced learned conditional computation through sparse Mixture of Experts. Starting from the dense SwiGLU path, it developed token-level routing, top-k expert selection, dispatch/gather, router gradients, load balancing, capacity constraints, and the distinction between total and active parameters.

Reusable MoE components live in:

```text
src/llmfp/nn/moe.py
├── SparseMoE
├── expert_utilization
├── load_balancing_loss
└── expert_capacity
```

Lesson 18 begins the reinforcement-learning section from first principles. The goal is to understand the objects that later policy-gradient and PPO methods manipulate before introducing any policy-gradient estimator:

```text
state + action
    ↓
transition + reward
    ↓
trajectory
    ↓
return
    ↓
V(s), Q(s,a), advantage
    ↓
Bellman relationships
    ↓
Monte Carlo estimation
```

Only after these quantities are clear will the project move to policy gradients and REINFORCE.
