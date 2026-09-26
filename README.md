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

The project therefore starts below the Transformer itself: tensor operations, numerical stability, autodiff, tokenization, batching, losses, optimizers, and training loops — before building attention and eventually a complete decoder-only language model.

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

The notebooks are intended to be read in order.

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
| 15 | Linear Attention | 🚧 In Progress |

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

The long-term target is a small, extensible decoder-only language model.

The path is deliberately incremental:

```text
Foundations
├── tensors
├── scalar autograd
├── tokenization
├── sequences and batching
├── embeddings
├── language-modeling objectives
├── optimizers
└── training loop

Transformer fundamentals
├── causal self-attention
├── multi-head attention
├── normalization
├── residual connections
├── MLP / SwiGLU
├── positional information
└── decoder block

Tiny GPT
├── complete decoder-only model
├── end-to-end training
├── checkpointing
└── autoregressive generation

Inference
├── temperature
├── top-k / top-p sampling
└── KV cache

Attention variants
├── MHA
├── MQA
├── GQA
└── linear attention

Experiments
├── parameter count
├── training and validation loss
├── tokens / second
├── GPU memory
├── inference latency
└── KV-cache memory
```

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
│   └── 14_multi_query_and_grouped_query_attention.ipynb
│
├── src/
│   └── llmfp/
│       ├── nn/
│       │   ├── attention.py
│       │   ├── mlp.py
│       │   ├── rope.py
│       │   └── transformer.py
│       ├── models/
│       │   └── gpt.py
│       ├── generation/
│       │   └── cache.py
│       └── utils/
│           └── inspection.py
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

Lessons **00–14** are complete.

Currently working on:

```text
15_linear_attention.ipynb
```

The reusable attention implementation now treats the number of query heads and
key/value heads as independent architectural choices:

```text
H_KV = H_Q  → MHA
1 < H_KV < H_Q → GQA
H_KV = 1 → MQA
```

The verified `GroupedQueryAttention` implementation keeps the persistent KV
cache compact with shape

```text
(B, H_KV, T, D)
```

and expands KV heads only for the educational attention computation.

`GPTConfig` now exposes both `num_query_heads` and `num_kv_heads`, so the
same decoder-only model can be configured as MHA, GQA, or MQA.

The next lesson asks a different question:

> Can attention avoid explicitly constructing a full T × T attention matrix?

That leads to linear attention.
