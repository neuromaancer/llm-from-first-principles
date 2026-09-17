# LLM from First Principles

Building a small language model from scratch to understand every tensor, operation, and gradient.

The goal of this project is not to build the largest or most capable model possible.  
It is to understand how a modern decoder-only language model works by implementing its components step by step with minimal abstraction.

Instead of starting from high-level libraries such as Hugging Face `Trainer`, the project begins with basic tensor operations and gradually builds toward a trainable Transformer language model.

---

## Goals

By the end of this project, I want to be able to explain and implement:

- tensor shapes, dimensions, indexing, broadcasting, and matrix multiplication
- tokenization and vocabulary construction
- token and positional embeddings
- language modeling objectives and cross-entropy loss
- scaled dot-product attention
- causal self-attention
- multi-head attention (MHA)
- multi-query attention (MQA)
- grouped-query attention (GQA)
- Transformer blocks
- normalization, residual connections, and MLPs
- training loops and optimization
- autoregressive generation
- KV caching
- alternative attention mechanisms such as linear attention
- basic benchmarking of memory, throughput, and latency

The emphasis is always on understanding **what happens to the tensors**.

---

## Learning philosophy

This repository follows a few rules:

1. **Build from simple operations upward.**  
   Understand `reshape`, `transpose`, broadcasting, matrix multiplication, and softmax before using them inside attention.

2. **Predict tensor shapes before running the code.**  
   Shapes should become something that can be reasoned about, not something discovered through trial and error.

3. **Prefer explicit implementations first.**  
   For example, attention starts from:

   ```python
   q = x @ W_q
   k = x @ W_k
   v = x @ W_v

   scores = q @ k.transpose(-2, -1)
   weights = softmax(scores)
   output = weights @ v
   ```

   Higher-level abstractions come later.

4. **Refactor only after understanding the duplication.**  
   MHA, MQA, and GQA may initially be implemented separately before extracting their common structure.

5. **Keep lessons separate from production-style implementations.**  
   Educational code can be verbose and contain intermediate tensors and shape checks.

6. **Use AI as a teacher, not as an autocomplete engine.**  
   The coding environment used for this project disables AI code completion so that the implementation is typed and reasoned through manually.

---

## Development environment

Primary development environment:

- WSL2 / Linux
- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management
- PyTorch
- VS Code

The project environment is reproducible through:

```bash
uv sync
```

Run Python inside the project environment with:

```bash
uv run python
```

For example:

```bash
uv run python lessons/00_tensor_basics.py
```

The `.venv/` directory is local and is not committed to Git.

---

## VS Code setup

A dedicated VS Code profile is used for this project so that the learning environment remains minimal.

Recommended extensions:

- Python
- Pylance
- Python Debugger
- Ruff
- WSL
- Jupyter (optional, mainly for visualization and small experiments)

AI-assisted code completion is disabled for this profile.

Useful settings include:

```json
{
    "chat.disableAIFeatures": true,
    "editor.inlineSuggest.enabled": false,
    "python.analysis.typeCheckingMode": "basic",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "editor.rulers": [88],
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true
}
```

The project interpreter should point to:

```text
.venv/bin/python
```

---

## Roadmap

### Stage 0 — Tensor thinking

Learn the tensor operations that everything else will be built from:

- scalar, vector, matrix, tensor
- `ndim` and `shape`
- indexing and slicing
- `squeeze` / `unsqueeze`
- `reshape`
- `transpose`
- broadcasting
- element-wise operations
- matrix multiplication
- reductions
- softmax

### Stage 1 — Text and tokenization

Build the first language-model input pipeline:

- characters and tokens
- vocabulary
- encoding and decoding
- context windows
- input/target sequence construction
- batching

Start with a character-level tokenizer before moving to subword tokenization.

### Stage 2 — First language model

Implement a small baseline model:

- embedding lookup
- logits
- next-token prediction
- cross-entropy loss
- gradient descent
- training loop
- autoregressive sampling

A bigram language model provides the first end-to-end training system.

### Stage 3 — Attention

Build attention from tensor operations:

```text
input
  ↓
Q, K, V projections
  ↓
QKᵀ
  ↓
scaling
  ↓
causal mask
  ↓
softmax
  ↓
weighted sum of V
```

Then implement:

- scaled dot-product attention
- causal self-attention
- multi-head attention

### Stage 4 — Transformer

Combine the components into a decoder-only Transformer:

```text
tokens
  ↓
embeddings
  ↓
Transformer block × N
  ↓
normalization
  ↓
language-model head
  ↓
logits
```

Each block will eventually contain:

```text
x
├── normalization
├── attention
├── residual connection
├── normalization
├── MLP
└── residual connection
```

### Stage 5 — Train a Tiny GPT

Train a small decoder-only language model end to end.

Possible progression:

```text
tiny custom corpus
        ↓
Tiny Shakespeare
        ↓
TinyStories subset
        ↓
larger text corpora
```

The first useful models will remain deliberately small enough to train on modest hardware.

### Stage 6 — Modern attention variants

Once standard MHA is fully understood, extend the implementation with:

```text
attention/
├── mha.py
├── mqa.py
├── gqa.py
└── linear.py
```

The Transformer block should eventually depend only on a common attention interface rather than on a specific implementation.

This makes it possible to compare attention mechanisms while keeping the rest of the model fixed.

### Stage 7 — Benchmarking and deeper experiments

Compare variants using measurements such as:

- parameter count
- training loss
- validation loss
- tokens / second
- GPU memory usage
- inference latency
- KV-cache memory
- scaling with sequence length

Later experiments may also include:

- sliding-window attention
- alternative positional encodings
- RMSNorm vs LayerNorm
- different MLP activations
- weight tying
- mixed precision
- `torch.compile`
- custom kernels

---

## Planned repository structure

The repository will grow gradually rather than being generated all at once.

```text
llm-from-first-principles/
├── lessons/
│   ├── 00_tensor_basics.py
│   ├── 01_tokenization.py
│   ├── 02_embeddings.py
│   └── ...
│
├── src/
│   └── llmfp/
│       ├── data/
│       ├── tokenization/
│       ├── nn/
│       │   ├── attention/
│       │   ├── embedding.py
│       │   ├── normalization.py
│       │   ├── mlp.py
│       │   └── transformer_block.py
│       ├── models/
│       └── training/
│
├── scripts/
├── tests/
├── configs/
├── pyproject.toml
└── uv.lock
```

`lessons/` contains explicit educational implementations.

`src/` will contain cleaner reusable implementations created after the underlying concepts are understood.

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

Small teaching datasets may still be committed when they help explain an implementation.

---

## Git workflow

Development uses small, meaningful commits following the
[Conventional Commits](https://www.conventionalcommits.org/) format.

Examples:

```text
chore(env): initialize Python environment
feat(tensors): add tensor fundamentals lesson
feat(tokenizer): implement character tokenizer
feat(attention): implement scaled dot-product attention
test(attention): verify causal masking
refactor(attention): unify MHA and GQA projections
perf(attention): reduce KV cache memory usage
docs(readme): document project goals and roadmap
```

A change should normally be inspected before committing:

```bash
git status
git diff
git diff --staged
```

---

## Current status

The project is currently at:

```text
Stage 0 — Tensor thinking
```

The first goal is to develop an intuitive understanding of tensor dimensions and shape transformations before moving into matrix multiplication and attention.