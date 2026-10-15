# PyTorch Function Tracker

This file tracks PyTorch functions and methods encountered while building the model.

The purpose is not to document the whole PyTorch API. It is a learning index for this repository.

## Rule

When a new PyTorch function or method appears in a lesson:

1. explain what it does before relying on it,
2. identify its important input/output shapes,
3. show a minimal example when the behavior is not obvious,
4. record it here,
5. distinguish in-place operations such as `scatter_` from out-of-place operations.

A function can move from **Introduced** to **Understood** once its behavior has been explained and used deliberately.

## Status

- ✅ **Understood** — already explained and used intentionally.
- 🟡 **Introduced** — recently encountered; keep an eye on it in the current lesson.

## Tensor creation and shape operations

| Function / method | Status | What it does |
|---|---|---|
| `torch.tensor` | ✅ | Creates a tensor from explicit values. |
| `torch.arange` | ✅ | Creates evenly spaced integer-like values over a range. |
| `torch.linspace` | ✅ | Creates evenly spaced floating-point values between endpoints, including them by default. |
| `torch.randperm` | ✅ | Returns a random permutation of integer indices, used to shuffle PPO sequence minibatches. |
| `torch.zeros` | ✅ | Creates a tensor filled with zeros. |
| `torch.zeros_like` | ✅ | Creates zeros with the same shape, dtype, and device as another tensor. |
| `torch.ones` | ✅ | Creates a tensor filled with ones. |
| `torch.ones_like` | ✅ | Creates ones with the same shape, dtype, and device as another tensor. |
| `torch.full_like` | ✅ | Creates a tensor with the same shape, dtype, and device as another tensor, filled with a chosen value. |
| `torch.randn` | ✅ | Samples values from a standard normal distribution. |
| `torch.randint` | ✅ | Samples integer values from a specified range. |
| `Tensor.reshape` | ✅ | Changes tensor shape without changing the logical values. |
| `Tensor.expand` | ✅ | Broadcasts singleton dimensions without copying the underlying values. |
| `Tensor.numel` | ✅ | Returns the total number of tensor elements, useful for detecting empty masks. |
| `Tensor.unsqueeze` | ✅ | Inserts a size-1 dimension. |
| `Tensor.squeeze` | ✅ | Removes size-1 dimensions. |
| `Tensor.transpose` | ✅ | Swaps two tensor dimensions. |
| `Tensor.flatten` | ✅ | Collapses a range of dimensions into one. |
| `torch.stack` | ✅ | Creates a new dimension and stacks tensors along it. |
| `torch.cat` | ✅ | Concatenates tensors along an existing dimension. |
| `torch.repeat_interleave` | ✅ | Repeats tensor entries a fixed number of times along a selected dimension. |

## Numerical and reduction operations

| Function / method | Status | What it does |
|---|---|---|
| `torch.exp` | ✅ | Applies the exponential function element-wise. |
| `torch.log` | ✅ | Applies the natural logarithm element-wise. |
| `torch.sqrt` | ✅ | Applies square root element-wise. |
| `torch.clamp` | ✅ | Limits each value to given bounds; PPO clips importance ratios in its objective. |
| `torch.minimum` | ✅ | Takes the element-wise minimum of two tensors; PPO uses it for its clipped surrogate. |
| `Tensor.square` | ✅ | Squares every tensor element. |
| `Tensor.mean` | ✅ | Computes a mean over selected dimensions. |
| `Tensor.var` | ✅ | Computes variance over selected dimensions. |
| `Tensor.sum` | ✅ | Computes a sum over selected dimensions. |
| `torch.cumsum` | ✅ | Computes cumulative sums along a dimension. |
| `torch.linalg.vector_norm` | ✅ | Computes vector norms along a selected dimension. |
| `torch.isfinite` | ✅ | Checks whether values are finite rather than NaN or infinity. |

## Indexing, filtering, and comparison

| Function / method | Status | What it does |
|---|---|---|
| `torch.maximum` | ✅ | Computes the element-wise maximum. |
| `torch.any` | ✅ | Checks whether at least one Boolean condition is true. |
| `torch.all` | ✅ | Checks whether all Boolean conditions are true. |
| `torch.where` | ✅ | Selects values element-wise according to a Boolean condition. |
| `Tensor.masked_fill` | ✅ | Replaces values where a Boolean mask is true. |
| `torch.topk` | ✅ | Returns the largest or smallest k values and their original indices. |
| `torch.sort` | ✅ | Sorts values and returns both sorted values and their original indices. |
| `Tensor.scatter_` | ✅ | Writes source values into selected destination indices in-place. |
| `torch.allclose` | ✅ | Checks approximate numerical equality between tensors. |
| `torch.equal` | ✅ | Checks whether two tensors have the same shape and exactly equal elements. |

## Probability and sampling

| Function / method | Status | What it does |
|---|---|---|
| `torch.softmax` / `F.softmax` | ✅ | Converts logits into a normalized probability distribution. |
| `torch.log_softmax` / `F.log_softmax` | ✅ | Converts logits into numerically stable log-probabilities along a chosen dimension. |
| `F.cross_entropy` | ✅ | Computes cross-entropy loss from logits and class targets. |
| `torch.multinomial` | ✅ | Samples indices according to categorical weights or probabilities. |
| `torch.argmax` | ✅ | Returns the index of the largest value. |

## Autograd and parameter handling

| Function / method | Status | What it does |
|---|---|---|
| `Tensor.backward` | ✅ | Runs reverse-mode automatic differentiation from a scalar loss. |
| `torch.autograd.grad` | ✅ | Computes gradients of selected outputs with respect to selected inputs without relying on parameter `.grad` accumulation. |
| `Tensor.detach` | ✅ | Returns a tensor detached from the autograd graph. |
| `Tensor.clone` | ✅ | Creates a separate tensor containing copied values. |
| `torch.no_grad` | ✅ | Disables autograd graph construction inside a context. |

## Optimization and training

| Function / method | Status | What it does |
|---|---|---|
| `torch.optim.SGD` | ✅ | PyTorch stochastic-gradient-descent optimizer used after manually deriving SGD in an earlier lesson. |
| `torch.optim.AdamW` | ✅ | PyTorch AdamW optimizer used for the end-to-end pretraining experiment. |
| `Optimizer.zero_grad` | ✅ | Clears or resets parameter gradients before the next backward pass. |
| `Optimizer.step` | ✅ | Applies one optimizer update using the currently accumulated gradients. |
| `torch.nn.utils.clip_grad_norm_` | ✅ | Clips the global parameter-gradient norm in-place to limit unusually large training updates. |

## Neural-network utilities

| Function / method | Status | What it does |
|---|---|---|
| `nn.Embedding` | ✅ | Maps integer token IDs to learned embedding vectors. |
| `nn.Linear` | ✅ | Applies an affine transformation to the final feature dimension. |
| `nn.Parameter` | ✅ | Registers a tensor as a trainable model parameter. |
| `nn.init.zeros_` | ✅ | Fills a parameter tensor with zeros in-place; used to initialize tabular actor/critic parameters deliberately. |
| `nn.ModuleList` | ✅ | Stores and registers a list of child modules. |
| `nn.RMSNorm` | ✅ | Applies RMS normalization over the final feature dimension. |
| `F.elu` | ✅ | Applies the ELU activation; used after deriving ELU(x) + 1 manually for the positive linear-attention feature map. |
| `F.gelu` | ✅ | Applies the GELU activation. |
| `F.silu` | ✅ | Applies the SiLU / Swish activation. |
| `F.scaled_dot_product_attention` | ✅ | Computes scaled dot-product attention using an optimized PyTorch primitive. |

## Lesson 13 — Generation additions

The following functions were introduced or revisited during autoregressive sampling:

- `torch.argmax`
- `torch.multinomial`
- `torch.topk`
- `torch.sort`
- `torch.cumsum`
- `torch.full_like`
- `Tensor.scatter_`
- `torch.equal`

These functions should be understood before the generation notebook moves on to KV caching.


## Lesson 14 — GQA additions

- `torch.repeat_interleave` — introduced after first implementing KV-head expansion manually.


## Lesson 15 — Linear-attention additions

- `F.elu` — used in reusable source only after the notebook derived the positive feature map manually.
- `torch.cumsum` — reused to construct causal prefix states for parallel linear attention.


## Lesson 19 — Policy-gradient additions

- `torch.log_softmax` — used to keep sampled-action log-probabilities differentiable and numerically stable.
- `torch.autograd.grad` — used to inspect score-function gradients such as `d log pi(a) / d theta` directly.
- `torch.multinomial` — revisited as categorical action sampling; the sampled index is discrete and gradients do not pass through the sampling operation.
- `Tensor.detach` — revisited to separate stochastic sampling / credit weights from the differentiable log-probability path.


## Lesson 20 — Actor-critic and GAE additions

- `nn.init.zeros_` — used to initialize the tabular actor and critic at a known neutral starting point; the trailing underscore indicates an in-place operation.
- `torch.no_grad` — revisited to freeze bootstrap targets and rollout-time measurements during critic/actor updates.
- `Tensor.detach` — revisited to keep TD/advantage credit signals fixed when optimizing the actor.
- `torch.zeros_like` — reused to allocate GAE advantage tensors with matching shape, dtype, and device.

## Lesson 21A — PPO from Scratch additions

- `torch.clamp` and `torch.minimum` — construct the advantage-aware clipped PPO surrogate; clipping is not a hard constraint on updated policy probabilities.
- `torch.randperm` — shuffle complete trajectory indices before selecting aligned sequence minibatches.
- `torch.linspace` — display ratio-dependent objective curves for positive and negative advantages.
- `Tensor.expand` — share a small position template across a batch without duplicating data.
- `Tensor.numel` — reject masked reductions with no valid actions.
- `torch.any` and `torch.all` — validate action position, padding, and termination invariants.

The RL primitives extracted after Lesson 21A live in `src/llmfp/rl/`; their regression tests use only the Python standard library and PyTorch.

---

## API Reference — Examples and Parameter Guides

This is the detailed companion to the quick-index tables above. Every API entry includes the call signature, descriptions of its parameters (including optional arguments), one runnable usage example, and shape/semantic notes. **All examples assume the following imports:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Defaults can evolve between PyTorch releases; the examples target the repository's PyTorch API. A displayed `out=None` or `generator=None` is an explicit demonstration of an optional argument. Some overloads or mutually exclusive options require separate calls; parameter descriptions cover their intended meaning, and examples choose one valid combination. `Tensor` denotes a method on an existing tensor and `Optimizer` denotes an optimizer instance. Individual examples are meant to be run with their included local setup lines after the imports.

### A. Tensor creation and shape operations

#### `torch.tensor`

**Signature:** `torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)`

**Parameters:** `data` — scalar, list or array-like source; `dtype` — requested element type (otherwise inferred); `device` — CPU/CUDA destination; `requires_grad` — record floating/complex gradients; `pin_memory` — allocate pinned CPU memory.

```python
x = torch.tensor(data=[[1., 2.], [3., 4.]], dtype=torch.float32,
                 device="cpu", requires_grad=False, pin_memory=False)
```

**Shape / use:** (B=2, C=2): explicit 2×2 tensor.

#### `torch.arange`

**Signature:** `torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`

**Parameters:** `start` — inclusive beginning; `end` — exclusive stop; `step` — signed spacing; `out` — optional preallocated output; `dtype` — element type; `layout` — storage layout (usually strided); `device` — destination; `requires_grad` — autograd tracking.

```python
indices = torch.arange(start=1, end=7, step=2, out=None,
                       dtype=torch.int64, layout=torch.strided,
                       device="cpu", requires_grad=False)
```

**Shape / use:** (T=3,): `[1,3,5]`, end excluded.

#### `torch.linspace`

**Signature:** `torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`

**Parameters:** `start` — inclusive first value; `end` — inclusive last value; `steps` — output count; `out` — optional destination; `dtype` — element type; `layout` — tensor layout; `device` — destination; `requires_grad` — gradient tracking.

```python
grid = torch.linspace(start=0., end=1., steps=5, out=None,
                      dtype=torch.float32, layout=torch.strided,
                      device="cpu", requires_grad=False)
```

**Shape / use:** (N=5,): five values from 0 to 1.

#### `torch.randperm`

**Signature:** `torch.randperm(n, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)`

**Parameters:** `n` — number of shuffled indices 0…n−1; `generator` — optional reproducible RNG; `out` — preallocated destination; `dtype` — integer type; `layout` — layout; `device` — destination; `requires_grad` — gradient tracking (normally False); `pin_memory` — pinned CPU allocation.

```python
rng = torch.Generator(device="cpu").manual_seed(42)
order = torch.randperm(n=5, generator=rng, out=None,
                       dtype=torch.int64, layout=torch.strided,
                       device="cpu", requires_grad=False, pin_memory=False)
```

**Shape / use:** (B=5,): one permutation for trajectory minibatches.

#### `torch.zeros`

**Signature:** `torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)`

**Parameters:** `*size` — dimensions (tuple or positional); `out` — optional destination; `dtype` — element type; `layout` — storage format; `device` — CPU/CUDA; `requires_grad` — track gradients; `pin_memory` — pinned CPU allocation.

```python
z = torch.zeros(2, 3, out=None, dtype=torch.float32,
                layout=torch.strided, device="cpu",
                requires_grad=False, pin_memory=False)
```

**Shape / use:** (B=2, T=3): zero-valued reward table.

#### `torch.zeros_like`

**Signature:** `torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)`

**Parameters:** `input` — reference shape; `dtype` — override dtype or inherit; `layout` — override storage layout; `device` — override device; `requires_grad` — gradient flag; `memory_format` — layout preservation policy.

```python
reference = torch.ones(2, 3)
z = torch.zeros_like(input=reference, dtype=torch.float32,
                     layout=torch.strided, device="cpu",
                     requires_grad=False, memory_format=torch.preserve_format)
```

**Shape / use:** (B=2, T=3): zero tensor matching a reference.

#### `torch.ones`

**Signature:** `torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)`

**Parameters:** `*size` — output dimensions; `out` — optional buffer; `dtype` — element type; `layout` — storage layout; `device` — destination; `requires_grad` — gradient tracking; `pin_memory` — pinned CPU allocation.

```python
mask = torch.ones(2, 3, out=None, dtype=torch.float32,
                  layout=torch.strided, device="cpu",
                  requires_grad=False, pin_memory=False)
```

**Shape / use:** (B=2, T=3): bootstrap switches initially set to one.

#### `torch.ones_like`

**Signature:** `torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)`

**Parameters:** `input` — reference shape; `dtype` — dtype choice; `layout` — tensor layout; `device` — CPU/CUDA; `requires_grad` — gradient tracking; `memory_format` — layout preservation.

```python
reference = torch.zeros(2, 3)
mask = torch.ones_like(input=reference, dtype=torch.float32,
                       layout=torch.strided, device="cpu",
                       requires_grad=False, memory_format=torch.preserve_format)
```

**Shape / use:** (B=2, T=3): same shape as reference.

#### `torch.full_like`

**Signature:** `torch.full_like(input, fill_value, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)`

**Parameters:** `input` — reference tensor; `fill_value` — scalar used everywhere; `dtype` — type; `layout` — layout; `device` — device; `requires_grad` — gradient flag; `memory_format` — storage-memory policy.

```python
reference = torch.zeros(2, 3)
padded = torch.full_like(input=reference, fill_value=-1.,
                         dtype=torch.float32, layout=torch.strided,
                         device="cpu", requires_grad=False,
                         memory_format=torch.preserve_format)
```

**Shape / use:** (B=2, T=3): filled with −1.

#### `torch.randn`

**Signature:** `torch.randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)`

**Parameters:** `*size` — output dimensions; `generator` — optional RNG; `out` — optional destination; `dtype` — element type; `layout` — storage layout; `device` — device; `requires_grad` — autograd; `pin_memory` — pinned CPU allocation.

```python
noise = torch.randn(2, 3, generator=None, out=None,
                    dtype=torch.float32, layout=torch.strided,
                    device="cpu", requires_grad=False, pin_memory=False)
```

**Shape / use:** (B=2, C=3): standard-normal random numbers.

#### `torch.randint`

**Signature:** `torch.randint(low, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`

**Parameters:** `low` — inclusive smallest integer; `high` — exclusive upper bound; `size` — output shape; `generator` — RNG; `out` — optional destination; `dtype` — integer type; `layout` — storage layout; `device` — device; `requires_grad` — gradient tracking (normally False).

```python
kinds = torch.randint(low=1, high=3, size=(8,), generator=None,
                      out=None, dtype=torch.int64, layout=torch.strided,
                      device="cpu", requires_grad=False)
```

**Shape / use:** (B=8,): prompt kinds 1 or 2.

#### `Tensor.reshape`

**Signature:** `x.reshape(*shape)`

**Parameters:** `*shape` — target dimensions; `one dimension may be −1 for inferenc` — e dimension may be −1 for inference; `new dimensions must preserve element coun` — w dimensions must preserve element count.

```python
x = torch.arange(6)
y = x.reshape(2, 3)
```

**Shape / use:** (B=2, T=3): reorganizes six elements; may return a view or copy.

#### `Tensor.expand`

**Signature:** `x.expand(*sizes)`

**Parameters:** `*sizes` — expanded shape; `only singleton dimensions can be broadcas` — ly singleton dimensions can be broadcast; `expansion does not copy storag` — pansion does not copy storage.

```python
x = torch.tensor([[2], [3]])  # (B=2, 1)
y = x.expand(2, 4)         # (B=2, T=4)
```

**Shape / use:** Expanded entries share storage; avoid in-place writes.

#### `Tensor.numel`

**Signature:** `x.numel()`

**Parameters:** `No parameters` — returns the total element count, even for tensors with zero dimensions..

```python
valid = torch.tensor([True, False, True])
count = valid.numel()  # 3
```

**Shape / use:** Returns Python `int`; for count of valid actions use `valid.sum()` instead.

#### `Tensor.unsqueeze`

**Signature:** `x.unsqueeze(dim)`

**Parameters:** `dim` — insertion index for a new size-1 axis; `negative axes count from the en` — gative axes count from the end.

```python
x = torch.tensor([1, 2, 3])  # (T=3,)
y = x.unsqueeze(dim=0)     # (B=1, T=3)
```

**Shape / use:** Creates a singleton axis.

#### `Tensor.squeeze`

**Signature:** `x.squeeze(dim=None)`

**Parameters:** `dim` — optional axis (or tuple of axes in recent PyTorch); `None removes all size-1 dimension` — ne removes all size-1 dimensions.

```python
x = torch.ones(2, 1, 3)
y = x.squeeze(dim=1)  # (B=2, C=3)
```

**Shape / use:** Prefer an explicit `dim` to avoid accidentally removing batch size 1.

#### `Tensor.transpose`

**Signature:** `x.transpose(dim0, dim1)`

**Parameters:** `dim0` — first axis to swap; `dim1` — second axis to swap.

```python
x = torch.zeros(2, 3, 4)  # (B,T,C)
y = x.transpose(dim0=1, dim1=2)  # (B,C,T)
```

**Shape / use:** Swaps exactly two axes; often creates a view.

#### `Tensor.flatten`

**Signature:** `x.flatten(start_dim=0, end_dim=-1)`

**Parameters:** `start_dim` — first axis to merge; `end_dim` — last axis to merge (inclusive).

```python
x = torch.zeros(2, 3, 4)        # (B,T,C)
y = x.flatten(start_dim=1, end_dim=2)  # (B,T*C)
```

**Shape / use:** Collapses a contiguous range of logical dimensions.

#### `torch.stack`

**Signature:** `torch.stack(tensors, dim=0, *, out=None)`

**Parameters:** `tensors` — same-shaped sequence of tensors; `dim` — new dimension insertion position; `out` — optional output tensor.

```python
rows = [torch.tensor([1, 2]), torch.tensor([3, 4])]
y = torch.stack(tensors=rows, dim=0, out=None)
```

**Shape / use:** (B=2, T=2): creates a NEW axis.

#### `torch.cat`

**Signature:** `torch.cat(tensors, dim=0, *, out=None)`

**Parameters:** `tensors` — nonempty sequence, same shape except concat axis; `dim` — EXISTING concatenation axis; `out` — optional destination.

```python
a = torch.tensor([[1, 2]])  # (B=1, T=2)
b = torch.tensor([[3]])     # (B=1, T=1)
y = torch.cat(tensors=(a, b), dim=1, out=None)
```

**Shape / use:** (B=1, T=3): appends a token, not a batch.

#### `torch.repeat_interleave`

**Signature:** `torch.repeat_interleave(input, repeats, dim=None, *, output_size=None)`

**Parameters:** `input` — source tensor; `repeats` — scalar count or per-element counts; `dim` — repeat axis (None flattens); `output_size` — optional total output length on that axis.

```python
x = torch.tensor([[10, 20]])
y = torch.repeat_interleave(input=x, repeats=2, dim=1,
                            output_size=4)
```

**Shape / use:** (B=1, T=4): `[10,10,20,20]`; also used for KV head expansion.

### B. Numerical operations, reductions and indexing

#### `torch.exp`

**Signature:** `torch.exp(input, *, out=None)`

**Parameters:** `input` — tensor of exponent arguments; `out` — optional output buffer.

```python
x = torch.tensor([0., 1.])
y = torch.exp(input=x, out=None)
```

**Shape / use:** (N=2,): element-wise eˣ, used to convert log-ratios to importance ratios.

#### `torch.log`

**Signature:** `torch.log(input, *, out=None)`

**Parameters:** `input` — tensor of positive numbers; `out` — optional output buffer.

```python
p = torch.tensor([0.2, 0.8])
log_p = torch.log(input=p, out=None)
```

**Shape / use:** (N=2,): natural logarithms; zero yields −inf.

#### `torch.sqrt`

**Signature:** `torch.sqrt(input, *, out=None)`

**Parameters:** `input` — tensor of nonnegative real numbers; `out` — optional destination.

```python
x = torch.tensor([4., 9.])
y = torch.sqrt(input=x, out=None)
```

**Shape / use:** (N=2,): square roots `[2,3]`; used in attention scaling.

#### `torch.clamp`

**Signature:** `torch.clamp(input, min=None, max=None, *, out=None)`

**Parameters:** `input` — tensor to bound; `min` — optional lower bound; `max` — optional upper bound; `out` — optional output buffer.

```python
ratios = torch.tensor([0.5, 1.0, 1.5])
clipped = torch.clamp(input=ratios, min=0.8, max=1.2, out=None)
```

**Shape / use:** (T=3,): PPO probability-ratio clipping `[0.8,1.0,1.2]`.

#### `torch.minimum`

**Signature:** `torch.minimum(input, other, *, out=None)`

**Parameters:** `input` — first tensor; `other` — second tensor (broadcastable); `out` — optional result buffer.

```python
raw = torch.tensor([3., -2.])
clipped = torch.tensor([2., -1.])
surrogate = torch.minimum(input=raw, other=clipped, out=None)
```

**Shape / use:** (T=2,): pointwise pessimistic PPO objective.

#### `Tensor.square`

**Signature:** `x.square()`

**Parameters:** `No parameters` — squares every element of the receiver tensor.

```python
x = torch.tensor([-2., 3.])
y = x.square()
```

**Shape / use:** (N=2,): `[4,9]`; useful for squared value error.

#### `Tensor.mean`

**Signature:** `x.mean(dim=None, keepdim=False, *, dtype=None)`

**Parameters:** `dim` — axis or axes to average, None means all values; `keepdim` — retain reduced axes as length 1; `dtype` — optional accumulation/output type.

```python
x = torch.tensor([[2., 4.], [6., 8.]])
y = x.mean(dim=1, keepdim=True, dtype=torch.float32)
```

**Shape / use:** (B=2, 1): averages across sequence positions.

#### `Tensor.var`

**Signature:** `x.var(dim=None, correction=1, keepdim=False)`

**Parameters:** `dim` — axes to reduce, None reduces all; `correction` — degrees-of-freedom adjustment (1 for sample variance, 0 for population variance); `keepdim` — preserve size-1 axes.

```python
x = torch.tensor([[1., 2., 3.], [2., 4., 6.]])
y = x.var(dim=1, correction=0, keepdim=False)
```

**Shape / use:** (B=2,): population variance within each sequence.

#### `Tensor.sum`

**Signature:** `x.sum(dim=None, keepdim=False, dtype=None)`

**Parameters:** `dim` — axis or axes, None reduces entire tensor; `keepdim` — retain reduced axis; `dtype` — optional accumulation/output dtype.

```python
rewards = torch.tensor([[0., 1.], [0., 0.5]])
y = rewards.sum(dim=1, keepdim=False, dtype=torch.float32)
```

**Shape / use:** (B=2,): total observed reward per trajectory.

#### `torch.cumsum`

**Signature:** `torch.cumsum(input, dim, *, dtype=None, out=None)`

**Parameters:** `input` — source tensor; `dim` — dimension along which to accumulate; `dtype` — optionally change accumulation type; `out` — optional output buffer.

```python
x = torch.tensor([1, 2, 3])
prefix = torch.cumsum(input=x, dim=0, dtype=torch.int64, out=None)
```

**Shape / use:** (T=3,): prefix sums `[1,3,6]`, used in causal prefix computations.

#### `torch.linalg.vector_norm`

**Signature:** `torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None)`

**Parameters:** `x` — input; `ord` — norm order (2 is Euclidean); `dim` — axes to reduce; `keepdim` — preserve reduced axes; `dtype` — optional computation type; `out` — optional result buffer.

```python
x = torch.tensor([[3., 4.], [5., 12.]])
norms = torch.linalg.vector_norm(x=x, ord=2, dim=1,
                                 keepdim=False, dtype=torch.float32, out=None)
```

**Shape / use:** (B=2,): Euclidean norms `[5,13]`.

#### `torch.isfinite`

**Signature:** `torch.isfinite(input)`

**Parameters:** `input` — tensor to test for finite numbers (rejects NaN, +inf and −inf).

```python
x = torch.tensor([1., float("nan"), float("inf")])
finite = torch.isfinite(input=x)
```

**Shape / use:** (N=3,): Boolean mask `[True,False,False]`.

#### `torch.maximum`

**Signature:** `torch.maximum(input, other, *, out=None)`

**Parameters:** `input` — first operand; `other` — broadcastable second operand; `out` — optional destination.

```python
a = torch.tensor([-1., 3.])
b = torch.zeros(2)
y = torch.maximum(input=a, other=b, out=None)
```

**Shape / use:** (N=2,): element-wise larger of two values.

#### `torch.any`

**Signature:** `torch.any(input, dim=None, keepdim=False, *, out=None)`

**Parameters:** `input` — Boolean/numeric conditions; `dim` — optional reduction axis (None means all); `keepdim` — preserve axis; `out` — optional result tensor.

```python
invalid = torch.tensor([[False, True], [False, False]])
flag = torch.any(input=invalid, dim=1, keepdim=False, out=None)
```

**Shape / use:** (B=2,): True if any invalid position in each row.

#### `torch.all`

**Signature:** `torch.all(input, dim=None, keepdim=False, *, out=None)`

**Parameters:** `input` — Boolean/numeric condition tensor; `dim` — optional axis (None checks all); `keepdim` — preserve axis; `out` — optional destination.

```python
valid = torch.tensor([[True, True], [True, False]])
flags = torch.all(input=valid, dim=1, keepdim=False, out=None)
```

**Shape / use:** (B=2,): True only for fully valid rows.

#### `torch.where`

**Signature:** `torch.where(condition, input, other, *, out=None)`

**Parameters:** `condition` — Boolean selection mask; `input` — values where True; `other` — values where False; `out` — optional destination; `a one-argument overload instead returns indice` — one-argument overload instead returns indices.

```python
mask = torch.tensor([True, False])
chosen = torch.where(condition=mask,
                     input=torch.tensor([5, 5]),
                     other=torch.tensor([0, 0]), out=None)
```

**Shape / use:** (N=2,): `[5,0]`, useful for selecting valid action positions.

#### `Tensor.masked_fill`

**Signature:** `x.masked_fill(mask, value)`

**Parameters:** `mask` — broadcastable Boolean mask; `value` — scalar replacement for True positions.

```python
x = torch.tensor([[1., 2., 3.]])
y = x.masked_fill(mask=torch.tensor([[False, True, False]]),
                  value=0.0)
```

**Shape / use:** (B=1,T=3): masks one padded action.

#### `torch.topk`

**Signature:** `torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)`

**Parameters:** `input` — source scores; `k` — number to select; `dim` — selection axis; `largest` — True chooses largest; `sorted` — sort selected scores; `out` — optional pair of buffers (values, indices).

```python
scores = torch.tensor([[0.2, 0.5, 0.3]])
values, ids = torch.topk(input=scores, k=2, dim=-1,
                         largest=True, sorted=True, out=None)
```

**Shape / use:** (B=1,K=2): highest scores and matching vocabulary indices.

#### `torch.sort`

**Signature:** `torch.sort(input, dim=-1, descending=False, stable=False, *, out=None)`

**Parameters:** `input` — tensor to sort; `dim` — sorting axis; `descending` — reverse order; `stable` — preserve order of equal keys; `out` — optional (values,indices) buffers.

```python
scores = torch.tensor([0.3, 0.1, 0.2])
values, indices = torch.sort(input=scores, dim=0,
                             descending=True, stable=True, out=None)
```

**Shape / use:** (N=3,): sorted scores and their original indices.

#### `Tensor.scatter_`

**Signature:** `x.scatter_(dim, index, src, reduce=None)`

**Parameters:** `dim` — destination axis; `index` — integer destination indices; `src` — tensor source or scalar value; `reduce` — optional legacy reduction (None, 'add' or 'multiply'; `prefer `scatter_reduce_` for modern reductions` — efer `scatter_reduce_` for modern reductions); `trailing underscore means in-plac` — ailing underscore means in-place.

```python
x = torch.zeros(1, 4)
indices = torch.tensor([[1, 3]])
source = torch.tensor([[5., 6.]])
x.scatter_(dim=1, index=indices, src=source)
# Optional legacy reduction uses reduce="add" (deprecated).
# Prefer scatter_reduce_ for reduction-based writes.
```

**Shape / use:** (B=1,V=4): writes 5 into position 1 and 6 into position 3.

#### `torch.allclose`

**Signature:** `torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)`

**Parameters:** `input` — first tensor; `other` — reference tensor; `rtol` — relative tolerance; `atol` — absolute tolerance; `equal_nan` — treat NaNs at identical positions as equal.

```python
is_close = torch.allclose(input=torch.tensor([1.000001]),
                          other=torch.tensor([1.0]),
                          rtol=1e-5, atol=1e-8, equal_nan=False)
```

**Shape / use:** Returns Python `bool`, useful for numerical invariants.

#### `torch.equal`

**Signature:** `torch.equal(input, other)`

**Parameters:** `input` — first tensor; `other` — second tensor, must match shape and all elements exactly.

```python
same = torch.equal(input=torch.tensor([1, 2]),
                   other=torch.tensor([1, 2]))
```

**Shape / use:** Returns Python `bool`; unlike allclose, no numeric tolerance.

### C. Probability, sampling and cross-entropy

#### `torch.softmax / F.softmax`

**Signature:** `torch.softmax(input, dim, dtype=None); F.softmax(input, dim=None, _stacklevel=3, dtype=None)`

**Parameters:** `input` — unnormalized logits; `dim` — vocabulary/normalization axis; `dtype` — optional computation/output dtype; `_stacklevel` — internal warning stack-depth argument of F.softmax, normally left at 3.

```python
logits = torch.tensor([[1., 2., 3.]])  # (B,V)
p = torch.softmax(input=logits, dim=-1, dtype=torch.float32)
p_f = F.softmax(input=logits, dim=-1, _stacklevel=3,
                dtype=torch.float32)
```

**Shape / use:** (B=1,V=3): probabilities sum to one over the vocabulary axis.

#### `torch.log_softmax / F.log_softmax`

**Signature:** `torch.log_softmax(input, dim, dtype=None); F.log_softmax(input, dim=None, _stacklevel=3, dtype=None)`

**Parameters:** `input` — logits tensor; `dim` — softmax axis; `dtype` — optional computation type; `_stacklevel` — F.log_softmax's internal warning stack-depth argument, usually 3.

```python
logits = torch.tensor([[1., 2., 3.]])  # (B,V)
log_p = torch.log_softmax(input=logits, dim=-1, dtype=torch.float32)
log_p_f = F.log_softmax(input=logits, dim=-1, _stacklevel=3,
                        dtype=torch.float32)
```

**Shape / use:** (B=1,V=3): numerically stable log-probabilities.

#### `F.cross_entropy`

**Signature:** `F.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)`

**Parameters:** `input` — class logits `(B,V)` or `(B,V,...)`; `target` — class indices `(B,)` or class probabilities; `weight` — per-class weights; `size_average` — deprecated reduction override, use None; `ignore_index` — target index excluded from loss; `reduce` — deprecated reduction override, use None; `reduction` — 'none' / 'mean' / 'sum'; `label_smoothing` — mixing of one-hot targets with uniform probability.

```python
logits = torch.tensor([[1., 2., 3.], [2., 1., 0.]]) # (B,V)
targets = torch.tensor([2, 0])                       # (B,)
loss = F.cross_entropy(input=logits, target=targets,
                       weight=None, size_average=None,
                       ignore_index=-100, reduce=None,
                       reduction="mean", label_smoothing=0.0)
```

**Shape / use:** Scalar NLL/cross-entropy for two labeled sequences; deprecated flags left `None`.

#### `torch.multinomial`

**Signature:** `torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)`

**Parameters:** `input` — nonnegative categorical weights (normalization not required); `num_samples` — draws per row; `replacement` — whether a category may be sampled twice; `generator` — optional reproducible RNG; `out` — optional output tensor.

```python
weights = torch.tensor([[0.1, 0.2, 0.7]])  # (B,V)
draws = torch.multinomial(input=weights, num_samples=1,
                          replacement=False, generator=None, out=None)
```

**Shape / use:** (B=1,K=1): sampled integer token IDs; the sampling operation is discrete.

#### `torch.argmax`

**Signature:** `torch.argmax(input, dim=None, keepdim=False)`

**Parameters:** `input` — values/scores; `dim` — axis to search (None flattens); `keepdim` — retain selected axis of length 1.

```python
logits = torch.tensor([[0.1, 0.9, 0.3]])
ids = torch.argmax(input=logits, dim=-1, keepdim=True)
```

**Shape / use:** (B=1,1): greedy next-token index, not a differentiable sample.

### D. Autograd and parameter handling

#### `Tensor.backward`

**Signature:** `x.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None)`

**Parameters:** `gradient` — upstream gradient for non-scalar x (None for scalar); `retain_graph` — keep forward graph after backward; `create_graph` — track gradients of derivatives for higher order differentiation; `inputs` — optional subset of leaf tensors that receive gradients.

```python
x = torch.tensor([1., 2.], requires_grad=True)
loss = x.square().sum()  # scalar
loss.backward(gradient=None, retain_graph=False,
              create_graph=False, inputs=[x])
```

**Shape / use:** `x.grad` becomes `[2,4]`; default backward accumulates into `.grad`.

#### `torch.autograd.grad`

**Signature:** `torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False)`

**Parameters:** `outputs` — differentiated output(s); `inputs` — tensor(s) to differentiate against; `grad_outputs` — upstream vector(s) for nonscalar outputs; `retain_graph` — preserve backward graph; `create_graph` — construct higher-order gradient graph; `only_inputs` — legacy flag (ignored; `gradients only for inputs` — adients only for inputs); `allow_unused` — whether disconnected inputs are allowed (None follows materialize_grads); `is_grads_batched` — first grad_outputs dimension indexes independent vector-Jacobian products; `materialize_grads` — return zeros for unused inputs instead of None.

```python
x = torch.tensor([1., 2.], requires_grad=True)
loss = x.square().sum()
(grad,) = torch.autograd.grad(outputs=loss, inputs=(x,),
    grad_outputs=None, retain_graph=False, create_graph=False,
    only_inputs=True, allow_unused=False, is_grads_batched=False,
    materialize_grads=False)
```

**Shape / use:** (N=2,): returns gradients without accumulating into `x.grad`.

#### `Tensor.detach`

**Signature:** `x.detach()`

**Parameters:** `No parameters` — returns a tensor sharing data but severed from its autograd history.

```python
x = torch.tensor([1., 2.], requires_grad=True)
y = (x * 2).detach()
```

**Shape / use:** (N=2,): gradient does not flow from `y` back to `x`; mutable storage can be shared.

#### `Tensor.clone`

**Signature:** `x.clone(*, memory_format=torch.preserve_format)`

**Parameters:** `memory_format` — requested memory arrangement, typically torch.preserve_format.

```python
x = torch.tensor([1., 2.], requires_grad=True)
y = x.clone(memory_format=torch.preserve_format)
```

**Shape / use:** (N=2,): copies storage while retaining autograd connection unless detached.

#### `torch.no_grad`

**Signature:** `torch.no_grad()`

**Parameters:** `No parameters` — context manager or function decorator that disables reverse-mode autograd recording.

```python
x = torch.tensor([1.], requires_grad=True)
with torch.no_grad():
    frozen = x * 2  # requires_grad=False
```

**Shape / use:** Use for rollout collection and fixed targets; does not put the module in eval mode.

### E. Optimization and optimizers

#### `torch.optim.SGD`

**Signature:** `torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None)`

**Parameters:** `params` — iterable of trainable parameters or parameter groups; `lr` — learning rate; `momentum` — velocity mixing; `dampening` — reduces new momentum contribution; `weight_decay` — L2 regularization; `nesterov` — Nesterov-style momentum (requires positive momentum and zero dampening); `maximize` — gradient ascent instead of descent; `foreach` — fused tensor-list kernel preference; `differentiable` — track optimizer step for higher-order gradients; `fused` — fused kernel preference when supported.

```python
p = nn.Parameter(torch.tensor([1.]))
opt = torch.optim.SGD(params=[p], lr=0.1, momentum=0.9,
    dampening=0.0, weight_decay=0.0, nesterov=True,
    maximize=False, foreach=None, differentiable=False, fused=None)
```

**Shape / use:** Creates SGD optimizer for a single trainable weight.

#### `torch.optim.AdamW`

**Signature:** `torch.optim.AdamW(params, lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)`

**Parameters:** `params` — trainable parameter iterable / groups; `lr` — step size; `betas` — exponential moving average rates for first/second moments; `eps` — denominator stability; `weight_decay` — decoupled regularization strength; `amsgrad` — use max-of-second-moments variant; `maximize` — maximize objective rather than minimize; `foreach` — batched tensor-kernel option; `capturable` — permit accelerator graph capture; `differentiable` — enable differentiation through optimizer step; `fused` — fused implementation when available.

```python
p = nn.Parameter(torch.tensor([1.]))
opt = torch.optim.AdamW(params=[p], lr=1e-3, betas=(0.9,0.999),
    eps=1e-8, weight_decay=0.01, amsgrad=False, maximize=False,
    foreach=None, capturable=False, differentiable=False, fused=None)
```

**Shape / use:** Optimizer used for Tiny GPT and PPO Actor/Critic updates.

#### `Optimizer.zero_grad`

**Signature:** `opt.zero_grad(set_to_none=True)`

**Parameters:** `set_to_none` — True removes stored gradient tensors, whereas False fills existing gradients with zero.

```python
p = nn.Parameter(torch.tensor([1.]))
opt = torch.optim.SGD([p], lr=0.1)
opt.zero_grad(set_to_none=True)
```

**Shape / use:** Run before each independent backward/update; some unused gradients remain `None`.

#### `Optimizer.step`

**Signature:** `opt.step(closure=None)`

**Parameters:** `closure` — optional function that recomputes loss and gradients (needed by some optimizers such as LBFGS); `None for common SGD/AdamW loop` — ne for common SGD/AdamW loops.

```python
p = nn.Parameter(torch.tensor([1.]))
opt = torch.optim.SGD([p], lr=0.1)
(p.square().sum()).backward()
opt.step(closure=None)
```

**Shape / use:** Applies accumulated gradients to parameter values; no graph retained by default.

#### `torch.nn.utils.clip_grad_norm_`

**Signature:** `torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None)`

**Parameters:** `parameters` — iterable/tensor of model parameters with gradients; `max_norm` — allowed total norm; `norm_type` — p-norm order (2 is Euclidean, inf uses max); `error_if_nonfinite` — fail on NaN/inf total norm; `foreach` — implementation preference.

```python
model = nn.Linear(2, 1)
model(torch.ones(1, 2)).sum().backward()
original_norm = torch.nn.utils.clip_grad_norm_(
    parameters=model.parameters(), max_norm=1.0,
    norm_type=2.0, error_if_nonfinite=True, foreach=None)
```

**Shape / use:** Returns norm before clipping; trailing underscore denotes in-place modification of gradients.

### F. Neural-network layers and operations

#### `nn.Embedding`

**Signature:** `nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)`

**Parameters:** `num_embeddings` — vocabulary/table row count; `embedding_dim` — vector width; `padding_idx` — optionally excluded padding row from gradient updates; `max_norm` — optionally renormalize rows on forward; `norm_type` — p-norm used with max_norm; `scale_grad_by_freq` — rescale by inverse batch frequency; `sparse` — sparse weight gradients; `_weight` — optional supplied weight tensor; `_freeze` — internal option to freeze embedding weights; `device` — device; `dtype` — parameter type.

```python
embedding = nn.Embedding(num_embeddings=5, embedding_dim=8,
    padding_idx=None, max_norm=None, norm_type=2.0,
    scale_grad_by_freq=False, sparse=False,
    _weight=None, _freeze=False, device="cpu", dtype=torch.float32)
token_vectors = embedding(torch.tensor([[0, 1, 2]]))
```

**Shape / use:** (B=1,T=3,C=8): token IDs become trainable dense vectors.

#### `nn.Linear`

**Signature:** `nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)`

**Parameters:** `in_features` — input channel width; `out_features` — output channel width; `bias` — learn additive offset; `device` — location of parameters; `dtype` — parameter type.

```python
head = nn.Linear(in_features=8, out_features=5,
                 bias=False, device="cpu", dtype=torch.float32)
logits = head(torch.randn(2, 3, 8))
```

**Shape / use:** (B=2,T=3,V=5): final feature dimension is projected.

#### `nn.Parameter`

**Signature:** `nn.Parameter(data=None, requires_grad=True)`

**Parameters:** `data` — initial tensor values (or None); `requires_grad` — whether this value should be optimized.

```python
weight = nn.Parameter(data=torch.zeros(3, 4), requires_grad=True)
```

**Shape / use:** (3,4): registered as a trainable parameter when assigned to a Module attribute.

#### `nn.init.zeros_`

**Signature:** `nn.init.zeros_(tensor)`

**Parameters:** `tensor` — tensor/parameter to fill with zeros in place; `initialization is executed without autograd histor` — itialization is executed without autograd history.

```python
layer = nn.Linear(2, 1)
nn.init.zeros_(tensor=layer.weight)
```

**Shape / use:** Sets all selected parameter entries to zero; used for deliberate neutral baselines.

#### `nn.ModuleList`

**Signature:** `nn.ModuleList(modules=None)`

**Parameters:** `modules` — iterable of nn.Module children; `all children are registered as part of the mode` — l children are registered as part of the model.

```python
layers = nn.ModuleList(modules=[nn.Linear(4, 4), nn.Linear(4, 4)])
```

**Shape / use:** Useful for stacked Transformer blocks; unlike a plain Python list, registers parameters.

#### `nn.RMSNorm`

**Signature:** `nn.RMSNorm(normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None)`

**Parameters:** `normalized_shape` — trailing feature shape to normalize; `eps` — numerical stabilization value; `elementwise_affine` — learn scale weights; `device` — parameter device; `dtype` — parameter dtype.

```python
norm = nn.RMSNorm(normalized_shape=8, eps=1e-5,
                  elementwise_affine=True, device="cpu",
                  dtype=torch.float32)
y = norm(torch.randn(2, 3, 8))
```

**Shape / use:** (B=2,T=3,C=8): normalizes final feature dimension.

#### `F.elu`

**Signature:** `F.elu(input, alpha=1.0, inplace=False)`

**Parameters:** `input` — activations; `alpha` — negative-branch saturation magnitude; `inplace` — whether to overwrite input.

```python
x = torch.tensor([-1., 0., 1.])
y = F.elu(input=x, alpha=1.0, inplace=False)
```

**Shape / use:** (N=3,): positive feature map can use `F.elu(x) + 1`.

#### `F.gelu`

**Signature:** `F.gelu(input, approximate='none')`

**Parameters:** `input` — preactivation tensor; `approximate` — 'none' for exact erf form or 'tanh' for fast approximation.

```python
x = torch.tensor([-1., 0., 1.])
y = F.gelu(input=x, approximate="tanh")
```

**Shape / use:** (N=3,): smooth gating/activation.

#### `F.silu`

**Signature:** `F.silu(input, inplace=False)`

**Parameters:** `input` — preactivation tensor; `inplace` — overwrite input if True.

```python
x = torch.tensor([-1., 0., 1.])
y = F.silu(input=x, inplace=False)
```

**Shape / use:** (N=3,): x·sigmoid(x), used in SwiGLU.

#### `F.scaled_dot_product_attention`

**Signature:** `F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)`

**Parameters:** `query` — `(B,Hq,Tq,D)` query vectors; `key` — `(B,Hkv,Tk,D)` keys; `value` — `(B,Hkv,Tk,Dv)` values; `attn_mask` — optional Boolean/additive bias mask; `dropout_p` — dropout probability (set 0.0 explicitly for eval); `is_causal` — use triangular causal mask (normally cannot combine with attn_mask); `scale` — optional score multiplier (default 1/√D); `enable_gqa` — repeat KV heads internally when permitted.

```python
q = torch.randn(2, 4, 3, 8)  # (B,H,T,D)
k = torch.randn(2, 4, 3, 8)
v = torch.randn(2, 4, 3, 8)
y = F.scaled_dot_product_attention(query=q, key=k, value=v,
    attn_mask=None, dropout_p=0.0, is_causal=True,
    scale=None, enable_gqa=False)
```

**Shape / use:** (B=2,H=4,T=3,D=8): causal attention; masks and dropout rules matter.

---

**Reading note.** Entries here describe the parameters of the illustrated API signatures, including optional flags and important overloads. Internal or deprecated options are labeled. To inspect additional backend-specific overloads, consult the installed PyTorch version's API documentation. When copying examples to a notebook, preserve the indicated tensor shapes and respect model/device consistency.
