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
| `torch.zeros` | ✅ | Creates a tensor filled with zeros. |
| `torch.zeros_like` | ✅ | Creates zeros with the same shape, dtype, and device as another tensor. |
| `torch.ones` | ✅ | Creates a tensor filled with ones. |
| `torch.ones_like` | ✅ | Creates ones with the same shape, dtype, and device as another tensor. |
| `torch.full_like` | ✅ | Creates a tensor with the same shape, dtype, and device as another tensor, filled with a chosen value. |
| `torch.randn` | ✅ | Samples values from a standard normal distribution. |
| `torch.randint` | ✅ | Samples integer values from a specified range. |
| `Tensor.reshape` | ✅ | Changes tensor shape without changing the logical values. |
| `Tensor.unsqueeze` | ✅ | Inserts a size-1 dimension. |
| `Tensor.squeeze` | ✅ | Removes size-1 dimensions. |
| `Tensor.transpose` | ✅ | Swaps two tensor dimensions. |
| `Tensor.flatten` | ✅ | Collapses a range of dimensions into one. |
| `torch.stack` | ✅ | Creates a new dimension and stacks tensors along it. |
| `torch.cat` | ✅ | Concatenates tensors along an existing dimension. |

## Numerical and reduction operations

| Function / method | Status | What it does |
|---|---|---|
| `torch.exp` | ✅ | Applies the exponential function element-wise. |
| `torch.log` | ✅ | Applies the natural logarithm element-wise. |
| `torch.sqrt` | ✅ | Applies square root element-wise. |
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
| `F.cross_entropy` | ✅ | Computes cross-entropy loss from logits and class targets. |
| `torch.multinomial` | ✅ | Samples indices according to categorical weights or probabilities. |
| `torch.argmax` | ✅ | Returns the index of the largest value. |

## Autograd and parameter handling

| Function / method | Status | What it does |
|---|---|---|
| `Tensor.backward` | ✅ | Runs reverse-mode automatic differentiation from a scalar loss. |
| `Tensor.detach` | ✅ | Returns a tensor detached from the autograd graph. |
| `Tensor.clone` | ✅ | Creates a separate tensor containing copied values. |
| `torch.no_grad` | ✅ | Disables autograd graph construction inside a context. |

## Neural-network utilities

| Function / method | Status | What it does |
|---|---|---|
| `nn.Embedding` | ✅ | Maps integer token IDs to learned embedding vectors. |
| `nn.Linear` | ✅ | Applies an affine transformation to the final feature dimension. |
| `nn.Parameter` | ✅ | Registers a tensor as a trainable model parameter. |
| `nn.ModuleList` | ✅ | Stores and registers a list of child modules. |
| `nn.RMSNorm` | ✅ | Applies RMS normalization over the final feature dimension. |
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
