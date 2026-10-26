# PyTorch Function Reference

A learning reference for **LLM from First Principles** — **81 tracked API entries**, reviewed through completed Lesson 21B.

Open a function to see its **signature → parameters → example → output**. The examples show common calls; the parameter tables cover the optional arguments separately.

| Browse | Entries |
|:--|--:|
| [Tensor creation & shapes](#tensor-creation-and-shape-operations) | 24 |
| [Numerical operations & reductions](#numerical-and-reduction-operations) | 17 |
| [Indexing & comparisons](#indexing-filtering-and-comparison) | 11 |
| [Probability & sampling](#probability-and-sampling) | 5 |
| [Autograd](#autograd-and-parameter-handling) | 5 |
| [Optimization](#optimization-and-training) | 9 |
| [Neural-network utilities](#neural-network-utilities) | 10 |

## Shared setup

Run this once before any example. Each example is otherwise independent.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False, linewidth=80)
```

**Dimensions:** `B` sequences · `T` token/action positions · `V` vocabulary · `C` features · `H` heads · `D` head width. `()` means a scalar.

**Notation:** `x` is an existing tensor; `optimizer` is an optimizer instance. `*` introduces keyword-only arguments; `*size` accepts multiple dimensions. Arguments with no default are required. A trailing `_` usually denotes an in-place operation.

**Output check:** All 81 examples were executed on CPU with **PyTorch 2.10.0 / Python 3.13.5**. The repository requests `torch>=2.14.0`; these outputs are not a test of that environment. Seeded random values and numerical formatting can differ across versions/devices. Internal, deprecated, and overload-specific options are labeled rather than forced into the examples.

## Tensor creation and shape operations

<details>
<summary><code>torch.tensor</code> — Create a tensor by copying explicit data.</summary>

**Signature**

```python
torch.tensor(data, *, dtype=None, device=None,
             requires_grad=False, pin_memory=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `data` | Required | Scalar, list, tuple, or array-like source. |
| `dtype` | `None` | Element type; inferred from data when omitted. |
| `device` | `None` | Output device; follows the current default when omitted. |
| `requires_grad` | `False` | Record gradients for floating-point or complex values. |
| `pin_memory` | `False` | Allocate pinned CPU memory. |

**Example**

```python
# Purpose: Create a two-sequence integer token batch.
tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
# (B=2, T=3): sequences, token positions
print(tokens)
```

**Output**

```text
tensor([[1, 2, 3],
        [4, 5, 6]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html)

</details>

<details>
<summary><code>torch.arange</code> — Create evenly spaced values with an exclusive endpoint.</summary>

**Signature**

```python
torch.arange(start, end, step=1, *, out=None, dtype=None,
             layout=torch.strided, device=None, requires_grad=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `start` | 0 in end-only call | Inclusive start; omit it in torch.arange(end). |
| `end` | Required | Exclusive upper or lower endpoint. |
| `step` | `1` | Nonzero signed spacing. |
| `out` | `None` | Optional destination tensor. |
| `dtype` | `None` | Inferred from start, end, and step. |
| `layout` | `torch.strided` | Storage layout. |
| `device` | `None` | Output device. |
| `requires_grad` | `False` | Enable gradient recording. |

**Example**

```python
# Purpose: Build every other token position.
positions = torch.arange(1, 7, step=2)
# (T=3,): selected positions
print(positions)
```

**Output**

```text
tensor([1, 3, 5])
```

The end-only overload is `torch.arange(end, ...)`; it starts at zero.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.arange.html)

</details>

<details>
<summary><code>torch.linspace</code> — Create a fixed number of evenly spaced values, including both endpoints.</summary>

**Signature**

```python
torch.linspace(start, end, steps, *, out=None, dtype=None,
               layout=torch.strided, device=None, requires_grad=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `start` | Required | First value. |
| `end` | Required | Last value when steps is greater than one. |
| `steps` | Required | Number of output values. |
| `out` | `None` | Optional destination tensor. |
| `dtype` | `None` | Output type; inferred from endpoints and the default dtype. |
| `layout` | `torch.strided` | Storage layout. |
| `device` | `None` | Output device. |
| `requires_grad` | `False` | Enable gradient recording. |

**Example**

```python
# Purpose: Make five points for a probability plot.
grid = torch.linspace(0.0, 1.0, steps=5)
# (N=5,): grid points
print(grid)
```

**Output**

```text
tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.linspace.html)

</details>

<details>
<summary><code>torch.randperm</code> — Shuffle the integers from zero through n minus one.</summary>

**Signature**

```python
torch.randperm(n, *, generator=None, out=None, dtype=torch.int64,
               layout=torch.strided, device=None, requires_grad=False,
               pin_memory=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `n` | Required | Number of indices to permute. |
| `generator` | `None` | Optional random-number generator. |
| `out` | `None` | Optional destination tensor. |
| `dtype` | `torch.int64` | Output element type. |
| `layout` | `torch.strided` | Storage layout. |
| `device` | `None` | Output device. |
| `requires_grad` | `False` | Keep False for integer indices. |
| `pin_memory` | `False` | Allocate pinned CPU memory. |

**Example**

```python
# Purpose: Shuffle trajectory indices reproducibly.
rng = torch.Generator().manual_seed(42)
order = torch.randperm(5, generator=rng)
# (B=5,): sequence indices in shuffled order
print(order)
```

**Output**

```text
tensor([2, 4, 3, 0, 1])
```

A seeded random output is a reference run, not a cross-version guarantee.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.randperm.html)

</details>

<details>
<summary><code>torch.manual_seed</code> — Set the default RNG seed for reproducible PyTorch sampling.</summary>

**Signature**

```python
torch.manual_seed(seed)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `seed` | Required | Integer seed for PyTorch RNGs; reproducibility also depends on device and algorithms. |

**Example**

```python
# Purpose: Confirm that reseeding repeats the same random permutation.
torch.manual_seed(123)
first = torch.randperm(5)   # (B=5,): first random order
torch.manual_seed(123)
second = torch.randperm(5)  # (B=5,): reproduced order
print(torch.equal(first, second))
```

**Output**

```text
True
```

Returns a `torch.Generator`; changing the global seed is not the same as maintaining an independent `Generator`.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html)

</details>



<details>
<summary><code>torch.zeros</code> — Create a tensor filled with zeros.</summary>

**Signature**

```python
torch.zeros(*size, out=None, dtype=None, layout=torch.strided,
            device=None, requires_grad=False, pin_memory=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `*size` | Required | Output dimensions, passed separately or as a tuple. |
| `out` | `None` | Optional destination tensor. |
| `dtype` | `None` | Uses the default floating-point dtype when omitted. |
| `layout` | `torch.strided` | Storage layout. |
| `device` | `None` | Output device. |
| `requires_grad` | `False` | Enable gradient recording. |
| `pin_memory` | `False` | Pinned CPU allocation option. |

**Example**

```python
# Purpose: Initialize a two-sequence, three-action table.
x = torch.zeros(2, 3)
# (B=2, T=3): sequences, action positions
print(x)
```

**Output**

```text
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

`pin_memory` is an allocation option accepted by the tested implementation; it may be absent from the shorter public documentation signature.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html)

</details>

<details>
<summary><code>torch.zeros_like</code> — Create a filled tensor matching a reference tensor.</summary>

**Signature**

```python
torch.zeros_like(input, *, dtype=None, layout=None, device=None,
                 requires_grad=False, memory_format=torch.preserve_format)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Reference tensor; supplies shape and default dtype/device. |
| `dtype` | `None` | Override the reference dtype. |
| `layout` | `None` | Inherit the reference layout unless overridden. |
| `device` | `None` | Inherit the reference device unless overridden. |
| `requires_grad` | `False` | Set independently of the reference gradient flag. |
| `memory_format` | `torch.preserve_format` | Preserve the reference memory format where possible. |

**Example**

```python
# Purpose: Allocate values with the same shape as a token-score table.
reference = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
# (B=2, T=2): sequences, action scores
print(torch.zeros_like(reference))
```

**Output**

```text
tensor([[0., 0.],
        [0., 0.]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.zeros_like.html)

</details>

<details>
<summary><code>torch.ones</code> — Create a tensor filled with ones.</summary>

**Signature**

```python
torch.ones(*size, out=None, dtype=None, layout=torch.strided,
            device=None, requires_grad=False, pin_memory=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `*size` | Required | Output dimensions, passed separately or as a tuple. |
| `out` | `None` | Optional destination tensor. |
| `dtype` | `None` | Uses the default floating-point dtype when omitted. |
| `layout` | `torch.strided` | Storage layout. |
| `device` | `None` | Output device. |
| `requires_grad` | `False` | Enable gradient recording. |
| `pin_memory` | `False` | Pinned CPU allocation option. |

**Example**

```python
# Purpose: Initialize a two-sequence, three-action table.
x = torch.ones(2, 3)
# (B=2, T=3): sequences, action positions
print(x)
```

**Output**

```text
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

`pin_memory` is an allocation option accepted by the tested implementation; it may be absent from the shorter public documentation signature.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.ones.html)

</details>

<details>
<summary><code>torch.ones_like</code> — Create a filled tensor matching a reference tensor.</summary>

**Signature**

```python
torch.ones_like(input, *, dtype=None, layout=None, device=None,
                 requires_grad=False, memory_format=torch.preserve_format)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Reference tensor; supplies shape and default dtype/device. |
| `dtype` | `None` | Override the reference dtype. |
| `layout` | `None` | Inherit the reference layout unless overridden. |
| `device` | `None` | Inherit the reference device unless overridden. |
| `requires_grad` | `False` | Set independently of the reference gradient flag. |
| `memory_format` | `torch.preserve_format` | Preserve the reference memory format where possible. |

**Example**

```python
# Purpose: Allocate values with the same shape as a token-score table.
reference = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
# (B=2, T=2): sequences, action scores
print(torch.ones_like(reference))
```

**Output**

```text
tensor([[1., 1.],
        [1., 1.]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.ones_like.html)

</details>

<details>
<summary><code>torch.full_like</code> — Create a filled tensor matching a reference tensor.</summary>

**Signature**

```python
torch.full_like(input, fill_value, *, dtype=None, layout=None, device=None,
                 requires_grad=False, memory_format=torch.preserve_format)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Reference tensor; supplies shape and default dtype/device. |
| `fill_value` | Required | Scalar value used to fill the output. |
| `dtype` | `None` | Override the reference dtype. |
| `layout` | `None` | Inherit the reference layout unless overridden. |
| `device` | `None` | Inherit the reference device unless overridden. |
| `requires_grad` | `False` | Set independently of the reference gradient flag. |
| `memory_format` | `torch.preserve_format` | Preserve the reference memory format where possible. |

**Example**

```python
# Purpose: Allocate values with the same shape as a token-score table.
reference = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
# (B=2, T=2): sequences, action scores
print(torch.full_like(reference, -1.0))
```

**Output**

```text
tensor([[-1., -1.],
        [-1., -1.]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.full_like.html)

</details>

<details>
<summary><code>torch.randn</code> — Sample standard-normal random values.</summary>

**Signature**

```python
torch.randn(*size, generator=None, out=None, dtype=None,
            layout=torch.strided, device=None, requires_grad=False,
            pin_memory=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `*size` | Required | Output dimensions. |
| `generator` | `None` | Optional random-number generator. |
| `out` | `None` | Optional destination tensor. |
| `dtype` | `None` | Output floating-point or complex dtype. |
| `layout` | `torch.strided` | Storage layout. |
| `device` | `None` | Output device. |
| `requires_grad` | `False` | Enable gradient recording. |
| `pin_memory` | `False` | Allocate pinned CPU memory. |

**Example**

```python
# Purpose: Generate a small, reproducible noise vector.
rng = torch.Generator().manual_seed(42)
noise = torch.randn(3, generator=rng)
# (C=3,): feature values
print(noise)
```

**Output**

```text
tensor([0.3367, 0.1288, 0.2345])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.randn.html)

</details>

<details>
<summary><code>torch.randint</code> — Sample integers between an inclusive low and exclusive high.</summary>

**Signature**

```python
torch.randint(low, high, size, *, generator=None, out=None,
              dtype=None, layout=torch.strided, device=None,
              requires_grad=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `low` | 0 in high-only call | Inclusive lower bound. |
| `high` | Required | Exclusive upper bound. |
| `size` | Required | Output shape. |
| `generator` | `None` | Optional random-number generator. |
| `out` | `None` | Optional destination tensor. |
| `dtype` | `None` | Defaults to torch.int64. |
| `layout` | `torch.strided` | Storage layout. |
| `device` | `None` | Output device. |
| `requires_grad` | `False` | Keep False for integer samples. |

**Example**

```python
# Purpose: Sample the two toy prompt types.
rng = torch.Generator().manual_seed(42)
kinds = torch.randint(1, 3, (6,), generator=rng)
# (B=6,): one prompt type per sequence
print(kinds)
```

**Output**

```text
tensor([1, 2, 1, 1, 1, 2])
```

The high-only overload is `torch.randint(high, size, ...)`; low is then zero.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.randint.html)

</details>

<details>
<summary><code>Tensor.reshape</code> — Change shape while preserving element order and count.</summary>

**Signature**

```python
x.reshape(*shape)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `*shape` | Required | Target dimensions; one may be -1 to infer its size. |

**Example**

```python
# Purpose: Turn a flat token vector into a batch.
x = torch.arange(6)  # (N=6,): flat token IDs
y = x.reshape(2, 3)  # (B=2, T=3): sequences, tokens
print(y)
```

**Output**

```text
tensor([[0, 1, 2],
        [3, 4, 5]])
```

May return a view or a copy; do not assume independent storage.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html)

</details>

<details>
<summary><code>Tensor.expand</code> — Broadcast singleton dimensions without copying data.</summary>

**Signature**

```python
x.expand(*sizes)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `*sizes` | Required | Expanded dimensions; -1 keeps an existing dimension unchanged. |

**Example**

```python
# Purpose: Reuse one action-position template across two sequences.
positions = torch.tensor([[2, 3]])  # (1, T=2)
print(positions.expand(2, -1))      # (B=2, T=2)
```

**Output**

```text
tensor([[2, 3],
        [2, 3]])
```

Expanded entries may share storage. Avoid in-place writes to the expanded view.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html)

</details>

<details>
<summary><code>Tensor.numel</code> — Return the total number of tensor elements.</summary>

**Signature**

```python
x.numel()
```

**Parameters:** None.

**Example**

```python
# Purpose: Count all elements, not only True entries.
mask = torch.tensor([[True, False, True]])  # (B=1, T=3)
print(mask.numel())
```

**Output**

```text
3
```

Use `mask.sum()` to count True entries instead.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numel.html)

</details>

<details>
<summary><code>Tensor.item</code> — Convert a single-element tensor into a Python number.</summary>

**Signature**

```python
x.item()
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `—` | — | No parameters. |

**Example**

```python
# Purpose: Report a scalar training loss as a Python float.
loss = torch.tensor(0.125)  # (): one scalar loss
print(loss.item())
```

**Output**

```text
0.125
```

Requires exactly one element; the resulting Python number is not differentiable.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html)

</details>

<details>
<summary><code>Tensor.tolist</code> — Convert tensor values into nested Python lists.</summary>

**Signature**

```python
x.tolist()
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `—` | — | No parameters. |

**Example**

```python
# Purpose: Inspect token IDs in familiar Python-list form.
tokens = torch.tensor([[1, 2], [3, 4]])  # (B=2, T=2)
print(tokens.tolist())
```

**Output**

```text
[[1, 2], [3, 4]]
```

Converts tensor values to Python objects; CUDA tensors are moved to CPU when needed. This is for inspection, not differentiable computation.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html)

</details>





<details>
<summary><code>Tensor.unsqueeze</code> — Insert a dimension of size one.</summary>

**Signature**

```python
x.unsqueeze(dim)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `dim` | Required | Axis at which to insert the singleton dimension; may be negative. |

**Example**

```python
# Purpose: Add a batch dimension to one token sequence.
x = torch.tensor([1, 2, 3])  # (T=3,)
print(x.unsqueeze(0))        # (B=1, T=3)
```

**Output**

```text
tensor([[1, 2, 3]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html)

</details>

<details>
<summary><code>Tensor.squeeze</code> — Remove selected dimensions of size one.</summary>

**Signature**

```python
x.squeeze(dim=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `dim` | `None` | Axis or tuple of axes to squeeze; None removes all size-one axes. |

**Example**

```python
# Purpose: Remove only the singleton sampling dimension.
x = torch.tensor([[3], [4]])  # (B=2, K=1)
print(x.squeeze(dim=1))       # (B=2,)
```

**Output**

```text
tensor([3, 4])
```

Specify dim so a batch of size one is not accidentally removed.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.squeeze.html)

</details>

<details>
<summary><code>Tensor.transpose</code> — Swap two tensor dimensions.</summary>

**Signature**

```python
x.transpose(dim0, dim1)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `dim0` | Required | First axis to swap. |
| `dim1` | Required | Second axis to swap. |

**Example**

```python
# Purpose: Swap the two axes of a small matrix.
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (B=2, T=3)
print(x.transpose(0, 1))                  # (T=3, B=2)
```

**Output**

```text
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.transpose.html)

</details>

<details>
<summary><code>Tensor.flatten</code> — Merge a range of dimensions into one.</summary>

**Signature**

```python
x.flatten(start_dim=0, end_dim=-1)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `start_dim` | `0` | First axis to merge. |
| `end_dim` | `-1` | Last axis to merge, inclusive. |

**Example**

```python
# Purpose: Flatten token features while preserving the batch axis.
x = torch.arange(8).reshape(2, 2, 2)  # (B=2, T=2, C=2)
print(x.flatten(start_dim=1))         # (B=2, T*C=4)
```

**Output**

```text
tensor([[0, 1, 2, 3],
        [4, 5, 6, 7]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.flatten.html)

</details>

<details>
<summary><code>torch.stack</code> — Join same-shaped tensors along a new dimension.</summary>

**Signature**

```python
torch.stack(tensors, dim=0, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `tensors` | Required | Nonempty sequence of tensors with identical shapes. |
| `dim` | `0` | Position of the new dimension. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Combine two trajectories into a batch.
a = torch.tensor([1, 2])  # (T=2,): first trajectory
b = torch.tensor([3, 4])  # (T=2,): second trajectory
print(torch.stack([a, b], dim=0))  # (B=2, T=2)
```

**Output**

```text
tensor([[1, 2],
        [3, 4]])
```

stack creates an axis; cat extends an existing axis.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.stack.html)

</details>

<details>
<summary><code>torch.cat</code> — Concatenate tensors along an existing dimension.</summary>

**Signature**

```python
torch.cat(tensors, dim=0, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `tensors` | Required | Tensors with matching shapes outside the concatenation axis. |
| `dim` | `0` | Existing axis along which to concatenate. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Append one sampled token to each sequence.
tokens = torch.tensor([[1, 2], [3, 4]])  # (B=2, T=2)
next_token = torch.tensor([[5], [6]])    # (B=2, 1)
print(torch.cat([tokens, next_token], dim=1))  # (B=2, T=3)
```

**Output**

```text
tensor([[1, 2, 5],
        [3, 4, 6]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.cat.html)

</details>

<details>
<summary><code>torch.repeat_interleave</code> — Repeat individual entries along a chosen axis.</summary>

**Signature**

```python
torch.repeat_interleave(input, repeats, dim=None, *, output_size=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Source tensor. |
| `repeats` | Required | Integer repeat count or tensor of per-entry counts. |
| `dim` | `None` | Axis to repeat; None flattens input first. |
| `output_size` | `None` | Optional total output length along the repeated axis. |

**Example**

```python
# Purpose: Repeat each KV-head identifier twice.
heads = torch.tensor([0, 1])  # (H_KV=2,)
print(torch.repeat_interleave(heads, repeats=2, dim=0))  # (H_Q=4,)
```

**Output**

```text
tensor([0, 0, 1, 1])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.repeat_interleave.html)

</details>

## Numerical and reduction operations

<details>
<summary><code>torch.exp</code> — Exponentiate each value.</summary>

**Signature**

```python
torch.exp(input, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Input tensor. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Apply exp element by element.
x = torch.tensor([0.0, 1.0, 2.0])  # (N=3,)
print(torch.exp(x))
```

**Output**

```text
tensor([1.0000, 2.7183, 7.3891])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.exp.html)

</details>

<details>
<summary><code>torch.log</code> — Take the natural logarithm of each value.</summary>

**Signature**

```python
torch.log(input, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Input tensor. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Apply log element by element.
x = torch.tensor([0.25, 0.5, 1.0])  # (N=3,)
print(torch.log(x))
```

**Output**

```text
tensor([-1.3863, -0.6931,  0.0000])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.log.html)

</details>

<details>
<summary><code>torch.sqrt</code> — Take the square root of each value.</summary>

**Signature**

```python
torch.sqrt(input, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Input tensor. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Apply sqrt element by element.
x = torch.tensor([1.0, 4.0, 9.0])  # (N=3,)
print(torch.sqrt(x))
```

**Output**

```text
tensor([1., 2., 3.])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.sqrt.html)

</details>

<details>
<summary><code>torch.clamp</code> — Limit each element to a lower and/or upper bound.</summary>

**Signature**

```python
torch.clamp(input, min=None, max=None, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Values to bound. |
| `min` | `None` | Lower bound; scalar or tensor. |
| `max` | `None` | Upper bound; scalar or tensor. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Clip PPO importance ratios for the surrogate objective.
ratios = torch.tensor([0.5, 1.0, 1.5])  # (T=3,)
print(torch.clamp(ratios, min=0.8, max=1.2))
```

**Output**

```text
tensor([0.8000, 1.0000, 1.2000])
```

At least one bound is required. Clipping this tensor does not constrain the model parameters.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html)

</details>

<details>
<summary><code>torch.minimum</code> — Take the element-wise smaller of two tensors.</summary>

**Signature**

```python
torch.minimum(input, other, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | First tensor. |
| `other` | Required | Second tensor; shapes must be broadcastable. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Select the pessimistic PPO surrogate per action.
raw = torch.tensor([3.0, -2.0])      # (T=2,)
clipped = torch.tensor([2.4, -0.8])  # (T=2,)
print(torch.minimum(raw, clipped))
```

**Output**

```text
tensor([ 2.4000, -2.0000])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.minimum.html)

</details>

<details>
<summary><code>Tensor.square</code> — Square every element.</summary>

**Signature**

```python
x.square()
```

**Parameters:** None.

**Example**

```python
# Purpose: Compute squared prediction errors.
errors = torch.tensor([-2.0, 3.0])  # (T=2,)
print(errors.square())
```

**Output**

```text
tensor([4., 9.])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.square.html)

</details>

<details>
<summary><code>Tensor.float</code> — Cast tensor elements to 32-bit floating point.</summary>

**Signature**

```python
x.float()
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `—` | — | No parameters. |

**Example**

```python
# Purpose: Convert integer token labels to floating-point values.
ids = torch.tensor([1, 2], dtype=torch.int64)  # (T=2,)
print(ids.float())
```

**Output**

```text
tensor([1., 2.])
```

Equivalent to `x.to(torch.float32)`; integer-to-float conversion does not create gradients for the integer input.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.float.html)

</details>

<details>
<summary><code>Tensor.abs</code> — Compute absolute values element-wise.</summary>

**Signature**

```python
x.abs()
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `—` | — | No parameters. |

**Example**

```python
# Purpose: Measure unsigned prediction errors.
errors = torch.tensor([-3., 2.])  # (T=2,)
print(errors.abs())
```

**Output**

```text
tensor([3., 2.])
```

Preserves tensor shape; often combined with `.max()` to inspect the largest absolute parameter change.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.abs.html)

</details>

<details>
<summary><code>Tensor.max</code> — Find largest values, optionally returning their indices on a chosen axis.</summary>

**Signature**

```python
x.max(dim=None, keepdim=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `dim` | `None` | If omitted, return one scalar maximum; if supplied, return `(values, indices)` along the selected dimension. |
| `keepdim` | `False` | Keep the reduced dimension of length 1 when `dim` is provided. |

**Example**

```python
# Purpose: Find the highest score and its position in each sequence.
scores = torch.tensor([[1., 3., 2.], [4., 0., 6.]])
# (B=2, V=3): sequences, candidate scores
values, positions = scores.max(dim=1)
print(values)      # (B=2,): largest values
print(positions)   # (B=2,): indices of largest values
```

**Output**

```text
tensor([3., 6.])
tensor([1, 2])
```

`x.max()` with no `dim` returns a scalar tensor. Do not confuse this reduction with `torch.maximum(a, b)`, which is element-wise.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.max.html)

</details>

<details>
<summary><code>Tensor.min</code> — Find smallest values, optionally returning their indices.</summary>

**Signature**

```python
x.min(dim=None, keepdim=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `dim` | `None` | If omitted, return a scalar minimum; with an axis, return `(values, indices)`. |
| `keepdim` | `False` | Preserve a size-1 reduced axis when `dim` is provided. |

**Example**

```python
# Purpose: Check the lowest value in a small sample.
scores = torch.tensor([3, 1, 2])  # (T=3,)
print(scores.min())
```

**Output**

```text
tensor(1)
```

For the smallest pairwise elements of two tensors, use `torch.minimum(a, b)` instead.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.min.html)

</details>

<details>
<summary><code>Tensor.norm</code> — Compute a tensor or vector norm.</summary>

**Signature**

```python
x.norm(p='fro', dim=None, keepdim=False, dtype=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `p` | `'fro'` | Norm order: `2` for Euclidean vector norm; `'fro'` for Frobenius norm. |
| `dim` | `None` | Dimension or tuple of dimensions; omit to reduce the whole tensor. |
| `keepdim` | `False` | Retain reduced dimensions as size 1. |
| `dtype` | `None` | Optional computation/output dtype. |

**Example**

```python
# Purpose: Inspect the magnitude of a two-component gradient.
grad = torch.tensor([3., 4.])  # (C=2,)
print(grad.norm(p=2))
```

**Output**

```text
tensor(5.)
```

For new code, prefer the more explicit `torch.linalg.vector_norm` when working with vectors.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.norm.html)

</details>











<details>
<summary><code>Tensor.mean</code> — Average values over selected dimensions.</summary>

**Signature**

```python
x.mean(dim=None, keepdim=False, *, dtype=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `dim` | `None` | Axis or tuple of axes to reduce; None reduces all elements. |
| `keepdim` | `False` | Retain reduced axes with size one. |
| `dtype` | `None` | Optional dtype used before the reduction. |

**Example**

```python
# Purpose: Average token scores separately for each sequence.
scores = torch.tensor([[2.0, 4.0], [6.0, 8.0]])  # (B=2, T=2)
print(scores.mean(dim=1, keepdim=True))          # (B=2, 1)
```

**Output**

```text
tensor([[3.],
        [7.]])
```

An empty floating-point tensor has mean NaN; reject empty masks before reducing.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.mean.html)

</details>

<details>
<summary><code>Tensor.var</code> — Measure variance over selected dimensions.</summary>

**Signature**

```python
x.var(dim=None, *, correction=1, keepdim=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `dim` | `None` | Axis or axes to reduce; None reduces all elements. |
| `correction` | `1` | Degrees-of-freedom correction; use 0 for population variance. |
| `keepdim` | `False` | Retain reduced axes with size one. |

**Example**

```python
# Purpose: Compute population variance, not sample variance.
x = torch.tensor([1.0, 2.0, 3.0])  # (N=3,)
print(x.var(correction=0))          # (): scalar
```

**Output**

```text
tensor(0.6667)
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.var.html)

</details>

<details>
<summary><code>Tensor.sum</code> — Sum values over selected dimensions.</summary>

**Signature**

```python
x.sum(dim=None, keepdim=False, dtype=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `dim` | `None` | Axis or axes to reduce; None sums all elements. |
| `keepdim` | `False` | Retain reduced axes with size one. |
| `dtype` | `None` | Optional dtype used before accumulation. |

**Example**

```python
# Purpose: Total the rewards within each trajectory.
rewards = torch.tensor([[0.0, 1.0], [0.0, 0.5]])  # (B=2, T=2)
print(rewards.sum(dim=1))                         # (B=2,)
```

**Output**

```text
tensor([1.0000, 0.5000])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.sum.html)

</details>

<details>
<summary><code>torch.cumsum</code> — Compute cumulative sums along an axis.</summary>

**Signature**

```python
torch.cumsum(input, dim, *, dtype=None, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Source tensor. |
| `dim` | Required | Axis along which to accumulate. |
| `dtype` | `None` | Optional dtype used before accumulation. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Build prefix sums.
x = torch.tensor([1, 2, 3])  # (T=3,)
print(torch.cumsum(x, dim=0))
```

**Output**

```text
tensor([1, 3, 6])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.cumsum.html)

</details>

<details>
<summary><code>torch.linalg.vector_norm</code> — Compute a vector norm over chosen dimensions.</summary>

**Signature**

```python
torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False,
                         *, dtype=None, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `x` | Required | Floating-point or complex tensor. |
| `ord` | `2` | Norm order; 2 is Euclidean and float('inf') takes the maximum magnitude. |
| `dim` | `None` | Axis or axes treated as a vector; None flattens all dimensions. |
| `keepdim` | `False` | Retain reduced axes with size one. |
| `dtype` | `None` | Optional computation dtype. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Measure the norm of each feature vector.
x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])  # (B=2, C=2)
print(torch.linalg.vector_norm(x, dim=1))    # (B=2,)
```

**Output**

```text
tensor([ 5., 13.])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html)

</details>

<details>
<summary><code>torch.isfinite</code> — Test for values other than NaN or infinity.</summary>

**Signature**

```python
torch.isfinite(input)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Tensor to check. |

**Example**

```python
# Purpose: Locate invalid numerical values.
x = torch.tensor([1.0, float("nan"), float("inf")])  # (N=3,)
print(torch.isfinite(x))
```

**Output**

```text
tensor([ True, False, False])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html)

</details>

## Indexing, filtering, and comparison

<details>
<summary><code>torch.maximum</code> — Take the element-wise larger of two tensors.</summary>

**Signature**

```python
torch.maximum(input, other, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | First tensor. |
| `other` | Required | Second tensor; shapes must be broadcastable. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Compare two score vectors element by element.
a = torch.tensor([-1.0, 3.0])  # (N=2,)
b = torch.tensor([0.0, 2.0])   # (N=2,)
print(torch.maximum(a, b))
```

**Output**

```text
tensor([0., 3.])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.maximum.html)

</details>

<details>
<summary><code>torch.any</code> — Test whether at least one element is True.</summary>

**Signature**

```python
torch.any(input, dim=None, keepdim=False, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Boolean or numeric tensor; nonzero means True. |
| `dim` | `None` | Axis or axes to reduce; None checks all elements. |
| `keepdim` | `False` | Retain reduced axes with size one. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Check each sequence's validity flags.
mask = torch.tensor([[True, True], [True, False]])  # (B=2, T=2)
print(torch.any(mask, dim=1))  # (B=2,)
```

**Output**

```text
tensor([True, True])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.any.html)

</details>

<details>
<summary><code>torch.all</code> — Test whether every element is True.</summary>

**Signature**

```python
torch.all(input, dim=None, keepdim=False, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Boolean or numeric tensor; nonzero means True. |
| `dim` | `None` | Axis or axes to reduce; None checks all elements. |
| `keepdim` | `False` | Retain reduced axes with size one. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Check each sequence's validity flags.
mask = torch.tensor([[True, True], [True, False]])  # (B=2, T=2)
print(torch.all(mask, dim=1))  # (B=2,)
```

**Output**

```text
tensor([ True, False])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.all.html)

</details>

<details>
<summary><code>torch.where</code> — Choose between two values according to a Boolean condition.</summary>

**Signature**

```python
torch.where(condition, input, other, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `condition` | Required | Boolean selection mask; shapes must be broadcastable. |
| `input` | Required | Tensor or scalar selected where condition is True. |
| `other` | Required | Tensor or scalar selected where condition is False. |
| `out` | `None` | Optional destination tensor. |

**Example**

```python
# Purpose: Keep valid action scores and replace padding with zero.
scores = torch.tensor([0.7, 99.0, 0.4])  # (T=3,)
mask = torch.tensor([True, False, True])
print(torch.where(mask, scores, 0.0))
```

**Output**

```text
tensor([0.7000, 0.0000, 0.4000])
```

The one-argument overload `torch.where(condition)` returns indices instead of selected values.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.where.html)

</details>

<details>
<summary><code>Tensor.masked_fill</code> — Return a copy with selected entries replaced.</summary>

**Signature**

```python
x.masked_fill(mask, value)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `mask` | Required | Broadcastable Boolean mask; True means replace. |
| `value` | Required | Scalar replacement value. |

**Example**

```python
# Purpose: Clear a padded action slot.
x = torch.tensor([[1.0, 2.0, 3.0]])            # (B=1, T=3)
padding = torch.tensor([[False, True, False]])
print(x.masked_fill(padding, 0.0))
```

**Output**

```text
tensor([[1., 0., 3.]])
```

The mask direction is important: True entries are replaced, not retained.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)

</details>

<details>
<summary><code>torch.topk</code> — Return the k largest or smallest values and their indices.</summary>

**Signature**

```python
torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Source tensor. |
| `k` | Required | Number of entries to select. |
| `dim` | `None` | Selection axis; defaults to the last axis. |
| `largest` | `True` | Select largest values; False selects smallest. |
| `sorted` | `True` | Sort the selected values. |
| `out` | `None` | Optional (values, indices) output buffers. |

**Example**

```python
# Purpose: Find the two most likely vocabulary items.
scores = torch.tensor([0.2, 0.5, 0.3])  # (V=3,)
values, indices = torch.topk(scores, k=2)
print(values)
print(indices)
```

**Output**

```text
tensor([0.5000, 0.3000])
tensor([1, 2])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.topk.html)

</details>

<details>
<summary><code>torch.sort</code> — Sort values and return their original indices.</summary>

**Signature**

```python
torch.sort(input, dim=-1, descending=False, *, stable=False, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Source tensor. |
| `dim` | `-1` | Axis to sort. |
| `descending` | `False` | True sorts largest first. |
| `stable` | `False` | Preserve original order among equal elements. |
| `out` | `None` | Optional (values, indices) output buffers. |

**Example**

```python
# Purpose: Rank vocabulary scores in descending order.
scores = torch.tensor([0.3, 0.1, 0.2])  # (V=3,)
values, indices = torch.sort(scores, descending=True)
print(values)
print(indices)
```

**Output**

```text
tensor([0.3000, 0.2000, 0.1000])
tensor([0, 2, 1])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.sort.html)

</details>

<details>
<summary><code>Tensor.scatter_</code> — Write source values into indexed positions in place.</summary>

**Signature**

```python
x.scatter_(dim, index, src, *, reduce=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `dim` | Required | Destination axis along which index selects positions. |
| `index` | Required | Integer destination indices. |
| `src` | Required | Source tensor; a scalar-value overload is also available. |
| `reduce` | None in docs; omit in call | Legacy 'add' or 'multiply' reduction; deprecated for tensor src. |

**Example**

```python
# Purpose: Write two selected scores into a vocabulary-sized tensor.
x = torch.zeros(1, 4)                  # (B=1, V=4)
indices = torch.tensor([[1, 3]])       # (B=1, K=2)
source = torch.tensor([[5.0, 6.0]])    # (B=1, K=2)
x.scatter_(dim=1, index=indices, src=source)
print(x)
```

**Output**

```text
tensor([[0., 5., 0., 6.]])
```

Omit reduce for ordinary writes; explicitly passing None is not accepted by every overload. Repeated indices can make ordinary writes nondeterministic.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html)

</details>

<details>
<summary><code>torch.allclose</code> — Check approximate numerical equality.</summary>

**Signature**

```python
torch.allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Tensor to check. |
| `other` | Required | Reference tensor. |
| `rtol` | `1e-5` | Relative tolerance, scaled by the magnitude of other. |
| `atol` | `1e-8` | Absolute tolerance. |
| `equal_nan` | `False` | Treat NaNs at matching positions as equal. |

**Example**

```python
# Purpose: Accept a small floating-point discrepancy.
a = torch.tensor([1.000001])  # (N=1,)
b = torch.tensor([1.0])       # (N=1,)
print(torch.allclose(a, b, atol=1e-5))
```

**Output**

```text
True
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.allclose.html)

</details>

<details>
<summary><code>torch.testing.assert_close</code> — Assert numerical agreement and raise an informative error if it fails.</summary>

**Signature**

```python
torch.testing.assert_close(actual, expected, *, allow_subclasses=True, rtol=None, atol=None, equal_nan=False, check_device=True, check_dtype=True, check_layout=True, check_stride=False, msg=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `actual` | Required | Result tensor or other supported structure being tested. |
| `expected` | Required | Reference result or structure. |
| `allow_subclasses` | `True` | Allow related tensor subclasses. |
| `rtol` | `None` | Relative tolerance; defaults depend on data type. |
| `atol` | `None` | Absolute tolerance; defaults depend on data type. |
| `equal_nan` | `False` | Allow NaNs at corresponding locations. |
| `check_device` | `True` | Require matching tensor devices. |
| `check_dtype` | `True` | Require matching dtypes. |
| `check_layout` | `True` | Require matching layouts. |
| `check_stride` | `False` | Optionally require matching strides. |
| `msg` | `None` | Custom error text or callable to produce it. |

**Example**

```python
# Purpose: Assert that two slightly different probabilities agree.
actual = torch.tensor([1.000001])  # (T=1,)
expected = torch.tensor([1.0])     # (T=1,)
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
print("Close tensors verified.")
```

**Output**

```text
Close tensors verified.
```

Returns `None` on success and raises `AssertionError` on failure; unlike `torch.allclose`, which returns a Boolean.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)

</details>



<details>
<summary><code>torch.equal</code> — Check identical shapes and element values without a tolerance.</summary>

**Signature**

```python
torch.equal(input, other)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | First tensor. |
| `other` | Required | Second tensor. |

**Example**

```python
# Purpose: Verify an unchanged rollout snapshot.
a = torch.tensor([1, 2, 3])  # (T=3,)
b = a.clone()
print(torch.equal(a, b))
```

**Output**

```text
True
```

This compares values and shapes, not dtype identity; NaNs are not equal.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.equal.html)

</details>

## Probability and sampling

<details>
<summary><code>torch.softmax / F.softmax</code> — Convert logits into normalized probabilities.</summary>

**Signature**

```python
torch.softmax(input, dim, *, dtype=None)
F.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Unnormalized logits. |
| `dim` | Required for torch; None for F | Normalization axis; always provide it explicitly. |
| `dtype` | `None` | Optional dtype used before computing probabilities. |
| `_stacklevel` | 3; F only | Internal warning stack depth; normally leave unchanged. |

**Example**

```python
# Purpose: Normalize across vocabulary tokens, not sequences.
logits = torch.tensor([[1.0, 2.0, 3.0]])  # (B=1, V=3)
print(torch.softmax(logits, dim=-1))
print(F.softmax(logits, dim=-1))  # Equivalent alias
```

**Output**

```text
tensor([[0.0900, 0.2447, 0.6652]])
tensor([[0.0900, 0.2447, 0.6652]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)

</details>

<details>
<summary><code>torch.log_softmax / F.log_softmax</code> — Convert logits into numerically stable log-probabilities.</summary>

**Signature**

```python
torch.log_softmax(input, dim, *, dtype=None)
F.log_softmax(input, dim=None, _stacklevel=3, dtype=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Unnormalized logits. |
| `dim` | Required for torch; None for F | Normalization axis; always provide it explicitly. |
| `dtype` | `None` | Optional dtype used before computing probabilities. |
| `_stacklevel` | 3; F only | Internal warning stack depth; normally leave unchanged. |

**Example**

```python
# Purpose: Normalize across vocabulary tokens, not sequences.
logits = torch.tensor([[1.0, 2.0, 3.0]])  # (B=1, V=3)
print(torch.log_softmax(logits, dim=-1))
print(F.log_softmax(logits, dim=-1))  # Equivalent alias
```

**Output**

```text
tensor([[-2.4076, -1.4076, -0.4076]])
tensor([[-2.4076, -1.4076, -0.4076]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html)

</details>

<details>
<summary><code>F.cross_entropy</code> — Compute classification loss directly from logits.</summary>

**Signature**

```python
F.cross_entropy(input, target, weight=None, size_average=None,
                ignore_index=-100, reduce=None, reduction="mean",
                label_smoothing=0.0)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Logits; e.g., (B, V) or (B, V, ...). |
| `target` | Required | Class indices or same-shaped class probabilities. |
| `weight` | `None` | Optional rescaling weight per class. |
| `size_average` | `None` | Deprecated; use reduction instead. |
| `ignore_index` | `-100` | Ignore this label when target contains class indices. |
| `reduce` | `None` | Deprecated; use reduction instead. |
| `reduction` | `"mean"` | 'none', 'mean', or 'sum'. |
| `label_smoothing` | `0.0` | Mix targets with a uniform distribution. |

**Example**

```python
# Purpose: Score two next-token predictions against target IDs.
logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
# (B=2, V=3): sequences, vocabulary logits
targets = torch.tensor([2, 0])  # (B=2,): correct token IDs
print(F.cross_entropy(logits, targets))  # (): mean loss
```

**Output**

```text
tensor(0.4076)
```

Pass logits, not already-softmaxed probabilities.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

</details>

<details>
<summary><code>torch.multinomial</code> — Sample category indices from nonnegative weights.</summary>

**Signature**

```python
torch.multinomial(input, num_samples, replacement=False,
                  *, generator=None, out=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Finite nonnegative weights; each sampled row needs a positive sum. |
| `num_samples` | Required | Number of draws per row. |
| `replacement` | `False` | Allow repeated categories within a row when True. |
| `generator` | `None` | Optional random-number generator. |
| `out` | `None` | Optional integer output tensor. |

**Example**

```python
# Purpose: Sample token IDs from a categorical policy.
rng = torch.Generator().manual_seed(7)
probs = torch.tensor([[0.1, 0.2, 0.7]])  # (B=1, V=3)
actions = torch.multinomial(probs, 4, replacement=True, generator=rng)
print(actions)  # (B=1, K=4): sampled token IDs
```

**Output**

```text
tensor([[1, 1, 2, 2]])
```

Weights need not sum to one. Gradients do not flow through the sampled integer indices.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html)

</details>

<details>
<summary><code>torch.argmax</code> — Return the index of the largest value.</summary>

**Signature**

```python
torch.argmax(input, dim=None, keepdim=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Source scores. |
| `dim` | `None` | Axis to search; None flattens all dimensions. |
| `keepdim` | `False` | Retain the selected axis with size one. |

**Example**

```python
# Purpose: Choose the most likely token greedily.
logits = torch.tensor([[0.1, 0.9, 0.3]])  # (B=1, V=3)
print(torch.argmax(logits, dim=-1, keepdim=True))  # (B=1, 1)
```

**Output**

```text
tensor([[1]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.argmax.html)

</details>

## Autograd and parameter handling

<details>
<summary><code>Tensor.backward</code> — Accumulate gradients in leaf tensors connected to this output.</summary>

**Signature**

```python
x.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `gradient` | `None` | Upstream gradient; required for most nonscalar outputs. |
| `retain_graph` | `None` | Keep the graph; defaults to create_graph. |
| `create_graph` | `False` | Build a graph of derivatives for higher-order differentiation. |
| `inputs` | `None` | Optional tensors whose .grad fields receive accumulated gradients. |

**Example**

```python
# Purpose: Differentiate a scalar loss with respect to two weights.
x = torch.tensor([1.0, 2.0], requires_grad=True)  # (C=2,)
loss = x.square().sum()  # (): scalar loss
loss.backward()
print(x.grad)
```

**Output**

```text
tensor([2., 4.])
```

Gradients accumulate across backward calls unless cleared.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)

</details>

<details>
<summary><code>torch.autograd.grad</code> — Return requested gradients without accumulating into .grad.</summary>

**Signature**

```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None,
                    create_graph=False, only_inputs=True, allow_unused=None,
                    is_grads_batched=False, materialize_grads=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `outputs` | Required | Differentiated tensor or sequence of tensors. |
| `inputs` | Required | Tensor or sequence to differentiate against; newer versions also accept named dictionaries. |
| `grad_outputs` | `None` | Upstream vectors for the outputs. |
| `retain_graph` | `None` | Keep the graph; defaults to create_graph. |
| `create_graph` | `False` | Track derivative computations for higher-order gradients. |
| `only_inputs` | `True` | Deprecated and ignored. |
| `allow_unused` | `None` | Allow disconnected inputs; defaults to materialize_grads. |
| `is_grads_batched` | `False` | First grad_outputs axis indexes a batch of vector-Jacobian products. |
| `materialize_grads` | `False` | Return zeros instead of None for allowed unused inputs. |

**Example**

```python
# Purpose: Inspect a gradient without changing x.grad.
x = torch.tensor([1.0, 2.0], requires_grad=True)  # (C=2,)
(gradient,) = torch.autograd.grad(x.square().sum(), x)
print(gradient)
print(x.grad)
```

**Output**

```text
tensor([2., 4.])
None
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)

</details>

<details>
<summary><code>Tensor.detach</code> — Remove the autograd connection while sharing underlying storage.</summary>

**Signature**

```python
x.detach()
```

**Parameters:** None.

**Example**

```python
# Purpose: Freeze a computed value for use as a learning target.
x = torch.tensor([1.0, 2.0], requires_grad=True)  # (T=2,)
target = (2 * x).detach()
print(target)
print(target.requires_grad)
```

**Output**

```text
tensor([2., 4.])
False
```

Use `x.detach().clone()` when the frozen snapshot also needs independent storage.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)

</details>

<details>
<summary><code>Tensor.clone</code> — Copy tensor storage while preserving the gradient connection.</summary>

**Signature**

```python
x.clone(*, memory_format=torch.preserve_format)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `memory_format` | `torch.preserve_format` | Memory arrangement for the copy. |

**Example**

```python
# Purpose: Copy values so an in-place edit cannot change the original.
x = torch.tensor([1.0, 2.0])  # (T=2,)
y = x.clone()
y[0] = 9.0
print(x)
print(y)
```

**Output**

```text
tensor([1., 2.])
tensor([9., 2.])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.clone.html)

</details>

<details>
<summary><code>torch.no_grad</code> — Disable reverse-mode autograd recording in a block or decorated function.</summary>

**Signature**

```python
torch.no_grad()
```

**Parameters:** None.

**Example**

```python
# Purpose: Compute collection-time values without building a graph.
x = torch.tensor([1.0], requires_grad=True)  # (N=1,)
with torch.no_grad():
    frozen = 2 * x
print(frozen)
print(frozen.requires_grad)
```

**Output**

```text
tensor([2.])
False
```

Does not call model.eval(), clear existing gradients, or override explicit requires_grad on factory-created tensors.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)

</details>

## Optimization and training

<details>
<summary><code>torch.optim.SGD</code> — Update trainable parameters with stochastic gradient descent.</summary>

**Signature**

```python
torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0,
                weight_decay=0, nesterov=False, *, maximize=False,
                foreach=None, differentiable=False, fused=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `params` | Required | Parameter iterable or parameter-group dictionaries. |
| `lr` | `0.001` | Learning rate. |
| `momentum` | `0` | Momentum coefficient. |
| `dampening` | `0` | Dampening of new gradient contributions to momentum. |
| `weight_decay` | `0` | L2 penalty coefficient. |
| `nesterov` | `False` | Nesterov momentum; requires positive momentum and zero dampening. |
| `maximize` | `False` | Maximize rather than minimize the objective. |
| `foreach` | `None` | Use tensor-list kernels when supported. |
| `differentiable` | `False` | Record optimizer operations for higher-order gradients. |
| `fused` | `None` | Select a fused implementation when supported. |

**Example**

```python
# Purpose: Take one gradient-descent step on a scalar weight.
p = nn.Parameter(torch.tensor([1.0]))  # (N=1,)
optimizer = torch.optim.SGD([p], lr=0.1)
optimizer.zero_grad(set_to_none=True)
p.square().sum().backward()
optimizer.step()
print(p.detach())
```

**Output**

```text
tensor([0.8000])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)

</details>

<details>
<summary><code>torch.optim.AdamW</code> — Apply adaptive updates with decoupled weight decay.</summary>

**Signature**

```python
torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                  weight_decay=0.01, amsgrad=False, *, maximize=False,
                  foreach=None, capturable=False, differentiable=False,
                  fused=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `params` | Required | Parameter iterable or parameter-group dictionaries. |
| `lr` | `0.001` | Learning rate. |
| `betas` | (0.9, 0.999) | Decay rates for first and second gradient moments. |
| `eps` | `1e-8` | Numerical-stability term in the update denominator. |
| `weight_decay` | `0.01` | Decoupled weight-decay coefficient. |
| `amsgrad` | `False` | Use the maximum historical second-moment estimate. |
| `maximize` | `False` | Maximize rather than minimize the objective. |
| `foreach` | `None` | Use tensor-list kernels when supported. |
| `capturable` | `False` | Support graph capture on compatible devices. |
| `differentiable` | `False` | Record optimizer operations for higher-order gradients. |
| `fused` | `None` | Select a fused implementation when supported. |

**Example**

```python
# Purpose: Take one adaptive update without weight decay.
p = nn.Parameter(torch.tensor([1.0]))  # (N=1,)
optimizer = torch.optim.AdamW([p], lr=0.1, weight_decay=0.0)
optimizer.zero_grad(set_to_none=True)
p.square().sum().backward()
optimizer.step()
print(p.detach())
```

**Output**

```text
tensor([0.9000])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

</details>

<details>
<summary><code>Optimizer.zero_grad</code> — Clear gradients before the next independent backward pass.</summary>

**Signature**

```python
optimizer.zero_grad(set_to_none=True)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `set_to_none` | `True` | True removes gradient tensors; False zeros existing gradients. |

**Example**

```python
# Purpose: Clear a gradient that was accumulated by backward.
p = nn.Parameter(torch.tensor([2.0]))  # (N=1,)
optimizer = torch.optim.SGD([p], lr=0.1)
p.square().sum().backward()
print(p.grad)
optimizer.zero_grad(set_to_none=True)
print(p.grad)
```

**Output**

```text
tensor([4.])
None
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)

</details>

<details>
<summary><code>Optimizer.step</code> — Update parameters using their currently stored gradients.</summary>

**Signature**

```python
optimizer.step(closure=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `closure` | `None` | Optional callable that recomputes loss and gradients; optimizer-specific. |

**Example**

```python
# Purpose: Apply an already-computed SGD gradient.
p = nn.Parameter(torch.tensor([1.0]))  # (N=1,)
optimizer = torch.optim.SGD([p], lr=0.1)
p.square().sum().backward()
optimizer.step()
print(p.detach())
```

**Output**

```text
tensor([0.8000])
```

Signature shown for SGD/AdamW. Other optimizer subclasses can have different requirements.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html)

</details>

<details>
<summary><code>nn.Module.eval</code> — Switch a module and its children to evaluation behavior.</summary>

**Signature**

```python
module.eval()
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `—` | — | No parameters. |

**Example**

```python
# Purpose: Disable dropout behavior while collecting a rollout.
layer = nn.Dropout(p=0.5)
layer.eval()
print(layer.training)
```

**Output**

```text
False
```

Equivalent to `module.train(False)`. Crucially, `eval()` does NOT disable autograd; use `torch.no_grad()` separately when gradients are unwanted.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)

</details>

<details>
<summary><code>nn.Module.train</code> — Set training or evaluation behavior for a module and its children.</summary>

**Signature**

```python
module.train(mode=True)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `mode` | `True` | `True` enables training-specific behavior (e.g., dropout); `False` sets evaluation behavior. |

**Example**

```python
# Purpose: Restore training behavior after evaluating.
layer = nn.Linear(2, 1)
layer.eval()
layer.train(mode=True)
print(layer.training)
```

**Output**

```text
True
```

Train mode affects module behavior, not whether gradients are tracked.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train)

</details>

<details>
<summary><code>nn.Module.parameters</code> — Iterate over registered parameters for optimizers or gradient checks.</summary>

**Signature**

```python
module.parameters(recurse=True)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `recurse` | `True` | Include parameters of all descendant submodules when enabled. |

**Example**

```python
# Purpose: Inspect registered weight and bias shapes.
model = nn.Linear(2, 1)
print([tuple(p.shape) for p in model.parameters(recurse=True)])
```

**Output**

```text
[(1, 2), (1,)]
```

Returns an iterator of `nn.Parameter` objects; use this when constructing AdamW or checking separate Actor/Critic parameter sets.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters)

</details>

<details>
<summary><code>nn.Module.zero_grad</code> — Clear gradient fields of all registered module parameters.</summary>

**Signature**

```python
module.zero_grad(set_to_none=True)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `set_to_none` | `True` | Delete accumulated gradients (`.grad = None`) rather than fill them with zeros. |

**Example**

```python
# Purpose: Clear gradients after a single backward pass.
layer = nn.Linear(2, 1)
layer(torch.ones(1, 2)).sum().backward()
layer.zero_grad(set_to_none=True)
print(layer.weight.grad is None)
```

**Output**

```text
True
```

Similar to `optimizer.zero_grad`, but affects every parameter of the module rather than only those owned by one optimizer.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.zero_grad)

</details>









<details>
<summary><code>torch.nn.utils.clip_grad_norm_</code> — Limit the joint norm of parameter gradients in place.</summary>

**Signature**

```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0,
                               error_if_nonfinite=False, foreach=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `parameters` | Required | Parameter tensor or iterable with .grad values. |
| `max_norm` | Required | Maximum total gradient norm. |
| `norm_type` | `2.0` | Norm order; float('inf') uses the maximum absolute entry. |
| `error_if_nonfinite` | `False` | Raise on a nonfinite total norm. |
| `foreach` | `None` | Select tensor-list kernels when supported. |

**Example**

```python
# Purpose: Reduce a gradient vector's norm from ten to one.
p = nn.Parameter(torch.tensor([3.0, 4.0]))  # (C=2,)
p.square().sum().backward()  # Gradient is [6, 8].
before = torch.nn.utils.clip_grad_norm_(
    [p], max_norm=1.0, error_if_nonfinite=True
)
print(before)
print(p.grad)
```

**Output**

```text
tensor(10.)
tensor([0.6000, 0.8000])
```

Returns the norm BEFORE clipping. This is different from PPO probability-ratio clipping.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)

</details>

## Neural-network utilities

<details>
<summary><code>nn.Embedding</code> — Look up a learned vector for each integer token ID.</summary>

**Signature**

```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,
             max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
             sparse=False, _weight=None, _freeze=False,
             device=None, dtype=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `num_embeddings` | Required | Number of rows, usually vocabulary size V. |
| `embedding_dim` | Required | Vector width C. |
| `padding_idx` | `None` | Row excluded from ordinary gradient updates. |
| `max_norm` | `None` | Optional row-norm limit, applied during forward. |
| `norm_type` | `2.0` | Norm used with max_norm. |
| `scale_grad_by_freq` | `False` | Divide updates by token frequency within the batch. |
| `sparse` | `False` | Produce sparse weight gradients. |
| `_weight` | `None` | Internal constructor option for initial weights. |
| `_freeze` | `False` | Internal constructor option to disable weight gradients. |
| `device` | `None` | Parameter device. |
| `dtype` | `None` | Parameter dtype. |

**Example**

```python
# Purpose: Retrieve two token vectors from a known embedding table.
embedding = nn.Embedding(3, 2)  # (V=3, C=2): weight shape
with torch.no_grad():
    embedding.weight.copy_(torch.tensor([[0., 0.], [1., 2.], [3., 4.]]))
tokens = torch.tensor([[1, 2]])  # (B=1, T=2)
print(embedding(tokens).detach())  # (B=1, T=2, C=2)
```

**Output**

```text
tensor([[[1., 2.],
         [3., 4.]]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

</details>

<details>
<summary><code>nn.Linear</code> — Project the final feature dimension with an affine transformation.</summary>

**Signature**

```python
nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `in_features` | Required | Input feature width. |
| `out_features` | Required | Output feature width. |
| `bias` | `True` | Learn an additive output bias. |
| `device` | `None` | Parameter device. |
| `dtype` | `None` | Parameter dtype. |

**Example**

```python
# Purpose: Project three input features into two outputs.
head = nn.Linear(3, 2, bias=False)
with torch.no_grad():
    head.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 1.]]))
x = torch.tensor([[1., 2., 3.]])  # (B=1, C_in=3)
print(head(x).detach())           # (B=1, C_out=2)
```

**Output**

```text
tensor([[1., 5.]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)

</details>

<details>
<summary><code>nn.Parameter</code> — Register a tensor as a model parameter when assigned to a module.</summary>

**Signature**

```python
nn.Parameter(data=None, requires_grad=True)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `data` | `None` | Initial tensor values. |
| `requires_grad` | `True` | Enable optimization through gradient recording. |

**Example**

```python
# Purpose: Register a custom trainable scale vector.
model = nn.Module()
model.scale = nn.Parameter(torch.ones(2))  # (C=2,)
print(list(dict(model.named_parameters())))
print(model.scale.requires_grad)
```

**Output**

```text
['scale']
True
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Parameter.html)

</details>

<details>
<summary><code>nn.init.zeros_</code> — Fill a tensor with zeros in place without recording initialization gradients.</summary>

**Signature**

```python
nn.init.zeros_(tensor)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `tensor` | Required | Tensor or parameter to initialize in place. |

**Example**

```python
# Purpose: Initialize a neutral value-head weight vector.
weight = nn.Parameter(torch.tensor([[2.0, -3.0]]))  # (1, C=2)
nn.init.zeros_(weight)
print(weight.detach())
```

**Output**

```text
tensor([[0., 0.]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.zeros_)

</details>

<details>
<summary><code>nn.ModuleList</code> — Store child modules while registering their parameters.</summary>

**Signature**

```python
nn.ModuleList(modules=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `modules` | `None` | Iterable of child modules; omitted means an empty collection. |

**Example**

```python
# Purpose: Register two layers as a reusable stack.
layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 1)])
print(layers)
```

**Output**

```text
ModuleList(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=1, bias=True)
)
```

ModuleList stores modules; the enclosing model still defines how to run them.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)

</details>

<details>
<summary><code>nn.RMSNorm</code> — Normalize trailing features by their root-mean-square magnitude.</summary>

**Signature**

```python
nn.RMSNorm(normalized_shape, eps=None, elementwise_affine=True,
           device=None, dtype=None)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `normalized_shape` | Required | Trailing dimension size or shape to normalize. |
| `eps` | `None` | Stability term; None uses machine epsilon for the input dtype. |
| `elementwise_affine` | `True` | Learn one scale per normalized element. |
| `device` | `None` | Parameter device. |
| `dtype` | `None` | Parameter dtype. |

**Example**

```python
# Purpose: Normalize a two-feature vector without a learned scale.
norm = nn.RMSNorm(2, eps=1e-5, elementwise_affine=False)
x = torch.tensor([[3.0, 4.0]])  # (B=1, C=2)
print(norm(x))
```

**Output**

```text
tensor([[0.8485, 1.1314]])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)

</details>

<details>
<summary><code>F.elu</code> — Keep positive values and apply an exponential branch to negative values.</summary>

**Signature**

```python
F.elu(input, alpha=1.0, inplace=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Activation tensor. |
| `alpha` | `1.0` | Scale of the negative branch. |
| `inplace` | `False` | Overwrite input when True. |

**Example**

```python
# Purpose: Apply ELU to negative, zero, and positive inputs.
x = torch.tensor([-1.0, 0.0, 1.0])  # (C=3,)
print(F.elu(x))
```

**Output**

```text
tensor([-0.6321,  0.0000,  1.0000])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.elu.html)

</details>

<details>
<summary><code>F.gelu</code> — Apply the Gaussian error linear activation.</summary>

**Signature**

```python
F.gelu(input, approximate="none")
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Activation tensor. |
| `approximate` | `"none"` | 'none' uses the erf form; 'tanh' uses an approximation. |

**Example**

```python
# Purpose: Apply the common tanh approximation to GELU.
x = torch.tensor([-1.0, 0.0, 1.0])  # (C=3,)
print(F.gelu(x, approximate="tanh"))
```

**Output**

```text
tensor([-0.1588,  0.0000,  0.8412])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html)

</details>

<details>
<summary><code>F.silu</code> — Multiply each input by its sigmoid gate.</summary>

**Signature**

```python
F.silu(input, inplace=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `input` | Required | Activation tensor. |
| `inplace` | `False` | Overwrite input when True. |

**Example**

```python
# Purpose: Apply the activation used inside SwiGLU.
x = torch.tensor([-1.0, 0.0, 1.0])  # (C=3,)
print(F.silu(x))
```

**Output**

```text
tensor([-0.2689,  0.0000,  0.7311])
```

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.silu.html)

</details>

<details>
<summary><code>F.scaled_dot_product_attention</code> — Compute scaled attention over keys and values.</summary>

**Signature**

```python
F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                               dropout_p=0.0, is_causal=False,
                               scale=None, enable_gqa=False)
```

**Parameters**

| Parameter | Default | Meaning |
|:--|:--|:--|
| `query` | Required | Query vectors, e.g. (B, H_Q, T_Q, D). |
| `key` | Required | Key vectors, e.g. (B, H_KV, T_K, D). |
| `value` | Required | Value vectors, e.g. (B, H_KV, T_K, D_V). |
| `attn_mask` | `None` | Broadcastable Boolean or additive mask; Boolean True means allowed. |
| `dropout_p` | `0.0` | Attention dropout probability; explicitly use zero during evaluation. |
| `is_causal` | `False` | Apply a causal mask; do not combine with attn_mask in this interface. |
| `scale` | `None` | Score multiplier; None uses 1 / sqrt(D). |
| `enable_gqa` | `False` | Enable grouped-query attention when head counts/backends support it. |

**Example**

```python
# Purpose: Show causal attention with uniform scores over available keys.
q = torch.zeros(1, 1, 2, 2)  # (B=1, H=1, T=2, D=2)
k = torch.zeros_like(q)
v = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]])
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(y)  # First token sees itself; second averages both values.
```

**Output**

```text
tensor([[[[10.,  0.],
          [ 5., 10.]]]])
```

Dropout follows dropout_p, not the parent module's train/eval flag. scale and enable_gqa are keyword-only.

[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

</details>

## Learning history

<details>
<summary>Lesson notes and maintenance rules</summary>

### Maintenance

Keep each new entry in the same format: one-line purpose, signature, one row per parameter, one common example, and its captured output. Explain important tensor dimensions and distinguish in-place operations. Keep optional/internal arguments in the reference table, not in the common example unless relevant.

**Status:** Understood means explained and used deliberately; Introduced means encountered but still under review. All 81 tracked entries are now marked Understood after the Lesson 21B review; this records the examples and semantics learned here, not exhaustive coverage of every PyTorch overload.

### Lesson 13 — Generation additions

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

### Lesson 14 — GQA additions

- `torch.repeat_interleave` — introduced after first implementing KV-head expansion manually.

### Lesson 15 — Linear-attention additions

- `F.elu` — used in reusable source only after the notebook derived the positive feature map manually.
- `torch.cumsum` — reused to construct causal prefix states for parallel linear attention.

### Lesson 19 — Policy-gradient additions

- `torch.log_softmax` — used to keep sampled-action log-probabilities differentiable and numerically stable.
- `torch.autograd.grad` — used to inspect score-function gradients such as `d log pi(a) / d theta` directly.
- `torch.multinomial` — revisited as categorical action sampling; the sampled index is discrete and gradients do not pass through the sampling operation.
- `Tensor.detach` — revisited to separate stochastic sampling / credit weights from the differentiable log-probability path.

### Lesson 20 — Actor-critic and GAE additions

- `nn.init.zeros_` — used to initialize the tabular actor and critic at a known neutral starting point; the trailing underscore indicates an in-place operation.
- `torch.no_grad` — revisited to freeze bootstrap targets and rollout-time measurements during critic/actor updates.
- `Tensor.detach` — revisited to keep TD/advantage credit signals fixed when optimizing the actor.
- `torch.zeros_like` — reused to allocate GAE advantage tensors with matching shape, dtype, and device.

### Lesson 21A — PPO from Scratch additions

- `torch.clamp` and `torch.minimum` — construct the advantage-aware clipped PPO surrogate; clipping is not a hard constraint on updated policy probabilities.
- `torch.randperm` — shuffle complete trajectory indices before selecting aligned sequence minibatches.
- `torch.linspace` — display ratio-dependent objective curves for positive and negative advantages.
- `Tensor.expand` — share a small position template across a batch without duplicating data.
- `Tensor.numel` — reject masked reductions with no valid actions.
- `torch.any` and `torch.all` — validate action position, padding, and termination invariants.

The RL primitives extracted after Lesson 21A live in `src/llmfp/rl/`; their regression tests use only the Python standard library and PyTorch.

### Lesson 21B — Tiny GPT PPO integration (complete)

- `torch.manual_seed` — reproducible sampling and minibatch-order experiments.
- `torch.testing.assert_close` — raise an informative error when causal rescoring and saved old log-probabilities disagree.
- `nn.Module.eval` / `nn.Module.train` — control dropout-like behavior without enabling or disabling autograd.
- `nn.Module.parameters` / `nn.Module.zero_grad` — inspect independently owned Actor/Critic parameters and clear gradients.
- `Tensor.abs`, `Tensor.max`, `Tensor.min`, `Tensor.norm` — inspect parameter differences, ratio ranges, and gradient magnitudes.
- `Tensor.item`, `Tensor.tolist`, `Tensor.float` — extract scalar diagnostics, display token sequences, and convert numeric dtypes.

**Chapter closure (October 2026):** All 13 APIs added during Lesson 21B have been explained with a signature, parameter descriptions, a runnable common example, and its output. No additional PyTorch API entries were needed for the final lesson audit.

</details>

## Reference notes

The API selection, categories, and lesson history come from the previous tracker. Parameter descriptions have been separated into rows, malformed text has been repaired, and signatures use explicit overload notes where needed. The linked PyTorch documentation and the locally installed API were consulted for ambiguous signatures; this is not an exhaustive audit of every optional-argument combination.

`torch.zeros` and `torch.ones` retain the previously listed `pin_memory` allocation option, with its documentation caveat. `Tensor.scatter_` distinguishes the displayed `reduce=None` signature from an ordinary call that omits `reduce`. Newer `torch.autograd.grad` documentation also describes dictionary inputs; the example here deliberately uses the tensor form supported by the tested runtime.
