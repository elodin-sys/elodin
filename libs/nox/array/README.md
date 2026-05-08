# nox_array

## Description
A small building block of the `nox` crate for representing n‑dimensional arrays (tensors) in a zero‑copy friendly way thanks to `ArrayView`.
`ArrayView` does not own data: it borrows a buffer (`&[T]` or `&[u8]`) and associates it with a shape (`&[usize]`). This is enough to describe and manipulate multi‑dimensional arrays even in constrained environments (embedded, GPU, WASM), without allocation and without heavy dependencies like ndarray.

## Simple example
```rust
use nox_array::ArrayView;

let data = [1, 2, 3, 4, 5, 6];
let shape = [2, 3];

let view = ArrayView::from_buf_shape_unchecked(&data, &shape);

assert_eq!(view.len(), 6);
assert_eq!(view.shape(), &shape);
// Interpreted as [[1,2,3], [4,5,6]] in contiguous memory
```

## Key points
- Zero‑copy: no allocation, just a view.
- `no_std` by default: works in firmware, WASM, kernels, etc.
- Safe but minimal: you must guarantee that `buf.len()` equals the product of dimensions in shape.
- Adds pretty printing similar to how `NumPy` prints arrays, with ellipses for large arrays.

## Dynamic broadcasting in `nox::Array`

The parent `nox` crate uses dynamic array shapes for runtime tensor values. Binary operations on dynamic `nox::Array` values (`add`, `sub`, `mul`, `div`) follow the same right-aligned broadcasting rule used by NumPy and JAX:

- A scalar shape `[]` broadcasts to any output shape.
- Two dimensions are compatible when they are equal or one of them is `1`.
- Shapes are compared from the trailing dimension toward the leading dimension.
- Missing leading dimensions behave like `1`.
- Fallible APIs (`try_add`, `try_sub`, `try_mul`, `try_div`) return a controlled error for incompatible shapes.
- Non-fallible APIs (`add`, `sub`, `mul`, `div`) remain compatibility wrappers and expect the shapes to be broadcastable.

The dynamic broadcasting tests in `../src/array/mod.rs` and `../src/array/broadcast.rs` document the expected behavior with concrete examples:

| Scenario | Expected shape |
| --- | --- |
| `scalar * [3]` | `[3]` |
| `[3] * scalar` | `[3]` |
| `[3] + [2, 3]` | `[2, 3]` |
| `[2, 3] + [3]` | `[2, 3]` |
| `[2, 3] + [2, 1]` | `[2, 3]` |
| `[2, 3] + [2, 3]` | `[2, 3]` |
| `scalar * [2, 3]` | `[2, 3]` |
| `[2, 3] * scalar` | `[2, 3]` |
| `[1, 3] + [2, 1, 3]` | `[2, 1, 3]` |
| `[2, 3] + [3, 2]` | error |

Dynamic allocation uses the product of all dimensions. For example, `Array::zeroed([2, 3])` allocates `6` elements.

## More examples

### From raw bytes
```rust
use nox_array::ArrayView;
use zerocopy::byteorder::{LE, U32};

// [1u32, 2u32] as little-endian bytes
let bytes: &[u8] = &[
    0x01, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00,
];

let view = ArrayView::<U32<LE>>::from_bytes_shape_unchecked(bytes, &[2]).unwrap();
assert_eq!(view.len(), 2);
```

### Pretty printing
```rust
#[cfg(feature = "std")]
{
    use nox_array::ArrayView;

    let data: Vec<i32> = (0..20).collect();
    let v = ArrayView::from_buf_shape_unchecked(&data, &[4, 5]);

    println!("{}", v);   // uses ellipses if needed
    println!("{:#}", v); // shows everything without ellipses
}
```
