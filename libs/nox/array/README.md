# nox/array

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

## More examples

### From raw bytes
```rust
use nox_array::ArrayView;
use zerocopy::{Immutable, TryFromBytes};


#[derive(TryFromBytes, Immutable)]
#[repr(C)]
struct F32(f32);


let bytes: &[u8] = &[0, 0, 0, 0, 0, 0, 0x80, 0x3f]; // [0.0, 1.0] as f32 LE
let view = ArrayView::<F32>::from_bytes_shape_unchecked(bytes, &[2]).unwrap();
assert_eq!(view.len(), 2);
```

### Pretty printing
```rust
#[cfg(feature = "std")]
{
   let data: Vec<i32> = (0..20).collect();
   let v = ArrayView::from_buf_shape_unchecked(&data, &[4, 5]);

   println!("{}", v); // uses ellipses if needed
   println!("{:#}", v); // shows everything without ellipses
}
```
