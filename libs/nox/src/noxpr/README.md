# noxpr — Tensor Expression IR

## Description
`noxpr` lets you build tensor compute graphs in Rust.
It is a subsystem of **Nox** (not a standalone crate) and lives in `libs/nox/src/noxpr/`.

## Features

1. **Build compute graphs in Rust** — Create expressions with Rust types and methods (no text parsing).
   Use `Scalar`, `Vector`, `Matrix`, and `Tensor<_, _, Op>` to build a graph (`Noxpr` / `NoxprNode`).

2. **Lower to JAX** — With the `jax` feature, `JaxTracer` converts the graph to JAX operations
   for compilation via StableHLO and IREE.

3. **Rich, typed ops** — Elementwise math, linear algebra (`dot`, `dot_general`), shape transforms
   (`reshape`, `transpose`, `broadcast`), indexing (`slice`, `gather`, `dynamic_slice`),
   control flow (`scan`, `select`), type casts (`convert`), and `cholesky`.

## Snippets

All snippets are supposed to be within the `nox` crate (or its internal modules),
since `noxpr` is not publicly exported.

---

### Build a graph in Rust

> Example: define `a`, `b` (f32 vectors of length 3), then `expr = dot(((a + 1) * b), a)`
> and wrap it in a `NoxprFn`.

```rust
use nox::{Noxpr, NoxprFn, ArrayTy, NoxprTy, NoxprScalarExt, ElementType};
use smallvec::smallvec;

// Parameters
let a = Noxpr::parameter(
    0,
    NoxprTy::ArrayTy(ArrayTy::new(ElementType::F32, smallvec![3])),
    "a".into()
);
let b = Noxpr::parameter(
    1,
    NoxprTy::ArrayTy(ArrayTy::new(ElementType::F32, smallvec![3])),
    "b".into()
);

// ((a + 1) * b).dot(a)
let expr = ((a.clone() + 1.0f32.constant()) * b.clone()).dot(&a);

// Function wrapper
let f = NoxprFn::new(vec![a, b], expr);

// Human-readable IR preview
println!("{}", f);
```

---

### Debug & helpers

> Pretty-print the graph (human-readable IR)

```rust
use nox::PrettyPrintTracer;

let mut pp = PrettyPrintTracer::default();
let mut s = String::new();
let f = nox::doctest::noxpr::example_function();
pp.visit(&f.inner, &mut s).unwrap();
println!("{s}");
```
