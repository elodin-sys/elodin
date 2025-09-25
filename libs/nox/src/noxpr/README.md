# noxpr — Tensor Expression IR & XLA Tracer

## Description
`noxpr` lets you build tensor compute graphs in Rust and run them fast with XLA (Accelerated Linear Algebra).  
It is a subsystem of **Nox** (not a standalone crate) and lives in `libs/nox/src/noxpr/`.

## Features

1. **Build compute graphs in Rust** — Create expressions with Rust types and methods (no text parsing).  
   Use `Scalar`, `Vector`, `Matrix`, and `Tensor<_, _, Op>` to build a graph (`Noxpr` / `NoxprNode`). 

2. **Compile to XLA (Accelerated Linear Algebra)** — `XlaTracer` lowers the graph to an `xla::XlaComputation`.  
   You can also dump **HLO** (High Level Optimizer) text to see what will run.  
   
3. **Run on CPU/GPU via PJRT (Portable Just‑In‑Time Runtime)** — The `Client` compiles and executes on CPU (Central Processing Unit) and, if enabled, GPU (Graphics Processing Unit).  
   
4. **Rich, typed ops** — Elementwise math, linear algebra (`dot`, `dot_general`), shape transforms (`reshape`, `transpose`, `broadcast`), indexing (`slice`, `gather`, `dynamic_slice`), control flow (`scan`, `select`), type casts (`convert`), and `cholesky`.  

5. **Optional JAX bridge** — With the `jax` feature, interoperate with **JAX** (Python numerical library). Handy for mixing ecosystems or debugging.

## Snippets
All snippets are supposed to be within the `nox` crate (or its internal modules), since `noxpr` is not publicly exported.

---

###  Build a graph in Rust

> Example: define `a`, `b` (f32 vectors of length 3), then `expr = dot(((a + 1) * b), a)` and wrap it in a `NoxprFn`.

```rust
use crate::nox::{Noxpr, NoxprFn, ArrayTy, NoxprTy, NoxprScalarExt};
use smallvec::smallvec;
use xla::ElementType;

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

// Function wrapper (for compile / pretty-print / etc.)
let f = NoxprFn::new(vec![a, b], expr);

// Human-readable IR preview (see also “Debug” section below)
println!("{}", f);
```

---

### Lower to XLA + peek at HLO

> Starting from a `NoxprFn`, produce an `xla::XlaOp` then an `xla::XlaComputation`.  

```rust
use nox::NoxprFn;

# fn main() -> Result<(), nox::Error> {
# let f = nox::doctest::noxpr::example_function();

let xla_op = f.build("feat1_example")?;   // Built via XlaTracer
let comp   = xla_op.build()?;             // xla::XlaComputation
println!("{}", comp.to_hlo_text()?);
# Ok(())
# }
```

---

### Run via PJRT (CPU/GPU)

> Minimal execution skeleton with `Client` (CPU by default, GPU if enabled).  
> *Note:* input types/shapes must match the XLA signature.

```rust
use nox::{Client, tensor};
# fn main() -> Result<(), nox::Error> {
let client = Client::cpu()?;

# let f = nox::doctest::noxpr::example_function();
# let xla_op = f.build("feat1_example")?;   // Built via XlaTracer
# let comp   = xla_op.build()?;             // xla::XlaComputation

let exec   = client.compile(&comp)?;
let input  = tensor![1.0f32, 2.0, 3.0];
let out    = exec.execute_buffers(input)?.to_host();
# Ok(())
# }
```

---

### Typed ops (reshape/transpose/slice/gather/dot_general/select/cholesky)

> Tiny recipes around typed operators. All start from `Noxpr`.

```rust
use nox::{Noxpr, ArrayTy, NoxprTy};
use smallvec::smallvec;
use xla::ElementType;

// x: f32[3, 1]
let x = Noxpr::parameter(
    0,
    NoxprTy::ArrayTy(ArrayTy::new(ElementType::F32, smallvec![3, 1])),
    "x".into()
);

// Broadcast (3,1) -> (3,4) along dim 1
let y = x.clone().broadcast_in_dim(smallvec![3, 4], smallvec![0, 1]);

// Transpose (2D)
let y_t = y.transpose(smallvec![1, 0]);

// Reshape (example) to (12,)
let z = y_t.reshape(smallvec![12]);
```

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
