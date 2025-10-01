# nox-ecs

**nox-ecs** provides the core traits and types for the ECS-like layer of the
[Nox engine](../nox).

Nox is a Rust implementation of [JAX](https://github.com/google/jax), inspired by
NumPy for expressive numerical computing, but built in Rust for performance,
strong typing, and integration with flight software.  
For a detailed introduction, see
[NumPy, JAX & Nox](https://github.com/elodin-sys/elodin/blob/main/docs/public/content/home/tao/jax-nox.md).

## What this crate provides

- **`Component` trait** – the base interface to register a Rust type as a Nox component (stable name, Impeller schema, introspection).
- **`Archetype`** – aggregates multiple components in a struct, with insertion and introspection into a `World`.
- **`ComponentGroup`** – logical grouping of components, supports iteration and can produce a computational representation (`Noxpr::tuple`).
- **`IntoOp` / `FromOp` conversions** – bridges between Rust objects and Nox operators (`Noxpr`).
- **World** – container and organizer for components, initializes systems, supports introspection and execution.

## Usage with `nox-ecs-macros`

This crate is designed to be used together with
[`nox-ecs-macros`](../nox-ecs-macros).  
The derive macros save you from writing verbose and repetitive code:

- `#[derive(Component)]`
- `#[derive(Archetype)]`
- `#[derive(ComponentGroup)]`
- `#[derive(IntoOp)]`, `#[derive(FromOp)]`
- etc.

In practice you declare a `struct`, add a `#[derive(...)]`, and the connection with the Nox engine is generated automatically.

## Some examples

### Defining a custom component
Rather than hand-implementing multiple traits, you can derive them automatically with the `#[derive(Component, ReprMonad)]` macros from `nox-ecs-macros`:

```rust
use nox_ecs::{Op, OwnedRepr};       // core types re-exported by nox-ecs
use nox_ecs::nox::Scalar;           // scalar tensor type from the underlying Nox library
use nox_ecs_macros::ReprMonad;      // derive macro for symbolic representation

// `#[derive(nox_ecs::Component, ReprMonad)]`:
// - implements the Component trait (so `Mass` can live in a World)
// - implements the ReprMonad trait (so `Mass` can participate in symbolic ops)
#[derive(nox_ecs::Component, ReprMonad)]
struct Mass<R: OwnedRepr = Op>(Scalar<f64, R>);
```

### Using a concrete alias for everyday code
The first example showed how to define a generic component with `#[derive(Component, ReprMonad)]`.
In practice, you often want a concrete type that is directly usable in systems or worlds. By fixing the representation to `Op`, you get a type alias that feels natural to manipulate while still keeping the generic definition for flexibility.
```rust
use nox_ecs::{Op, OwnedRepr};       // core types from nox-ecs
use nox_ecs::nox::Scalar;           // scalar tensor type from Nox
use nox_ecs_macros::ReprMonad;      // derive macro for symbolic representation

#[derive(nox_ecs::Component, ReprMonad)]
struct Mass<R: OwnedRepr = Op>(Scalar<f64, R>);

// Concrete alias with Op as the backend representation
type MassOp = Mass<Op>;
```

### Working with symbolic components

By fixing the backend to `Op`, components become symbolic nodes in the `Nox computation graph`.
This lets you build graphs that can be inspected, transformed, or differentiated.
In the example below, two constant masses are added as a symbolic sum; to get a numeric value, the graph is lowered to a `NoxprFn` and evaluated with the `noxpr` execution pipeline.
```rust
use nox_ecs::{Op, OwnedRepr};
use nox_ecs::nox::Scalar;
use nox_ecs_macros::ReprMonad;

use nox::{Noxpr, NoxprFn, NoxprScalarExt, Client, Tensor, FromTypedBuffers};
use nox::xla::BufferArgsRef;

#[derive(nox_ecs::Component, ReprMonad)]
struct Mass<R: OwnedRepr = Op>(Scalar<f64, R>);

// Constant nodes in the computation graph (symbolic under `Op`)
let m1: Mass<Op> = Mass(1.0.into());
let m2: Mass<Op> = Mass(2.5.into());

// This builds a graph expression (not an immediate f64)
let _total_symbolic = Mass(m1.0 + m2.0);

// Build the expression (constants) and wrap it in a NoxprFn
let expr = 1.0f64.constant() + 2.5f64.constant();
let f = NoxprFn::new(vec![], expr);

// Trace to XLA then compile with PJRT (CPU)
let xla_op = f.build("sum_constants").unwrap();
let comp   = xla_op.build().unwrap();
let client = Client::cpu().unwrap();
let exec   = client.compile(&comp).unwrap();

// Execute 
let args = BufferArgsRef::default();
let mut outs = exec.execute_buffers(args).unwrap();

// Convert the PJRT output to a host Tensor and verify the result
let out = Tensor::from_typed_buffers(&Tensor::from_pjrt_buffers(&mut outs)).unwrap();
assert_eq!(out, Tensor::from(3.5f64));
```