# nox


## Description
`nox` is the **core crate of the Nox engine**, a Rust implementation inspired by [NumPy](https://numpy.org) and [JAX](https://github.com/google/jax). 
It provides the **symbolic backend**, **tensor representation**, and **differentiable computation primitives** that power the higher-level Nox crates.


For a broader introduction to the project, see the [overview documentation](../../docs/public/content/home/tao/jax-nox.md).


## Role of this crate
- Defines the fundamental tensor types (`Scalar`, `Vector`, `Matrix`, `Tensor<_, _, Op>`). 
- Provides an Intermediate Representation (IR) for mathematical expressions, compiled to cranelift (default) or executed via JAX for simulation backends. 
- Implements differentiable programming utilities used across the Nox ecosystem. 
- Serves as the foundation for domain-specific layers (ECS, world management, bindings, etc.).


Most users will **not depend on `nox` directly**. Instead, they will interact with one of the specialized crates or subsystems built on top of it.

## Dynamic array broadcasting

Dynamic `nox::Array` binary operations (`add`, `sub`, `mul`, `div`) follow the same right-aligned broadcasting rule used by NumPy and JAX:

- A scalar shape `[]` broadcasts to any output shape.
- Two dimensions are compatible when they are equal or one of them is `1`.
- Shapes are compared from the trailing dimension toward the leading dimension.
- Missing leading dimensions behave like `1`.
- Fallible APIs (`try_add`, `try_sub`, `try_mul`, `try_div`) return a controlled error for incompatible shapes. The non-fallible APIs remain compatibility wrappers.

The dynamic broadcasting tests in `src/array/mod.rs` document the expected behavior with concrete examples:

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

`Array::zeroed([2, 3])` is also covered by tests and must allocate the product of the dimensions (`6` elements), not their sum.


## Related crates and subsystems
- [array](array) – array and tensor utilities.
- [noxpr](src/noxpr) – subsystem of nox (not standalone) for building tensor compute graphs in Rust, lowered to JAX for StableHLO/IREE compilation. 
- [nox-py](../nox-py) – Python bindings (includes ECS layer in `src/`).
   - [elodin-macros](../elodin-macros) – derive macros for components and archetypes.


### Visual overview
```text
nox (core crate: tensors, symbolic backend, differentiation)
├── array (tensor/array utilities)
├── src/noxpr (subsystem: tensor IR → JAX → StableHLO → Cranelift)
├── nox-py (Python bindings; ECS in src/)
└── elodin-macros (derive macros for components/archetypes)
```
