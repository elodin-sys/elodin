# nox


## Description
`nox` is the **core crate of the Nox engine**, a Rust implementation inspired by [NumPy](https://numpy.org) and [JAX](https://github.com/google/jax). 
It provides the **symbolic backend**, **tensor representation**, and **differentiable computation primitives** that power the higher-level Nox crates.


For a broader introduction to the project, see the [overview documentation](../../docs/public/content/home/tao/jax-nox.md).


## Role of this crate
- Defines the fundamental tensor types (`Scalar`, `Vector`, `Matrix`, `Tensor<_, _, Op>`). 
- Provides an Intermediate Representation (IR) for mathematical expressions and integrates with XLA via the `noxla` crate for compilation and execution. 
- Implements differentiable programming utilities used across the Nox ecosystem. 
- Serves as the foundation for domain-specific layers (ECS, world management, bindings, etc.).


Most users will **not depend on `nox` directly**. Instead, they will interact with one of the specialized crates or subsystems built on top of it.


## Related crates and subsystems
- [array](array) – array and tensor utilities.
- [noxpr](src/noxpr) – subsystem of nox (not standalone) for building tensor compute graphs in Rust and lowering them to XLA. 
- [nox-ecs](../nox-ecs) – ECS-like layer and world management. 
   - [nox-ecs-macros](../nox-ecs-macros) – derive macros for components and archetypes. 
- [nox-py](../nox-py) – Python bindings. 
- [noxla](../noxla) – minimal integration layer with XLA. 


### Visual overview
```text
nox (core crate: tensors, symbolic backend, differentiation)
├── array (tensor/array utilities)
├── src/noxpr (subsystem: tensor IR + XLA lowering)
├── nox-ecs (ECS-like layer & world management)
│ └── nox-ecs-macros (derive macros for components/archetypes)
├── nox-py (Python bindings)
└── noxla (minimal XLA integration)
```
