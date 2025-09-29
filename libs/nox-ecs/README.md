# nox-ecs

**nox-ecs** provides the core traits and types for the ECS-like layer of the
[Nox engine](../nox).<!-- TODO: replace with link to Nox README once available -->

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
[`nox-ecs-macros`](../nox-ecs-macros/README.md).  
The derive macros save you from writing verbose and repetitive code:

- `#[derive(Component)]`
- `#[derive(Archetype)]`
- `#[derive(ComponentGroup)]`
- `#[derive(IntoOp)]`, `#[derive(FromOp)]`
- etc.

In practice: you declare a `struct`, add a `#[derive(...)]`, and the connection with the Nox engine is generated automatically.
