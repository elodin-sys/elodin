# Elodin

Elodin is a platform for rapid design, testing, and simulation of
drones, satellites, and aerospace control systems. This repository
is a collection of core libraries:

- `libs/nox`: Tensor library that compiles to XLA (like
JAX, but for Rust).
- `libs/nox-ecs`: ECS framework built to work with Jax and Nox,
that allows you to build your own physics engine.
- `libs/nox-ecs-macros`: Derive macros to generate implementations of
ECS and Nox traits.
- `libs/conduit`: Column-based protocol for transferring ECS data
between different systems.
- `libs/xla-rs`: Rust bindings to XLA's C++ API (originally based on
https://github.com/LaurentMazare/xla-rs).

## License

Licensed under either of

 * [Apache License, Version 2.0](LICENSES/Apache-2.0.txt)
 * [MIT License](LICENSES/MIT.txt)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
