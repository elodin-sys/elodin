# noxla

noxla is a Rust wrapper around XLA, a compiler for machine learning and linear algebra. The goal of this project is to create a set of safe bindings close to XLA's C++ API. The intended use of this crate is in other higher-level linear algebra and ML crates. It can be used on its own, but it doesn't promise to be the most ergonomic or well-documented.

The bindings are rewritten to use the cpp and cxx crates. This allows our bindings to be written in line with Rust code, and to have more clear typing.
