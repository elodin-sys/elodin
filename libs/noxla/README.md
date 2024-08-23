# noxla

> Originally based on https://github.com/LaurentMazare/xla-rs

noxla is a Rust wrapper around XLA, a compiler for machine learning and linear algebra. The goal of this project is to create a set of safe bindings close to XLA's C++ API. The intended use of this crate is in other higher-level linear algebra and ML crates. It can be used on its own, but it doesn't promise to be the most ergonomic or well-documented.


## Why Fork?

This crate differs from the original xla-rs in a few key ways. The biggest is that the bindings are rewritten to use the cpp and cxx crates. This allows our bindings to be written in line with Rust code, and to have more clear typing. The other large change is that we are using a fork of [`xla_extension`](https://github.com/elodin-sys/xla), which allows for building a fully static binary. This fits better the intended use of noxla as a low-level crate, by other higher-level libraries.
