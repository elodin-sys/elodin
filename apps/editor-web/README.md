# Requirements

- [`Rust`](https://www.rust-lang.org/tools/install)
- [`just`](https://just.systems/man/en/chapter_1.html)
- `wasm32` target for rust: 

```bash
rustup target install wasm32-unknown-unknown
```

- [`wasm-bindgen`](https://github.com/rustwasm/wasm-bindgen):

```bash
cargo install wasm-bindgen-cli
```

# How to Build

- Run `just build` in this directory.
