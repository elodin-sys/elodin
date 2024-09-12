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


## WASM debugging:
Debugging the WASM bundle can be challenging. For issues that only present in the browser but work fine in the desktop client, a method:

- find & uncomment this line `tracing_wasm::set_as_global_default()` in [./src/main.rs](./src/main.rs), this will great increase the verbosity of browser console logging output.
- update the [Justfile](./Justfile) to build debug instead of release, this further increases the useful detail you'll find in the logs.
- find & uncomment this line `Error.stackTraceLimit = Infinity` in [index.html](./index.html) if you need even more logs from the error stack trace.
- from this folder, build the bundle with `just build`, which will copy the bundle alongside the `index.html` test page.
- start a new web sandbox via `app.elodin.dev`, and extract the `data-ws-url` from it to replace the one found in [index.html](./index.html)
- serve & load the test page with a simple web server such as `python3 -m http.server`
