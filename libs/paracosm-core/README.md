# Paracosm Core

### Building & Testing Examples for WebAssembly

Pre-requisites:
Sourced from [here](https://github.com/bevyengine/bevy/tree/main/examples#wasm)
- [rustup](https://rustup.rs/) - Install Rust & Cargo
- [basic-http-server](https://crates.io/crates/basic-http-server) - Add to your path
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) - Add to your path
- [wasm-opt](https://github.com/WebAssembly/binaryen/releases) - download and place bin/wasm-opt and lib in your path


```bash
# Build an example with no features, wasm release profile
cargo build --no-default-features --profile wasm-release --example solar_system --target wasm32-unknown-unknown

# Transpile to wasm
wasm-bindgen --out-name paracosm \
  --out-dir . \
  --target web ../../target/wasm32-unknown-unknown/wasm-release/examples/solar_system.wasm
  
# Optimize runtime size & speed
wasm-opt -Oz -ol 100 -s 100 -o ./paracosm_bg.wasm paracosm_bg.wasm

# Serve from root directory & open in browser
basic-http-server
```
