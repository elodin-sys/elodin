# Elodin Editor

### Install

Install the editor using the standalone installer script:

```sh
# Install the latest version
curl -LsSf https://storage.googleapis.com/elodin-releases/install-editor.sh | sh

# Install a specific version (e.g., 0.13.3)
curl -LsSf https://storage.googleapis.com/elodin-releases/install-editor.sh | sh -s v0.13.3
```

Alternatively, you can download the latest portable binary for your platform:

- [macOS (arm64)](https://storage.googleapis.com/elodin-releases/latest/elodin-aarch64-apple-darwin.tar.gz)
- [Linux (x86_64)](https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-unknown-linux-musl.tar.gz)
- [Linux (arm64)](https://storage.googleapis.com/elodin-releases/latest/elodin-aarch64-unknown-linux-musl.tar.gz)
- [Windows (x86_64)](https://storage.googleapis.com/elodin-releases/latest/elodin-x86_64-pc-windows-msvc.zip)

### How to run locally

Install nox-py:

```sh
cd libs/nox-py
uv venv
# Following command should be the one mentioned in previous output, like `Activate with: source .venv/bin/activate.fish`
source .venv/bin/activate.fish
uvx maturin develop --uv
```

Run `three-body.py` example in the editor:

```sh
# run from `libs/nox-py`
cargo run --manifest-path=../../apps/elodin/Cargo.toml editor examples/three-body.py
```

Run `three-body.py` example while watching for editor code changes (requires [cargo-watch](https://crates.io/crates/cargo-watch)):

```sh
# run from `libs/nox-py`
cargo watch --watch ../../libs/elodin-editor \
    -x 'run --manifest-path=../../apps/elodin/Cargo.toml editor examples/three-body.py'
```
