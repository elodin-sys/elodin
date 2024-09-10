
# Elodin Editor

## How to run locally

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
