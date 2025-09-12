# Elodin Editor

### Install
Install the editor from the [releases](https://github.com/elodin-sys/elodin/releases) page.

### How to run locally
To run the editor locally and test the included examples, follow these steps:

```bash
# Move into the nox-py library
cd libs/nox-py

# Create a new virtual environment using uv
uv venv

# Activate the environment (Bash)
source .venv/bin/activate

# Build and install nox-py in development mode
uvx maturin develop --uv

# Run the three-body example from the editor (execute inside libs/nox-py)
cargo run --manifest-path=../../apps/elodin/Cargo.toml editor examples/three-body.py
```

### Run the example with live code watching
Run `three-body.py` example while watching for editor code changes (requires [cargo-watch](https://crates.io/crates/cargo-watch)):

```bash
# Execute this command from inside the libs/nox-py directory
cargo watch --watch ../../libs/elodin-editor \
    -x 'run --manifest-path=../../apps/elodin/Cargo.toml editor examples/three-body.py'
```
