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

### `ELODIN_ASSETS_DIR`

The `ELODIN_ASSETS_DIR` environment variable can specify an "assets" folder,
used to find external files like meshes or images. If no environment variable is
set, it will look for an "assets" folder in the current working directory.

### No variable
This is probably the most typical usage.
```sh
$ elodin; # Will look for "assets" folder.
INFO ELODIN_ASSETS_DIR defaulted to "assets"
```
### Set variable for one run
```sh
$ ELODIN_ASSETS_DIR=my-assets elodin
INFO ELODIN_ASSETS_DIR set to "my-assets"
WARN ELODIN_ASSETS_DIR "/Users/shane/Projects/elodin_sim/my-assets" does not exist.
```
### Set variable for session
```sh
$ export ELODIN_ASSETS_DIR=/path/to/my-assets; # Or place this in your shell's rc file.
$ elodin
INFO ELODIN_ASSETS_DIR set to "/path/to/my-assets"
```

