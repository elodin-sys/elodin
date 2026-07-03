# Elodin Editor

### Install
Install the editor from the [releases](https://github.com/elodin-sys/elodin/releases) page.

### How to run locally
To run the editor locally and test the included examples, follow these steps:

```bash
# Create a new virtual environment using uv
uv venv

# Activate the environment (Bash)
source .venv/bin/activate

# Build and install nox-py in development mode
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml

# Run the three-body example from the editor (execute inside libs/nox-py)
cargo run --bin elodin editor examples/three-body/main.py
```

### Run the example with live code watching
Run `three-body` example while watching for editor code changes (requires [cargo-watch](https://crates.io/crates/cargo-watch)):

```bash
# Execute this command 
cargo watch --watch libs/elodin-editor \
    -x 'run --bin elodin editor examples/three-body/main.py'
```

### `ELODIN_ASSETS`

The `ELODIN_ASSETS` environment variable specifies an "assets" directory,
used to store external files like meshes or images. If no environment variable
is set, it looks for an "assets" directory in the current directory.

### No variable
This is probably the most typical usage.
```sh
$ elodin; # Will look for "assets" directory.
INFO Assets directory defaulted to "assets"
```
### Set variable for one run
```sh
$ ELODIN_ASSETS=my-assets elodin
INFO ELODIN_ASSETS set to "my-assets"
WARN Assets directory "/Users/shane/Projects/elodin_sim/my-assets" does not exist.
```
### Set variable for session
```sh
$ export ELODIN_ASSETS=/path/to/my-assets; # Or place this in your shell's rc file.
$ elodin
INFO ELODIN_ASSETS set to "/path/to/my-assets"
```

### `ELODIN_KDL_DIR`

The `ELODIN_KDL_DIR` environment variable specifies for schematic files, i.e.,
files with a "kdl" extension. If no environment variable is set, it looks in the
current working directory.

### No variable
This is probably the most typical usage.
```sh
$ elodin; # Looks for ".kdl"s in current working directory.
```
### Set variable for one run
```sh
$ ELODIN_KDL_DIR=my-kdls elodin
```
### Set variable for session
```sh
$ export ELODIN_KDL_DIR=/path/to/my-kdls; # Or place this in your shell's rc file.
$ elodin
```

### `BLOCKADE_API_KEY`

The `BLOCKADE_API_KEY` environment variable enables Skybox AI generation from
the editor command palette (`Skybox...` → `Generate Skybox...`). Selecting
existing cached skyboxes does not require it.

Create a Skybox AI API key from Blockade Labs at
https://skybox.blockadelabs.com/api, then pass it through your shell:

```sh
$ BLOCKADE_API_KEY=your_key_here elodin editor examples/rc-jet/main.py
```

Do not commit API keys or put them in schematic files.
