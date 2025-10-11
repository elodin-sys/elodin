<h1 align="center">
  <a href="https://www.elodin.systems/">
    <img alt="elodin-banner" src="https://assets.elodin.systems/assets/elodin-banner.png">
  </a>
</h1>

This monorepo contains the source code for all Elodin simulation and flight software:

- Flight software
  - [`aleph-os`](./images/aleph): Aleph NixOS modules for composing a Linux flight software stack that runs on the Orin.
  - [`elodin-db`](./libs/db) (FSW application): A time-series database which functions as a central telemetry store and message bus.
  - [`serial-bridge`](./fsw/serial-bridge) (FSW application): Reads sensor data being streamed over the serial port and writes it to [`elodin-db`](./libs/db).
  - [`mekf`](./fsw/mekf) (FSW application): A Multiplicative Extended Kalman Filter implementation that fuses sensor data to estimate vehicle attitude.
  - [`sensor-fw`](./fsw/sensor-fw): Aleph expansion board firmware that streams sensor data (IMU, mag, baro) to the Orin over USB/UART.
- Simulation software
  - [`nox-py`](./libs/nox-py): Python version of `nox-ecs`, that works with JAX.
- [Editor](./apps/elodin): 3D viewer and graphing tool for visualizing both simulation and flight data.

<h2 align="center">
  <a href="https://www.elodin.systems/">
    <img alt="elodin-stack" src="assets/elodin-stack.png">
  </a>
</h2>

## Dependencies

Rust 1.90.0
Preference for Arm-based Macs

## Building

> [!NOTE]
> These instructions were validated on M1 architecture, macOS 15.1.1 on 2025-08-26.

### Before Cloning

Before cloning the source, ensure `git-lfs` is installed. 

``` sh
brew install gstreamer python gfortran openblas uv git-lfs;
git lfs install; # This is idempotent; you can run it again.
```

### Source

``` sh
git clone https://github.com/elodin-sys/elodin.git
```


### Build & Run
```sh
cd elodin
just install
```

#### Elodin App & SDK Development
(See [apps/elodin/README.md](apps/elodin/README.md))
``` sh

cd libs/nox-py
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv
uv sync

cargo run --manifest-path=../../apps/elodin/Cargo.toml editor examples/three-body.py
```

Alternatively, install [Determinate Systems Nix](https://determinate.systems/nix-installer/) which will give you exactly the same development environment that we are using. Once you have Nix installed, switch to the top of the Elodin repo. Then you can do 

```
nix develop
```

This unified shell includes all tools needed for:
- Rust development (cargo, clippy, nextest)
- Python development (uv, maturin, ruff)
- C/C++ compilation
- Cloud operations (kubectl, gcloud, azure)
- Documentation (zola, typos)
- Version control (git-lfs for large files)

### Build & Install Elodin & Elodin DB
```sh
just install
```

#### Elodin App & SDK Development
(See [apps/elodin/README.md](apps/elodin/README.md))
``` sh
nix develop

cd libs/nox-py
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv
uv sync

cargo run --manifest-path=../../apps/elodin/Cargo.toml editor examples/three-body.py
```
