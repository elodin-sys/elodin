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

### Core Requirements
- **Rust 1.90.0** (managed via `rust-toolchain.toml`)
  - The project uses Rust 2024 edition (stable since Rust 1.85.0)
  - Includes targets: `wasm32-unknown-unknown`, `thumbv7em-none-eabihf`
- **Python 3.12** (standardizing across the project)
- **Preference for ARM-based Macs** (M1/M2/M3)

### macOS

Validated on M1 architecture, macOS 15.1.1 on 2025-09-28.

#### System Dependencies
Install required system packages using Homebrew:

```sh
# One-liner to install all dependencies
brew install pkg-config gstreamer ffmpeg python@3.12 gfortran openblas uv
```

### Build & Run

#### Quick Install
```sh
just install
```

#### Full Development Setup
(See [apps/elodin/README.md](apps/elodin/README.md) for detailed instructions)

```sh
# Set up Python environment
cd libs/nox-py
uv venv --python 3.12
source .venv/bin/activate

# Build and install the Python bindings
uvx maturin develop --uv
uv sync

# Run the Elodin editor with an example
cargo run --manifest-path=../../apps/elodin/Cargo.toml editor examples/three-body.py
```
