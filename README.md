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

## Getting Started

### Prerequisites
- **Recommended**: [Determinate Systems Nix](https://determinate.systems/nix-installer/) for a consistent development environment
- **Alternative**: Manual setup on macOS (see [Local Setup](#local-setup-macos-only) below)

## Development Setup (Recommended: Nix)

The Elodin repository uses Nix to provide a consistent, reproducible development environment across all platforms. This is the same environment our team uses daily.

### 1. Install Nix
```sh
# Install Determinate Systems Nix (recommended)
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

### 2. Clone the Repository
```sh
git clone https://github.com/elodin-sys/elodin.git
cd elodin
```

### 3. Enter the Development Shell
```sh
nix develop --command bash
```

This unified shell automatically provides:
- ✅ **Enhanced Terminal** (Works with your existing zsh/bash configuration)
- ✅ **Modern CLI Tools** (eza, bat, delta, fzf, ripgrep, zoxide)
- ✅ **Rust toolchain** (cargo, clippy, nextest) - pinned to 1.90.0
- ✅ **Python environment** (uv, maturin, ruff, pytest)
- ✅ **C/C++ compilers** (clang, gcc, cmake)
- ✅ **Cloud tools** (kubectl, gcloud, azure-cli)
- ✅ **Documentation tools** (zola, typos)
- ✅ **Version control** (git-lfs, git-filter-repo)
- ✅ **All system dependencies** (gstreamer, ffmpeg, openssl, etc.)

> [!TIP]
> The Nix shell runs Oh My Zsh + Powerlevel 10k, and will run configuration setup on first run if not installed

### 4. Build and Install Elodin Editor and Elodin DB into your path
```sh
# In the Nix shell
just install
```

### 5. Develop the Elodin simulation server & example

#### Python SDK Development
```sh
# Still in the Nix shell
cd libs/nox-py
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv
python3 examples/rocket.py
```

Open the Elodin editor ("elodin" in your terminal) and connect to the local server

---

## Alternative Local Setup (macOS Only)

> [!WARNING]
> This setup is more complex and may lead to inconsistent environments across developers. We strongly recommend using Nix instead.

If you cannot use Nix, you can manually install dependencies on macOS:

### Prerequisites
```sh
# Install required tools via Homebrew
brew install gstreamer python gfortran openblas uv git-lfs rust

# Initialize git-lfs
git lfs install
```

### Build and Run
```sh
git clone https://github.com/elodin-sys/elodin.git
cd elodin
just install
```

### Python Development (Local Setup)
```sh
cd libs/nox-py
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv
uv sync

cargo run --manifest-path=../../apps/elodin/Cargo.toml editor examples/three-body.py
```

> [!NOTE]
> Local setup instructions were validated on M1 architecture, macOS 15.1.1 on 2025-08-26.

## Additional Resources

- [Elodin App Documentation](apps/elodin/README.md)
- [Python SDK Documentation](libs/nox-py/README.md)
- [Internal Nix Documentation](docs/internal/nix.md)
