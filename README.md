<h1 align="center">
  <a href="https://www.elodin.systems/">
    <img alt="elodin-banner" src="https://assets.elodin.systems/assets/elodin-banner.png">
  </a>
</h1>

This monorepo contains the source code for all Elodin simulation and flight software:

- Flight software
  - [`aleph-os`](./aleph): Aleph NixOS modules for composing a Linux flight software stack that runs on the Orin.
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
- [Determinate Systems Nix](https://determinate.systems/nix-installer/) for a consistent development environment
- [Just](https://just.systems/man/en/): ( `brew install just` / `apt install just` )
- [git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage): ( `brew install git-lfs` / `apt install git-lfs` ), and make sure to activate it globally with `git lfs install` from the terminal.
- **STRONGLY** prefer an Arm-based MacOS, nix works on x86 & Ubuntu but at much slower build speeds and worse DX.

---

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

### 3. Enter the Nix Development Shell (this can take awhile to build the first time)
```sh
nix develop
```
> [!TIP]
> The Nix shell supports Oh My Zsh + Powerlevel 10k; for first time configuration run: `p10k configure`
>

### 4. Build and Install Elodin Editor and Elodin DB into your path
```sh
just install

elodin --version

python3 examples/rocket/main.py run
```

Open the Elodin editor in a new nix develop shell and connect to the local server

---

## Python SDK Development Setup (Nix-based continued)

### 5. Enter *another* Nix Development Shell, run the server & example simulation

#### Python SDK Development
```sh
# In a new terminal
nix develop
# build the SDK python wheel
cd libs/nox-py && \
uv venv --python 3.12 && \
source .venv/bin/activate && \
uvx maturin develop --uv
# use the newly built wheel
python3 ../../examples/rocket/main.py run
```

Open the Elodin editor and connect to the local server

> [!NOTE]
> Local setup instructions were validated on Arm M2 MacOS & Intel x86 Ubuntu 24.04 on 2025-10-12.

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
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml

cargo run --manifest-path=apps/elodin/Cargo.toml editor examples/three-body/main.py
```

> [!NOTE]
> Local setup instructions were validated on M1 architecture, macOS 15.6.1 on 2025-11-03.

## Additional Resources

- [Elodin App Documentation](apps/elodin/README.md)
- [Python SDK Documentation](libs/nox-py/README.md)
- [Internal Nix Documentation](docs/internal/nix.md)
