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

Install [Determinate Systems Nix](https://determinate.systems/nix-installer/) which will give you exactly the same development environment that we are using. Once installed, then you can set up a development environment with all the requisite packages by running  

```sh
nix develop .#rust
cargo ...
```

or, for Python, 

```sh
nix develop .#python 
python ...
```

Note that the former will also give you the Python binary and packages. 

### Build & Install Elodin & Elodin DB
```sh
just install
```

#### Elodin App & SDK Development
(See [apps/elodin/README.md](apps/elodin/README.md))

If you are using Nix and don't want to install the editor binary then run 

```sh
nix run .#elodin-cli editor libs/nox-py/examples/three-body.py
```

or

```sh
nix develop .#python 

cd libs/nox-py
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv
uv sync

cargo run --manifest-path=../../apps/elodin/Cargo.toml editor examples/three-body.py
```
