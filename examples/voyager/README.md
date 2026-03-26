# Voyager

This example simulates Voyager 1 and Voyager 2 under gravity from
the Sun and major planets, while also drawing SPICE-driven "truth"
trajectories for comparison. The planets and truth probes are updated
directly from NASA SPICE kernels each tick, and the simulated probes
are integrated by Elodin.

SPICE is NASA's toolkit and data format for spacecraft geometry,
time systems, and ephemerides (time-indexed descriptions of where
celestial bodies and spacecraft are, and how fast they are moving).
In this example it provides reference
positions and velocities for the planets and the Voyager spacecraft
from published files (aka SPICE kernels).

This example is a work in progress. Right now the simulated probes do
not make it to Saturn. Future work is needed to isolate the error
sources and improve the simulation.


## Setup

Create the repo-local Python venv if needed:

```bash
cd elodin
uv venv --python=3.13 python-env
```

Install `spiceypy` into that venv:

```bash
source python-env/bin/activate
uv pip install spiceypy
```

Download the required SPICE kernels:

```bash
cd examples/voyager
./download_spice_data.sh
```

This writes the kernels into `examples/voyager/nasa_spice_data/`,
which [main.py](/mnt/share/elodin/code/elodin-agent1/examples/voyager/main.py)
loads at startup.


## Run

```bash
python examples/voyager/main.py run
```
