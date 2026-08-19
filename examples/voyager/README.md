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

This example is a work in progress. The simulated probes still do not
reproduce the gravity assists or reach Saturn. Chapter 2 below fixes
one real error in the Sun-centered force model; it does not turn this
into a mission reconstruction.

The editor exposes that divergence numerically as two telemetry signals
for each simulated probe:

- `position_error_km`: Euclidean distance from the matching SPICE truth
  position, in kilometers.
- `velocity_error_mps`: Euclidean difference from the matching SPICE truth
  velocity, in meters per second.

The default schematic graphs both signals for Voyager 1 and Voyager 2.
These diagnostics make it possible to see when the trajectory starts
diverging instead of relying only on the red simulated and green truth
paths in the 3D viewport.


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
which `main.py` loads at startup.


## Run

Chapter 1 is the original model and the default:

```bash
python examples/voyager/main.py run
```

Chapter 2 uses the same kernels, masses, RK4 integrator, and timestep,
but subtracts each planet's acceleration of the Sun. The states are
Sun-relative (`r = R_probe - R_sun`), so the force model needs

```text
mu_i * ((r_i - r) / |r_i - r|^3 - r_i / |r_i|^3)
```

```bash
python examples/voyager/chapter_2.py run
```

That correction reduces long-span disagreement with the published
supertrajectory, but the probes still do not slingshot correctly. This
example does not reconstruct maneuvers or claim navigation-grade
accuracy.
