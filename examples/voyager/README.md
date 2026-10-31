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

## Current model limits

The simulation is still gravity-only. That gets us pretty far, but it
leaves out a few things that matter once the propagation gets longer.

Solar radiation pressure is one of them. It is just the small force from
sunlight hitting the spacecraft, but over time it can add up. I tested it
separately during the Voyager validation work and it did reduce some of
the remaining error, but I am leaving it out of the main model for now.

We also are not replaying the full history of Voyager thrust events and
attitude-control activity. Because of that, a difference from the SPICE
trajectory does not automatically mean the gravity model or integrator
is wrong.

For validation, shorter maneuver-free windows are more useful than trying
to score the entire mission at once. They let us change one part of the
model, start from the same SPICE state, and see what actually improved
without mixing a bunch of effects together.

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

```bash
python examples/voyager/main.py run
```


## Chapter 1

`chapter1.py` runs Voyager 1 from Feb 22 to Feb 28, 1979 using
just the Sun and Jupiter's gravity. It starts from a SPICE state and
compares the result with the SPICE trajectory.

From `examples/voyager`:

```bash
bash download_spice_data.sh
python chapter1.py run
```

Voyager is integrated with RK4 in one-hour steps. Each step's four
stages evaluate gravity at different times, so Jupiter's position is
interpolated linearly to each stage's time (`el.StageTime`) between its
SPICE positions at the start and end of the step. That keeps the
integration error to a few meters over the six days; holding Jupiter at
the midpoint of each step instead costs about 650 m.

## Chapter 2

`chapter2.py` uses the same dates and SPICE data as Chapter 1.
It also accounts for Jupiter pulling on the Sun.

From `examples/voyager`:

```bash
python chapter2.py run
```

The remaining error against SPICE, about 1 km, comes from the model
(only the Sun and Jupiter), not the integration.
