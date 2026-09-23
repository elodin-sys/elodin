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

The current dynamics are still gravity-only, so there are real effects
missing from the model.

One of them is solar radiation pressure (SRP), the small push from
sunlight on the spacecraft. It is much weaker than gravity, but over
longer propagation arcs it can still build up and show up in the
remaining position and velocity error.

SRP was tested separately during the Voyager validation work and moved
the residual in the right direction, but it is not being added to the
simulation here yet. For now this is just documented as one likely
missing effect instead of mixing another force-model change into the
same validation work.

There are other limits too:

- historical trajectory-correction and attitude-control thrust are not
  modeled by the current gravity-only propagation;
- the interactive example uses SPICE trajectories as the reference, but
  matching those trajectories does not mean every real mission force or
  maneuver has been reconstructed;
- error over a long arc can come from more than one source, so a remaining
  SPICE difference should not automatically be blamed on gravity or the
  integrator alone;
- the goal right now is to improve the model one piece at a time and keep
  each change small enough to validate separately.

This also means a short clean arc is more useful for testing one physics
change than treating the full mission trajectory as a single pass/fail
check. The Voyager validation work is using that approach so changes can
be compared against the same SPICE reference without mixing several new
effects together.

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

Chapter 2 (`python examples/voyager/chapter_2.py run`) subtracts each
planet's acceleration of the Sun. Same kernels, masses, RK4, and
timestep. It does not reproduce the gravity assists.
