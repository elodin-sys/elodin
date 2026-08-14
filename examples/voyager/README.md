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

This example is a work in progress. It is organized as an educational
progression rather than a high-fidelity reconstruction of the Voyager
missions. Chapter 1 intentionally keeps the dynamics simple; Chapter 2
adds one subtle correction to the same model.

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

Run Chapter 1, the original Newtonian-gravity "hello world":

```bash
python examples/voyager/main.py run
```

Run Chapter 2, which corrects the dynamics for the accelerating
Sun-centered origin:

```bash
python examples/voyager/chapter_2.py run
```

## Chapter 2: the Sun-centered origin accelerates

Chapter 1 looks reasonable: put the Sun at the origin, then add the Sun's and
planets' direct gravity. But what happens to the Sun?

SPICE supplies the Voyager and planetary states relative to the Sun in
`ECLIPJ2000`. Its axes are nonrotating, but its origin is attached to the Sun,
and the planets accelerate that origin.

Let the heliocentric probe position be

```text
r = R_probe - R_sun
```

Differentiating twice gives

```text
r'' = R_probe'' - R_sun''
```

For a planet at heliocentric position `r_i`, Chapter 2 therefore uses

```text
mu_i * ((r_i - r) / |r_i - r|^3 - r_i / |r_i|^3)
```

The first term is the planet's direct acceleration of Voyager. The second is
the same planet's acceleration of the Sun, which must be subtracted to obtain
relative acceleration in a Sun-centered frame.

The experiment changes only this equation. It uses the same kernels, constants,
RK4 integrator, and 3600-second timestep for a 400-day native Elodin run:

| Probe | Chapter 1 position / velocity error | Chapter 2 position / velocity error |
| --- | ---: | ---: |
| Voyager 1 | 122,783.010 km / 9.08717 m/s | 32,101.379 km / 1.94987 m/s |
| Voyager 2 | 121,720.889 km / 8.05398 m/s | 34,124.210 km / 1.75436 m/s |

That is about 72--74% less position disagreement and 78% less velocity
disagreement. Tens of thousands of kilometers still remain. Later chapters can
address gravitational parameters, encounter-specific force models, maneuvers,
and other fidelity limits without obscuring the lesson in Chapters 1 and 2.

Run either command above and compare the existing position- and velocity-error
graphs for the two probes to reproduce the experiment interactively.
