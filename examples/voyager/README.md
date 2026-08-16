# Voyager

This example simulates Voyager 1 and Voyager 2 under gravity from the Sun and
major planets, while drawing NASA SPICE trajectories for comparison.

The example is an educational journey. `main.py` is the intentionally simple
Chapter 1 model; later chapters add one astrodynamics idea at a time.

## What you will see

The editor shows:

- red propagated Voyager trajectories;
- green SPICE reference trajectories;
- `position_error_km` for Voyager 1 and Voyager 2;
- `velocity_error_mps` for Voyager 1 and Voyager 2.

The error telemetry makes the difference between the simulated and reference
trajectories visible without relying only on the 3D paths.

## Prerequisites

Run these commands from the repository root unless noted otherwise.

Create the repo-local Python environment if needed:

```bash
uv venv --python=3.13 python-env
```

Activate it:

```bash
source python-env/bin/activate
```

Install `spiceypy`:

```bash
uv pip install spiceypy
```

## Download the SPICE data

From the Voyager example directory:

```bash
cd examples/voyager
./download_spice_data.sh
cd ../..
```

The script writes the required kernels to:

```text
examples/voyager/nasa_spice_data/
```

The example loads:

- the NAIF leap-seconds kernel;
- DE440 planetary ephemerides;
- Voyager 1 trajectory data;
- Voyager 2 trajectory data.

## Run Chapter 1

From the repository root:

```bash
python examples/voyager/main.py run
```

Chapter 1 is the original gravity "hello world": the Sun is placed at the
origin and the probes receive direct gravitational attraction from the Sun and
major planets.

Read the Chapter 1 notes after you have the example running:

[Chapter 1 — Simple interplanetary gravity](docs/chapter_1.md)

## Run Chapter 2

From the repository root:

```bash
python examples/voyager/chapter_2.py run
```

Chapter 2 keeps the same kernels, constants, timestep, integrator, telemetry,
and visualization, but adds the heliocentric indirect acceleration required
when the coordinate origin follows the accelerating Sun.

Read the derivation, experiment, and measured results here:

[Chapter 2 — Heliocentric relative dynamics](docs/chapter_2.md)

## Educational journey

| Chapter | Topic | Run |
| --- | --- | --- |
| [1](docs/chapter_1.md) | Simple interplanetary gravity | `python examples/voyager/main.py run` |
| [2](docs/chapter_2.md) | Heliocentric relative dynamics | `python examples/voyager/chapter_2.py run` |

The chapters intentionally build on one another rather than turning the first
example into a single high-fidelity mission reconstruction.

## Truth reference and reproducible validation

The interactive journey uses long-span merged Voyager SPKs for mission-scale
visualization. A separate headless benchmark uses NAIF's current-best
reconstructed Jupiter and Saturn encounter SPKs, initializing and scoring from
the same solution.

[Read the truth-reference contract and Chapter 1–2 revalidation](docs/truth_reference.md)

## Troubleshooting

If SPICE reports a missing kernel, rerun:

```bash
cd examples/voyager
./download_spice_data.sh
```

If Python cannot import `spiceypy`, activate `python-env` and install it with:

```bash
uv pip install spiceypy
```

If you want a fresh persistent simulation database, set `DB_PATH` to a new
location before running the example.

## Next step

Start with Chapter 1, watch the position and velocity error graphs, and then
continue to Chapter 2 to see how one coordinate-system correction changes the
long-horizon trajectory.
