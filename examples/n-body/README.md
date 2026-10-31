# Solar System Truth Demo

This example simulates a solar system with full all-pairs gravity and overlays
truth trajectories loaded from configured CSV files.

What it demonstrates:
- Full graph gravity over all loaded bodies with `el.Integrator.Verlet`. Gravity depends only on
  position, so Verlet's energy error stays bounded over long runs instead of drifting, and it
  evaluates gravity half as often as `el.Integrator.Rk4`.
- Truth-vs-sim visual overlay (ghost bodies + trails). The truth CSVs are heliocentric, so a
  system places each truth position relative to the simulated Sun, which drifts slightly. It runs
  inside the simulation, so truth is recorded in every run mode, including `bench`.
- Higher simulation cadence than telemetry cadence.
- Headless and editor-compatible execution with unified backend selection.

## Time Semantics

- Simulation starts at the truth epoch (`2020-01-01T00:00:00Z`) via `start_timestamp`.
- `simulation_rate` and `telemetry_rate` are in **Hz** (per second).
- Timeline is configured with `follow_latest=#true` so live view stays current.
- Schematic `coordinate frame=ENU` (Z-up). Viewport `pos` / `look_at` are in that
  frame, not Bevy. `(-6,6,6)` is a corner view above the XY orbit plane.

## Run

From repo root:

```bash
nix develop
```

Editor (live visualization):

```bash
elodin editor examples/n-body/main.py
```

Headless:

```bash
elodin run examples/n-body/main.py
```

Useful env vars:
- `ELODIN_BACKEND`: `cranelift` (default), `jax-cpu`, `jax-gpu`
- `ELODIN_NBODY_MAX_TICKS`: override max ticks for quick runs
- `DBNAME`: explicit DB directory path

By default the example loads `examples/n-body/planets_truth.csv`; `body_metadata.py` also lists
`examples/n-body/moons_truth.csv`, commented out.

## Backend Benchmark

Run the matrix:

```bash
ELODIN_NBODY_BENCH_TICKS=20000 ELODIN_NBODY_BENCH_REPEATS=1 uv run examples/n-body/benchmark_backends.py
```

The benchmark script prints `db_path[backend]` for each run and stores logs at:
- `/tmp/n-body-bench-cranelift.log`
- `/tmp/n-body-bench-jax-cpu.log`

## Accuracy Report

Export one benchmark DB and compute the aggregate coefficient:

```bash
elodin-db export <db_path_from_benchmark_output> --format csv --output /tmp/n-body-export.csv
ELODIN_NBODY_EXPORT_DIR=/tmp/n-body-export.csv uv run examples/n-body/accuracy_report.py
```

Recorded output for the full 1826-day timeline (`main.py bench --ticks 43824`):
- Bodies compared: 9
- Global RMS error (AU): `2.838570921e-05`
- Accuracy coefficient: `0.999994334`

Mercury, the fastest orbit, dominates at `8.2e-05` AU RMS. With `el.Integrator.Rk4` it drops to
`1.4e-05` AU, the model's own floor (no moons, asteroids, or relativity), where the other planets
already sit; that is the accuracy Verlet trades for half the gravity evaluations.
