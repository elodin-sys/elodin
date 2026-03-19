# Solar System Truth Demo

This example simulates a solar system with full all-pairs gravity and overlays
truth trajectories loaded from configured CSV files.

What it demonstrates:
- Full graph gravity over all loaded bodies with `el.Integrator.Rk4`.
- Truth-vs-sim visual overlay (ghost bodies + trails).
- Higher simulation cadence than telemetry cadence.
- Headless and editor-compatible execution with unified backend selection.

## Time Semantics

- Simulation starts at the truth epoch (`2020-01-01T00:00:00Z`) via `start_timestamp`.
- `simulation_rate` and `telemetry_rate` are in **Hz** (per second).
- Timeline is configured with `follow_latest=#true` so live view stays current.

## Run

From repo root:

```bash
nix develop
```

Editor (live visualization):

```bash
elodin editor examples/n-body/main.py
```

Headless (with GPU):

```bash
ELODIN_BACKEND=iree-gpu elodin run examples/n-body/main.py
```

Useful env vars:
- `ELODIN_BACKEND`: `iree-cpu`, `iree-gpu`, `jax-cpu`, `jax-gpu`
- `ELODIN_NBODY_MAX_TICKS`: override max ticks for quick runs
- `DBNAME`: explicit DB directory path

By default the example combines:
- `examples/n-body/planets_truth.csv`
- `examples/n-body/moons_truth.csv`

## Backend Benchmark (All 4 Backends)

Run the matrix:

```bash
ELODIN_NBODY_BENCH_TICKS=20000 ELODIN_NBODY_BENCH_REPEATS=1 uv run examples/n-body/benchmark_backends.py
```

Recorded median results from this branch (`ticks=20000`, `repeats=1`):

| backend | tick_ms | kernel_ms | h2d_ms | d2h_ms | real_time_factor |
|---|---:|---:|---:|---:|---:|
| `iree-cpu` | 2321.497 | 2321.487 | 0.004 | 0.007 | 37217.361 |
| `iree-gpu` | 11915.744 | 11912.800 | 0.199 | 2.745 | 7250.911 |
| `jax-cpu` | 591.461 | 590.688 | 0.056 | 0.717 | 146078.977 |
| `jax-gpu` | 1603.125 | 1595.964 | 5.423 | 1.738 | 53894.735 |

The benchmark script prints `db_path[backend]` for each run and stores logs at:
- `/tmp/n-body-bench-iree-cpu.log`
- `/tmp/n-body-bench-iree-gpu.log`
- `/tmp/n-body-bench-jax-cpu.log`
- `/tmp/n-body-bench-jax-gpu.log`

## Accuracy Report

Export one benchmark DB and compute the aggregate coefficient:

```bash
elodin-db export <db_path_from_benchmark_output> --format csv --output /tmp/n-body-export.csv
ELODIN_NBODY_EXPORT_DIR=/tmp/n-body-export.csv uv run examples/n-body/accuracy_report.py
```

Recorded output snapshot from this branch:
- Bodies compared: depends on loaded CSV body set
- Global RMS error (AU): `1.551175590e-01`
- Accuracy coefficient: `0.984776664`
