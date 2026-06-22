# Native Monte Carlo SITL Example

This example demonstrates `elodin monte-carlo` with a minimal software-in-the-loop
setup:

- `main.py` runs the Elodin plant and bridges `post_step` state to a controller.
- `controller.py` is the external flight software process, launched with
  `world.recipe(...)` and connected over UDP.
- `sim.py` declares tunable parameters with `el.monte_carlo.params_spec(...)`
  and includes a scalable baked constant so shared-constant memory behavior is
  visible in campaign metrics.
- `campaign.toml`, `spec.toml`, `plan.csv`, and `hooks/` demonstrate 100 runs,
  sampling, per-run scoring, and post-campaign reporting. Worker and
  orchestrator thread counts auto-size from available CPUs unless overridden.

Run from the repository root:

```sh
elodin monte-carlo run examples/monte-carlo/main.py \
  --campaign examples/monte-carlo/campaign.toml \
  --spec examples/monte-carlo/spec.toml \
  --out dbs/monte-carlo-demo
```

For a deterministic hand-authored plan:

```sh
elodin monte-carlo run examples/monte-carlo/main.py \
  --campaign examples/monte-carlo/campaign.toml \
  --plan examples/monte-carlo/plan.csv \
  --out dbs/monte-carlo-demo
```

The terminal shows campaign progress, success/failure counts, and a final
campaign summary. Individual simulation stdout/stderr is written to per-run log
files under `runs/<run_id>/logs/` instead of being interleaved in the terminal.
At startup, `elodin monte-carlo run` reaps prior campaign-scoped cgroups so
stale sidecars from interrupted runs cannot occupy worker ports. Pass
`--keep-existing` only if you intentionally want to manage those processes
yourself. Per-worker UDP ports are declared in `[resources.ports]` and consumed
with `el.monte_carlo.port("state")` / `ELODIN_MC_PORT_STATE`.

The campaign output includes per-run databases, `sim_summary.json`, `results.csv`,
`perf.csv`, `resources.csv`, `campaign_summary.txt`, and `summary.json`.
`summary.json` always includes total campaign wall time,
aggregate/average per-run wall time, worker parallel efficiency, disk usage,
CPU/RAM resource rollups, promoted hook metrics, invalid-run counts, and the
merged simulation phase summary.

## Scaling vs. Memory Profiles

The example has two useful modes:

- **Scaling profile:** this is the default. It uses a small table and disables
  the sparse page-fault probe, measuring campaign scheduling, process startup,
  compilation, and SITL overhead without intentionally saturating memory
  bandwidth.

  When `--workers` is omitted, the runner sizes workers from the s10 admission
  budget: `floor(S10_MAX_INFLIGHT / per-run recipe weight)`, capped by the plan
  size. `S10_MAX_INFLIGHT` defaults to logical cores; raise it to oversubscribe
  I/O-bound SITL recipes.

  ```sh
  elodin monte-carlo run examples/monte-carlo/main.py \
    --campaign examples/monte-carlo/campaign.toml \
    --plan examples/monte-carlo/plan.csv \
    --out dbs/monte-carlo-scaling
  ```

- **Memory-sharing profile:** use a large table and a wide sparse probe. This
  faults pages in the content-addressed mmap cache so `memory.json` shows PSS
  sharing across workers. `--memory-probe` enables the expensive `/proc` PSS
  sampler and per-run process-tree sampling; leave it off for scaling runs.

  ```sh
  ELODIN_MONTE_CARLO_GRID_SIZE=16777216 \
  ELODIN_MONTE_CARLO_PROBE_ROWS=65536 \
  elodin monte-carlo run examples/monte-carlo/main.py \
    --campaign examples/monte-carlo/campaign.toml \
    --plan examples/monte-carlo/plan.csv \
    --memory-probe \
    --out dbs/monte-carlo-memory
  ```

Set `ELODIN_MONTE_CARLO_CONTROLLER=0` to disable the external UDP controller and
use an in-process control law. This isolates campaign/process overhead from the
SITL two-process lockstep path.

To repeat the scaling investigation across worker counts and ablations:

```sh
python monte_carlo_scaling_sweep.py \
  --workers 1,5,10,20 \
  --controllers 0,1 \
  --out dbs/monte-carlo-scaling
```

The sweep writes `scaling.csv` with wall time, speedup, tick-loop cost, phase
attribution, CPU/load/context-switch metrics, and concurrency data.
