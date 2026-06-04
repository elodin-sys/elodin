# Native Monte Carlo SITL Example

This example demonstrates `elodin monte-carlo` with a minimal software-in-the-loop
setup:

- `main.py` runs the Elodin plant and bridges `post_step` state to a controller.
- `controller.py` is the external flight software process, launched with
  `world.recipe(...)` and connected over UDP.
- `sim.py` declares tunable parameters with `el.monte_carlo.params_spec(...)`
  and includes a scalable baked constant so shared-constant memory behavior is
  visible in campaign metrics.
- `campaign.toml`, `spec.toml`, `plan.csv`, and `hooks/` demonstrate 100 runs
  across 10 worker slots, sampling, per-run scoring, and post-campaign reporting.

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

The campaign output includes per-run databases, `results.csv`, `perf.csv`,
`resources.csv`, `summary.json`, and `memory.json`. `summary.json` always
includes total campaign wall time, aggregate/average per-run wall time, worker
parallel efficiency, disk usage, and CPU/RAM resource rollups.
