# Simulation Regression CI

This directory contains the simulation regression gate used by Buildkite.

## What CI does

For each example (`ball`, `drone`, `rocket`, `three-body`, `cube-sat`), CI runs:

1. benchmark run (`bench --ticks 100`, profiling enabled by default)
2. DB export (`elodin-db export --format csv --flatten`)
3. telemetry comparison (`compare_baseline_csv.py`)
4. performance A/B against the base commit (`compare_profile_metrics.py`, see
   [Performance A/B](#performance-ab))

The comparison is tolerance-based for numeric values and ignores columns named
`time` to avoid wall-clock timestamp noise.

## Files

- `regress.sh` - end-to-end runner used by Buildkite
- `compare_baseline_csv.py` - tolerance-based CSV comparator
- `compare_profile_metrics.py` - compares median profile metrics of candidate vs base runs
- `extract_profile_metrics.py` - run-log parser for profile metrics
- `baseline/tolerances.json` - default and per-example tolerances
- `baseline/<example>/...` or `baseline/<example>-csv/...` - known-good CSV exports

## Local usage

From repository root:

```bash
nix develop .#run --command bash -lc "bash ./scripts/ci/regress.sh ball examples/ball/main.py"
```

To check all currently baselined regression examples:

```bash
nix develop .#run --command bash -lc "bash ./scripts/ci/regress.sh --all"
```

To refresh one example baseline in place:

```bash
nix develop .#run --command bash -lc "bash ./scripts/ci/regress.sh --update ball examples/ball/main.py"
```

To refresh all currently baselined regression examples:

```bash
nix develop .#run --command bash -lc "bash ./scripts/ci/regress.sh --all --update"
```

`--all` only runs examples that already have a baseline directory under
`scripts/ci/baseline`. It does not scan every `examples/*/main.py`.

## Baseline layout

The runner looks for per-example baseline directories under `scripts/ci/baseline`.
`--all` scans its subdirectories, normalizes compatibility names such as
`ball-csv` back to `ball`, and runs
`examples/<example>/main.py`.

Preferred naming is one directory per example:

- `scripts/ci/baseline/ball`
- `scripts/ci/baseline/drone`
- `scripts/ci/baseline/rocket`
- `scripts/ci/baseline/three-body`
- `scripts/ci/baseline/cube-sat`

It also accepts compatibility names like `<example>-csv` and
`<example>-csv-100`. The checked-in baselines currently use those compatibility
names.

## Refreshing baselines

Refresh one example baseline:

```bash
nix develop .#run --command bash -lc "bash ./scripts/ci/regress.sh --update ball examples/ball/main.py"
```

Refresh all example baselines:

```bash
nix develop .#run --command bash -lc "bash ./scripts/ci/regress.sh --all --update"
```

`--update` rewrites the baseline for that example after exporting fresh flattened
CSV output. If no per-example baseline directory exists yet, it creates one under
`scripts/ci/baseline/<example>`. Performance has no checked-in baseline.

## Performance A/B

From the wall time of each complete batch of ticks, the bench prints:

- `warm-up`: the first batch (lazy JIT compilation, first allocations), excluded
  from every other statistic.
- `tick time`: mean batch wall time over the remaining batches, and their median
  (over the last 1024 batches).
- `real_time_factor`: simulated time per batch over the mean batch wall time
  (steady-state throughput).
- `real_time_factor_median`: the same over the median batch wall time. It ignores
  preemption spikes and is the number CI compares.

Absolute numbers swing by tens of percent with load on the shared CI host, so
`regress.sh` compares against the base commit's `elodin-py` in the same job:

1. The base is `REGRESSION_BASE` if set, else `git merge-base HEAD origin/main`,
   or `HEAD^` when `HEAD` is already on `main`.
2. Its wheel comes from the binary cache only
   (`nix build --max-jobs 0 "git+file://$PWD?rev=<base>&allRefs=1#elodin-py"`); CI
   built and pushed it when it tested that commit. This relies on `elodin-py`'s
   source being identical in a checkout and a `?rev=` export (no LFS files or
   submodule contents, see `nix/pkgs/src.nix`).
3. Per example, candidate and base runs alternate `REGRESSION_RUNS` times
   (default 5). Base runs put the base wheel first on `PYTHONPATH`.
4. `compare_profile_metrics.py` compares the median of each side.

The A/B is skipped with a `WARN` line, while the CSV comparison still runs, when
the base commit cannot be resolved or its wheel is not in the cache (for example,
its CI build has not finished). `--update` never runs it.

## Tolerance config

`baseline/tolerances.json` supports:

- `default.abs_tol` / `default.rel_tol`
- `examples.<name>.abs_tol` / `examples.<name>.rel_tol`
- `examples.<name>.files.<file>.abs_tol` / `examples.<name>.files.<file>.rel_tol`
- `performance.default.<metric>.abs_tol` / `performance.default.<metric>.rel_tol`
- `performance.examples.<name>.<metric>.abs_tol` / `performance.examples.<name>.<metric>.rel_tol`

For telemetry CSVs, precedence is file override -> example override -> default.

For profile metrics, each side is the median over its runs:

- `build_time_ms` fails only when the candidate is slower than the base beyond
  tolerance.
- `real_time_factor_median` fails only when the candidate drops below the base
  beyond tolerance.
- A metric missing from a base run is skipped with a `WARN`; missing from a
  candidate run fails.
- `ticks` must match across all runs.

## Failure triage

When CI fails, the comparator reports:

- missing/extra CSV files
- header/row shape mismatches
- first row+column value exceeding tolerance with abs/rel diff and limits
- profile metric regressions with base median, candidate median, and allowed slack

Use that output to decide whether to:

1. fix a regression in code, or
2. intentionally regenerate baselines and/or adjust tolerances.
