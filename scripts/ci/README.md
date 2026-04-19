# Simulation Regression CI

This directory contains the simulation regression gate used by Buildkite.

## What CI does

For each example (`ball`, `drone`, `rocket`, `three-body`, `cube-sat`), CI runs:

1. benchmark run (`bench --ticks 100`, profiling enabled by default)
2. DB export (`elodin-db export --format csv --flatten`)
3. telemetry comparison (`compare_baseline_csv.py`)
4. profile metric comparison (`compare_profile_metrics.py`)

The comparison is tolerance-based for numeric values and ignores columns named
`time` to avoid wall-clock timestamp noise.

## Files

- `regress.sh` - end-to-end runner used by Buildkite
- `compare_baseline_csv.py` - tolerance-based CSV comparator
- `compare_profile_metrics.py` - tolerance-based profile metric comparator
- `extract_profile_metrics.py` - run-log parser for profile metrics
- `baseline/tolerances.json` - default and per-example tolerances
- `baseline/<example>/...` or `baseline/<example>-csv/...` - known-good CSV exports plus `profile-metrics.json`

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
`--all` scans subdirectories that contain `profile-metrics.json`, normalizes
compatibility names such as `ball-csv` back to `ball`, and runs
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
CSV output and capturing `profile-metrics.json`. If no per-example baseline
directory exists yet, it creates one under `scripts/ci/baseline/<example>`.

## Tolerance config

`baseline/tolerances.json` supports:

- `default.abs_tol` / `default.rel_tol`
- `examples.<name>.abs_tol` / `examples.<name>.rel_tol`
- `examples.<name>.files.<file>.abs_tol` / `examples.<name>.files.<file>.rel_tol`
- `performance.default.<metric>.abs_tol` / `performance.default.<metric>.rel_tol`
- `performance.examples.<name>.<metric>.abs_tol` / `performance.examples.<name>.<metric>.rel_tol`

For telemetry CSVs, precedence is file override -> example override -> default.

For profile metrics:

- `build_time_ms` fails only when the candidate is slower than baseline beyond
  tolerance.
- `real_time_factor` fails only when the candidate drops below baseline beyond
  tolerance.
- `ticks` must match exactly between baseline and candidate.

## Failure triage

When CI fails, the comparator reports:

- missing/extra CSV files
- header/row shape mismatches
- first row+column value exceeding tolerance with abs/rel diff and limits
- profile metric regressions with baseline value, candidate value, and allowed slack

Use that output to decide whether to:

1. fix a regression in code, or
2. intentionally regenerate baselines and/or adjust tolerances.
