# Simulation Regression CI

This directory contains the simulation regression gate used by Buildkite.

## What CI does

For each example (`ball`, `drone`, `rocket`, `three-body`, `cube-sat`), CI runs:

1. benchmark run (`bench --ticks 100`, profiling enabled by default)
2. DB export (`elodin-db export --format csv --flatten`)
3. baseline comparison (`compare_baseline_csv.py`)

The comparison is tolerance-based for numeric values and ignores columns named
`time` to avoid wall-clock timestamp noise.

## Files

- `regress.sh` - end-to-end runner used by Buildkite
- `compare_baseline_csv.py` - tolerance-based CSV comparator
- `baseline/tolerances.json` - default and per-example tolerances
- `baseline/<example>/...` - known-good CSV exports

## Local usage

From repository root:

```bash
nix develop .#run --command bash -lc "bash ./scripts/ci/regress.sh ball examples/ball/main.py"
```

To run without profile collection:

```bash
nix develop .#run --command bash -lc "REGRESSION_ENABLE_PROFILE=0 bash ./scripts/ci/regress.sh ball examples/ball/main.py"
```

## Baseline layout

The runner looks for per-example baseline directories under `scripts/ci/baseline`.
Recommended naming is one directory per example:

- `scripts/ci/baseline/ball`
- `scripts/ci/baseline/drone`
- `scripts/ci/baseline/rocket`
- `scripts/ci/baseline/three-body`
- `scripts/ci/baseline/cube-sat`

It also accepts compatibility names like `<example>-csv` and `<example>-csv-100`.

## Refreshing baselines

Refresh one example baseline:

```bash
nix develop .#run --command bash -lc '
set -euo pipefail
example=ball
entrypoint=examples/ball/main.py
tmp="$(mktemp -d -t elodin-baseline-${example}.XXXXXX)"
db_path="${tmp}/db"
out_dir="scripts/ci/baseline/${example}"
rm -rf "${out_dir}"
mkdir -p "${out_dir}"
ELODIN_DB_PATH="${db_path}" uv run "${entrypoint}" bench --ticks 100 --profile
elodin-db export --format csv --flatten --output "${out_dir}" "${db_path}"
'
```

Refresh all example baselines:

```bash
nix develop .#run --command bash -lc '
set -euo pipefail
for item in \
  "ball:examples/ball/main.py" \
  "drone:examples/drone/main.py" \
  "rocket:examples/rocket/main.py" \
  "three-body:examples/three-body/main.py" \
  "cube-sat:examples/cube-sat/main.py"
do
  example="${item%%:*}"
  entrypoint="${item#*:}"
  tmp="$(mktemp -d -t elodin-baseline-${example}.XXXXXX)"
  db_path="${tmp}/db"
  out_dir="scripts/ci/baseline/${example}"
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  ELODIN_DB_PATH="${db_path}" uv run "${entrypoint}" bench --ticks 100 --profile
  elodin-db export --format csv --flatten --output "${out_dir}" "${db_path}"
done
'
```

## Tolerance config

`baseline/tolerances.json` supports:

- `default.abs_tol` / `default.rel_tol`
- `examples.<name>.abs_tol` / `examples.<name>.rel_tol`
- `examples.<name>.files.<file>.abs_tol` / `examples.<name>.files.<file>.rel_tol`

Precedence is file override -> example override -> default.

## Failure triage

When CI fails, the comparator reports:

- missing/extra CSV files
- header/row shape mismatches
- first row+column value exceeding tolerance with abs/rel diff and limits

Use that output to decide whether to:

1. fix a regression in code, or
2. intentionally regenerate baselines and/or adjust tolerances.
