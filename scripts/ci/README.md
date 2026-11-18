# Simulation Regression CI

This directory contains the simulation regression gate used by Buildkite.

## What CI does

For each example (`ball`, `drone`, `rocket`, `three-body`, `cube-sat`), CI runs:

1. benchmark run (`bench --ticks 100`, profiling enabled by default)
2. DB export (`elodin-db export --format csv --flatten`)
3. telemetry comparison (`compare_baseline_csv.py`)
4. the same run, export, and comparison with `ELODIN_BACKEND=jax-cpu`, so both
   backends must match the baseline
5. performance A/B against the base commit (`compare_profile_metrics.py`, see
   [Performance A/B](#performance-ab))

The comparison is tolerance-based for numeric values and ignores columns named
`time` to avoid wall-clock timestamp noise. When every row of a file has a
timestamp from its export's `globals.tick.csv`, rows are compared tick by tick,
carrying unchanged values forward. Otherwise consecutive duplicate rows are
collapsed and the remaining change rows are compared in order.

## Files

- `regress.sh` - end-to-end runner used by Buildkite
- `compare_baseline_csv.py` - tolerance-based CSV comparator
- `compare_profile_metrics.py` - compares median profile metrics of candidate vs base runs
- `extract_profile_metrics.py` - run-log parser for profile metrics
- `betaflight_perf.sh` / `betaflight_perf.py` - real headless Betaflight C0 performance gate
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

## Betaflight headless performance gate

A separate Buildkite performance job runs the actual default scripted Betaflight
SITL flight: 15 simulated seconds at 8 kHz, production real-time pacing, real
pinned/patched firmware, UDP lockstep, physics, DB commits, and Python `post_step`.
It uses `python main.py run` with the SDK's built-in s10 supervision (equivalent
to the headless `elodin run` path); no editor, camera, GPU, or fake controller.
The firmware is built once and its EEPROM initialized **and verified** once,
outside measurements. Every flight starts from a copy of that verified EEPROM.

The gated metric is the **paired median candidate/base non-pacing work-time
ratio**, higher being worse. Each run computes mean work ns per telemetry cycle as
`(total_cycle.sum_ns - real_time_pacing.sum_ns) / cycles` from
`ELODIN_SIM_SUMMARY_JSON`; intentional real-time sleep is excluded, but physics,
DB, callback, firmware round-trip waits, scheduler stalls during work, and final
callback cleanup are included. Warmup is pinned to the first 100 simulation steps,
regardless of inherited environment. Inherited pacing lead/grace overrides and
pacing-trace/detailed-timing diagnostics are cleared to use production defaults.
One telemetry cycle executes one simulation step here; each complete
run must report 119900 measured cycles/steps plus 100 excluded steps = 120000
steps. Phase samples, final tick 119999, C0 PASS, >=99% successful lockstep
responses, recorded DB state, and process exit are required. Missing/malformed or
incomplete metrics are hard failures, not skipped data. Achieved RTF is reported
separately as measured simulated seconds / measured wall seconds; a paced flight
can still achieve ~1x while its work cost regresses.

The Betaflight-only initial threshold is a **10% paired median work-time increase**,
not a 10% throughput drop. It is configurable with `BETAFLIGHT_PERF_THRESHOLD=0.10`
and does not change the generic examples' tolerances. Normally five runs per side
alternate base/candidate, reversing pair order every other pair. For each adjacent
pair, divide candidate work ns/cycle by base work ns/cycle; fail when the median
ratio is **greater than** `1 + threshold` (exactly 10% is allowed). Reports retain
individual pair increases and each side's median work/RTF as diagnostics; the gate
is **not** the ratio of those side medians. Pairing reduces sensitivity to changing
host-speed regimes, but these initial measurements do not establish a CI
false-positive bound. Only a measured regression triggers **one fresh full A/B
confirmation comparison**; both attempts are retained, and a second pass is
explicitly reported as `PASS on confirmation`. Setup, timeout, process, contract,
and schema failures immediately fail without a performance retry.

Baseline revision selection and the cache-only SDK lookup match `regress.sh`
(`REGRESSION_BASE`, otherwise merge-base with `origin/main`, or `HEAD^` on main).
Unlike generic wheel-only A/B, this job also snapshots the **base revision's
example source**, including its callback and schematic. Candidate source is the
tracked working tree. Both sides use the same candidate-pinned firmware binary,
verified initial EEPROM and current assets to isolate Elodin, not firmware or
asset changes. Separate source/run directories prevent dirty schematic, EEPROM,
and default DB-path collisions. If base selection or cache lookup is unavailable,
the job validates one candidate C0 flight and explicitly skips only performance
A/B. A resolved base with incompatible source/SDK/summary fails rather than
silently skipping metrics.

The constant Buildkite `betaflight-sitl-performance` concurrency group with
`concurrency=1` covers the **entire** job, including firmware setup, both attempts
and cleanup across builds/PRs. Other CI jobs remain parallel. Per-flight timeout
is 120 seconds, the shell wrapper has a 40-minute timeout in CI, and Buildkite has
a 45-minute job timeout. Logs, raw summaries, per-run metrics, firmware/EEPROM
hashes, revisions and comparison reports are uploaded from
`profile_output/betaflight-perf-ci/run.*`; scratch DBs and processes are cleaned
up, not uploaded.

From the repository root:

```bash
nix develop .#run --command bash scripts/ci/betaflight_perf.sh
python3 scripts/ci/test_betaflight_perf.py
```

For shorter local comparisons, set `BETAFLIGHT_PERF_RUNS=1`. Other overrides:
`PYTHON`, `BETAFLIGHT_PERF_RUN_TIMEOUT` (seconds), `BETAFLIGHT_PERF_OUTPUT` (must
be empty), and `BETAFLIGHT_PERF_SKIP_BUILD=1` (only with a known-current firmware
build). Without Nix, `BETAFLIGHT_PERF_BASE_SITE=/absolute/base/site-packages`
provides an explicitly prepared base SDK; still set `REGRESSION_BASE` to its
matching source revision. It does not build/install another wheel. Do not run
this command concurrently with another Betaflight process on the same host:
firmware ports and the example's existing stale-process cleanup are global.

This draft is not a dedicated/isolated runner benchmark: unrelated shared-host
load can still add noise despite serialization and same-job A/B. It does not
measure startup/compile time, editor rendering, camera throughput, firmware
regressions, or telemetry against a large checked-in golden baseline. It adds
no pinning, load daemon or historical dashboard.

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
