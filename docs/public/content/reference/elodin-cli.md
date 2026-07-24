+++
title = "Elodin CLI"
description = "Elodin CLI and Elodin DB CLI Reference"
draft = false
weight = 104
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 4
+++

# Command-Line Help

This document contains the help content for the Elodin command-line programs.

**Command Overview:**

* [`elodin`↴](#elodin)
* [`elodin editor`↴](#elodin-editor)
* [`elodin run`↴](#elodin-run)
* [`elodin-db`↴](#elodin-db)
* [`elodin-db run`↴](#elodin-db-run)
* [`elodin-db lua`↴](#elodin-db-lua)
* [`elodin-db gen-cpp`↴](#elodin-db-gen-cpp)
* [`elodin-db fix-timestamps`↴](#elodin-db-fix-timestamps)
* [`elodin-db merge`↴](#elodin-db-merge)
* [`elodin-db prune`↴](#elodin-db-prune)
* [`elodin-db compact`↴](#elodin-db-compact)
* [`elodin-db truncate`↴](#elodin-db-truncate)
* [`elodin-db time-align`↴](#elodin-db-time-align)
* [`elodin-db drop`↴](#elodin-db-drop)
* [`elodin-db info`↴](#elodin-db-info)
* [`elodin-db query`↴](#elodin-db-query)
* [`elodin-db export`↴](#elodin-db-export)
* [`elodin-db export-videos`↴](#elodin-db-export-videos)

---

## `elodin`

**Usage:** `elodin [OPTIONS] [COMMAND]`

###### **Subcommands**

* `editor` — Launch the Elodin editor (default)
* `run` — Run an Elodin simulation in headless mode
* `monte-carlo` — Run Monte Carlo campaigns

###### **Options**

* `-u`, `--url <URL>`

  Default value: `https://app.elodin.systems`



## `elodin editor`

Launch the Elodin editor (default)

**Usage:** `elodin editor [--kdl KDL-PATH] [addr/path]`

###### **Arguments**

* `<addr/path>` — Optional connection target or simulation to run. Can be:
  - A socket address (e.g., `127.0.0.1:2240`) to connect to a running Elodin DB
  - A Python file (e.g., `main.py`) to run a simulation
  - A TOML file (e.g., `s10.toml`) to run from a plan
  - A directory containing `main.py` or `s10.toml`
* `--kdl <PATH>` — Optional parameter that will load a specific schematic KDL
  after connecting to a database.

###### **Environment**

* `BLOCKADE_API_KEY` — Optional. Enables Skybox AI generation from the command palette
  (`Skybox...` → `Generate Skybox...`). Existing cached skyboxes and KDL `skybox name="..."`
  activation do not require this key.
* `ELODIN_ASSETS` — Optional. Overrides the asset root. Skybox assets are read from and
  generated into `$ELODIN_ASSETS/skyboxes`; otherwise `./assets/skyboxes` is used.

## `elodin run`

Run an Elodin simulation in headless mode (not available on Windows)

**Usage:** `elodin run [addr/path]`

###### **Arguments**

* `<addr/path>` — Simulation to run. Can be:
  - A Python file (e.g., `main.py`)
  - A TOML file (e.g., `s10.toml`)
  - A directory containing `main.py` or `s10.toml`

For campaigns, use `elodin monte-carlo run` (below).

## `elodin monte-carlo`

Run a simulation campaign with a bounded worker pool. Each worker owns a
deterministic resource slot (DB port and user-defined SITL ports), and the
runner recycles those slots across arbitrarily many runs. The campaign pins a
shared `ELODIN_CACHE_DIR` so large Cranelift constants are mapped once across
workers. Concurrency comes from `--workers N` (or `workers = N` in
`campaign.toml`): exactly N runs execute at once regardless of how many
processes each run spawns. When neither is set, the runner sizes itself from
logical cores; `S10_MAX_INFLIGHT` remains a low-level escape hatch for
budgeting by process count instead of run count. Only a numeric
`S10_MAX_INFLIGHT` overrides `campaign.toml`'s `workers`; `off` (or an
unparsable value) disables admission limiting without overriding the
configured worker count.

Before anything launches, the runner takes an exclusive lock on the out dir
(`campaign.lock`) so two campaigns cannot interleave in the same output,
validates the entire static port plan for every worker (u16 overflow,
cross-name collisions), warns when planned ports fall inside the kernel
ephemeral range, raises its own file-descriptor limit, and reaps prior
campaign-scoped cgroups plus campaign-marked processes still bound to a
campaign port. Foreign port owners block startup with pid/name details instead
of being killed. Each run preflight-probes its static ports and reports
squatters by pid/name (`port 20034 already bound by pid 320389 (weaverd)`). On timeout the
runner tears the run down via its cgroup when one is available and via
per-recipe process groups otherwise; on Linux hosts without a delegated cgroup
(e.g. a plain ssh session) the campaign transparently re-executes itself under
`systemd-run --user --scope` — for both `run` and `resume` (opt out with
`--no-self-scope`).

Every worker slot also reserves `db_port + 1` for elodin-db's always-on asset
server (the headless sensor-camera renderer fetches scene assets from it):
the assets port is validated, preflight-probed, and exported as
`ELODIN_MC_PORT_DB_ASSETS`, and a `db_port = "auto"` allocation always yields
a consecutive db/assets port pair.

**Usage:** `elodin monte-carlo <COMMAND>`

###### **Subcommands**

* `quickstart` — Scaffold a runnable campaign (spec + campaign + hooks) from a simulation
* `template` — Generate a starter plan or sampling spec from declared simulation params
* `sample` — Materialize a sampling spec into a plan CSV
* `run` — Execute a campaign
* `resume` — Re-run missing or failed runs from a previous campaign
* `report` — Rebuild campaign reports

### `elodin monte-carlo quickstart`

The fastest way to get a new simulation into a Monte Carlo campaign. Reads the
sim's declared params (`python SIM.py params`) and writes a ready-to-run
skeleton you then edit:

```bash
elodin monte-carlo quickstart examples/drone/main.py campaigns/drone
```

**Prerequisite:** the simulation must declare its tunable parameters with
`el.monte_carlo.params_spec(...)` before running quickstart — that declaration
is what populates `spec.toml`. A sim with no declared params produces an empty
`[monte_carlo.variables]` (quickstart prints a warning). Declare them in the
sim, for example:

```python
import elodin as el

def params_spec():
    return el.monte_carlo.params_spec(
        thrust=el.monte_carlo.Param(default=1.0, min=0.8, max=1.2),
        mass=el.monte_carlo.Param(default=12.0, min=11.0, max=13.0),
    )
```

This generates:

```text
campaigns/drone/
  spec.toml        # one variable per param: uniform[min,max] when bounds are
                   # declared, else fixed at the default
  campaign.toml    # worker pool + post_run/post_campaign hooks
  hooks/score.py   # post_run: records result.json metrics, marks pass
  hooks/gate.py    # post_campaign: raises when any run failed (CI gate)
```

Each `el.monte_carlo.Param(..., min=..., max=...)` becomes a `uniform`
variable; params without bounds are emitted as `fixed`. Edit the ranges and the
`hooks/score.py` pass criterion, then run the printed `elodin monte-carlo run`
command. The campaign keeps the default exit code (0 even with partial
failures); `hooks/gate.py` is what turns a failure count into a non-zero exit
for a gated pipeline.

### `elodin monte-carlo run`

```bash
elodin monte-carlo run examples/monte-carlo/main.py \
  --campaign examples/monte-carlo/campaign.toml \
  --spec examples/monte-carlo/spec.toml \
  --out dbs/monte-carlo-demo
```

Key options:

- `--workers <N>`: run exactly N runs at once. Wins over `S10_MAX_INFLIGHT`
  and `campaign.toml`'s `workers`. Default: sized from logical cores.
- `--plan <PLAN.csv>`: materialized one-row-per-run plan. If a sibling
  `spec.toml` is newer than the plan, the runner warns that the plan is stale.
- `--spec <SPEC.toml>`: sampling spec; sampled into a plan before execution.
- `--campaign <CAMPAIGN.toml>`: workers, resource slots, hooks, retries,
  timeouts, retention, quality gates, scratch dir, `[[build]]` steps, and a
  campaign-wide `[env]` table.
- `--scratch-dir <DIR|auto>`: run per-run IO (including the embedded DB) on a
  fast scratch filesystem; each run's surviving artifacts move to `--out`
  (sparse-aware) as it finishes. `auto` picks `/dev/shm` when present. Use
  this when the artifact volume cannot sustain `workers x` DB write IOPS
  (network/EBS-class disks). When `auto` finds no `/dev/shm`, the campaign
  logs that per-run IO is staying on the artifact volume. If a run's final
  move fails (e.g. the artifact volume fills up), the run is marked failed
  with the destination error and its scratch copy is preserved — the campaign
  never deletes a scratch tree that still holds run artifacts, and logs the
  path to recover them from. The scratch location is deterministic per out
  dir, so `resume` finds preserved passed runs and finishes the move instead
  of re-running them (a fresh `run` clears the campaign's scratch tree first).
- `--cache-dir <DIR>`: override the compile cache. The default lives in
  `~/.cache/elodin/monte-carlo/const-cache` (content-addressed, shared across
  campaigns) so `--clean` and fresh out dirs never cause a compile storm.
- `--strict-ports`: error (instead of warn) when planned ports fall inside the
  kernel ephemeral range (`/proc/sys/net/ipv4/ip_local_port_range`).
- `--no-self-scope`: do not re-exec under `systemd-run --user --scope` when no
  delegated cgroup is available.
- `--runtime-threads <N>`: override the orchestrator I/O thread pool. When unset
  (or `0`) it is auto-sized from the worker budget, capped at logical cores.
- `--memory-probe`: enable expensive shared-constant PSS sampling and
  `memory.json`/`processes.csv` output. Leave this off for scaling benchmarks.
- `--keep-existing`: do not reap prior campaign cgroups or campaign-marked
  processes bound to campaign ports at startup.
- `--fail-on-errors`: exit non-zero when any run failed or missed scoring.
  Off by default so exploratory campaigns can finish with partial failures.
  Also configurable as `fail_on_run_errors = true` in `campaign.toml`. For CI
  gates, prefer a `post_campaign` hook that raises on `summary.failed` (see
  `examples/apollo-lander/hooks/ci_gate.py`) instead of relying on this flag.
- `--post-run <HOOK.py>` / `--post-campaign <HOOK.py>`: plain-Python lifecycle hooks.
- `--clean`: prune `runs/` directories that are not part of the active plan.
- Campaigns always display a live progress TUI with aggregate counts and active
  worker progress while they run.

`campaign.toml` additions beyond the flags above:

- `[resources.ports]` values may be a numeric base (shifted by
  `worker_id * port_stride`, validated up front for every worker) or `"auto"`
  (allocated dynamically per run, collision-free by construction). Sims read
  them via `el.monte_carlo.port("name")` / `ELODIN_MC_PORT_<NAME>` either way.
  `db_port` also accepts `"auto"` (allocated together with its `db_port + 1`
  assets port). Placing a named port on `db_port + 1` is rejected — that port
  always belongs to the asset server.
- `[env]`: extra environment variables for every run's processes.
- `[[build]]`: any number of one-time build steps run before workers start.
- `[retention]`: `keep_run_db = "always" | "never" | "on-fail"`, plus
  `prune_on_pass` / `prune_on_fail` glob lists (relative to the run dir)
  removed after scoring, and `compact_run_db` (default `true`) which truncates
  kept DBs' preallocated files to their real size.
- `[quality]`: `max_behind_deadline_frac` / `max_real_time_factor` mark runs
  whose real-time pacing degraded as `degraded` (they are excluded from
  passes; `fail_on_degraded = true` also counts them toward
  `fail_on_run_errors`). Use this to keep oversubscribed campaigns from
  silently ingesting load-skewed samples.

The runner also injects machine-sympathy defaults into every run when the user
has not set them: `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`MKL_NUM_THREADS` and XLA CPU thread flags sized to `max(1, cores / workers)`,
so N concurrent sims do not each size their thread pools to every core.

Simulations that ingest parameters from a file (rather than via
`el.monte_carlo.params(...)`) can configure `[params_delivery]` in
`campaign.toml`: the runner writes each run's sampled params to a JSON/TOML file
and sets the env vars the simulation expects (with `{seed}` / `{db_path}` /
`{run_id}` / `{run_dir}` placeholders).

Outputs include per-run databases under `runs/`, `results.csv`, `perf.csv`,
`resources.csv`, `campaign_summary.txt`, and `summary.json`. With
`--memory-probe`, the runner also writes `memory.json` and `processes.csv`.
Each run's child process output is captured in `runs/<run_id>/logs/`, and the
per-run simulation timing snapshot is written to `runs/<run_id>/sim_summary.json`
for the final campaign rollup. Every failed, invalid, or degraded run carries a
one-line machine-readable `failure_reason` in `results.csv` / `metrics.json`
(timeouts, which leaf recipe failed and how, readiness-gate timeouts, port
conflicts with the owning pid), echoed on the live `[failed]` reporter line.
Setup-only failures (a run whose port preflight failed, so no process ever
spawned) skip the `post_run` hook entirely — there is no run database to score.
Real-time-paced runs also record `behind_deadline_frac`, `real_time_factor`,
and `drift_resets`, with worst-run callouts in the campaign summary.

## Python Simulation Subcommands

When you run a simulation Python file directly (for example `python examples/drone/main.py ...`), the embedded simulation CLI supports additional subcommands:

- `run` (default): run the simulation normally.
- `bench`: run a fixed-tick benchmark and print runtime metrics.
- `params`: print the simulation's declared Monte Carlo parameter schema as JSON.

### `bench` options

- `--ticks <N>`: number of ticks to run (default: `1000`)
- `--profile`: enable full profiling output (including HLO/graph analysis output)
- `--detail`: include per-phase timing breakdown such as upload, kernel, and download times

### Examples

```bash
# CPU benchmark
python examples/drone/main.py bench --ticks 1000

# GPU benchmark with detailed timing
ELODIN_BACKEND=jax-gpu python examples/drone/main.py bench --ticks 1000 --detail

# Full profiling output
python examples/drone/main.py bench --ticks 1000 --profile
```

---

## `elodin-db`

**Usage:** `elodin-db <COMMAND>`

###### **Subcommands**

* `run` — Run the Elodin database server
* `lua` — Run a Lua script or launch a REPL
* `gen-cpp` — Generate C++ header files
* `fix-timestamps` — Fix monotonic timestamps in a database
* `merge` — Merge two databases into one with optional prefixes
* `prune` — Remove empty components from a database
* `compact` — Truncate preallocated (sparse) database files to their real size
* `truncate` — Clear all data from a database, preserving schemas
* `time-align` — Align component timestamps to a target timestamp
* `drop` — Drop (delete) components from a database
* `info` — Display information about a database
* `query` — Run an EQL or SQL query and print results (table, CSV, parquet, or arrow-ipc)
* `export` — Export database contents to parquet, arrow-ipc, or csv files
* `export-videos` — Export video message logs to MP4 files


## `elodin-db run`

Run the Elodin database server

**Usage:** `elodin-db run [OPTIONS] [ADDR] [PATH]`

###### **Arguments**

* `<ADDR>` — Address to bind the server to

  Default value: `[::]:2240`

* `<PATH>` — Path to the data directory. If not specified, uses the system default data directory.

###### **Options**

* `--log-level <LOG_LEVEL>` — Log level (error, warn, info, debug, trace)

  Default value: `info`

* `--start-timestamp <TIMESTAMP>` — Start timestamp in microseconds

* `--config <PATH>` — Path to a Lua configuration file to run on startup

* `--http-addr <ADDR>` — Address to bind the HTTP server to (enables HTTP API)

* `--replay` — Replay recorded data as live telemetry. The database advances `last_updated` with playback so connected editors see data "arriving" over time. Requires an existing database with recorded data.

* `--follows <ADDR>` — Follow another elodin-db instance, replicating all components, messages, and metadata over a single TCP connection. The local instance still accepts its own connections and data writers.

* `--follow-packet-size <BYTES>` — Target packet size for follow streaming. Small updates are buffered to this size before sending, reducing network overhead when the source has many components.

  Default value: `1500`


## `elodin-db lua`

Run a Lua script or launch an interactive REPL for querying and interacting with the database.

**Usage:** `elodin-db lua [OPTIONS] [CONFIG] [-- <LUA_ARGS>...]`

###### **Arguments**

* `<CONFIG>` — Path to a Lua script to execute. If not provided, launches an interactive REPL.

* `<LUA_ARGS>` — Arguments to pass to the Lua script (available via the `arg` table)

###### **Options**

* `--db <PATH>` — Path to an Elodin database directory

###### **REPL Commands**

When running in interactive mode, the following commands are available:

* `:sql [addr]` — Connect to a database and enter SQL mode (default: `localhost:2240`)
* `:exit` — Exit SQL mode or quit the REPL
* `:help` or `:h` — Show help information

###### **Lua API**

* `connect(addr)` — Connect to a database server, returns a Client
* `Client:sql(query)` — Execute a SQL query and print results
* `Client:get_time_series(component_id, start, stop)` — Get time series data
* `Client:stream(stream)` — Stream data from the database
* `Client:get_msgs(msg_id, start, stop)` — Get messages by ID
* `Client:save_archive(path, format)` — Save database to arrow-ipc or parquet files


## `elodin-db gen-cpp`

Generate C++ header files for the Elodin database protocol. Outputs to stdout.

**Usage:** `elodin-db gen-cpp`

This command generates C++ header files containing type definitions and serialization code for interacting with the Elodin database from C++ applications.


## `elodin-db fix-timestamps`

Fix monotonic timestamps in a database. This is useful when a database contains timestamps from different clock sources (wall-clock vs monotonic) that need to be normalized.

**Usage:** `elodin-db fix-timestamps [OPTIONS] <PATH>`

###### **Arguments**

* `<PATH>` — Path to the database directory

###### **Options**

* `--dry-run` — Show what would be changed without modifying the database

* `-y`, `--yes` — Skip the confirmation prompt

* `--reference <REFERENCE>` — Clock to use as reference when computing offsets

  Default value: `wall-clock`

  Possible values: `wall-clock`, `monotonic`

###### **Example**

```bash
# Preview changes without modifying
elodin-db fix-timestamps --dry-run ./my-database

# Apply fixes using wall-clock as reference
elodin-db fix-timestamps -y ./my-database

# Apply fixes using monotonic clock as reference
elodin-db fix-timestamps -y --reference monotonic ./my-database
```


## `elodin-db merge`

Merge two databases into one with optional prefixes. This enables viewing simulation and real-world telemetry data simultaneously in the Elodin Editor.

**Usage:** `elodin-db merge [OPTIONS] <DB1> <DB2> --output <OUTPUT>`

###### **Arguments**

* `<DB1>` — Path to the first source database

* `<DB2>` — Path to the second source database

###### **Options**

* `-o`, `--output <PATH>` — Path for the merged output database (required)

* `--prefix1 <PREFIX>` — Prefix to apply to first database component names (e.g., `sim`)

* `--prefix2 <PREFIX>` — Prefix to apply to second database component names (e.g., `truth`)

* `--dry-run` — Show what would be merged without creating output

* `-y`, `--yes` — Skip the confirmation prompt

* `--align1 <MICROSECONDS>` — Alignment timestamp (in microseconds) for an event in DB1

* `--align2 <MICROSECONDS>` — Alignment timestamp (in microseconds) for the same event in DB2. DB2 is shifted to align its anchor with DB1's anchor.

###### **Component Naming**

When prefixes are applied, component names are transformed using an underscore separator:
- `rocket.velocity` with prefix `sim` becomes `sim_rocket.velocity`
- `rocket.velocity` with prefix `truth` becomes `truth_rocket.velocity`

###### **Time Alignment**

The `--align1` and `--align2` options allow you to align two databases based on a common event (e.g., launch, ignition, or simply the start of recording). Both options must be provided together. Timestamps are specified in microseconds for precise alignment.

When alignment is specified:
- **DB1 is never shifted** - it serves as the reference
- **DB2 is shifted** to align its anchor (`--align2`) with DB1's anchor (`--align1`)
- The shift can be **forward** (positive offset) or **backward** (negative offset)

This is particularly useful for aligning:
- A simulation database (monotonic timestamps starting at 0) with real-world telemetry (wall-clock timestamps)
- Two recordings of the same event captured with different clock sources

###### **Example**

```bash
# Basic merge with prefixes
elodin-db merge ./sim-db ./flight-db -o ./merged-db --prefix1 sim --prefix2 truth

# Merge with time alignment (align "launch" event at 15s in sim with 45s in flight)
# Timestamps are in microseconds: 15s = 15000000, 45s = 45000000
# DB2 (flight) is shifted backward by 30s to align
elodin-db merge ./sim-db ./flight-db -o ./merged-db \
  --prefix1 sim --prefix2 truth \
  --align1 15000000 --align2 45000000

# Align wall-clock timestamps to monotonic (start DB2 at 0)
# DB2 starts at 4884937s (4884937000000us), align with DB1's start at 0
# DB2 is shifted backward by ~4.8M seconds
elodin-db merge ./sitl-db ./real-db -o ./merged-db \
  --prefix1 sitl --prefix2 real \
  --align1 0 --align2 4884937000000

# Preview merge without creating output
elodin-db merge ./sim-db ./flight-db -o ./merged-db --dry-run
```


## `elodin-db prune`

Remove empty components from a database. Empty components are those that have been registered but contain no data entries.

**Usage:** `elodin-db prune [OPTIONS] <PATH>`

###### **Arguments**

* `<PATH>` — Path to the database directory

###### **Options**

* `--dry-run` — Show what would be pruned without modifying the database

* `-y`, `--yes` — Skip the confirmation prompt

###### **Example**

```bash
# Preview what would be pruned
elodin-db prune --dry-run ./my-database

# Prune empty components
elodin-db prune -y ./my-database
```


## `elodin-db compact`

Truncate a database's preallocated (sparse) storage files to their committed length. Elodin DB preallocates each component's `data`/`index` files as 8 GB sparse files, so a recorded database's *apparent* size can be hundreds of gigabytes while its real size is under one — and anything that walks it naively (rsync, tar, S3 upload, CI artifact collection) processes the apparent size. After compaction, apparent size matches real size.

Compacted databases stay fully readable (open, export, query, replay, editor playback). Further *writes* need the headroom that compaction removed, so only compact databases that are done recording, and never one that a live server has open. Monte Carlo campaigns compact retained run databases automatically (`[retention] compact_run_db`, on by default).

**Usage:** `elodin-db compact [OPTIONS] <PATH>`

###### **Arguments**

* `<PATH>` — Path to the database directory

###### **Options**

* `--dry-run` — Show how much apparent size would be reclaimed without modifying

###### **Example**

```bash
# Preview reclaimable space
elodin-db compact --dry-run ./my-database

# Truncate preallocated files to their real size
elodin-db compact ./my-database
```


## `elodin-db truncate`

Clear all data from a database while preserving component schemas and metadata. This effectively resets the database to an empty state, ready for fresh data collection.

**Usage:** `elodin-db truncate [OPTIONS] <PATH>`

###### **Arguments**

* `<PATH>` — Path to the database directory

###### **Options**

* `--dry-run` — Show what would be truncated without modifying the database

* `-y`, `--yes` — Skip the confirmation prompt

###### **What is Preserved**

When truncating a database, the following are preserved:
- Component schemas (data type definitions)
- Component metadata (names, IDs)
- Message log metadata

###### **What is Removed**

- All time-series data entries
- All message log entries

###### **Example**

```bash
# Preview what would be truncated
elodin-db truncate --dry-run ./my-database

# Truncate all data (requires confirmation)
elodin-db truncate ./my-database

# Truncate without confirmation prompt
elodin-db truncate -y ./my-database
```


## `elodin-db time-align`

Align component timestamps to a target timestamp. This is useful when a database contains components that were recorded at the same real-world moment but have different timestamp offsets.

**Usage:** `elodin-db time-align [OPTIONS] --timestamp <TIMESTAMP> <PATH>`

###### **Arguments**

* `<PATH>` — Path to the database directory

###### **Options**

* `--timestamp <SECONDS>` — Target timestamp (in seconds) to align the first sample to (required)

* `--all` — Align all components in the database

* `--component <NAME>` — Align only a specific component by name

* `--dry-run` — Show what would be changed without modifying the database

* `-y`, `--yes` — Skip the confirmation prompt

###### **Component Selection**

You must specify either `--all` or `--component`:
- `--all` aligns every component in the database, shifting each so its first timestamp matches the target
- `--component <NAME>` aligns only the named component

###### **How It Works**

For each selected component, the command:
1. Finds the first (minimum) timestamp in the component
2. Calculates the offset needed to shift that timestamp to the target
3. Applies the offset to all timestamps in the component

Each component is aligned independently, so if two components have different starting timestamps, they will both end up with their first timestamp at the target, but the relative timing within each component is preserved.

###### **Example**

```bash
# Preview alignment of all components to t=0
elodin-db time-align --timestamp 0.0 --all --dry-run ./my-database

# Align all components to start at t=0
elodin-db time-align --timestamp 0.0 --all -y ./my-database

# Align a specific component to t=0
elodin-db time-align --timestamp 0.0 --component "rocket.velocity" -y ./my-database

# Align all components to start at t=10.5 seconds
elodin-db time-align --timestamp 10.5 --all -y ./my-database
```


## `elodin-db drop`

Drop (delete) components from a database. Supports fuzzy name matching, glob patterns, and bulk removal with confirmation before deletion.

**Usage:** `elodin-db drop [OPTIONS] <PATH>`

###### **Arguments**

* `<PATH>` — Path to the database directory

###### **Options**

* `--component <NAME>` — Component name to match using fuzzy matching. All matching components will be dropped.

* `--pattern <PATTERN>` — Glob pattern to match component names. Supports `*` (any characters) and `?` (single character).

* `--all` — Drop all components in the database

* `--dry-run` — Show what would be dropped without modifying the database

* `-y`, `--yes` — Skip the confirmation prompt

###### **Matching Modes**

You must specify exactly one of `--component`, `--pattern`, or `--all`:

| Option | Behavior |
|--------|----------|
| `--component` | Fuzzy match against component names (e.g., "rocket.vel" matches "rocket.velocity") |
| `--pattern` | Glob pattern match (e.g., "rocket.*" matches all components starting with "rocket.") |
| `--all` | Drop all components in the database |

###### **Fuzzy Matching**

When using `--component`, the command uses fuzzy matching to find components:
- Case-insensitive matching (unless pattern contains uppercase)
- Matches partial strings (e.g., "vel" matches "velocity")
- Results are ranked by match score, best matches first

###### **Glob Pattern Matching**

When using `--pattern`, the following wildcards are supported:
- `*` — matches any sequence of characters
- `?` — matches exactly one character

Examples:
- `rocket.*` — matches "rocket.velocity", "rocket.position", etc.
- `*.velocity` — matches "rocket.velocity", "drone.velocity", etc.
- `comp?` — matches "comp1", "comp2", but not "comp10"

###### **Safety**

This command permanently deletes data and cannot be undone. The command:
- Shows all matching components and their entry counts before deletion
- Requires explicit confirmation (unless `-y` is passed)
- Supports `--dry-run` to preview what would be deleted

###### **Example**

```bash
# Preview what would be dropped using fuzzy match
elodin-db drop --component "rocket.vel" --dry-run ./my-database

# Drop components matching fuzzy pattern (with confirmation)
elodin-db drop --component "rocket" ./my-database

# Drop components matching glob pattern
elodin-db drop --pattern "rocket.*" -y ./my-database

# Drop all velocity components
elodin-db drop --pattern "*.velocity" -y ./my-database

# Drop all components (dangerous!)
elodin-db drop --all -y ./my-database
```


## `elodin-db info`

Display information about a database, including recording state, time step configuration, and metadata.

**Usage:** `elodin-db info [PATH]`

###### **Arguments**

* `<PATH>` — Path to the database directory. If not provided, uses the standard location (`~/.local/share/elodin/db/data`).

###### **Example**

```bash
# Display info for the default database
elodin-db info

# Display info for a specific database
elodin-db info ./my-database
```


## `elodin-db query`

Run an EQL (Elodin Query Language) or raw SQL query against a database and print the result. Output can be a terminal table, CSV, or binary formats (arrow-ipc, parquet). Does not require a running database server.

**Usage:** `elodin-db query [OPTIONS] --eql <EQL> | --sql <SQL> <DBFILE>`

###### **Arguments**

* `<DBFILE>` — Path to the database directory

###### **Query source (required, mutually exclusive)**

You must provide exactly one of:

* `--eql <EQL>` — EQL query (e.g. a component name like `rocket.world_pos`, or an expression like `(rocket.world_pos[0], rocket.world_pos[1])`)
* `--sql <SQL>` — Raw SQL query

###### **Options**

* `--offset <N|DURATION>` — Skip rows before returning results. Can be:
  - An integer: skip that many rows. Negative values count from the end (e.g. `-10` = start 10 rows before the last).
  - A duration: e.g. `2.6s`, `340000ms`, `53000ns`. Negative duration means from the end (e.g. `-1s` = start 1 second before the last entry).

* `--limit <N|DURATION>` — Return at most this many rows or this duration. Same formats as `--offset` (e.g. `10`, `2.6s`, `340000ms`, `53000ns`).

* `-f`, `--format <FORMAT>` — Output format.

  Default value: `table`

  Possible values: `table`, `csv`, `arrow-ipc`, `parquet`

  For `arrow-ipc` and `parquet`, output is binary; pipe to a file (e.g. `... > out.arrow`).

* `--flatten` — Flatten vector columns into separate scalar columns (e.g. `vel` → `vel.0`, `vel.1`, …). Uses component metadata for column names when available (e.g. `q0`, `q1`, `x`, `y`, `z`).

* `--time-format <FORMAT>` — How to show the time column (when present).

  Possible values: `omit`, `datetime`, `s` (seconds), `ms` (milliseconds), `us` (microseconds; aliases: `µs`). If not set, defaults to seconds, or to the unit implied by a duration in `--offset` or `--limit` (e.g. `--limit 500ms` → time in ms).

* `-v`, `--verbose` — Print the SQL used (EQL conversion or raw SQL) to stderr.

* `-p`, `--precision <N|full>` — Decimal places for floating-point values in table/CSV. Use a number (default `6`) or `full` to show all digits. When not `full`, a note is printed to stderr suggesting `--precision full` for full data.

* `--row-index` — Add a first column `index` with the 0-based row index in the full result set (useful with `--offset`/`--limit` to see which rows are shown).

###### **Time column**

For EQL queries, a `time` column is included by default (first column) unless `--time-format omit` is used. The time column header includes the unit (e.g. `time (s)`, `time (μs)`, `time (UTC)`). Elodin stores time in microseconds since epoch.

###### **Example**

```bash
# First 10 rows of a component as a table
elodin-db query --eql "rocket.world_pos" --limit 10 ./my-database

# Last 10 rows, show EQL→SQL on stderr
elodin-db query --eql "rocket.world_pos" --offset -10 -v ./my-database

# Tuple expression, flattened, with row index and full precision
elodin-db query --eql "(rocket.world_pos[0], rocket.world_pos[1])" --limit 5 --flatten --row-index -p full ./my-database

# Raw SQL, output as CSV
elodin-db query --sql "SELECT time, rocket_world_pos FROM rocket_world_pos LIMIT 100" -f csv ./my-database

# Duration-based slice: last 2.5 seconds of data, time in seconds
elodin-db query --eql "drone.position" --offset -2.5s --limit 2.5s ./my-database

# Export slice to Parquet (binary; pipe to file)
elodin-db query --eql "rocket.world_pos" --limit 1000 -f parquet ./my-database > out.parquet
```


## `elodin-db export`

Export database contents to parquet, arrow-ipc, csv, or a Foxglove-compatible mcap file without requiring a running server. This is useful for analyzing telemetry data with external tools like pandas, DuckDB, or other data analysis frameworks, or for reviewing and sharing recordings in [Foxglove](https://foxglove.dev).

The export runs in parallel across components and is dramatically faster than the historical single-threaded path; on a 20-core machine the customer's ~3 GB CSV export went from ~32 s (default formatting) to ~2.2 s (`--csv-fast-floats`), a 14× speedup.

**Usage:** `elodin-db export [OPTIONS] --output <OUTPUT> <PATH>`

###### **Arguments**

* `<PATH>` — Path to the database directory

###### **Options**

* `-o`, `--output <PATH>` — Output directory for exported files (required)

* `--format <FORMAT>` — Export format

  Default value: `parquet`

  Possible values: `parquet`, `arrow-ipc`, `csv`, `mcap` (alias: `foxglove`)

* `--flatten` — Flatten vector columns to separate columns (e.g., `vel_ned` becomes `vel_ned.x`, `vel_ned.y`, `vel_ned.z`)

* `--pattern <PATTERN>` — Filter components by glob pattern (e.g., `NavNED.*`, `*.velocity`)

* `--join` — Group components by name prefix (everything before the last `.`) and emit one file per group. Components in a group are joined on time. See [Component Joining](#component-joining) below.

* `--csv-fast-floats` — CSV-only: format `f32` and `f64` values via `ryu` instead of Rust's `Display`. Much faster (around 2x on float-heavy components) but produces a slightly different (still round-trippable) text format (e.g. `0.0000001` becomes `1e-7`, `1.0` stays `1.0` instead of `1`). Off by default so existing pipelines that consume the CSV see no format change.

* `--mono-ns` — Replace the time column with integer nanoseconds since unix epoch. The column is renamed `time_ns` and changes type from `Timestamp(Microsecond)` to `Int64`. Mutually exclusive with `--mono-us`. Applies to all formats (CSV, Parquet, Arrow IPC).

* `--mono-us` — Replace the time column with integer microseconds since unix epoch. The column is renamed `time_us` and changes type from `Timestamp(Microsecond)` to `Int64`. Mutually exclusive with `--mono-ns`. Applies to all formats.

* `--include-private` — Include components whose metadata has `private: true`. Off by default — those components are skipped (see [Private Components](#private-components) below).

* `--all-assets` — MCAP-only: attach every file under `{db}/assets/` instead of only schematic-referenced assets.

* `--epoch-offset-us <i64>` — MCAP-only: add this offset (µs) to every sample timestamp before writing MCAP `log_time`/`publish_time`. When omitted and the earliest sample is pre-1970 (negative Unix µs), the exporter auto-rebases so earliest becomes `t=0` and records the shift in metadata key `elodin.time_offset_us`. The same auto-rebase runs if a requested offset would leave any sample pre-epoch (MCAP `log_time` is unsigned and cannot store absolute 1969-era times) — so `--epoch-offset-us 0` on Apollo-style data still preserves playback ordering rather than collapsing every sample to `t=0`.

* `--max-embed-mb <u64>` — MCAP-only: maximum GLB size (MiB) to base64-embed inside `/scene` `SceneUpdate` messages (default `32`). Larger GLBs are still attached to the MCAP but their model primitive is omitted entirely (no empty-`data` model). The viewport follow-entity's mesh is always embedded regardless of this limit.

###### **Export Formats**

| Format | Extension | Description |
|--------|-----------|-------------|
| `parquet` | `.parquet` | Columnar format with compression. Best for large datasets and analytics tools. |
| `arrow-ipc` | `.arrow` | Arrow IPC format. Fast to read/write, good for streaming data between processes. |
| `csv` | `.csv` | Plain text format. Universal compatibility but larger file sizes. |
| `mcap` | `.mcap` | Foxglove-compatible MCAP recording (single file, zstd-compressed JSON channels). See [Foxglove MCAP Export](#foxglove-mcap-export) below. |

###### **Foxglove MCAP Export**

`--format mcap` (alias `--format foxglove`) writes a single `{db_name}.mcap` plus a generated `{db_name}.foxglove-layout.json`, ready to open in [Foxglove](https://app.foxglove.dev) or upload to the Foxglove Data Platform:

- Every component becomes a JSON channel (`drone.world_pos` → `/drone/world_pos`) with fields named after the component's `element_names`; dotted names nest (`e.r` → `.e.r`).
- Pose components (`*.world_pos`, 7 elements) additionally publish `foxglove.FrameTransforms` on `/tf` (`world` → entity), driving the Foxglove 3D panel.
- Schematic `object_3d` meshes/GLBs (including literal-pose entities), `line_3d` trajectories (decimated, pixel-width scale-invariant lines), constant `vector_arrow`s, and `world_mesh "globe"` (swapped to `earth.glb` on the Earth frame) become `foxglove.SceneUpdate` entities — **one topic per entity** (`/scene/<entity-id>`), because Foxglove backfills only the latest message per topic when a 3D panel (re)mounts. Data-driven `vector_arrow`s (≤30 Hz) publish one topic each (`/scene_dynamic/<name>`). Multiple `object_3d` on the same entity get unique ids (`{frame}-model`, `{frame}-model-2`, …). Literal poses compose with GLB `translate`/`rotate`.
- Viewport `near`/`far` are honored in the 3D panel `cameraState`; `far` is clamped to ≥4× the camera distance (derived from the viewport `pos` offset, including `translate_world(x,y,z)`-style method chains). Each 3D panel follows its own `look_at`/`pos` subject and subscribes to every scene topic.
- A schematic `coordinate lat=… lon=… alt=…` node emits static world→`NED`/`ENU` anchor transforms and re-parents entities whose `object_3d` declares `frame="NED"`/`"ENU"` under them.
- Message logs export as `foxglove.CompressedVideo` (native H.264 and sensor-camera RGBA re-encoded via openh264 when the `video-export` feature is on), `foxglove.RawImage` (sensor cameras without video-export), `foxglove.Log` (LogEntry streams → Log/`RosOut` panel), or base64 JSON (other).
- Schematic KDL files and referenced GLB assets travel along as MCAP attachments; DB and component metadata as MCAP metadata records (including `elodin.time_offset_us` when timestamps are rebased).
- The generated layout mirrors the Elodin schematic: tabs/splits map to Foxglove tabs/splits, `graph` EQL expressions expand to Plot panel series, `viewport` becomes a 3D panel following the vehicle, `component_monitor` becomes Raw Messages, `log_stream` becomes a Log panel. SQL `query_plot`, icons, thrusters, bloom/hdr, and non-globe `world_mesh` regions are skipped with a console note.

```bash
# Export and open locally (drag into app.foxglove.dev or the desktop app)
elodin-db export --format mcap --output ./fg ./my-database

# One-shot upload + layout + view URL
scripts/foxglove-upload.sh \
  --mcap ./fg/my-database.mcap \
  --layout ./fg/my-database.foxglove-layout.json \
  --device elodin-my-example \
  --key elodin-my-example-v1 \
  --layout-name "Elodin My Example"
```

###### **Vector Column Handling**

By default, vector columns (e.g., 3D positions, quaternions) are exported as fixed-size lists. When exporting to CSV without `--flatten`, these appear as JSON-like strings (e.g., `[1.0, 2.0, 3.0]`).

With `--flatten`, vector columns are split into separate scalar columns with element names as suffixes:
- `position` (3D vector) → `position.x`, `position.y`, `position.z`
- `quaternion` (4D vector) → `quaternion.q0`, `quaternion.q1`, `quaternion.q2`, `quaternion.q3`

This is particularly useful for CSV export or when working with tools that don't support nested types.

###### **Glob Pattern Filtering**

The `--pattern` option supports standard glob wildcards:
- `*` — matches any sequence of characters
- `?` — matches exactly one character

Examples:
- `rocket.*` — export only components starting with "rocket."
- `*.velocity` — export only velocity components
- `NavNED.*` — export only NavNED components

###### **Component Joining**

By default each component is exported to its own file. With `--join`, components are grouped by their **name prefix** (everything before the last `.`) and joined on time into a single file per group:

| Component name | Group | Short name |
|---|---|---|
| `TARGETMESSAGE.POS_ECEF` | `TARGETMESSAGE` | `POS_ECEF` |
| `TARGETMESSAGE.VEL_ECEF` | `TARGETMESSAGE` | `VEL_ECEF` |
| `rocket.set_control` | `rocket` | `set_control` |
| `tick` | `tick` | `tick` |

The example above produces three output files: `TARGETMESSAGE.csv` (with `time, POS_ECEF, VEL_ECEF` columns), `rocket.csv`, and `tick.csv`. Components in a group with **identical** timestamp arrays are zipped onto a shared time axis; components with disjoint timestamps go through a sorted union with NULL fill for the missing rows. Composes with `--flatten`, `--csv-fast-floats`, `--mono-ns`/`--mono-us`, and `--pattern`.

###### **Time Column Format**

The default time column is a `Timestamp(Microsecond)` field named `time` rendered as ISO 8601 in CSV (e.g. `1970-01-01T00:00:00.019`). The two `--mono-*` flags swap it for an integer column to make it directly comparable to user-stored monotonic timestamps:

| Flag | Column header | Type | Example value |
|---|---|---|---|
| (none, default) | `time` | `Timestamp(Microsecond)` | `1970-01-01T00:00:00.019` |
| `--mono-us` | `time_us` | `Int64` | `19000` |
| `--mono-ns` | `time_ns` | `Int64` | `19000000` |

The conversion is exact within the database's microsecond storage precision: `time_ns = time_us * 1000`. If your data has its own monotonic field in nanoseconds (e.g. `TIME_MONOTONIC`), `--mono-ns` makes the values comparable cell-for-cell — typically matching every row to within ~1 µs (any difference is sub-microsecond rounding from the on-disk storage).

###### **Private Components**

Components whose metadata contains `"private": "true"` are skipped during export by default. This lets simulation authors mark intermediate or sensitive components (e.g. internal Kalman filter state, scratch buffers, large covariance matrices) so that downstream pipelines never see them when re-running `elodin-db export` against a recorded DB. See the Python API reference for how to set this metadata key on a component.

When the flag is honored, the export prints a one-line skip message per component:

```
  Skipping drone.estimate_covariance (private)
```

Pass `--include-private` to override the filter and export every component regardless of metadata (useful for forensic or full-fidelity exports).

###### **Example**

```bash
# Export all components to parquet (default format)
elodin-db export ./my-database -o ./export

# Export to CSV with flattened vectors
elodin-db export ./my-database -o ./export --format csv --flatten

# Export to Arrow IPC format
elodin-db export ./my-database -o ./export --format arrow-ipc

# Export only specific components using glob pattern
elodin-db export ./my-database -o ./export --pattern "rocket.*"

# Export velocity components as flattened CSV
elodin-db export ./my-database -o ./export --format csv --flatten --pattern "*.velocity"

# Export with maximum CSV throughput (ryu floats + integer-nanosecond time column)
elodin-db export ./my-database -o ./export --format csv --csv-fast-floats --mono-ns

# Group components by message family into one file per group
elodin-db export ./my-database -o ./export --format csv --join

# Same, with flattened vectors and integer-microsecond time
elodin-db export ./my-database -o ./export --format csv --join --flatten --mono-us

# Include components flagged `private: true` in the export
elodin-db export ./my-database -o ./export --format csv --include-private
```


## `elodin-db export-videos`

Export video message logs to MP4 files. This command supports both H.264 streams stored as timestamped Annex B payloads and `sensor_camera` streams stored as raw RGBA frames. H.264 streams are muxed directly into MP4; `sensor_camera` frames are encoded to H.264 during export and then muxed into standards-compliant MP4 files (e.g. for playback in QuickTime, VLC, or other players).

**Usage:** `elodin-db export-videos [OPTIONS] --output <OUTPUT> <PATH>`

###### **Arguments**

* `<PATH>` — Path to the database directory

###### **Options**

* `-o`, `--output <PATH>` — Output directory for MP4 files (required)

* `--pattern <PATTERN>` — Filter message logs by name glob (e.g. `test-*`). If not set, all video message logs in the database are exported.

* `--fps <FPS>` — Default frame rate when the H.264 stream’s SPS (Sequence Parameter Set) has no timing info, or when a `sensor_camera` stream has invalid FPS metadata. Valid `sensor_camera` metadata uses the camera’s configured `fps`.

  Default value: `30`

###### **How video gets into the database**

Video is stored in Elodin DB as **message logs**: each frame is a timestamped binary payload. H.264 video logs store Annex B NAL units; `sensor_camera` logs store raw RGBA frame buffers and rely on the database's `sensor_cameras` metadata for width, height, and FPS. Typical ingestion paths:

- **GStreamer + elodinsink**: A GStreamer pipeline (e.g. `videotestsrc` → `x264enc` → `h264parse` → `elodinsink`) sends H.264 frames over TCP to the database. The `elodinsink` plugin uses a configurable message name (e.g. `test-video`) that becomes the log name. See the Video Streaming Example in the repository (`examples/video-stream/`) for a full setup.
- **Sensor cameras**: `world.sensor_camera(...)` registers camera metadata and the headless render server writes raw RGBA frames into the database. `export-videos` detects these streams from `sensor_cameras` metadata and encodes them during export.
- **Schematic**: A `video_stream "name"` entry in the schematic ties a video tile in the Elodin Editor to that message name; the same name is used when exporting with `export-videos`.

For H.264 streams, resolution and frame rate are read from the SPS in the first keyframe and no re-encoding is done. For `sensor_camera` streams, resolution and frame rate come from the camera metadata and raw RGBA frames are encoded during export. Output files use fast-start layout for web and player compatibility.

###### **Example**

```bash
# Export all video streams to an output directory
elodin-db export-videos ./my-database -o ./videos

# Export only streams matching a glob (e.g. names starting with "test-")
elodin-db export-videos ./my-database -o ./videos --pattern "test-*"

# Use a specific default FPS when SPS has no timing info
elodin-db export-videos ./my-database -o ./videos --fps 60
```
