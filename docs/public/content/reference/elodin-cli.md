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

# Command-Line Help for `elodin`

This document contains the help content for the `elodin` command-line program.

**Command Overview:**

* [`elodin`↴](#elodin)
* [`elodin editor`↴](#elodin-editor)
* [`elodin run`↴](#elodin-run)

## `elodin`

**Usage:** `elodin [OPTIONS] [COMMAND]`

###### **Subcommands**

* `editor` — Launch the Elodin editor (default)
* `run` — Run an Elodin simulation in headless mode

###### **Options**

* `-u`, `--url <URL>`

  Default value: `https://app.elodin.systems`



## `elodin editor`

Launch the Elodin editor (default)

**Usage:** `elodin editor [addr/path]`

###### **Arguments**

* `<addr/path>` — Address of a running database server (e.g., `localhost:2240`) or path to a simulation script

  Default value: ``



## `elodin run`

Run an Elodin simulation in headless mode

**Usage:** `elodin run [addr/path]`

###### **Arguments**

* `<addr/path>` — Path to a simulation script to run

  Default value: ``


---

# Command-Line Help for `elodin-db`

This document contains the help content for the `elodin-db` command-line program, which provides the Elodin telemetry database server and utilities for managing database files.

**Command Overview:**

* [`elodin-db run`↴](#elodin-db-run)
* [`elodin-db lua`↴](#elodin-db-lua)
* [`elodin-db gen-cpp`↴](#elodin-db-gen-cpp)
* [`elodin-db fix-timestamps`↴](#elodin-db-fix-timestamps)
* [`elodin-db merge`↴](#elodin-db-merge)
* [`elodin-db prune`↴](#elodin-db-prune)
* [`elodin-db truncate`↴](#elodin-db-truncate)
* [`elodin-db time-align`↴](#elodin-db-time-align)
* [`elodin-db drop`↴](#elodin-db-drop)

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

* `--align1 <SECONDS>` — Alignment timestamp (in seconds) for an event in DB1

* `--align2 <SECONDS>` — Alignment timestamp (in seconds) for the same event in DB2. When both alignment options are provided, the database with the earlier anchor is shifted forward to align timestamps.

###### **Component Naming**

When prefixes are applied, component names are transformed using an underscore separator:
- `rocket.velocity` with prefix `sim` becomes `sim_rocket.velocity`
- `rocket.velocity` with prefix `truth` becomes `truth_rocket.velocity`

###### **Time Alignment**

The `--align1` and `--align2` options allow you to align two databases based on a common event (e.g., launch, ignition). Both options must be provided together. The database with the earlier anchor timestamp is shifted forward so that both anchors occur at the same time in the merged output.

###### **Example**

```bash
# Basic merge with prefixes
elodin-db merge ./sim-db ./flight-db -o ./merged-db --prefix1 sim --prefix2 truth

# Merge with time alignment (align "launch" event at 15s in sim with 45s in flight)
elodin-db merge ./sim-db ./flight-db -o ./merged-db \
  --prefix1 sim --prefix2 truth \
  --align1 15.0 --align2 45.0

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
