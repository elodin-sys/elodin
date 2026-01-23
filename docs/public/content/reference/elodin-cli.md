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
