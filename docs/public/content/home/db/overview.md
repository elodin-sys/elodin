+++
title = "Overview"
description = "Overview of Elodin DB"
draft = false
weight = 104
sort_by = "weight"

[extra]
lead = "Overview of Elodin DB"
toc = true
top = false
order = 7
icon = ""
+++

Elodin DB is a time-series database for physical systems. It serves as a central telemetry store and message bus -- on a flight computer, in a ground station, or alongside a simulation. Flight software, sensors, and simulations publish and subscribe to telemetry data and messages through it.

Elodin DB is based on the same ECS system as Elodin Sim (and the rest of the Elodin ecosystem). Elodin DB sorts data into "entities" and "components". Entities are best thought as objects that emit telemetry. For instance an individual sensor would be an entity. A component is a piece of telemetry data that is associated with an entity. For example an accelerator's acceleration reading would be a component. Entities can have multiple components, and each component can be associated with multiple entities.

Elodin DB chiefly communicates over `impeller` -- Elodin's lightweight message protocol designed for flight software. In addition to `impeller`, Elodin DB can be queries via SQL and results are returned in the Arrow IPC format.

## Quick Start

Install Elodin DB from the [releases](https://github.com/elodin-sys/elodin/releases) page.


To start a new instance of Elodin DB, use the following command:
```sh
elodin-db run
```

### Lua REPL

The easiest way to interact with Elodin DB is through its Lua REPL. To start the REPL, run the following command:

```sh
elodin-db lua
```

You can connect to the database by running:

```lua
client = connect("localhost:2240")
```


### Editor

Connect the Elodin Editor to a running database to visualize telemetry and video in real time:

```sh
elodin editor 127.0.0.1:2240
```

### SQL

You can query Elodin DB via SQL. The easiest way to access this interface is through the REPL where you can run the following command to connect to the database using SQL

```lua
:sql [::]:2240
```

This will connect to your local instance of Elodin DB, and drop you into a SQL REPL.

To list the available tables, run:

```sql
show tables;
```

We use Datafusion to power the SQL interface. [Their docs](https://datafusion.apache.org/user-guide/sql/index.html) are the best place for details on the dialect.

## Key Capabilities

**Replay** -- Start a server with `--replay` to play back a recorded database as if data were arriving live. Connected editors scrub through the timeline with full fidelity.

**Follow mode** -- Start a second instance with `--follows <source>` to replicate all data from another running database over a single TCP connection. The follower accepts its own local connections and writers simultaneously.

```sh
elodin-db run [::]:2241 ./ground-station --follows 192.168.1.10:2240
```

**Video** -- H.264 video streams (e.g. from GStreamer via `elodinsink`) are stored as timestamped message logs and displayed in the Elodin Editor. Recorded video can be exported to MP4 with `elodin-db export-videos`.

**Export** -- Export component data to Parquet, Arrow IPC, or CSV without a running server: `elodin-db export ./db -o ./out`. Glob filtering and vector flattening are supported.

**Database tools** -- Offline commands for common post-processing tasks: `merge` (combine two databases with optional time alignment), `drop` (delete components by name or glob), `prune` (remove empties), `truncate` (clear data, keep schemas), `time-align` (shift timestamps), and `fix-timestamps` (normalize clock sources).
