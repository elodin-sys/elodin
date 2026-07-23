---
name: elodin-db
description: Work with Elodin-DB, the time-series telemetry database. Use when running elodin-db, writing client integrations (C, C++, Rust, Python), configuring replication/follow mode, the DB Asset Server / assets ingest, querying data via the Lua REPL, or connecting the Elodin Editor to a database.
---

# Elodin-DB

Elodin-DB is a high-performance time-series database for telemetry data. It stores components, messages, and metadata using the Impeller2 protocol, and serves as the central data bus between simulations, flight software, and the Elodin Editor.

## Quick Start

```bash
# Install (from source)
just install

# Run the database
elodin-db run [::]:2240 $HOME/.local/share/elodin/db --config libs/db/examples/db-config.lua --log-level warn

# Connect the Elodin Editor
elodin editor 127.0.0.1:2240

# Launch the Lua REPL
elodin-db lua
```

## Running the Database

```bash
elodin-db run <bind_addr> <data_dir> [--config <lua_file>] [--assets <dir>] [--log-level <level>]
```

| Parameter | Example | Purpose |
|-----------|---------|---------|
| `bind_addr` | `[::]:2240` | Listen address (IPv4/IPv6 + port) |
| `data_dir` | `$HOME/.local/share/elodin/db` | Storage directory |
| `--config` | `libs/db/examples/db-config.lua` | Lua configuration script |
| `--assets` | `/var/lib/elodin/assets` | Source `assets/` tree to ingest on **fresh** DB create (overrides `ELODIN_ASSETS`) |
| `--log-level` | `warn` | Log verbosity: error, warn, info, debug, trace |

Impeller TCP listens on port `N` (e.g. `2240`). The **DB Asset Server** always binds `N+1` (e.g. `2241`) and serves `{data_dir}/assets/` over HTTP. Do not put a follower Impeller listener on `N+1`.

## Assets and the DB Asset Server

Visual files for schematics live under `{db}/assets/{relative_key}` and are served at `http://host:(tcp_port+1)/{relative_key}` while the DB runs. Record once, copy the DB directory, replay anywhere — no separate `assets/` tree required on the consumer.

### Source asset root (ingest)

On first create of an empty DB, `elodin-db run` (and Python `world.run(..., db_path=…)`) copies a source tree into `{db}/assets/` **once**, then writes a `.elodin-ingested` marker. Later opens skip ingest so recorded/editor assets are never wiped.

Source resolution (CLI `--assets` wins when set; otherwise):

1. `$ELODIN_ASSETS`
2. `<sim_entry>/assets` (simulations only)
3. `<cwd>/assets`
4. Nearest ancestor `assets/` (simulations only)

```bash
# Seed a fresh DB from an explicit tree (Aleph / HITL pattern)
elodin-db run [::]:2240 ./my-db --assets /var/lib/elodin/assets
```

### Conventional keys inside `{db}/assets/`

Same layout as the simulation asset root (see elodin-simulation skill):

| Key prefix | Contents |
|------------|----------|
| `*.glb`, `meshes/…`, `models/…` | Meshes referenced by `glb path=` (rewritten to `db:…` in stored KDL) |
| `schematics/*.kdl` | Active schematic (default `schematics/main.kdl`) and window sub-schematics |
| `skyboxes/manifest.ron` + `*.cubemap.ktx2` | Named skyboxes |
| `terrains/…` | Terrain atlases for `world_mesh` |
| `color_schemes/…` | Optional theme JSON (local editor; not required for replay of built-in names) |

`schematic.active` metadata points at the active KDL asset key (usually `schematics/main.kdl`). Consumers fetch that KDL over HTTP — there is no inline KDL mirror in DB metadata.

### Paths and the `db:` scheme

At record/ingest, local paths like `models/jet.glb` become `db:models/jet.glb` in stored schematics. Already-`db:` / `http(s):` / `icon builtin=…` paths are left alone. Keys must not contain `..`.

### Follow mode and assets

Telemetry replicates over Impeller TCP. Assets do **not** — the follower `GET`s `http://source:(N+1)/__index__` and mirrors missing/changed keys into its own `{db}/assets/`, then serves them on `(follower_port+1)`.

```bash
elodin-db run 127.0.0.1:2240 ./source-db
elodin-db run 127.0.0.1:2242 ./follower-db --follows 127.0.0.1:2240   # assets on 2243
elodin editor 127.0.0.1:2242
```

Point `--follows` at the source **Impeller** port (`N`), not the asset port.

### Verify

```bash
curl -sf -o /dev/null -w "%{http_code}\n" http://127.0.0.1:2241/schematics/main.kdl
ls -lh "$DB_PATH/assets/"
```

Empty `assets/` after a sim usually means a temp DB (`world.run` without `db_path` / `ELODIN_DB_PATH`). Mesh 404s mean the source tree was never ingested — re-run against a fresh DB with the correct `--assets` / `ELODIN_ASSETS`.

Full reference: [docs/public/content/reference/db-asset-server.md](../../../docs/public/content/reference/db-asset-server.md)

## Lua REPL

Interactive database shell for debugging and exploration:

```bash
elodin-db lua
```

```lua
db> client = connect("127.0.0.1:2240")
db> client:dump_metadata()
db> :help
```

## Client Integration

### C Client

```bash
cc examples/client.c -lm -o /tmp/client && /tmp/client
```

See `libs/db/examples/client.c` for streaming fake sensor data.

### C++ Client

```bash
c++ -std=c++23 examples/client.cpp -o /tmp/client-cpp && /tmp/client-cpp
```

The C++ library is C++20 compatible. See `libs/db/examples/client.cpp` for subscription example.

### Rust Client

See `libs/db/examples/rust_client/` for a complete Rust client using Impeller2.

### C++ Header Generation

Generate the single-header C++20 library with message definitions:

```bash
cargo run --bin elodin-db gen-cpp > libs/db/examples/db.hpp
```

### Python (via Simulation)

Simulations automatically create an embedded database, or connect to an external one:

```python
# Embedded (temporary)
w.run(system)

# Explicit path
w.run(system, db_path="./my_data")

# Connect to external DB
w.run(system, db_addr="127.0.0.1:2240")
```

## Follow Mode (Replication)

Replicate data from one database to another over TCP:

```bash
# Source database
elodin-db run [::]:2240 $HOME/.local/share/elodin/source-db

# Follower database (replicates from source)
elodin-db run [::]:2241 $HOME/.local/share/elodin/follower-db --follows 127.0.0.1:2240
```

The follower:
1. Synchronizes all existing metadata and schemas
2. Backfills historical component data and message logs
3. Streams real-time updates as they arrive
4. Mirrors schematic assets from the source DB Asset Server on port `N+1` (see Assets above)

### Packet Size Tuning

Default batches outgoing data into ~1500-byte TCP writes (standard Ethernet MTU):

```bash
elodin-db run [::]:2241 ./follower-db --follows 127.0.0.1:2240 --follow-packet-size 9000
```

### Dual-Source Example

Run a simulation on source, follow on target, and add local streams:

```bash
# Source machine
elodin editor examples/video-stream/main.py

# Target — follow source
elodin-db run [::]:2241 ./follower-db --follows SOURCE_IP:2240

# Target — connect editor
elodin editor 127.0.0.1:2241

# Target — add local video stream
examples/video-stream/stream-video.sh
```

## Merging Databases

Combine two databases (e.g. SITL and real telemetry) with optional time alignment and component prefixes.

```bash
# Basic merge with prefixes
elodin-db merge -o merged --prefix1 sitl --prefix2 real ./sitl-db ./real-db

# Align using timestamps from the Elodin Editor's playback timeline
elodin-db merge -o merged --prefix1 sitl --prefix2 real \
  --align1 15000000 --align2 14000000 --from-playback-start ./sitl-db ./real-db
```

Use `--from-playback-start` when alignment timestamps come from the Editor's playback timeline (relative to recording start). Without it, `--align1`/`--align2` are absolute timestamps.

## Exporting a Database

Offline export to analysis formats or a Foxglove-ready MCAP recording:

```bash
# Parquet / arrow-ipc / csv (one file per component)
elodin-db export --format parquet --output ./out ./my-db

# Foxglove-compatible MCAP + generated Foxglove layout JSON
elodin-db export --format mcap --output ./out ./my-db

# Pre-1970 epochs (e.g. Apollo 1969) auto-rebase to t=0 (also if --epoch-offset-us
# would leave samples pre-epoch — MCAP log_time is unsigned)
# Large GLBs (moon.glb) stay attached; model primitive omitted above --max-embed-mb (default 32)
# Follow-entity mesh always embeds. Dynamic arrows go to /scene_dynamic.
elodin-db export --format mcap --max-embed-mb 32 --output ./out ./apollo-db
```

The MCAP export maps components to JSON channels (`/drone/world_pos.q0` message
paths), emits `/tf` from `*.world_pos` poses (with world→NED/ENU anchors from a
schematic `coordinate` node), publishes `foxglove.SceneUpdate` **one topic per
entity** (`/scene/<id>`: GLBs, literal-pose objects with composed
translate/rotate, pixel-width `line_3d` trails, static arrows,
`world_mesh "globe"` → `earth.glb`) plus ≤30 Hz dynamic arrows on
`/scene_dynamic/<name>` (Foxglove backfills latest-per-topic on panel remount,
so shared scene topics silently drop entities), encodes sensor-camera RGBA to
H.264 `CompressedVideo` when `video-export` is enabled, attaches schematic
KDLs/assets, and generates `{db}.foxglove-layout.json` (per-viewport followTf
from `look_at`, camera offsets incl. `translate_world(...)`, far ≥ 4× distance).
Upload with [`scripts/foxglove-upload.sh`](../../../scripts/foxglove-upload.sh)
or manually via `POST /v1/data/upload` + `PUT` then `POST /v1/layouts`. See
[elodin-cli.md](../../../docs/public/content/reference/elodin-cli.md) `Foxglove MCAP Export`
and [foxglove-mcap-export-design.md](../../../ai-context/foxglove-mcap-export-design.md).

## Trimming a Database

Remove data from the beginning or end of a recording. Values are in microseconds. Without `--output`, modifies in place.

```bash
# Remove the first 3 minutes from a recording
elodin-db trim --from-start 180000000 ./my-db

# Remove the last 2 minutes from a recording
elodin-db trim --from-end 120000000 --output ./trimmed ./my-db

# Trim 1 minute from the start and 2 minutes from the end
elodin-db trim --from-start 60000000 --from-end 120000000 --output ./window ./my-db

# Preview without modifying
elodin-db trim --from-start 180000000 --dry-run ./my-db
```

## Editor Connection

The Elodin Editor connects to any running database:

```bash
elodin editor 127.0.0.1:2240
```

From a simulation, the editor connects automatically when launched via `elodin editor sim.py`.

## Architecture

Elodin-DB uses the Impeller2 protocol internally:
- **Components**: Time-series data indexed by entity + component name + timestamp
- **Messages**: Ordered log entries (commands, events)
- **Metadata**: Schema information, entity names, component types

Storage is append-only with configurable retention. The database supports concurrent readers and writers with lock-free data structures.

## Key References

- Full documentation: [libs/db/README.md](../../../libs/db/README.md)
- DB Asset Server: [docs/public/content/reference/db-asset-server.md](../../../docs/public/content/reference/db-asset-server.md)
- C client example: [libs/db/examples/client.c](../../../libs/db/examples/client.c)
- C++ client example: [libs/db/examples/client.cpp](../../../libs/db/examples/client.cpp)
- Rust client example: [libs/db/examples/rust_client/](../../../libs/db/examples/rust_client/)
- Lua config example: [libs/db/examples/db-config.lua](../../../libs/db/examples/db-config.lua)
- DB architecture docs: [docs/public/content/home/db/](../../../docs/public/content/home/db/)
