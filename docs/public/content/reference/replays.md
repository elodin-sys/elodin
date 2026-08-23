+++
title = "Replays"
description = "Replays"
draft = false
weight = 105
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 5
+++

Elodin records simulation telemetry, metadata, schematics, and visual assets in
an Elodin DB directory. The editor can serve and open that directory directly,
so replaying a recording does not require a separate `elodin-db` process.

## Record a simulation

Set `ELODIN_DB_PATH` when running a simulation to keep its database:

```bash
ELODIN_DB_PATH=dbs/apollo ELODIN_NON_INTERACTIVE=1 elodin run examples/apollo-lander/main.py
```

The recording includes a `db_state` file, component data, and an `assets/`
tree. Keep the whole directory when copying or sharing it.

## Open a recording

Pass the database directory to the editor:

```bash
elodin editor dbs/apollo
```

The editor starts an embedded database server, connects to it automatically,
and stops the server when the editor closes. The default Impeller address is
`[::]:2240`; assets use port 2241 and gRPC uses port 2242. Use `--addr` to
select another base address:

```bash
elodin editor dbs/apollo --addr 127.0.0.1:3000
```

The database directory cannot also be open by another `elodin-db` or `elodin`
process on the same ports.

To serve the recording without opening an editor, use:

```bash
elodin run dbs/apollo
```

## Replay mode

Replay mode reveals recorded data progressively as the playback marker
advances, simulating a live session:

```bash
elodin editor dbs/apollo --replay
```

Without `--replay`, the full recorded time range is available immediately.

## Choose a schematic

By default, the editor opens the active schematic stored in the database.
Override it with a local KDL file:

```bash
elodin editor dbs/apollo --replay --kdl schematics/review.kdl
```

For details about bundled GLB files, images, skyboxes, and schematics, see
[DB Asset Server](/reference/db-asset-server).
