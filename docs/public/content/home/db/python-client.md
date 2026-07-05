+++
title = "Python Client"
description = "Stream external telemetry from Python with elodin.db"
draft = false
weight = 106
sort_by = "weight"

[extra]
lead = "Write and read Elodin DB telemetry from any plain Python process with `elodin.db`."
toc = true
top = false
order = 7
icon = ""
+++

The `elodin` Python wheel ships a first-class Elodin DB client: `elodin.db`.
It lets any Python process — a hardware-in-the-loop rig, a third-party
simulator bridge, a flight-test logger, a notebook — start or connect to
Elodin DB, write telemetry, and read it back. No Elodin simulation required.
This page is a guided tour; the full API surface is documented in the
[Python API reference](/reference/python-api#elodin-db-client).

```sh
pip install elodin      # or use the repo's nix devshell + `just install`
```

## Quick start

```python
import elodin.db as edb

# Embedded server (tests, notebooks, single-box logging) — or connect to an
# external `elodin-db run` instance instead.
server = edb.Server.start("./dbs/run42", "127.0.0.1:2240")
client = edb.Client.connect("127.0.0.1:2240")

writer = client.table_writer({
    "drone.imu.accel": edb.f64[3].labeled("x", "y", "z"),
    "drone.cmd.throttle": edb.f64,
})
writer.write(timestamp_us=t_us, values={
    "drone.imu.accel": [0.0, 0.0, -9.81],
    "drone.cmd.throttle": 0.42,
})

ts, accel = client.time_series("drone.imu.accel", t0_us, t1_us)
table = client.sql("SELECT * FROM drone_imu_accel")   # pyarrow.Table
```

Everything written this way is a normal Elodin DB citizen: plottable live in
the Elodin Editor (`elodin editor 127.0.0.1:2240`, with `.labeled(...)` names
appearing as axis labels), queryable via `:sql` in `elodin-db lua`, and
replicable with follow mode.

## Writing telemetry

### Schema declaration

A writer is declared once with the component names, dtypes, shapes, and
optional element labels:

```python
writer = client.table_writer(
    {
        "drone.imu.accel":    edb.f64[3].labeled("x", "y", "z"),
        "drone.imu.gyro":     edb.f64[3],
        "drone.cmd.throttle": edb.f64,
        "drone.cmd.body_rate": edb.f32[3],
        "race.active_gate":   edb.i32,
        "nav.covariance":     edb.f64[3, 3],   # tensors up to rank 3
    },
)
```

All primitive types are supported: `f32/f64`, `i8..i64`, `u8..u64`, `bool_`.
Every write emits exactly **one** Impeller2 `Table` packet: a shared `i64`
timestamp followed by each field's values — one packet per tick, not one per
component. All declared fields are required on every write; use one writer
per rate group (e.g. one 110 Hz IMU writer, one 50 Hz control writer).

### Never block a control loop

`write` blocks until the row is handed to the socket and raises on failure.
`write_nowait` never blocks and never raises for transport reasons — rows are
shed according to the queue policy when the database is unreachable or the
bounded queue is full:

```python
writer = client.table_writer(schema, queue="drop-oldest", maxlen=2048)
writer.write_nowait(timestamp_us=t_us, values=...)   # microseconds-cheap
print(writer.dropped)      # rows shed so far
print(writer.state())      # "Connected" | "Disconnected"
print(writer.last_error)   # most recent transport error or DB rejection
```

Writers reconnect automatically with the component-metadata + vtable
handshake replayed. Server-side rejections (e.g. re-registering a component
with a different shape) surface asynchronously via `writer.last_error`.

For nanosecond sources, declare the writer with `timestamp="ns"` and pass
`timestamp_ns=` to writes; the database stores microseconds.

For one-off scripts there is also `client.send(name, values, timestamp_us)` —
a convenience single-component f64 write.

## Reading telemetry

```python
# Discovery
client.components()          # {name: ComponentInfo(prim_type, shape, element_names, ...)}
client.earliest_timestamp()  # first data timestamp (µs)

# Latest value (starts a background real-time subscription on first call).
# Values keep their true dtype and shape.
sample = client.latest("drone.imu.accel")   # Sample(name, timestamp_us, values)

# Historical range [start_us, stop_us) — paginated internally, returns numpy
ts, pos = client.time_series("drone.vio.position", t0_us, t1_us)

# Live stream (rows of the requested components as they arrive)
for row in client.stream(["drone.imu.accel", "drone.cmd.throttle"]):
    print(row.timestamp_us, row["drone.imu.accel"])

# Fixed-rate replay of recorded data
for row in client.stream(["drone.vio.position"], rate_hz=50, start="earliest"):
    ...

# SQL over the same socket; component names map to snake_case table names
name = edb.sql_table_name("drone.vio.position")     # "drone_vio_position"
df = client.sql(f"SELECT * FROM {name}").to_pandas()
```

`client.stream(...)` returns an iterator usable as a context manager;
iteration ends when the stream is closed or its connection fails. `start`
accepts `"earliest"`, `"latest"`, or an integer microsecond timestamp.

## Message logs (events)

Discrete events with variable-length payloads use the message-log API.
Payload encoding is a v1 convenience: `bytes` pass through untouched, `str`
is UTF-8, anything else is JSON.

```python
client.send_msg("race.collision", {"id": 1001, "impulse": 2.4}, timestamp_us=t_us)

msgs = client.get_msgs("race.collision", t0_us, t1_us)   # [(timestamp_us, payload)]

with client.msg_stream("race.collision") as stream:      # live, new messages only
    for t_us, payload in stream:
        ...
```

## Example: drive a 3D viewport from a flight controller

Anything that writes a `world_pos`-shaped component — 7 `f64`s holding a
scalar-last quaternion `(q0, q1, q2, q3)` followed by an `(x, y, z)` position
— can place a GLB model in an Editor viewport. Stream your attitude solution
under a namespaced entity, e.g. `drone.world_pos`:

```python
import elodin.db as edb

client = edb.Client.connect("127.0.0.1:2240")
writer = client.table_writer(
    {"drone.world_pos": edb.f64[7].labeled("q0", "q1", "q2", "q3", "x", "y", "z")},
)

# in the estimator loop (e.g. attitude determination on the flight controller):
writer.write_nowait(
    timestamp_us=t_us,
    values={"drone.world_pos": [qx, qy, qz, qw, x, y, z]},  # identity = 0,0,0,1
)
```

Then a KDL schematic (`drone.kdl`) loads a GLB model at that pose and keeps
the camera on it:

```kdl
coordinate frame=ENU
timeline

viewport name="chase" pos="drone.world_pos + (0,0,0,0, -6, 0, 2)" look_at="drone.world_pos" show_grid=#true

object_3d drone.world_pos {
    glb path="assets/drone.glb" scale=1.0
}
```

Open the Editor against the database with the schematic:

```sh
elodin editor 127.0.0.1:2240 --kdl drone.kdl
```

The `object_3d` element accepts any EQL expression that evaluates to a
`world_pos`-like value, so the same pattern works for estimator ghosts
(`drone.ekf_pos` vs `drone.vio_pos`), targets, or any other entity your
process publishes. See the [schematic reference](/reference/schematic) for
the full `viewport`/`object_3d` syntax.

## Lifecycle & troubleshooting

- `Server`, `Client`, `TableWriter`, and both stream types are context
  managers; `close()` is idempotent.
- Timestamps in the public API are **microseconds** (`int`), the database's
  native resolution.
- Set `ELODIN_DB_LOG=debug` (any `tracing` filter works) to surface the
  embedded server's and client's diagnostics from an `elodin.db`-using
  process.
