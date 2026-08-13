+++
title = "gRPC API"
description = "Complete Elodin DB gRPC service, RPC, and message reference"
draft = false
weight = 104
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 4
lead = "Reference for every elodin.db.v1 service, RPC, message, stream workflow, and wire-level contract."
+++

Elodin DB exposes a versioned, language-neutral gRPC API for telemetry ingest,
queries, playback, message logs, configuration, and assets. This page is the
wire-contract reference. See the [gRPC Client guide](/home/db/grpc-client) for
stub generation and runnable C++ and Python examples.

The protobuf package is `elodin.db.v1`. Its source files are under
`libs/db/proto/elodin/db/v1`:

- `common.proto` — shared values, acknowledgements, metadata, and configuration;
- `ingest.proto` — acknowledged component telemetry writes;
- `query.proto` — discovery, historical reads, downsampling, and SQL;
- `stream.proto` — live component/message streams and fixed-rate playback;
- `msg.proto` — message registration, publishing, and historical reads;
- `admin.proto` — configuration, metadata, and asset operations.

Every database starts gRPC on the native listener's IP at port `P + 2`. With the
default native address `127.0.0.1:2240`, gRPC listens on
`127.0.0.1:2242`. Standard gRPC health checking and v1/v1alpha server reflection
share this endpoint.

## Calling conventions

### RPC shapes

- **Unary** RPCs accept one request and return one response.
- **Server-streaming** RPCs accept one request and return zero or more response
  frames.
- **Bidirectional-streaming** RPCs keep independent request and response streams
  open. The first request frame establishes the session; subsequent frames carry
  data or controls.

An empty request message still must be sent. Fields inside a protobuf `oneof` are
mutually exclusive. A request that requires a `oneof` rejects an unset value.

### Scalar presence

Proto3 fields marked `optional` preserve the difference between omitted and
explicitly set to zero or `false`. Unmarked scalar fields use normal proto3
defaults: `0`, `false`, empty bytes, or an empty string. Repeated fields and maps
default to empty.

### Time

All fields ending in `*_ns` use signed or unsigned nanoseconds on the wire.
Elodin DB stores record time at microsecond resolution:

- write timestamps are floored to the containing microsecond;
- read ranges are half-open: `[start_ns, end_ns)`;
- sub-microsecond range bounds are rounded up to the next storage bucket, so
  only buckets fully inside the requested range are returned;
- returned timestamps are microsecond-aligned nanosecond values.

For example, `[100500, 102500)` ns returns buckets at `101000` and `102000` ns.

### Sequences and delivery

Ingest rows and published messages use client sequences beginning at 1.
Acknowledgements are cumulative. Both writer APIs provide at-least-once delivery:
after reconnecting, replay every item above the server's `resume_from_seq`.

A clean stream end persists the latest ingest position. Resume state is also
persisted while streams run. A server crash can therefore return a resume point
slightly behind the last acknowledgement, and clients must retain enough recent
items to replay that window. Complete rows in the ambiguous window may appear
twice.

### Authentication and transport

The endpoint is unauthenticated by default. When the server starts with
`--grpc-auth-token TOKEN`, application RPCs and reflection require this metadata:

```
authorization: Bearer TOKEN
```

Invalid or missing credentials return `UNAUTHENTICATED`. Health checking remains
unauthenticated. The server uses plaintext HTTP/2; bind it to loopback or a
trusted network, or terminate TLS in deployment infrastructure.

### Limits and structured errors

The encoded and decoded message cap is 16 MiB for every service. Streaming an
object does not remove the per-frame limit. `QueryService.GetServerInfo` reports
the active cap.

RPC failures use canonical gRPC status codes. Selected failures include
`google.rpc.ErrorInfo` with domain `db.elodin.systems` and one of these stable
reasons:

- `COMPONENT_NOT_FOUND`
- `MESSAGE_NOT_FOUND`
- `MESSAGE_SCHEMA_CONFLICT`
- `TIME_RANGE_EMPTY`

Branch on the status code and structured reason, not human-readable status text.

## Common types

### _enum_ `PrimType`

Primitive element type for a component. `PRIM_TYPE_UNSPECIFIED` is never valid in
an ingest schema.

- `PRIM_TYPE_UNSPECIFIED = 0`
- `PRIM_TYPE_U8 = 1`
- `PRIM_TYPE_U16 = 2`
- `PRIM_TYPE_U32 = 3`
- `PRIM_TYPE_U64 = 4`
- `PRIM_TYPE_I8 = 5`
- `PRIM_TYPE_I16 = 6`
- `PRIM_TYPE_I32 = 7`
- `PRIM_TYPE_I64 = 8`
- `PRIM_TYPE_BOOL = 9`
- `PRIM_TYPE_F32 = 10`
- `PRIM_TYPE_F64 = 11`

Multi-byte packed values use little-endian byte order. A packed boolean is one
byte and must be `0` or `1`.

### _message_ `ComponentValue`

One component in a typed ingest row.

- `component_index` : `uint32` — zero-based index into
  `MessageSchema.components`.
- `value` : `oneof` — exactly one scalar or array payload:
  - `f64` : `double`
  - `f32` : `float`
  - `i64` : `sint64`
  - `u64` : `uint64`
  - `b` : `bool`
  - `f64s` : `DoubleArray`
  - `f32s` : `FloatArray`
  - `i64s` : `Sint64Array`
  - `u64s` : `Uint64Array`
  - `bools` : `BoolArray`

Widths narrower than 64 bits still use the `i64`/`u64` alternatives and must fit
the component's declared type. Scalars require scalar alternatives; tensors
require array alternatives with exactly the schema's element count.

### _message_ `DoubleArray`

- `v` : `repeated double` — row-major `f64` tensor elements.

### _message_ `FloatArray`

- `v` : `repeated float` — row-major `f32` tensor elements.

### _message_ `Sint64Array`

- `v` : `repeated sint64` — row-major signed-integer tensor elements.

### _message_ `Uint64Array`

- `v` : `repeated uint64` — row-major unsigned-integer tensor elements.

### _message_ `BoolArray`

- `v` : `repeated bool` — row-major boolean tensor elements.

### _message_ `AckPolicy`

Controls cumulative acknowledgement cadence for ingest and message publishing.

- `max_unacked_rows` : `uint32` — acknowledge after this many newly processed
  items. `0` selects the default `256`; maximum `1,000,000`.
- `max_ack_delay_ms` : `uint32` — maximum delay before acknowledging pending
  items. `0` selects the default `100` ms; maximum `10,000` ms.

The server may acknowledge sooner. An omitted `AckPolicy` uses both defaults.

### _message_ `WriteAck`

- `through_seq` : `uint64` — every sequence less than or equal to this value has
  been processed.

“Processed” means applied and visible to readers, or rejected by a per-item error
sent before this acknowledgement. Durability follows the database's storage
flush policy; the acknowledgement is not an `fsync` guarantee.

### _message_ `DbConfig`

Current database configuration.

- `recording` : `bool` — current database recording flag.
- `default_stream_time_step_ns` : `uint64` — default playback step used by
  native clients; read-only through the current gRPC admin API.
- `metadata` : `map<string, string>` — database metadata.

### _message_ `ComponentMetadata`

- `name` : `string` — component name.
- `metadata` : `map<string, string>` — component metadata keys and values.

### _message_ `MessageMetadata`

- `name` : `string` — registered message-log name.
- `postcard_schema` : `bytes` — serialized Postcard `OwnedNamedType` schema.
- `metadata` : `map<string, string>` — message metadata keys and values.

Opaque and structured-log registrations also have an internal Postcard schema,
so discovery can expose this field for every registered log.

### _message_ `ComponentSchemaSnapshot`

- `name` : `string` — component name.
- `prim_type` : `PrimType` — element type.
- `dims` : `repeated uint64` — tensor dimensions; empty means scalar.
- `start_time_ns` : `sfixed64` — first stored sample timestamp.

## IngestService

`IngestService` writes fixed-schema telemetry rows. One bidirectional stream is
one logical writer session.

### _rpc_ `Ingest`

```
rpc Ingest(stream IngestRequest) returns (stream IngestResponse)
```

Workflow:

1. Send exactly one `SessionOpen` as the first frame.
2. Read either `SessionAccept` or `SessionReject`.
3. On acceptance, use the returned session-scoped message handles.
4. Start at `resume_from_seq + 1`, or replay retained batches whose sequences are
   above `resume_from_seq`.
5. Send ordered `TelemetryBatch` frames while independently reading
   `RowError` and `WriteAck` frames.
6. Half-close the request stream after all rows are sent, then drain responses.

The server skips sequences already covered by the session resume position.
Sequence gaps terminate the stream. A historical replay can repair components
left missing by a mid-row crash, but complete replayed rows may duplicate. New
identical rows at later sequences always remain distinct.

Specific failures:

- `INVALID_ARGUMENT` — missing or malformed open frame, identity, fingerprint,
  schema, ack policy, request oneof, or sequence overflow.
- `FAILED_PRECONDITION` — first frame is not `SessionOpen`, a second open frame
  appears, sequences start at zero, or a batch leaves a sequence gap.
- `INTERNAL` — a database storage failure or violated server invariant.
- A schema conflict is a successful `SessionReject` response frame, not a gRPC
  error status.

### _message_ `IngestRequest`

- `req` : `oneof`
  - `open` : `SessionOpen` — valid only as the first request.
  - `batch` : `TelemetryBatch` — valid only after acceptance.

### _message_ `IngestResponse`

- `resp` : `oneof`
  - `accept` : `SessionAccept` — the opening schema was registered.
  - `ack` : `WriteAck` — cumulative processed position.
  - `error` : `RowError` — non-terminal row rejection.
  - `reject` : `SessionReject` — opening schema conflicts with stored data.

### _message_ `SessionOpen`

- `client_name` : `string` — logical writer name; 1–128 non-control UTF-8 bytes.
- `schema_fingerprint` : `bytes` — exactly 32 bytes: SHA-256 of the encoded
  `SchemaSet`.
- `schema` : `SchemaSet` — required schema declaration.
- `ack_policy` : `AckPolicy` — optional message; omitted uses defaults.
- `client_instance_id` : `bytes` — 1–128 bytes. Keep stable across reconnects
  for one process and choose a new value after a process restart.

Resume state is keyed by `(client_name, client_instance_id)`. The server records
the schema fingerprint in database metadata, but always validates the full
schema on connection.

### _message_ `SchemaSet`

- `messages` : `repeated MessageSchema` — at least one message schema.

Message names must be unique. Component names must be unique across the entire
set, including components in different messages.

### _message_ `MessageSchema`

- `name` : `string` — 1–256 non-control UTF-8 bytes.
- `encoding` : `RowEncoding` — must be `PACKED` or `TYPED`.
- `packed_size` : `uint32` — exact packed-row byte length for `PACKED`; must be
  zero for `TYPED`. Packed rows are limited to 8 MiB.
- `components` : `repeated ComponentSchema` — one or more components in row
  order.

### _message_ `ComponentSchema`

- `name` : `string` — 1–256 non-control UTF-8 bytes.
- `prim_type` : `PrimType` — required element type.
- `dims` : `repeated uint64` — nonzero dimensions; empty means scalar. The
  product may not exceed `16,777,216` elements.
- `element_names` : `repeated string` — optional labels. When present, supply
  exactly one unique, nonempty label per element; labels cannot contain commas.
- `packed_offset` : `uint32` — byte offset for `PACKED`; must be aligned to the
  primitive type, in bounds, and non-overlapping. Must be zero for `TYPED`.
- `timestamp_source` : `bool` — marks a scalar `U64` or `I64` component as a row
  timestamp source.

Existing component names may be reopened only with the same primitive type and
shape. Timestamp-source flags and element-name metadata are synchronized without
clearing unrelated component metadata.

### _enum_ `RowEncoding`

- `ROW_ENCODING_UNSPECIFIED = 0` — invalid in a session schema.
- `ROW_ENCODING_PACKED = 1` — each row contains one fixed-layout byte buffer.
- `ROW_ENCODING_TYPED = 2` — each row contains indexed `ComponentValue`s.

### _message_ `SessionAccept`

- `message_handles` : `map<string, uint32>` — message name to session-scoped
  handle. Do not cache these handles across sessions.
- `resume_from_seq` : `uint64` — last persisted sequence for this session
  identity; resend every retained row above it.

### _message_ `SessionReject`

- `detail` : `string` — summary of why the schema cannot be opened.
- `conflicts` : `repeated ComponentSchemaConflict` — type/shape conflicts.

After this frame the response stream closes and no rows are applied.

### _message_ `ComponentSchemaConflict`

- `component` : `string` — conflicting component name.
- `expected_prim_type` : `PrimType` — type already stored by the database.
- `expected_dims` : `repeated uint64` — shape already stored.
- `actual_prim_type` : `PrimType` — type declared by this session.
- `actual_dims` : `repeated uint64` — shape declared by this session.

### _message_ `TelemetryBatch`

- `first_seq` : `uint64` — sequence of `rows[0]`; must begin at 1.
- `rows` : `repeated Row` — row `i` has sequence `first_seq + i`.

Batches may overlap the already processed prefix; those rows are skipped. Empty
batches are allowed but do not advance the session. Nonempty batches must not
start after `current_seq + 1`.

### _message_ `Row`

- `message_handle` : `uint32` — handle from `SessionAccept.message_handles`.
- `time_monotonic_ns` : `optional sfixed64` — explicit record time.
- `payload` : `oneof`
  - `packed` : `bytes` — exactly `MessageSchema.packed_size` bytes.
  - `typed` : `TypedValues` — one indexed value for every component.

`time_monotonic_ns` may be omitted only when at least one component is a
`timestamp_source`. If explicit and embedded timestamps are both present, all
must match exactly before microsecond flooring.

### _message_ `TypedValues`

- `values` : `repeated ComponentValue` — exactly one value for every component;
  order is arbitrary because `component_index` establishes placement.

Missing, duplicate, out-of-range, wrong-kind, wrong-length, and integer-overflow
values reject the row.

### _message_ `RowError`

- `seq` : `uint64` — rejected row's client sequence.
- `component` : `string` — affected component, or empty for a row-level error.
- `detail` : `string` — human-readable validation or time-ordering failure.

A `RowError` is non-terminal. Its sequence advances and is included in a later
`WriteAck`; retrying the same invalid row cannot repair the stream.

## QueryService

`QueryService` provides discovery and lossless historical reads.

### _rpc_ `GetServerInfo`

```
rpc GetServerInfo(GetServerInfoRequest) returns (GetServerInfoResponse)
```

Returns limits and stable feature identifiers. Feature-detect from `features`,
not `build_version`.

### _message_ `GetServerInfoRequest`

No fields.

### _message_ `GetServerInfoResponse`

- `build_version` : `string` — diagnostic package version.
- `max_message_size_bytes` : `uint32` — current per-message encode/decode cap.
- `features` : `repeated string` — stable capability identifiers. Current
  values are `sql-arrow-ipc`, `lttb-downsample`, and `message-resume`.

### _rpc_ `GetTimeRange`

```
rpc GetTimeRange(GetTimeRangeRequest) returns (GetTimeRangeResponse)
```

Returns the global data range across components and message logs. When
`has_data` is false, ignore both timestamp fields.

### _message_ `GetTimeRangeRequest`

No fields.

### _message_ `GetTimeRangeResponse`

- `has_data` : `bool` — whether the database contains component or message data.
- `earliest_ns` : `sfixed64` — earliest data timestamp.
- `last_updated_ns` : `sfixed64` — latest global data update.

### _rpc_ `DumpMetadata`

```
rpc DumpMetadata(DumpMetadataRequest) returns (DumpMetadataResponse)
```

Returns component metadata, registered message metadata, and current database
configuration. Component and message lists are sorted by name.

### _message_ `DumpMetadataRequest`

No fields.

### _message_ `DumpMetadataResponse`

- `components` : `repeated ComponentMetadata` — all component metadata.
- `messages` : `repeated MessageMetadata` — all registered message metadata.
- `config` : `DbConfig` — current database configuration.

### _rpc_ `DumpSchema`

```
rpc DumpSchema(DumpSchemaRequest) returns (DumpSchemaResponse)
```

Returns component type, shape, and series-start information sorted by name.

### _message_ `DumpSchemaRequest`

No fields.

### _message_ `DumpSchemaResponse`

- `components` : `repeated ComponentSchemaSnapshot` — all component schemas.

### _rpc_ `GetTimeSeries`

```
rpc GetTimeSeries(GetTimeSeriesRequest)
    returns (stream GetTimeSeriesResponse)
```

Reads one component in timestamp order. A successful response contains exactly
one `header` frame followed by zero or more `data` frames. Raw data frames target
about 1 MiB; chunk boundaries are not stable API behavior.

Specific failures:

- `INVALID_ARGUMENT` — empty component, `limit = 0`, `max_points < 3`, or an
  out-of-bounds `element_index` when downsampling is required.
- `NOT_FOUND` + `COMPONENT_NOT_FOUND` — component is unknown.
- `OUT_OF_RANGE` + `TIME_RANGE_EMPTY` — component exists but the selected range
  contains no samples.

### _message_ `GetTimeSeriesRequest`

- `component` : `string` — required component name.
- `start_ns` : `optional sfixed64` — inclusive lower bound; omitted means
  earliest.
- `end_ns` : `optional sfixed64` — exclusive upper bound; omitted means
  open-ended.
- `limit` : `optional uint64` — maximum source rows before downsampling; must be
  at least 1. Omitted means unlimited.
- `max_points` : `optional uint32` — LTTB target when selected rows exceed this
  count; must be at least 3. Omitted returns raw rows.
- `element_index` : `uint32` — flattened row-major tensor element used as the
  LTTB signal.

Downsampling selects timestamps using one element but returns the complete
component value for each selected row.

### _message_ `TimeSeriesHeader`

- `component` : `string` — component name.
- `prim_type` : `PrimType` — packed element type.
- `dims` : `repeated uint64` — tensor dimensions; empty means scalar.
- `element_names` : `repeated string` — flattened element labels, when present.

### _message_ `TimeSeriesData`

- `timestamps_ns` : `repeated sfixed64` — one timestamp per packed row.
- `packed_values` : `bytes` — concatenated, little-endian, row-major component
  values.

The row byte size is `sizeof(prim_type) * product(dims)`, with scalar product 1.
`packed_values` contains exactly `timestamps_ns.size * row_size` bytes.

### _message_ `GetTimeSeriesResponse`

- `chunk` : `oneof`
  - `header` : `TimeSeriesHeader` — first and exactly once.
  - `data` : `TimeSeriesData` — zero or more chunks.

### _rpc_ `Sql`

```
rpc Sql(SqlRequest) returns (stream SqlResponse)
```

Runs SQL through the database's DataFusion context. Each response is a
self-contained Arrow IPC stream containing a schema and one record batch.
Decode every response independently, then concatenate the decoded batches.
Component names become sanitized snake-case table names, such as
`sensor.pressure` → `sensor_pressure`; each table includes a microsecond
timestamp column named `time`.

An empty or whitespace-only query returns `INVALID_ARGUMENT`. Planning,
execution, and Arrow encoding failures arrive on the response stream as
`INTERNAL`.

### _message_ `SqlRequest`

- `sql` : `string` — nonempty DataFusion SQL statement.

### _message_ `SqlResponse`

- `ipc` : `bytes` — one complete Arrow IPC stream.

## StreamService

`StreamService` provides latest-value subscriptions and controllable playback.
It is not the lossless historical interface; use `QueryService` for complete
ranges.

### _message_ `RealTime`

- `immediate` : `bool` — when true, each component has its own wake path to
  reduce batching latency. Intermediate rows can still coalesce.

### _enum_ `InitialTimestamp`

- `INITIAL_TIMESTAMP_UNSPECIFIED = 0` — same as `LATEST`.
- `INITIAL_TIMESTAMP_EARLIEST = 1` — global earliest data timestamp.
- `INITIAL_TIMESTAMP_LATEST = 2` — global latest data timestamp.
- `INITIAL_TIMESTAMP_MANUAL = 3` — use `initial_timestamp_ns`.

### _message_ `FixedRate`

- `initial` : `InitialTimestamp` — initial cursor policy.
- `initial_timestamp_ns` : `sfixed64` — used only for `MANUAL`; floored to the
  microsecond grid.
- `timestep_ns` : `uint64` — cursor advance per frame; at least `1,000`.
- `frequency` : `uint64` — wall-clock frame rate from 1 through 1,000 Hz.

The stream starts in the playing state.

### _message_ `StreamControl`

- `playing` : `optional bool` — pause or resume playback.
- `seek_ns` : `optional sfixed64` — replace the cursor; floored to a microsecond.
- `timestep_ns` : `optional uint64` — replace the step; at least `1,000`.
- `frequency` : `optional uint64` — replace the frame rate; 1–1,000 Hz.

All present fields are validated before any are applied. A seek always forces
resampling, even when it targets the current cursor.

### _rpc_ `StreamComponents`

```
rpc StreamComponents(stream StreamComponentsRequest)
    returns (stream StreamComponentsResponse)
```

The first request must contain `StreamOpen`. An empty component list selects all
components, sorted by name. The server first sends one `TimeSeriesHeader` per
selected component.

In real-time mode, the stream then sends the latest row for each changed
component. It can skip intermediate rows under load. Existing latest values are
emitted when the stream starts; unchanged `(timestamp, value)` pairs are not
repeated.

In fixed-rate mode, headers are followed by `StreamOpened`. At each cursor, the
server emits the sample at or before the cursor for every component that has one,
then emits `StreamTimestamp`. The cursor advances by `timestep_ns` at
`frequency` frames per second while playing.

Only an owning fixed-rate component stream accepts controls. Its stream ID lets
a fixed-rate message stream share the same clock.

Specific failures:

- `INVALID_ARGUMENT` — missing/misordered open, invalid step/frequency, invalid
  later frame, or invalid control.
- `NOT_FOUND` — a selected component does not exist.
- `FAILED_PRECONDITION` — a real-time stream receives a control frame.

### _message_ `StreamOpen`

- `components` : `repeated string` — selected names; empty selects all.
- `behavior` : `oneof`
  - `real_time` : `RealTime`
  - `fixed_rate` : `FixedRate`

An omitted behavior uses batched real-time mode.

### _message_ `StreamComponentsRequest`

- `request` : `oneof`
  - `open` : `StreamOpen` — first frame only.
  - `control` : `StreamControl` — later frames on fixed-rate playback only.

### _message_ `ComponentUpdate`

- `component` : `string` — component name.
- `timestamp_ns` : `sfixed64` — sampled row timestamp, which can be older than
  the playback cursor.
- `packed_value` : `bytes` — one little-endian component value matching its
  preceding header.

### _message_ `StreamTimestamp`

- `timestamp_ns` : `sfixed64` — current fixed-rate playback cursor.

This frame closes one component playback tick.

### _message_ `StreamComponentsResponse`

- `response` : `oneof`
  - `header` : `TimeSeriesHeader` — schema for one selected component.
  - `update` : `ComponentUpdate` — latest or sampled component value.
  - `timestamp` : `StreamTimestamp` — fixed-rate cursor marker.
  - `opened` : `StreamOpened` — fixed-rate owner identity.

### _message_ `StreamOpened`

- `stream_id` : `uint64` — nonzero, process-local identifier for this fixed-rate
  component clock. Valid only while the owning stream is alive.

### _rpc_ `StreamMessages`

```
rpc StreamMessages(stream StreamMessagesRequest)
    returns (stream StreamMessagesResponse)
```

The first request must contain `MessageStreamOpen`. An empty message list selects
all registered logs, sorted by name.

Real-time mode primes every selected log with its latest stored message, then
delivers every append by index while connected. Reconnecting primes again, so
the first message may duplicate one already consumed. This mode has no resume
token.

Independent fixed-rate mode samples each log at or before its cursor and accepts
controls. It does not return a stream ID or cursor frames. Attached fixed-rate
mode follows a component stream's clock, including pauses and seeks, and ends
when the owner ends. Send controls to the owning component stream.

Specific failures:

- `INVALID_ARGUMENT` — missing/misordered open, invalid fixed-rate settings,
  zero playback ID, playback ID without fixed-rate behavior, or invalid control.
- `NOT_FOUND` — a selected message or referenced playback stream does not exist.
- `FAILED_PRECONDITION` — live or attached playback receives a control frame.

### _message_ `MessageStreamOpen`

- `messages` : `repeated string` — selected log names; empty selects all.
- `behavior` : `oneof`
  - `real_time` : `RealTime`
  - `fixed_rate` : `FixedRate`
- `playback_stream_id` : `optional uint64` — attach to the nonzero stream ID of
  an active fixed-rate component stream. Omitted creates an independent clock.

When `playback_stream_id` is present, `fixed_rate` is still required to select
fixed-rate behavior, but the owner's clock supplies the effective timing.
`RealTime.immediate` does not change message delivery.

### _message_ `StreamMessagesRequest`

- `request` : `oneof`
  - `open` : `MessageStreamOpen` — first frame only.
  - `control` : `StreamControl` — later frames only for an independent
    fixed-rate clock.

### _message_ `StreamMessagesResponse`

- `name` : `string` — message-log name.
- `timestamp_ns` : `sfixed64` — stored message timestamp.
- `payload` : `bytes` — stored encoded payload. Structured logs are not decoded
  on this streaming RPC.

### _rpc_ `WatchDb`

```
rpc WatchDb(WatchDbRequest) returns (stream WatchDbResponse)
```

Immediately sends the current `last_updated_ns` event followed by the current
`DbConfig`. It then sends another timestamp or config event whenever that value
changes. Use `GetTimeRange.has_data` when the empty-database distinction matters.

### _message_ `WatchDbRequest`

No fields.

### _message_ `WatchDbResponse`

- `event` : `oneof`
  - `last_updated_ns` : `sfixed64` — latest global data update.
  - `config` : `DbConfig` — complete current configuration.

## MessageService

`MessageService` stores variable-length events separately from fixed-schema
component time series.

### _rpc_ `Register`

```
rpc Register(RegisterRequest) returns (RegisterResponse)
```

Registers a named log and returns its stable handle. Re-registering the same name
and schema is idempotent; supplied metadata keys merge into existing metadata.
A name/hash or schema collision returns `ALREADY_EXISTS` with
`MESSAGE_SCHEMA_CONFLICT`.

### _message_ `OpaqueKind`

No fields. Selects arbitrary raw byte payloads.

### _message_ `LogKind`

No fields. Selects structured `LogPayload` values encoded by the server.

### _message_ `RegisterRequest`

- `name` : `string` — required message-log name.
- `kind` : `oneof`
  - `opaque` : `OpaqueKind`
  - `log` : `LogKind`
  - `postcard_schema` : `bytes` — serialized Postcard `OwnedNamedType`.
- `metadata` : `map<string, string>` — keys to merge into message metadata.

The selected kind is permanent for a registered name.

### _message_ `RegisterResponse`

- `message_handle` : `uint32` — stable name-derived handle, valid across
  sessions and server restarts. Current handles fit in `uint16`.

### _rpc_ `Publish`

```
rpc Publish(stream PublishRequest) returns (stream PublishResponse)
```

Workflow:

1. Send `PublishOpen` as the first frame.
2. Read `PublishAccept`.
3. Resume with retained messages above `resume_from_seq`.
4. Send ordered, nonempty `PublishBatch` frames while reading errors and acks.
5. Half-close and drain the final acknowledgement.

Validation failures produce `MessageError`, advance that sequence, and are
covered by a later acknowledgement. Storage failures terminate the stream
without advancing the failed message, so reconnecting retries it.

Specific failures:

- `INVALID_ARGUMENT` — missing/misordered open, invalid identity/ack policy,
  empty batch, or a malformed later request frame.
- `FAILED_PRECONDITION` — sequence zero or a sequence gap.
- Invalid handles, payload kinds, and log levels produce `MessageError` rather
  than a terminal status.
- `INTERNAL`, `PERMISSION_DENIED`, or a structured database error — storage
  failure; the failed sequence is not acknowledged.

### _message_ `PublishOpen`

- `client_name` : `string` — logical writer name; 1–128 non-control UTF-8 bytes.
- `client_instance_id` : `bytes` — 1–128 bytes; stable across reconnects for
  this logical session.
- `ack_policy` : `AckPolicy` — optional message; omitted uses defaults.

Resume state is independent from `IngestService` even if the identity bytes are
the same.

### _message_ `LogPayload`

- `level` : `uint32` — log level; must fit in `uint8`.
- `message` : `string` — log text.

### _message_ `OutgoingMessage`

- `message_handle` : `uint32` — handle returned by `Register`.
- `timestamp_ns` : `optional sfixed64` — explicit record timestamp; omitted asks
  the server to assign one.
- `payload` : `oneof`
  - `raw` : `bytes` — required for opaque and custom Postcard registrations.
  - `log` : `LogPayload` — required for structured-log registrations.

Explicit timestamps are floored to the containing microsecond.

### _message_ `PublishBatch`

- `first_seq` : `uint64` — sequence of `messages[0]`; starts at 1.
- `messages` : `repeated OutgoingMessage` — must be nonempty; message `i` has
  sequence `first_seq + i`.

Already processed overlap is skipped. A batch cannot leave a gap after the
current sequence.

### _message_ `PublishRequest`

- `request` : `oneof`
  - `open` : `PublishOpen` — first frame only.
  - `batch` : `PublishBatch` — later frames only.

### _message_ `PublishAccept`

- `resume_from_seq` : `uint64` — last persisted sequence for this identity.

### _message_ `MessageError`

- `seq` : `uint64` — rejected message sequence.
- `message` : `string` — registered message name when it can be resolved.
- `detail` : `string` — human-readable validation failure.

This response is non-terminal and precedes the acknowledgement that covers its
sequence.

### _message_ `PublishResponse`

- `response` : `oneof`
  - `accept` : `PublishAccept` — opening response.
  - `ack` : `WriteAck` — cumulative processed position.
  - `error` : `MessageError` — non-terminal item rejection.

### _rpc_ `GetMessages`

```
rpc GetMessages(GetMessagesRequest)
    returns (stream GetMessagesResponse)
```

Streams a lossless, timestamp-ordered historical range. An empty range returns
an empty successful stream.

Specific failures:

- `INVALID_ARGUMENT` — empty name or `limit = 0`.
- `NOT_FOUND` + `MESSAGE_NOT_FOUND` — log is not registered.
- `INTERNAL` — a stored structured-log payload cannot be decoded.

### _message_ `GetMessagesRequest`

- `name` : `string` — required registered log name.
- `start_ns` : `optional sfixed64` — inclusive lower bound; omitted means
  earliest.
- `end_ns` : `optional sfixed64` — exclusive upper bound; omitted means
  open-ended.
- `limit` : `optional uint64` — maximum messages, at least 1; omitted means
  unlimited.

### _message_ `GetMessagesResponse`

- `timestamp_ns` : `sfixed64` — stored message timestamp.
- `payload` : `oneof`
  - `raw` : `bytes` — opaque or custom Postcard bytes.
  - `log` : `LogPayload` — decoded structured log.

## AdminService

`AdminService` manages database configuration, component metadata, and asset
bytes.

### _rpc_ `GetDbConfig`

```
rpc GetDbConfig(GetDbConfigRequest) returns (GetDbConfigResponse)
```

Returns the complete current configuration.

### _message_ `GetDbConfigRequest`

No fields.

### _message_ `GetDbConfigResponse`

- `config` : `DbConfig` — complete current configuration.

### _rpc_ `SetDbConfig`

```
rpc SetDbConfig(SetDbConfigRequest) returns (SetDbConfigResponse)
```

Applies a patch and returns the complete resulting configuration. Omitted
recording state remains unchanged. Metadata keys merge; an empty value deletes a
key except `skybox.active`, where empty is a preserved explicit-clear signal.

On a read-only follower mirror, attempts to change replicated asset-pointer keys
return `PERMISSION_DENIED`.

### _message_ `SetDbConfigRequest`

- `recording` : `optional bool` — new recording state; omitted means unchanged.
- `metadata` : `map<string, string>` — metadata patch.

### _message_ `SetDbConfigResponse`

- `config` : `DbConfig` — complete configuration after the patch.

### _rpc_ `SetComponentMetadata`

```
rpc SetComponentMetadata(SetComponentMetadataRequest)
    returns (SetComponentMetadataResponse)
```

Sets the supplied component metadata and returns the stored value. The server
preserves internal metadata keys such as the timestamp-source flag.

Missing metadata or an empty component name returns `INVALID_ARGUMENT`.
Persistence errors map to canonical database statuses.

### _message_ `SetComponentMetadataRequest`

- `metadata` : `ComponentMetadata` — required metadata value.

### _message_ `SetComponentMetadataResponse`

- `metadata` : `ComponentMetadata` — stored value.

### _rpc_ `PutAsset`

```
rpc PutAsset(stream PutAssetRequest) returns (PutAssetResponse)
```

The first request must be one `header`; every later request must be `data`.
Half-close to commit the asset. The server buffers at most 256 MiB, writes the
asset under the database's asset directory, and increments `assets.revision`.

Specific failures:

- `INVALID_ARGUMENT` — empty stream/key, data before header, repeated header, or
  an invalid relative asset key.
- `RESOURCE_EXHAUSTED` — aggregate asset exceeds 256 MiB.
- `PERMISSION_DENIED` — asset storage is read-only, including follower mirrors.
- `INTERNAL` — storage or task failure.

### _message_ `PutAssetHeader`

- `key` : `string` — nonempty relative asset key. Absolute paths, traversal, and
  reserved internal keys are rejected.

### _message_ `PutAssetRequest`

- `chunk` : `oneof`
  - `header` : `PutAssetHeader` — first frame exactly once.
  - `data` : `bytes` — subsequent asset bytes.

Each data frame must also fit the 16 MiB gRPC message cap.

### _message_ `PutAssetResponse`

- `size` : `uint64` — committed asset size in bytes.
- `assets_revision` : `uint64` — database asset revision after the write.

### _rpc_ `GetAsset`

```
rpc GetAsset(GetAssetRequest) returns (stream GetAssetResponse)
```

Reads one asset in 1 MiB response chunks. Concatenate `data` in response order;
chunk boundaries are not file boundaries.

An unknown key returns `NOT_FOUND`; an invalid key returns `INVALID_ARGUMENT`;
storage failures return `INTERNAL`.

### _message_ `GetAssetRequest`

- `key` : `string` — relative asset key.

### _message_ `GetAssetResponse`

- `data` : `bytes` — next asset chunk.

### _rpc_ `ListAssets`

```
rpc ListAssets(ListAssetsRequest) returns (ListAssetsResponse)
```

Lists indexed assets whose keys begin with `prefix`. An empty prefix selects all
assets. Indexing failures return `INTERNAL`.

### _message_ `ListAssetsRequest`

- `prefix` : `string` — optional key prefix; empty means all.

### _message_ `AssetInfo`

- `key` : `string` — relative asset key.
- `size` : `uint64` — asset size in bytes.

### _message_ `ListAssetsResponse`

- `assets` : `repeated AssetInfo` — matching indexed assets.
