+++
title = "gRPC Client"
description = "Use Elodin DB from C++, Python, and other gRPC clients"
draft = false
weight = 107
sort_by = "weight"

[extra]
lead = "Use the versioned protobuf API to ingest, query, and stream Elodin DB data from any gRPC-supported language."
toc = true
top = false
order = 7
icon = ""
+++

Elodin DB exposes a first-class gRPC API for applications that cannot use its
native Rust and Impeller interfaces. The same `elodin.db.v1` contract generates
typed clients for C++, Python, and every other language supported by gRPC.

Use gRPC when you need a stable cross-language contract, acknowledged and
resumable writes, historical queries, playback streams, message logs, or
administrative access. Python-only applications that prefer a higher-level API
can use the [Python Client](/home/db/python-client). Native Elodin processes
should continue to use Impeller and VTables.

Start the optional endpoint alongside the normal database listener:

```sh
elodin-db run 127.0.0.1:2240 ./db --grpc-addr 127.0.0.1:50051
```

The endpoint serves the application services, standard gRPC health checking,
and server reflection. The source contracts live in
`libs/db/proto/elodin/db/v1`.

## Generate a client

### Python

Install the gRPC runtime and compiler, then generate modules from the checked-in
contracts:

```sh
python -m pip install grpcio grpcio-tools
mkdir -p generated
python -m grpc_tools.protoc \
  -I libs/db/proto \
  --python_out=generated \
  --grpc_python_out=generated \
  libs/db/proto/elodin/db/v1/*.proto
export PYTHONPATH="$PWD/generated${PYTHONPATH:+:$PYTHONPATH}"
```

Install `pyarrow` as well when decoding SQL results.

### C++

Elodin packages the generated protobuf and gRPC sources as
`elodin-db-protos`:

```cmake
find_package(elodin-db-protos REQUIRED CONFIG)
target_link_libraries(my_client PRIVATE elodin-db-protos::elodin-db-protos)
```

Headers use paths such as `elodin/db/v1/ingest.grpc.pb.h`. The complete
reference client is `libs/db/examples/grpc-client-batched.cpp`.

## Quick start: acknowledged telemetry

An ingest stream starts with exactly one `SessionOpen`. The server validates
the schema and returns session-scoped handles before accepting rows:

```python
import hashlib
import os
import queue
import struct
import time

import grpc
from elodin.db.v1 import common_pb2, ingest_pb2, ingest_pb2_grpc

channel = grpc.insecure_channel("127.0.0.1:50051")
grpc.channel_ready_future(channel).result(timeout=10)

schema = ingest_pb2.SchemaSet(messages=[
    ingest_pb2.MessageSchema(
        name="sensor.row",
        encoding=ingest_pb2.ROW_ENCODING_PACKED,
        packed_size=8,
        components=[
            ingest_pb2.ComponentSchema(
                name="sensor.pressure",
                prim_type=common_pb2.PRIM_TYPE_F64,
                packed_offset=0,
            )
        ],
    )
])

outgoing = queue.Queue()

def requests():
    while (request := outgoing.get()) is not None:
        yield request

responses = ingest_pb2_grpc.IngestServiceStub(channel).Ingest(requests())
outgoing.put(ingest_pb2.IngestRequest(open=ingest_pb2.SessionOpen(
    client_name="sensor-bridge",
    client_instance_id=os.urandom(16),
    schema=schema,
    schema_fingerprint=hashlib.sha256(
        schema.SerializeToString(deterministic=True)
    ).digest(),
    ack_policy=common_pb2.AckPolicy(max_unacked_rows=32, max_ack_delay_ms=20),
)))

accepted = next(responses).accept
handle = accepted.message_handles["sensor.row"]
sequence = accepted.resume_from_seq + 1

outgoing.put(ingest_pb2.IngestRequest(batch=ingest_pb2.TelemetryBatch(
    first_seq=sequence,
    rows=[ingest_pb2.Row(
        message_handle=handle,
        time_monotonic_ns=time.monotonic_ns(),
        packed=struct.pack("<d", 101_325.0),
    )],
)))
outgoing.put(None)

for response in responses:
    if response.HasField("error"):
        raise RuntimeError(f"row {response.error.seq}: {response.error.detail}")
    if response.HasField("ack") and response.ack.through_seq >= sequence:
        break
```

Reuse one channel and stream for sustained writers. The C++ reference uses one
reader thread and one writer thread so batches can remain in flight while
cumulative acknowledgements arrive.

## Ingest contracts

### Schemas and row encodings

A `SchemaSet` groups messages by rate and layout. Each `MessageSchema` contains
one or more fixed-shape components:

- `prim_type` supports all integer widths, `f32`, `f64`, and `bool`;
- an empty `dims` list is a scalar; non-empty dimensions describe a dense
  tensor;
- `element_names` label tensor elements for plotting;
- `timestamp_source` marks an `i64` or `u64` component whose nanosecond value
  supplies record time.

`ROW_ENCODING_PACKED` is the efficient path for fixed-layout producers. Every
component declares a byte offset, and each row carries one little-endian byte
buffer of exactly `packed_size`. `ROW_ENCODING_TYPED` carries a
`ComponentValue` per component and is convenient for dynamic clients.

All `*_ns` fields are nanoseconds on the wire. Elodin DB floors record time to
its microsecond storage grid while preserving a timestamp-source component's
original nanosecond value as data.

### Resume and acknowledgements

Rows are numbered from 1. Keep `client_name` and `client_instance_id` stable
while reconnecting the same logical writer. After `SessionAccept`, resend every
row above `resume_from_seq`.

Delivery is at least once. `WriteAck.through_seq` means every covered row was
processed: it is visible to readers, or a preceding `RowError` reported its
rejection. Resume positions persist periodically, so a server crash may resume
slightly before the last ack; replayed ingest rows are deduplicated by sequence
and, across that crash window, by complete row content.

`RowError` is non-terminal and always arrives before the ack that covers its
sequence. Schema conflicts reject the session before any row is applied.

## Discovery and historical reads

`QueryService` provides typed operations for common reads:

```python
from elodin.db.v1 import query_pb2, query_pb2_grpc

query = query_pb2_grpc.QueryServiceStub(channel)
info = query.GetServerInfo(query_pb2.GetServerInfoRequest())
print(info.build_version, info.max_message_size_bytes, info.features)

metadata = query.DumpMetadata(query_pb2.DumpMetadataRequest())
schemas = query.DumpSchema(query_pb2.DumpSchemaRequest())

responses = query.GetTimeSeries(query_pb2.GetTimeSeriesRequest(
    component="sensor.pressure",
    start_ns=1_000_000,
    end_ns=2_000_000,
))
for response in responses:
    if response.HasField("header"):
        print(response.header.prim_type, response.header.dims)
    else:
        print(response.data.timestamps_ns, response.data.packed_values)
```

Time ranges are half-open `[start_ns, end_ns)` and are evaluated on the
microsecond storage grid. Omit either bound for earliest or open-ended reads.
Omit `limit` for all matching rows. `GetTimeSeries` sends one header followed
by bounded data chunks; chunk boundaries are not part of the contract.

Set `max_points >= 3` to apply LTTB downsampling. For tensors,
`element_index` selects the element used as the downsampling signal while each
selected row still contains the complete tensor.

`Sql` streams one self-contained Arrow IPC stream (schema plus one record
batch) in each response:

```python
import pyarrow as pa

batches = []
for response in query.Sql(query_pb2.SqlRequest(
    sql="SELECT * FROM sensor_pressure"
)):
    batches.extend(pa.ipc.open_stream(response.ipc))
table = pa.Table.from_batches(batches)
```

## Live streams and playback

`StreamService.StreamComponents` is a latest-value stream, not a lossless
recording feed. Real-time mode may coalesce intermediate rows under load;
`immediate=true` reduces batching but retains the same latest-value contract.
Use `GetTimeSeries` when every historical row matters.

Fixed-rate mode samples each component at or before a shared playback cursor.
The server returns a `stream_id` and emits `StreamTimestamp` frames with the
sampled component updates. Subsequent request frames may pause, resume, seek,
or change the timestep and frequency:

```python
from elodin.db.v1 import stream_pb2

open_request = stream_pb2.StreamComponentsRequest(
    open=stream_pb2.StreamOpen(
        components=["sensor.pressure"],
        fixed_rate=stream_pb2.FixedRate(
            initial=stream_pb2.INITIAL_TIMESTAMP_EARLIEST,
            timestep_ns=20_000_000,
            frequency=50,
        ),
    )
)
pause = stream_pb2.StreamComponentsRequest(
    control=stream_pb2.StreamControl(playing=False)
)
```

The first request must be `StreamOpen`; controls are valid only for a stream
that owns a fixed-rate clock. Timestep must be at least 1,000 ns and frequency
must be 1–1,000 Hz. An invalid control terminates the stream with
`INVALID_ARGUMENT`.

`StreamMessages` can own an independent playback clock or attach to a component
stream via `playback_stream_id`. An attached stream mirrors pauses, seeks, and
rate changes, ends with the owner, and is controlled through the component
stream. `WatchDb` emits the current timestamp/config followed by changes.

## Message logs

Variable-length events, logs, and media use `MessageService`:

1. `Register` a name as opaque bytes, structured log, or postcard schema;
2. open `Publish` with a stable client identity;
3. send sequenced `PublishBatch` frames and process `MessageError`/`WriteAck`;
4. read a lossless historical range with `GetMessages`.

Message publish resumes across server restarts. As with telemetry, clients must
replay above `PublishAccept.resume_from_seq` and tolerate duplicates from an
ambiguous failure. `GetMessages` uses half-open nanosecond ranges and streams
payloads without buffering the full log.

Live message delivery is exposed through `StreamService.StreamMessages`.
Real-time mode first primes each selected log with its latest stored message,
then delivers every append while connected. Reconnecting primes again, so
consumers must tolerate that duplicate.

## Configuration, metadata, and assets

`AdminService` exposes online-safe administration:

- `GetDbConfig` and `SetDbConfig` read or change recording and metadata;
- `SetComponentMetadata` updates component metadata;
- `PutAsset` accepts a header frame followed by byte chunks;
- `GetAsset` streams chunks back;
- `ListAssets` lists keys by prefix.

Send the `PutAssetHeader` first. Asset chunk boundaries are transport details,
not file boundaries.

## Authentication, errors, and limits

The server is unauthenticated by default. Add a static bearer token:

```sh
elodin-db run 127.0.0.1:2240 ./db \
  --grpc-addr 127.0.0.1:50051 \
  --grpc-auth-token "$TOKEN"
```

Pass `authorization: Bearer TOKEN` in call metadata. Authentication covers
application RPCs and reflection; standard health checks stay unauthenticated
for load balancers.

The current transport is plaintext. Bind only to loopback or a trusted network,
or terminate TLS in deployment infrastructure. The server limits encoded and
decoded messages to 16 MiB; `GetServerInfo.max_message_size_bytes` reports the
active contract.

Terminal failures use canonical gRPC status codes. Common failures also carry
`google.rpc.ErrorInfo` details under domain `db.elodin.systems`, with stable
reasons such as `COMPONENT_NOT_FOUND`, `MESSAGE_NOT_FOUND`, and
`TIME_RANGE_EMPTY`. Branch on status and structured reasons, not human-readable
error text.

## Runnable references

- `libs/db/examples/grpc-client-batched.cpp`: packed C++ ingest with concurrent
  writes and acknowledgements;
- `libs/db/examples/grpc_gse_client.py`: typed ingest, structured logs, and
  restart/resume;
- `libs/db/examples/grpc_full_api_demo.py`: all services, health, reflection,
  auth, playback synchronization, SQL, downsampling, and chunked assets.

Run the complete integration demo from the repository root:

```sh
nix develop .#run --command scripts/ci/db_grpc_full_api_demo.sh
```
