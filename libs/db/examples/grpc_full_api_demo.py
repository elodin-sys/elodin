#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import queue
import struct
import time

import grpc
import pyarrow as pa
from elodin.db.v1 import (
    admin_pb2,
    admin_pb2_grpc,
    common_pb2,
    ingest_pb2,
    ingest_pb2_grpc,
    msg_pb2,
    msg_pb2_grpc,
    query_pb2,
    query_pb2_grpc,
    stream_pb2,
    stream_pb2_grpc,
)
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc


def requests_from(items):
    while True:
        item = items.get()
        if item is None:
            return
        yield item


def metadata(token):
    return (("authorization", f"Bearer {token}"),) if token else ()


def query_component(query, component, call_metadata, max_points=None, element_index=0):
    request = query_pb2.GetTimeSeriesRequest(component=component, element_index=element_index)
    if max_points is not None:
        request.max_points = max_points
    responses = list(query.GetTimeSeries(request, metadata=call_metadata))
    header = next(response.header for response in responses if response.HasField("header"))
    timestamps = [
        timestamp
        for response in responses
        if response.HasField("data")
        for timestamp in response.data.timestamps_ns
    ]
    values = b"".join(
        response.data.packed_values for response in responses if response.HasField("data")
    )
    return header, timestamps, values


def decode_component_rows(header, values):
    formats = {
        common_pb2.PRIM_TYPE_U8: "B",
        common_pb2.PRIM_TYPE_U16: "H",
        common_pb2.PRIM_TYPE_U32: "I",
        common_pb2.PRIM_TYPE_U64: "Q",
        common_pb2.PRIM_TYPE_I8: "b",
        common_pb2.PRIM_TYPE_I16: "h",
        common_pb2.PRIM_TYPE_I32: "i",
        common_pb2.PRIM_TYPE_I64: "q",
        common_pb2.PRIM_TYPE_BOOL: "?",
        common_pb2.PRIM_TYPE_F32: "f",
        common_pb2.PRIM_TYPE_F64: "d",
    }
    code = formats.get(header.prim_type)
    if code is None:
        raise RuntimeError(f"unsupported primitive type: {header.prim_type}")
    elements = 1
    for dimension in header.dims:
        elements *= dimension
    row = struct.Struct("<" + code * elements)
    if len(values) % row.size:
        raise RuntimeError("component payload is not row-aligned")
    return [row.unpack_from(values, offset) for offset in range(0, len(values), row.size)]


def exercise_ingest(channel, call_metadata):
    stub = ingest_pb2_grpc.IngestServiceStub(channel)
    schema = ingest_pb2.SchemaSet(
        messages=[
            ingest_pb2.MessageSchema(
                name="grpc.demo.row",
                encoding=ingest_pb2.ROW_ENCODING_PACKED,
                packed_size=8,
                components=[
                    ingest_pb2.ComponentSchema(
                        name="grpc.demo.ingested",
                        prim_type=common_pb2.PRIM_TYPE_F64,
                    )
                ],
            )
        ]
    )
    items = queue.Queue()
    responses = stub.Ingest(requests_from(items), metadata=call_metadata)
    items.put(
        ingest_pb2.IngestRequest(
            open=ingest_pb2.SessionOpen(
                client_name="grpc-full-api-demo",
                client_instance_id=os.urandom(16),
                schema=schema,
                schema_fingerprint=hashlib.sha256(
                    schema.SerializeToString(deterministic=True)
                ).digest(),
                ack_policy=common_pb2.AckPolicy(max_unacked_rows=1),
            )
        )
    )
    accepted = next(responses)
    if not accepted.HasField("accept"):
        raise RuntimeError(f"ingest rejected: {accepted}")
    handle = accepted.accept.message_handles["grpc.demo.row"]
    timestamp = time.monotonic_ns()
    items.put(
        ingest_pb2.IngestRequest(
            batch=ingest_pb2.TelemetryBatch(
                first_seq=1,
                rows=[
                    ingest_pb2.Row(
                        message_handle=handle,
                        time_monotonic_ns=timestamp,
                        packed=struct.pack("<d", 42.5),
                    )
                ],
            )
        )
    )
    items.put(None)
    if not any(
        response.HasField("ack") and response.ack.through_seq == 1 for response in responses
    ):
        raise RuntimeError("ingest row was not acknowledged")
    return timestamp


def exercise_messages(channel, call_metadata):
    stub = msg_pb2_grpc.MessageServiceStub(channel)
    handle = stub.Register(
        msg_pb2.RegisterRequest(name="grpc.demo.log", log=msg_pb2.LogKind()),
        metadata=call_metadata,
    ).message_handle
    items = queue.Queue()
    responses = stub.Publish(requests_from(items), metadata=call_metadata)
    items.put(
        msg_pb2.PublishRequest(
            open=msg_pb2.PublishOpen(
                client_name="grpc-full-api-demo",
                client_instance_id=os.urandom(16),
                ack_policy=common_pb2.AckPolicy(max_unacked_rows=1),
            )
        )
    )
    if not next(responses).HasField("accept"):
        raise RuntimeError("message publish session was not accepted")
    items.put(
        msg_pb2.PublishRequest(
            batch=msg_pb2.PublishBatch(
                first_seq=1,
                messages=[
                    msg_pb2.OutgoingMessage(
                        message_handle=handle,
                        timestamp_ns=time.monotonic_ns(),
                        log=msg_pb2.LogPayload(level=2, message="gRPC demo"),
                    )
                ],
            )
        )
    )
    items.put(None)
    if not any(
        response.HasField("ack") and response.ack.through_seq == 1 for response in responses
    ):
        raise RuntimeError("message was not acknowledged")
    stored = list(
        stub.GetMessages(
            msg_pb2.GetMessagesRequest(name="grpc.demo.log"),
            metadata=call_metadata,
        )
    )
    if not any(
        message.HasField("log") and message.log.message == "gRPC demo" for message in stored
    ):
        raise RuntimeError("published log was not returned")
    return len(stored)


def exercise_admin(channel, call_metadata):
    stub = admin_pb2_grpc.AdminServiceStub(channel)
    stream_stub = stream_pb2_grpc.StreamServiceStub(channel)
    events = stream_stub.WatchDb(stream_pb2.WatchDbRequest(), metadata=call_metadata, timeout=10)
    initial_events = {next(events).WhichOneof("event"), next(events).WhichOneof("event")}
    if initial_events != {"last_updated_ns", "config"}:
        raise RuntimeError(f"incomplete initial DB watch: {initial_events}")

    config = stub.GetDbConfig(admin_pb2.GetDbConfigRequest(), metadata=call_metadata).config
    config = stub.SetDbConfig(
        admin_pb2.SetDbConfigRequest(metadata={"grpc.demo": "true"}),
        metadata=call_metadata,
    ).config
    if config.metadata["grpc.demo"] != "true":
        raise RuntimeError("config patch was not applied")
    changed = next(events)
    if (
        changed.WhichOneof("event") != "config"
        or changed.config.metadata.get("grpc.demo") != "true"
    ):
        raise RuntimeError("WatchDb did not push the config mutation")
    events.cancel()

    f22 = stub.ListAssets(
        admin_pb2.ListAssetsRequest(prefix="f22.glb"), metadata=call_metadata
    ).assets
    if len(f22) != 1 or f22[0].key != "f22.glb":
        raise RuntimeError("recorded f22.glb asset was not listed")
    f22_chunks = list(
        stub.GetAsset(admin_pb2.GetAssetRequest(key="f22.glb"), metadata=call_metadata)
    )
    if len(f22_chunks) <= 1 or sum(len(chunk.data) for chunk in f22_chunks) != f22[0].size:
        raise RuntimeError("recorded asset did not exercise chunked reads")

    asset_size = 2 * 1024 * 1024 + 123
    pattern = bytes(range(256))
    asset = (pattern * ((asset_size + len(pattern) - 1) // len(pattern)))[:asset_size]

    def put_requests():
        yield admin_pb2.PutAssetRequest(header=admin_pb2.PutAssetHeader(key="grpc-demo/probe.bin"))
        for offset in range(0, len(asset), 512 * 1024):
            yield admin_pb2.PutAssetRequest(data=asset[offset : offset + 512 * 1024])

    stored = stub.PutAsset(
        put_requests(),
        metadata=call_metadata,
    )
    received = b"".join(
        response.data
        for response in stub.GetAsset(
            admin_pb2.GetAssetRequest(key="grpc-demo/probe.bin"),
            metadata=call_metadata,
        )
    )
    if received != asset:
        raise RuntimeError("asset round-trip mismatch")
    assets = stub.ListAssets(
        admin_pb2.ListAssetsRequest(prefix="grpc-demo/"), metadata=call_metadata
    )
    if not any(item.key == "grpc-demo/probe.bin" for item in assets.assets):
        raise RuntimeError("asset was absent from listing")
    stub.SetComponentMetadata(
        admin_pb2.SetComponentMetadataRequest(
            metadata=common_pb2.ComponentMetadata(
                name="grpc.demo.ingested", metadata={"source": "grpc-demo"}
            )
        ),
        metadata=call_metadata,
    )
    stub.SetDbConfig(
        admin_pb2.SetDbConfigRequest(metadata={"grpc.demo": ""}),
        metadata=call_metadata,
    )
    return stored.assets_revision, len(f22_chunks), len(asset)


def exercise_numeric_sql(query, call_metadata, expected_rows):
    tables = [
        pa.ipc.open_stream(pa.py_buffer(response.ipc)).read_all()
        for response in query.Sql(
            query_pb2.SqlRequest(
                sql=(
                    "SELECT bdx_world_pos.bdx_world_pos[5] AS x, "
                    "bdx_world_pos.bdx_world_pos[6] AS y FROM bdx_world_pos"
                )
            ),
            metadata=call_metadata,
        )
    ]
    rows = sum(table.num_rows for table in tables)
    if rows != expected_rows:
        raise RuntimeError(f"ground-track SQL returned {rows} rows, expected {expected_rows}")
    if not tables or tables[0].column_names != ["x", "y"]:
        raise RuntimeError("ground-track SQL returned an unexpected schema")
    return rows


def exercise_vector_downsample(query, call_metadata):
    header, timestamps, values = query_component(query, "bdx.world_pos", call_metadata)
    rows = decode_component_rows(header, values)
    if len(header.dims) != 1 or header.dims[0] < 7:
        raise RuntimeError(f"unexpected bdx.world_pos shape: {list(header.dims)}")
    reduced_header, reduced_timestamps, reduced_values = query_component(
        query,
        "bdx.world_pos",
        call_metadata,
        max_points=32,
        element_index=6,
    )
    reduced_rows = decode_component_rows(reduced_header, reduced_values)
    if len(reduced_timestamps) != 32:
        raise RuntimeError("vector-element downsampling did not return 32 points")
    if reduced_timestamps[0] != timestamps[0] or reduced_timestamps[-1] != timestamps[-1]:
        raise RuntimeError("vector-element downsampling did not preserve endpoints")
    altitude = [row[6] for row in rows]
    reduced_altitude = [row[6] for row in reduced_rows]
    if min(reduced_altitude) < min(altitude) or max(reduced_altitude) > max(altitude):
        raise RuntimeError("downsampled altitude escaped the raw value range")
    return len(timestamps), len(reduced_timestamps)


def exercise_streams(channel, call_metadata, component, timestamp):
    stub = stream_pb2_grpc.StreamServiceStub(channel)
    component_requests = queue.Queue()
    responses = stub.StreamComponents(
        requests_from(component_requests), metadata=call_metadata, timeout=15
    )
    component_requests.put(
        stream_pb2.StreamComponentsRequest(
            open=stream_pb2.StreamOpen(
                components=[component],
                fixed_rate=stream_pb2.FixedRate(
                    initial=stream_pb2.INITIAL_TIMESTAMP_MANUAL,
                    initial_timestamp_ns=timestamp,
                    timestep_ns=1_000_000,
                    frequency=100,
                ),
            )
        )
    )
    component_requests.put(
        stream_pb2.StreamComponentsRequest(
            control=stream_pb2.StreamControl(playing=False, seek_ns=timestamp)
        )
    )
    seen = set()
    stream_id = None
    for response in responses:
        kind = response.WhichOneof("response")
        seen.add(kind)
        if kind == "opened":
            stream_id = response.opened.stream_id
        if {"header", "update", "timestamp", "opened"} <= seen:
            break
    if not {"header", "update", "timestamp", "opened"} <= seen:
        raise RuntimeError(f"incomplete component stream: {seen}")
    message_stream = stub.StreamMessages(
        iter(
            [
                stream_pb2.StreamMessagesRequest(
                    open=stream_pb2.MessageStreamOpen(
                        messages=["grpc.demo.log"],
                        fixed_rate=stream_pb2.FixedRate(
                            initial=stream_pb2.INITIAL_TIMESTAMP_MANUAL,
                            initial_timestamp_ns=timestamp,
                            timestep_ns=1_000_000,
                            frequency=100,
                        ),
                        playback_stream_id=stream_id,
                    )
                )
            ]
        ),
        metadata=call_metadata,
        timeout=15,
    )
    # The demo log was published after `timestamp`, so the paused shared clock
    # has no message at or before its cursor yet; seek the owning component
    # stream forward and the follower emits the log.
    component_requests.put(
        stream_pb2.StreamComponentsRequest(
            control=stream_pb2.StreamControl(seek_ns=time.monotonic_ns())
        )
    )
    if next(message_stream).name != "grpc.demo.log":
        raise RuntimeError("message stream did not bind to component playback")
    message_stream.cancel()
    responses.cancel()
    component_requests.put(None)
    events = stub.WatchDb(stream_pb2.WatchDbRequest(), metadata=call_metadata, timeout=15)
    initial = {next(events).WhichOneof("event"), next(events).WhichOneof("event")}
    events.cancel()
    if initial != {"last_updated_ns", "config"}:
        raise RuntimeError(f"incomplete DB watch: {initial}")


def exercise_recorded_playback(channel, call_metadata, camera_messages):
    stub = stream_pb2_grpc.StreamServiceStub(channel)
    first_frame_ns = camera_messages[0].timestamp_ns
    second_frame_ns = camera_messages[1].timestamp_ns
    component_requests = queue.Queue()
    component_responses = stub.StreamComponents(
        requests_from(component_requests), metadata=call_metadata, timeout=15
    )
    component_requests.put(
        stream_pb2.StreamComponentsRequest(
            open=stream_pb2.StreamOpen(
                components=["bdx.world_pos"],
                fixed_rate=stream_pb2.FixedRate(
                    initial=stream_pb2.INITIAL_TIMESTAMP_MANUAL,
                    initial_timestamp_ns=first_frame_ns,
                    timestep_ns=1_000_000,
                    frequency=100,
                ),
            )
        )
    )
    component_requests.put(
        stream_pb2.StreamComponentsRequest(
            control=stream_pb2.StreamControl(
                playing=False,
                seek_ns=first_frame_ns,
            )
        )
    )
    stream_id = None
    latest_update = None

    def wait_for_component_clock(target_ns):
        nonlocal stream_id, latest_update
        for _ in range(64):
            response = next(component_responses)
            kind = response.WhichOneof("response")
            if kind == "opened":
                stream_id = response.opened.stream_id
            elif kind == "update":
                latest_update = response.update
            elif kind == "timestamp" and response.timestamp.timestamp_ns == target_ns:
                if latest_update is None or stream_id is None:
                    continue
                if latest_update.timestamp_ns > target_ns:
                    raise RuntimeError("component playback advanced past its clock")
                return
        raise RuntimeError(f"component playback did not reach {target_ns}")

    wait_for_component_clock(first_frame_ns)
    message_requests = queue.Queue()
    message_responses = stub.StreamMessages(
        requests_from(message_requests), metadata=call_metadata, timeout=15
    )
    message_requests.put(
        stream_pb2.StreamMessagesRequest(
            open=stream_pb2.MessageStreamOpen(
                messages=["bdx.fpv_cam"],
                fixed_rate=stream_pb2.FixedRate(
                    initial=stream_pb2.INITIAL_TIMESTAMP_MANUAL,
                    initial_timestamp_ns=first_frame_ns,
                    timestep_ns=1_000_000,
                    frequency=100,
                ),
                playback_stream_id=stream_id,
            )
        )
    )
    if next(message_responses).timestamp_ns != first_frame_ns:
        raise RuntimeError("FPV playback did not start at the component clock")
    component_requests.put(
        stream_pb2.StreamComponentsRequest(
            control=stream_pb2.StreamControl(
                playing=False,
                seek_ns=second_frame_ns,
            )
        )
    )
    wait_for_component_clock(second_frame_ns)
    if next(message_responses).timestamp_ns != second_frame_ns:
        raise RuntimeError("FPV playback did not follow the component seek")

    component_responses.cancel()
    message_responses.cancel()
    component_requests.put(None)
    message_requests.put(None)
    return second_frame_ns - first_frame_ns


def collect_live_component(stub, call_metadata, component, immediate):
    responses = stub.StreamComponents(
        iter(
            [
                stream_pb2.StreamComponentsRequest(
                    open=stream_pb2.StreamOpen(
                        components=[component],
                        real_time=stream_pb2.RealTime(immediate=immediate),
                    )
                )
            ]
        ),
        metadata=call_metadata,
        timeout=8,
    )
    timestamps = []
    for response in responses:
        if response.WhichOneof("response") != "update":
            continue
        timestamp = response.update.timestamp_ns
        if not timestamps or timestamp > timestamps[-1]:
            timestamps.append(timestamp)
        if len(timestamps) == 3:
            responses.cancel()
            return timestamps
    raise RuntimeError(f"live {component} stream ended before three updates")


def collect_live_camera(stub, call_metadata):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        responses = stub.StreamMessages(
            iter(
                [
                    stream_pb2.StreamMessagesRequest(
                        open=stream_pb2.MessageStreamOpen(
                            messages=["bdx.fpv_cam"],
                            real_time=stream_pb2.RealTime(),
                        )
                    )
                ]
            ),
            metadata=call_metadata,
            timeout=8,
        )
        try:
            timestamps = []
            for response in responses:
                if not timestamps or response.timestamp_ns > timestamps[-1]:
                    timestamps.append(response.timestamp_ns)
                if len(timestamps) == 2:
                    responses.cancel()
                    return timestamps
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.NOT_FOUND:
                raise
        time.sleep(0.05)
    raise RuntimeError("live FPV stream did not produce two frames")


def exercise_live(channel, call_metadata):
    query = query_pb2_grpc.QueryServiceStub(channel)
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        schema = query.DumpSchema(query_pb2.DumpSchemaRequest(), metadata=call_metadata)
        names = {component.name for component in schema.components}
        if {"bdx.world_pos", "bdx.mach"} <= names:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("live RC-jet schema did not become available")

    stream = stream_pb2_grpc.StreamServiceStub(channel)
    batched = collect_live_component(stream, call_metadata, "bdx.world_pos", False)
    immediate = collect_live_component(stream, call_metadata, "bdx.mach", True)
    frames = collect_live_camera(stream, call_metadata)
    events = stream.WatchDb(stream_pb2.WatchDbRequest(), metadata=call_metadata, timeout=8)
    last_updated = []
    for event in events:
        if event.WhichOneof("event") != "last_updated_ns":
            continue
        timestamp = event.last_updated_ns
        if not last_updated or timestamp > last_updated[-1]:
            last_updated.append(timestamp)
        if len(last_updated) == 3:
            events.cancel()
            break
    if len(last_updated) != 3:
        raise RuntimeError("WatchDb did not advance with the live simulation")
    return {
        "live_batched_updates": len(batched),
        "live_immediate_updates": len(immediate),
        "live_camera_frames": len(frames),
        "live_watch_updates": len(last_updated),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--token")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    channel = grpc.insecure_channel(args.target)
    grpc.channel_ready_future(channel).result(timeout=10)
    call_metadata = metadata(args.token)

    if args.token:
        unauthenticated = query_pb2_grpc.QueryServiceStub(channel)
        try:
            unauthenticated.GetTimeRange(query_pb2.GetTimeRangeRequest())
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.UNAUTHENTICATED:
                raise
        else:
            raise RuntimeError("server accepted a request without its bearer token")

    health = health_pb2_grpc.HealthStub(channel).Check(
        health_pb2.HealthCheckRequest(service="elodin.db.v1.QueryService"),
        metadata=call_metadata,
    )
    if health.status != health_pb2.HealthCheckResponse.SERVING:
        raise RuntimeError("QueryService is not serving")
    reflection = reflection_pb2_grpc.ServerReflectionStub(channel)
    reflected = next(
        reflection.ServerReflectionInfo(
            iter([reflection_pb2.ServerReflectionRequest(list_services="")]),
            metadata=call_metadata,
        )
    )
    services = [service.name for service in reflected.list_services_response.service]
    expected = {
        "elodin.db.v1.IngestService",
        "elodin.db.v1.QueryService",
        "elodin.db.v1.StreamService",
        "elodin.db.v1.MessageService",
        "elodin.db.v1.AdminService",
    }
    if not expected <= set(services):
        raise RuntimeError(f"reflection missing services: {expected - set(services)}")
    if args.live:
        print(json.dumps(exercise_live(channel, call_metadata), sort_keys=True))
        return

    query = query_pb2_grpc.QueryServiceStub(channel)
    time_range = query.GetTimeRange(query_pb2.GetTimeRangeRequest(), metadata=call_metadata)
    metadata_snapshot = query.DumpMetadata(query_pb2.DumpMetadataRequest(), metadata=call_metadata)
    schema_snapshot = query.DumpSchema(query_pb2.DumpSchemaRequest(), metadata=call_metadata)
    if not time_range.has_data or len(schema_snapshot.components) < 10:
        raise RuntimeError("rc-jet database did not contain the expected telemetry")
    _, control_timestamps, _ = query_component(query, "bdx.control_commands", call_metadata)
    if len(control_timestamps) < 100:
        raise RuntimeError("RC controller did not produce a useful data set")
    if "sensor_cameras" not in metadata_snapshot.config.metadata:
        raise RuntimeError("RC-jet sensor camera config is missing")
    message_query = msg_pb2_grpc.MessageServiceStub(channel)
    camera_messages = list(
        message_query.GetMessages(
            msg_pb2.GetMessagesRequest(name="bdx.fpv_cam", limit=2),
            metadata=call_metadata,
        )
    )
    if len(camera_messages) < 2:
        raise RuntimeError("headless renderer did not record two FPV frames")
    world_rows, vector_points = exercise_vector_downsample(query, call_metadata)
    playback_seek_ns = exercise_recorded_playback(channel, call_metadata, camera_messages)
    ground_track_rows = exercise_numeric_sql(query, call_metadata, world_rows)

    selected = None
    for schema in schema_snapshot.components:
        try:
            header, timestamps, values = query_component(query, schema.name, call_metadata)
        except grpc.RpcError as error:
            if error.code() == grpc.StatusCode.OUT_OF_RANGE:
                continue
            raise
        if len(timestamps) >= 8:
            selected = (header, timestamps, values)
            break
    if selected is None:
        raise RuntimeError("no populated rc-jet component found")
    header, timestamps, values = selected
    _, reduced, _ = query_component(query, header.component, call_metadata, max_points=4)
    if len(reduced) != 4 or not values:
        raise RuntimeError("time-series downsampling failed")

    sql_rows = 0
    for response in query.Sql(
        query_pb2.SqlRequest(
            sql="select table_name from information_schema.tables where table_schema = 'public'"
        ),
        metadata=call_metadata,
    ):
        sql_rows += pa.ipc.open_stream(pa.py_buffer(response.ipc)).read_all().num_rows
    if sql_rows == 0:
        raise RuntimeError("SQL returned no component tables")

    ingested_timestamp = exercise_ingest(channel, call_metadata)
    query_component(query, "grpc.demo.ingested", call_metadata)
    message_count = exercise_messages(channel, call_metadata)
    revision, asset_chunks, uploaded_asset_bytes = exercise_admin(channel, call_metadata)
    exercise_streams(channel, call_metadata, "grpc.demo.ingested", ingested_timestamp)

    print(
        json.dumps(
            {
                "components": len(metadata_snapshot.components),
                "control_rows": len(control_timestamps),
                "camera_messages": len(camera_messages),
                "playback_seek_ns": playback_seek_ns,
                "vector_points": vector_points,
                "world_rows": world_rows,
                "ground_track_rows": ground_track_rows,
                "asset_chunks": asset_chunks,
                "uploaded_asset_bytes": uploaded_asset_bytes,
                "schemas": len(schema_snapshot.components),
                "selected": header.component,
                "selected_rows": len(timestamps),
                "sql_rows": sql_rows,
                "messages": message_count,
                "assets_revision": revision,
                "auth": bool(args.token),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
