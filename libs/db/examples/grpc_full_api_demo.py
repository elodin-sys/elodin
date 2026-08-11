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


def query_component(query, component, call_metadata, max_points=0):
    responses = list(
        query.GetTimeSeries(
            query_pb2.GetTimeSeriesRequest(
                component=component,
                start_ns=-(2**63),
                max_points=max_points,
            ),
            metadata=call_metadata,
        )
    )
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
            msg_pb2.GetMessagesRequest(name="grpc.demo.log", start_ns=-(2**63)),
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
    config = stub.GetDbConfig(admin_pb2.GetDbConfigRequest(), metadata=call_metadata).config
    config = stub.SetDbConfig(
        admin_pb2.SetDbConfigRequest(metadata={"grpc.demo": "true"}),
        metadata=call_metadata,
    ).config
    if config.metadata["grpc.demo"] != "true":
        raise RuntimeError("config patch was not applied")
    asset = b"Elodin gRPC full API demo\n"
    stored = stub.PutAsset(
        iter(
            [
                admin_pb2.PutAssetRequest(
                    header=admin_pb2.PutAssetHeader(key="grpc-demo/probe.txt")
                ),
                admin_pb2.PutAssetRequest(data=asset),
            ]
        ),
        metadata=call_metadata,
    )
    received = b"".join(
        response.data
        for response in stub.GetAsset(
            admin_pb2.GetAssetRequest(key="grpc-demo/probe.txt"),
            metadata=call_metadata,
        )
    )
    if received != asset:
        raise RuntimeError("asset round-trip mismatch")
    assets = stub.ListAssets(
        admin_pb2.ListAssetsRequest(prefix="grpc-demo/"), metadata=call_metadata
    )
    if not any(item.key == "grpc-demo/probe.txt" for item in assets.assets):
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
    return stored.assets_revision


def exercise_streams(channel, call_metadata, component, timestamp):
    stub = stream_pb2_grpc.StreamServiceStub(channel)
    requests = iter(
        [
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
            ),
            stream_pb2.StreamComponentsRequest(
                control=stream_pb2.StreamControl(playing=False, seek_ns=timestamp)
            ),
        ]
    )
    responses = stub.StreamComponents(requests, metadata=call_metadata)
    seen = set()
    stream_id = None
    for response in responses:
        kind = response.WhichOneof("response")
        seen.add(kind)
        if kind == "opened":
            stream_id = response.opened.stream_id
        if {"header", "update", "timestamp", "opened"} <= seen:
            responses.cancel()
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
    )
    if next(message_stream).name != "grpc.demo.log":
        raise RuntimeError("message stream did not bind to component playback")
    message_stream.cancel()
    events = stub.WatchDb(stream_pb2.WatchDbRequest(), metadata=call_metadata)
    initial = {next(events).WhichOneof("event"), next(events).WhichOneof("event")}
    events.cancel()
    if initial != {"last_updated_ns", "config"}:
        raise RuntimeError(f"incomplete DB watch: {initial}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--token")
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
            msg_pb2.GetMessagesRequest(
                name="bdx.fpv_cam",
                start_ns=-(2**63),
                limit=1,
            ),
            metadata=call_metadata,
        )
    )
    if not camera_messages:
        raise RuntimeError("headless renderer did not record an FPV frame")

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
    revision = exercise_admin(channel, call_metadata)
    exercise_streams(channel, call_metadata, "grpc.demo.ingested", ingested_timestamp)

    print(
        json.dumps(
            {
                "components": len(metadata_snapshot.components),
                "control_rows": len(control_timestamps),
                "camera_messages": len(camera_messages),
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
