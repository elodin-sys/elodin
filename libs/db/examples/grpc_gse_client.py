#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import queue
import time
from pathlib import Path

import grpc
from elodin.db.v1 import (
    common_pb2,
    ingest_pb2,
    ingest_pb2_grpc,
    msg_pb2,
    msg_pb2_grpc,
    query_pb2,
    query_pb2_grpc,
)

ROWS_BY_PHASE = {"write1": 500, "write2": 1000}
LOGS_BY_PHASE = {"write1": 4, "write2": 8}
LOGS = [
    (2, "FSW v2.4.1 starting up"),
    (2, "Board: Aleph Orin NX rev3"),
    (1, "Loading flight parameters from EEPROM"),
    (2, "System ready, entering IDLE state"),
    (2, "State: ARMED"),
    (2, "Altitude: 1024m AGL, max-Q passed"),
    (3, "Battery voltage low: 11.2V"),
    (4, "Simulated anomaly: sensor timeout on ADC ch3"),
]


def requests_from(items):
    while True:
        item = items.get()
        if item is None:
            return
        yield item


def metadata(token):
    return (("authorization", f"Bearer {token}"),) if token else ()


def state_file(directory):
    return directory / "state.json"


def load_state(directory, create):
    path = state_file(directory)
    if path.exists():
        return json.loads(path.read_text())
    if not create:
        raise RuntimeError(f"GSE state does not exist: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "client_instance_id": os.urandom(16).hex(),
        "base_timestamp_ns": time.monotonic_ns(),
    }
    path.write_text(json.dumps(state))
    return state


def schema_set():
    return ingest_pb2.SchemaSet(
        messages=[
            ingest_pb2.MessageSchema(
                name="grpc.gse.row",
                encoding=ingest_pb2.ROW_ENCODING_TYPED,
                components=[
                    ingest_pb2.ComponentSchema(
                        name="grpc.gse.scalar",
                        prim_type=common_pb2.PRIM_TYPE_F64,
                    ),
                    ingest_pb2.ComponentSchema(
                        name="grpc.gse.vector",
                        prim_type=common_pb2.PRIM_TYPE_F32,
                        dims=[3],
                        element_names=["x", "y", "z"],
                    ),
                    ingest_pb2.ComponentSchema(
                        name="grpc.gse.enabled",
                        prim_type=common_pb2.PRIM_TYPE_BOOL,
                    ),
                    ingest_pb2.ComponentSchema(
                        name="grpc.gse.counter",
                        prim_type=common_pb2.PRIM_TYPE_U32,
                    ),
                ],
            )
        ]
    )


def typed_row(handle, sequence, base_timestamp_ns):
    value = float(sequence)
    return ingest_pb2.Row(
        message_handle=handle,
        time_monotonic_ns=base_timestamp_ns + sequence * 1_000_000,
        typed=ingest_pb2.TypedValues(
            values=[
                common_pb2.ComponentValue(component_index=0, f64=value * 0.5),
                common_pb2.ComponentValue(
                    component_index=1,
                    f32s=common_pb2.FloatArray(v=[value, value + 0.25, value + 0.5]),
                ),
                common_pb2.ComponentValue(component_index=2, b=sequence % 2 == 0),
                common_pb2.ComponentValue(component_index=3, u64=sequence),
            ]
        ),
    )


def wait_for_ack(responses, sequence):
    for response in responses:
        if response.HasField("error"):
            raise RuntimeError(f"row {response.error.seq}: {response.error.detail}")
        if response.HasField("ack") and response.ack.through_seq >= sequence:
            return
    raise RuntimeError(f"ingest closed before acknowledging {sequence}")


def write_telemetry(channel, call_metadata, state, target_rows):
    schema = schema_set()
    items = queue.Queue()
    responses = ingest_pb2_grpc.IngestServiceStub(channel).Ingest(
        requests_from(items), metadata=call_metadata
    )
    items.put(
        ingest_pb2.IngestRequest(
            open=ingest_pb2.SessionOpen(
                client_name="grpc-gse-client",
                client_instance_id=bytes.fromhex(state["client_instance_id"]),
                schema=schema,
                schema_fingerprint=hashlib.sha256(
                    schema.SerializeToString(deterministic=True)
                ).digest(),
                ack_policy=common_pb2.AckPolicy(max_unacked_rows=25),
            )
        )
    )
    accepted = next(responses)
    if not accepted.HasField("accept"):
        raise RuntimeError(f"ingest session rejected: {accepted}")
    resume = accepted.accept.resume_from_seq
    if resume > target_rows:
        raise RuntimeError(f"server resume {resume} exceeds target {target_rows}")
    handle = accepted.accept.message_handles["grpc.gse.row"]
    for first in range(resume + 1, target_rows + 1, 25):
        last = min(first + 24, target_rows)
        items.put(
            ingest_pb2.IngestRequest(
                batch=ingest_pb2.TelemetryBatch(
                    first_seq=first,
                    rows=[
                        typed_row(handle, sequence, state["base_timestamp_ns"])
                        for sequence in range(first, last + 1)
                    ],
                )
            )
        )
        wait_for_ack(responses, last)
    items.put(None)
    return resume


def write_logs(channel, call_metadata, state, target_logs):
    stub = msg_pb2_grpc.MessageServiceStub(channel)
    handle = stub.Register(
        msg_pb2.RegisterRequest(name="grpc.gse.log", log=msg_pb2.LogKind()),
        metadata=call_metadata,
    ).message_handle
    items = queue.Queue()
    responses = stub.Publish(requests_from(items), metadata=call_metadata)
    items.put(
        msg_pb2.PublishRequest(
            open=msg_pb2.PublishOpen(
                client_name="grpc-gse-client",
                client_instance_id=bytes.fromhex(state["client_instance_id"]),
                ack_policy=common_pb2.AckPolicy(max_unacked_rows=1),
            )
        )
    )
    accepted = next(responses)
    if not accepted.HasField("accept"):
        raise RuntimeError("message publish session was not accepted")
    resume = accepted.accept.resume_from_seq
    if resume < target_logs:
        items.put(
            msg_pb2.PublishRequest(
                batch=msg_pb2.PublishBatch(
                    first_seq=resume + 1,
                    messages=[
                        msg_pb2.OutgoingMessage(
                            message_handle=handle,
                            timestamp_ns=state["base_timestamp_ns"] + sequence * 10_000_000,
                            log=msg_pb2.LogPayload(
                                level=LOGS[sequence - 1][0],
                                message=LOGS[sequence - 1][1],
                            ),
                        )
                        for sequence in range(resume + 1, target_logs + 1)
                    ],
                )
            )
        )
        wait_for_ack(responses, target_logs)
    items.put(None)
    return resume


def component_count(query, call_metadata, name):
    return sum(
        len(response.data.timestamps_ns)
        for response in query.GetTimeSeries(
            query_pb2.GetTimeSeriesRequest(component=name),
            metadata=call_metadata,
        )
        if response.HasField("data")
    )


def verify(channel, call_metadata):
    query = query_pb2_grpc.QueryServiceStub(channel)
    counts = {
        name: component_count(query, call_metadata, name)
        for name in [
            "grpc.gse.scalar",
            "grpc.gse.vector",
            "grpc.gse.enabled",
            "grpc.gse.counter",
        ]
    }
    if set(counts.values()) != {ROWS_BY_PHASE["write2"]}:
        raise RuntimeError(f"unexpected GSE component counts: {counts}")
    logs = list(
        msg_pb2_grpc.MessageServiceStub(channel).GetMessages(
            msg_pb2.GetMessagesRequest(name="grpc.gse.log"),
            metadata=call_metadata,
        )
    )
    if [entry.log.message for entry in logs] != [message for _, message in LOGS]:
        raise RuntimeError("GSE logs did not survive restart exactly")
    return {"gse_rows": next(iter(counts.values())), "gse_logs": len(logs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["write1", "write2", "verify"], required=True)
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--token")
    args = parser.parse_args()
    state = load_state(args.state_dir, args.phase == "write1")
    channel = grpc.insecure_channel(args.target)
    grpc.channel_ready_future(channel).result(timeout=10)
    call_metadata = metadata(args.token)
    if args.phase == "verify":
        result = verify(channel, call_metadata)
    else:
        result = {
            "phase": args.phase,
            "ingest_resume": write_telemetry(
                channel, call_metadata, state, ROWS_BY_PHASE[args.phase]
            ),
            "message_resume": write_logs(channel, call_metadata, state, LOGS_BY_PHASE[args.phase]),
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
