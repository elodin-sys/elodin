# Elodin DB gRPC ingest benchmark

## Local result

Measured 2026-08-11 UTC on Linux 7.0.0-28-generic x86_64, Intel Core
Ultra 7 155H, rustc 1.95.0, release builds, and localhost. The database ran as
a separate process at nice level 5. Each client sent one row containing 40
contiguous F64 components at 250 Hz for 5 seconds: about 10,000 component
writes/s and one application batch per tick.

Impeller batch:

- 49,840 component writes; 9,862.8 writes/s.
- Send latency p50/p95/p99/max: 9/14/30/33 us.
- Client wall/user/sys: 5.36/0.01/0.03 s.
- DB wall/user/sys: 5.56/0.07/0.26 s.

gRPC packed:

- 50,000 component writes; 9,954.7 writes/s.
- Send latency p50/p95/p99/max: 3/8/9/10 us.
- Cumulative-ack latency p50/p95/p99/max: 91/263/370/627 us.
- Client wall/user/sys: 5.33/0.06/0.08 s.
- DB wall/user/sys: 5.51/0.07/0.14 s.

`send_latency` covers payload construction and the client send/enqueue call.
gRPC `ack_latency` is enqueue to observing the cumulative `WriteAck`. Neither
is a link RTT measurement.

Same-process packed/typed comparison on the same 40×250 Hz workload:

- Packed: 9,955.7 writes/s; send p50/p95/p99/max 3/11/14/15 us; ack
  195/375/502/769 us; wall/user/sys 5.33/0.29/0.24 s.
- Typed: 9,955.9 writes/s; send p50/p95/p99/max 7/9/10/11 us; ack
  132/378/472/810 us; wall/user/sys 5.33/0.30/0.24 s.

This short x86 run shows functional throughput and similar process CPU for the
two encodings. It does not establish a typed wire-size or embedded-CPU budget.

## Reproduce

```bash
nix develop --command env DB_BENCH_DURATION=5 scripts/ci/db_grpc_compare.sh

nix develop --command target/release/elodin-db-bench \
  --components 40 --frequency 250 --duration 5 --clients 1 \
  --mode grpc-typed --json
```

Comparison output includes benchmark JSON and separate client/DB wall/user/sys
times. Set `DB_BENCH_CAPTURE=1` for a loopback pcap, then inspect with:

```bash
python3 scripts/ci/db_grpc_packet_shape.py capture.pcap \
  --port 2242 --mss 1460
```

This host lacked `CAP_NET_RAW`/`CAP_NET_ADMIN`, so no packet-shape or netem
result is claimed here.

## Write shape and provisional default

One gRPC `Write` per complete `TelemetryBatch` is the default; do not cork that
single write. Use C++ `WriteOptions::set_corked()` only when splitting one
logical batch across multiple writes, and leave the final write uncorked.
`GRPC_ARG_HTTP2_WRITE_BUFFER_SIZE` sizes the HTTP/2 write buffer and does not
replace application batching.

Packed is the provisional default for high-rate and constrained-link ingest.
Typed remains supported once it meets the target CPU and captured wire-size
budgets.

## Remaining external gates

1. Privileged loopback netem + pcap via `db_grpc_netem_gate.sh`.
2. Remote-link pcap comparing packed vs typed payload bytes and ack latency.

```bash
ELODIN_RUN_NETEM=1 DB_BENCH_DURATION=60 \
  nix develop --command scripts/ci/db_grpc_netem_gate.sh
```

The gate refuses to replace an existing loopback qdisc and removes only the
netem qdisc it installed.

Remote DB:

```bash
target/release/elodin-db run 0.0.0.0:2240 "$DB_PATH"
```

Capture on the client or server host:

```bash
sudo tcpdump -i "$IFACE" -nn -s 0 -w grpc-link.pcap 'tcp port 2242'
```

From the client, run each encoding against that endpoint on a fresh DB path:

```bash
target/release/elodin-db-bench \
  --components 40 --frequency 250 --duration 60 --clients 1 \
  --mode grpc-packed --db-addr "$DB_HOST:2240" \
  --grpc-addr "$DB_HOST:2242" --json

target/release/elodin-db-bench \
  --components 40 --frequency 250 --duration 60 --clients 1 \
  --mode grpc-typed --db-addr "$DB_HOST:2240" \
  --grpc-addr "$DB_HOST:2242" --json
```

Require p99 ack below 50 ms at the target load, measure DB and client CPU
separately, and compare captured payload bytes with `db_grpc_packet_shape.py`
before selecting typed.
