"""Acceptance tests for `elodin.db` (see the AGP feature request document).

1. write/read snippet works against an embedded Server.start
2. write_nowait under a dead DB is cheap and never raises
3. batched writer: one shared-timestamp Table row per write
4. 1M-sample time_series read is fast
5. element_names metadata round-trips (Editor axis labels)
6. all prim types + shapes round-trip
7. data is queryable via SQL (Arrow IPC -> pyarrow)

Plus regression coverage for writer observability (last_error, state),
queue overflow policies, and multi-process writers sharing one database.
"""

import itertools
import shutil
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

import elodin.db as edb

_port = itertools.count(23310)


@pytest.fixture()
def server(tmp_path):
    addr = f"127.0.0.1:{next(_port)}"
    server = edb.Server.start(str(tmp_path / "db"), addr)
    time.sleep(0.3)
    yield server
    server.stop()


@pytest.fixture()
def client(server):
    with edb.Client.connect(server.addr) as client:
        yield client


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except RuntimeError:
            pass  # e.g. component not ingested yet
        time.sleep(0.05)
    return False


# ── acceptance 1: the documented snippet ─────────────────────────────────────
def test_snippet_roundtrip(client):
    writer = client.table_writer(
        {
            "drone.imu.accel": edb.f64[3].labeled("x", "y", "z"),
            "drone.cmd.throttle": edb.f64,
        }
    )
    t0 = 1_000_000
    for i in range(50):
        writer.write(
            timestamp_us=t0 + i * 20_000,
            values={
                "drone.imu.accel": [0.0, 0.0, -9.81 + i],
                "drone.cmd.throttle": i / 50.0,
            },
        )
    assert _wait_for(lambda: len(client.time_series("drone.imu.accel", 0, t0 + 10**6)[0]) == 50)
    ts, accel = client.time_series("drone.imu.accel", t0, t0 + 10**6)
    assert accel.shape == (50, 3)
    assert accel[10][2] == pytest.approx(-9.81 + 10)
    assert ts.dtype == np.int64 and ts[0] == t0
    writer.close()


# ── acceptance 2: write_nowait under a dead DB ───────────────────────────────
@pytest.mark.perf
def test_write_nowait_dead_db():
    client = edb.Client.connect(f"127.0.0.1:{next(_port)}")  # nothing listening
    writer = client.table_writer({"x.v": edb.f64[3]}, maxlen=64)
    values = {"x.v": np.zeros(3)}

    n = 2000
    start = time.perf_counter()
    for i in range(n):
        writer.write_nowait(timestamp_us=i, values=values)
    elapsed = time.perf_counter() - start
    per_call_us = elapsed / n * 1e6

    assert writer.dropped > 0, "rows must be dropped when the DB is unreachable"
    # Acceptance: < 5 us per call. The call is a numpy pack + bounded-queue
    # try_send; no syscalls, no GIL release.
    assert per_call_us < 5.0, f"write_nowait cost {per_call_us:.2f} us/call"
    writer.close()
    client.close()


# ── acceptance 3: batched writer -> one shared-timestamp row per write ───────
def test_batched_row_shares_timestamp(client):
    writer = client.table_writer({"b.a": edb.f64[3], "b.b": edb.f32, "b.c": edb.i64[2]})
    t0 = 5_000_000
    for i in range(20):
        writer.write(
            timestamp_us=t0 + i * 1000,
            values={"b.a": [1, 2, 3], "b.b": 0.5, "b.c": [i, -i]},
        )
    assert _wait_for(lambda: len(client.time_series("b.c", 0, 10**8)[0]) == 20)
    # every component carries the identical timestamp vector -> a single Table
    # row (shared timestamp op) produced all three samples
    ts_a, _ = client.time_series("b.a", 0, 10**8)
    ts_b, _ = client.time_series("b.b", 0, 10**8)
    ts_c, _ = client.time_series("b.c", 0, 10**8)
    assert np.array_equal(ts_a, ts_b) and np.array_equal(ts_b, ts_c)
    assert len(ts_a) == 20
    writer.close()


def test_missing_field_raises(client):
    writer = client.table_writer({"m.a": edb.f64, "m.b": edb.f64})
    with pytest.raises(ValueError, match="missing"):
        writer.write(timestamp_us=1, values={"m.a": 1.0})
    writer.close()


# ── acceptance 4: 1M-sample read performance ─────────────────────────────────
@pytest.mark.perf
@pytest.mark.slow
def test_time_series_1m_read(client):
    # drop-newest: on a full queue the *incoming* row is shed, so retrying the
    # same row until `dropped` stops moving is lossless. (With the default
    # drop-oldest policy a full queue would permanently shed an old row and
    # the retry would enqueue a duplicate instead.)
    # A modest maxlen bounds how far the producer outruns the socket, so at
    # most 8192 rows remain in flight after the loop — little enough for the
    # DB-side wait below to cover even a heavily loaded CI machine.
    writer = client.table_writer({"perf.v": edb.f64}, maxlen=8192, queue="drop-newest")
    n = 1_000_000
    # Establish the connection and component registration before the large
    # fire-and-forget burst so CI does not race writer startup.
    writer.write(timestamp_us=-1, values={"perf.v": -1.0})
    # write_nowait for throughput; retry any row shed by the bounded queue so
    # all n rows land (the server is alive, so drops can only mean queue-full).
    for i in range(n):
        values = {"perf.v": float(i)}
        while True:
            before = writer.dropped
            writer.write_nowait(timestamp_us=i, values=values)
            if writer.dropped == before:
                break
            time.sleep(0.001)
    # Poll the DB until the tail lands. (A blocking sentinel write is *less*
    # robust here: it queues behind the whole nowait backlog and gives up
    # after the writer's internal 10s timeout on a slow CI machine.)
    assert _wait_for(
        lambda: len(client.time_series("perf.v", n - 10, n)[0]) == 10, timeout_s=180
    ), f"tail not ingested; dropped={writer.dropped} last_error={writer.last_error}"
    start = time.perf_counter()
    ts, vals = client.time_series("perf.v", 0, n)
    elapsed = time.perf_counter() - start
    assert len(ts) == n and len(vals) == n
    assert vals[123456] == 123456.0
    assert elapsed < 1.0, f"1M-sample read took {elapsed:.2f}s"
    writer.close()


# ── acceptance 5: element_names metadata (Editor axis labels) ────────────────
def test_element_names_roundtrip(client):
    writer = client.table_writer({"lbl.gyro": edb.f64[3].labeled("p", "q", "r")})
    writer.write(timestamp_us=1, values={"lbl.gyro": [1, 2, 3]})
    assert _wait_for(lambda: "lbl.gyro" in client.components())
    info = client.components()["lbl.gyro"]
    assert info.element_names == ["p", "q", "r"]
    assert info.prim_type == "f64"
    assert list(info.shape) == [3]
    writer.close()


# ── acceptance 6: all prim types + shapes ────────────────────────────────────
@pytest.mark.parametrize(
    "field,values",
    [
        (edb.f64[3], [1.5, -2.5, 3.5]),
        (edb.f32[2], [0.5, -0.25]),
        (edb.i64[2], [2**40, -(2**40)]),
        (edb.i32, -123456),
        (edb.i16[3], [3, -2, 1]),
        (edb.i8[2], [-128, 127]),
        (edb.u64, 2**50),
        (edb.u32[2], [1, 2**31]),
        (edb.u16, 65535),
        (edb.u8[4], [0, 1, 254, 255]),
        (edb.bool_[3], [True, False, True]),
        (edb.f64[2, 2], [[1.0, 2.0], [3.0, 4.0]]),
        (edb.f32[2, 3, 2], np.arange(12, dtype=np.float32).reshape(2, 3, 2)),
    ],
    ids=lambda p: getattr(p, "prim", None) and f"{p.prim}{list(p.shape)}",
)
def test_prim_type_roundtrip(client, field, values):
    name = f"prim.{field.prim}_{'_'.join(map(str, field.shape)) or 's'}"
    writer = client.table_writer({name: field})
    writer.write(timestamp_us=42, values={name: values})
    assert _wait_for(lambda: len(client.time_series(name, 0, 100)[0]) == 1)
    ts, vals = client.time_series(name, 0, 100)
    assert ts[0] == 42
    expected = np.asarray(values, dtype=field.dtype)
    got = vals[0] if field.shape else vals[0]
    np.testing.assert_array_equal(np.asarray(got).reshape(expected.shape), expected)
    assert vals.dtype == field.dtype
    writer.close()


# ── acceptance 7: SQL over Arrow IPC ─────────────────────────────────────────
def test_sql_returns_pyarrow_table(client):
    import pyarrow as pa

    writer = client.table_writer({"sqltest.v": edb.f64})
    for i in range(10):
        writer.write(timestamp_us=1000 + i, values={"sqltest.v": float(i)})
    assert _wait_for(lambda: len(client.time_series("sqltest.v", 0, 10**6)[0]) == 10)
    table = client.sql(f"SELECT * FROM {edb.sql_table_name('sqltest.v')} ORDER BY time")
    assert isinstance(table, pa.Table)
    assert table.num_rows == 10
    assert "time" in table.column_names
    writer.close()


# ── latest-value subscription ────────────────────────────────────────────────
def test_latest_live_value(client):
    writer = client.table_writer({"live.v": edb.f64[2]})
    client.latest("live.v")  # starts the subscription
    time.sleep(0.5)
    for i in range(30):
        writer.write_nowait(timestamp_us=i * 1000, values={"live.v": [i, -i]})
        time.sleep(0.02)
    assert _wait_for(lambda: client.latest("live.v") is not None)
    sample = client.latest("live.v")
    assert sample.values.shape == (2,)
    writer.close()


def test_single_component_send(client):
    client.send("conv.v", [7.0, 8.0], timestamp_us=99)
    assert _wait_for(lambda: len(client.time_series("conv.v", 0, 1000)[0]) == 1)
    _, vals = client.time_series("conv.v", 0, 1000)
    np.testing.assert_array_equal(vals[0], [7.0, 8.0])


def test_single_component_send_creates_one_writer_under_threads(client):
    original_table_writer = client.table_writer
    calls = 0
    calls_lock = threading.Lock()

    def slow_table_writer(*args, **kwargs):
        nonlocal calls
        time.sleep(0.05)
        with calls_lock:
            calls += 1
        return original_table_writer(*args, **kwargs)

    client.table_writer = slow_table_writer
    start = threading.Barrier(8)
    errors = []

    def send_once():
        try:
            start.wait(timeout=5)
            client.send("conv.racy", 1.0, timestamp_us=101)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=send_once) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert calls == 1
    assert _wait_for(lambda: len(client.time_series("conv.racy", 0, 1000)[0]) >= 1)


# ── read API: pagination, earliest_timestamp, ns timestamps, sql_table_name ──
def test_time_series_pagination_equivalence(client):
    writer = client.table_writer({"page.v": edb.f64[2]})
    n = 100
    for i in range(n):
        writer.write(timestamp_us=1000 + i, values={"page.v": [i, -i]})
    assert _wait_for(lambda: len(client.time_series("page.v", 0, 10**6)[0]) == n)

    one_shot = client._c.time_series("page.v", 0, 10**6, None, None)
    paged = client._c.time_series("page.v", 0, 10**6, None, 7)  # tiny pages
    assert one_shot == paged

    # limit still respected across page boundaries
    ts, vals = client.time_series("page.v", 0, 10**6, limit=25)
    assert len(ts) == 25 and vals.shape == (25, 2)
    writer.close()


def test_time_series_pagination_duplicate_timestamps(client):
    writer = client.table_writer({"page.dup": edb.f64})
    timestamps = [1000] * 5 + [1001] * 7 + [1002] * 5 + [1003] * 6
    for i, timestamp_us in enumerate(timestamps):
        writer.write(timestamp_us=timestamp_us, values={"page.dup": float(i)})
    assert _wait_for(lambda: len(client.time_series("page.dup", 0, 10**6)[0]) == len(timestamps))

    one_shot = client._c.time_series("page.dup", 0, 10**6, None, None)
    paged = client._c.time_series("page.dup", 0, 10**6, None, 4)
    assert one_shot == paged

    ts_bytes, data_bytes, prim, dims = paged
    assert prim == "f64" and dims == []
    np.testing.assert_array_equal(np.frombuffer(ts_bytes, dtype=np.int64), timestamps)
    np.testing.assert_array_equal(
        np.frombuffer(data_bytes, dtype=np.float64), np.arange(len(timestamps))
    )
    writer.close()


def test_time_series_stop_is_half_open(client):
    writer = client.table_writer({"page.stop": edb.f64})
    rows = [(10, 1.0), (20, 2.0), (20, 3.0), (30, 4.0)]
    for timestamp_us, value in rows:
        writer.write(timestamp_us=timestamp_us, values={"page.stop": value})
    assert _wait_for(lambda: len(client.time_series("page.stop", 0, 40)[0]) == len(rows))

    ts_left, vals_left = client.time_series("page.stop", 0, 20)
    ts_right, vals_right = client.time_series("page.stop", 20, 40)
    np.testing.assert_array_equal(ts_left, [10])
    np.testing.assert_array_equal(vals_left, [1.0])
    np.testing.assert_array_equal(ts_right, [20, 20, 30])
    np.testing.assert_array_equal(vals_right, [2.0, 3.0, 4.0])

    ts_exact_stop, _ = client.time_series("page.stop", 0, 30)
    np.testing.assert_array_equal(ts_exact_stop, [10, 20, 20])
    ts_empty, vals_empty = client.time_series("page.stop", 30, 30)
    assert len(ts_empty) == 0 and len(vals_empty) == 0
    writer.close()


def test_time_series_empty_valid_ranges_return_empty_arrays(client):
    writer = client.table_writer({"page.empty": edb.f64[2]})
    writer.write(timestamp_us=100, values={"page.empty": [1.0, 2.0]})
    assert _wait_for(lambda: len(client.time_series("page.empty", 0, 200)[0]) == 1)

    for start_us, stop_us in [(0, 50), (150, 200)]:
        ts, vals = client.time_series("page.empty", start_us, stop_us)
        assert ts.shape == (0,)
        assert vals.shape == (0, 2)
        assert vals.dtype == np.float64
    writer.close()


def test_time_series_unknown_component_still_raises(client):
    with pytest.raises(RuntimeError, match="time_series"):
        client.time_series("page.missing", 0, 100)


def test_earliest_timestamp(client):
    writer = client.table_writer({"early.v": edb.f64})
    writer.write(timestamp_us=123, values={"early.v": 1.0})
    assert _wait_for(lambda: len(client.time_series("early.v", 0, 10**6)[0]) == 1)
    ts = client.earliest_timestamp()
    assert isinstance(ts, int)
    assert ts == client.earliest_timestamp()  # stable
    writer.close()


def test_nanosecond_timestamp_writer(client):
    writer = client.table_writer({"ns.v": edb.f64}, timestamp="ns")
    # 3 samples at 1 ms spacing, expressed in nanoseconds
    for i in range(3):
        writer.write(timestamp_ns=(1_000_000 + i * 1_000) * 1_000, values={"ns.v": float(i)})
    assert _wait_for(lambda: len(client.time_series("ns.v", 0, 10**8)[0]) == 3)
    ts, vals = client.time_series("ns.v", 0, 10**8)
    # the DB stores microseconds: ns / 1000
    np.testing.assert_array_equal(ts, [1_000_000, 1_001_000, 1_002_000])
    np.testing.assert_array_equal(vals, [0.0, 1.0, 2.0])
    # unit mismatch is an error
    with pytest.raises(TypeError, match="timestamp_ns"):
        writer.write(timestamp_us=1, values={"ns.v": 0.0})
    writer.close()


def test_sql_table_name_matches_server(client):
    name = "camelCase.gps2Fix"
    writer = client.table_writer({name: edb.f64})
    writer.write(timestamp_us=10, values={name: 4.2})
    assert _wait_for(lambda: len(client.time_series(name, 0, 10**6)[0]) == 1)
    table = client.sql(f"SELECT * FROM {edb.sql_table_name(name)}")
    assert table.num_rows == 1
    writer.close()


# ── latest(): type fidelity + shape ──────────────────────────────────────────
def test_latest_preserves_integer_precision_and_shape(client):
    big = 2**60 + 3  # not representable as f64
    writer = client.table_writer(
        {
            "typed.count": edb.u64,
            "typed.mat": edb.f64[2, 2],
        }
    )
    client.latest("typed.count")  # start the subscription
    client.latest("typed.mat")
    time.sleep(0.3)
    for i in range(20):
        writer.write_nowait(
            timestamp_us=i * 1000,
            values={"typed.count": big, "typed.mat": [[1.0, 2.0], [3.0, 4.0]]},
        )
        time.sleep(0.02)
    assert _wait_for(lambda: client.latest("typed.count") is not None)
    count = client.latest("typed.count")
    assert count.values.dtype == np.uint64
    assert int(count.values.reshape(-1)[0]) == big
    assert _wait_for(lambda: client.latest("typed.mat") is not None)
    mat = client.latest("typed.mat")
    assert mat.values.shape == (2, 2)
    np.testing.assert_array_equal(mat.values, [[1.0, 2.0], [3.0, 4.0]])
    writer.close()


def test_client_close_is_prompt_under_load(server):
    """close() must interrupt the subscription promptly even while the
    stream is busy (event-driven shutdown, no polling races)."""
    client = edb.Client.connect(server.addr)
    writer = client.table_writer({"busy.v": edb.f64[4]})
    client.latest("busy.v")
    stop = time.time() + 1.0
    i = 0
    while time.time() < stop:
        writer.write_nowait(timestamp_us=i, values={"busy.v": [i, i, i, i]})
        i += 1
    start = time.perf_counter()
    client.close()
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"close took {elapsed:.2f}s"
    writer.close()


# ── follow-mode replication ──────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.skipif(shutil.which("elodin-db") is None, reason="elodin-db binary not on PATH")
def test_follow_mode_replicates_python_writes(server, tmp_path):
    """Data written by elodin.db must replicate to a follower instance with
    metadata (element_names) intact — no client-specific special-casing."""
    with edb.Client.connect(server.addr) as client:
        writer = client.table_writer({"fol.gyro": edb.f64[3].labeled("p", "q", "r")})
        for i in range(20):
            writer.write(timestamp_us=1000 + i * 100, values={"fol.gyro": [i, -i, 2.0 * i]})
        assert _wait_for(lambda: len(client.time_series("fol.gyro", 0, 10**6)[0]) == 20)
        writer.close()

    follower_addr = f"127.0.0.1:{next(_port)}"
    follower = subprocess.Popen(
        [
            "elodin-db",
            "run",
            follower_addr,
            str(tmp_path / "follower"),
            "--follows",
            server.addr,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with edb.Client.connect(follower_addr) as fclient:

            def replicated():
                ts, vals = fclient.time_series("fol.gyro", 0, 10**6)
                return len(ts) == 20

            assert _wait_for(replicated, timeout_s=30), "follower should replicate all samples"
            _, vals = fclient.time_series("fol.gyro", 0, 10**6)
            np.testing.assert_array_equal(vals[3], [3.0, -3.0, 6.0])
            info = fclient.components()["fol.gyro"]
            assert info.element_names == ["p", "q", "r"]
    finally:
        follower.terminate()
        follower.wait(timeout=10)


# ── message logs ─────────────────────────────────────────────────────────────
def test_msg_roundtrip_and_range(client):
    for i in range(5):
        client.send_msg("evt.collision", {"id": 1000 + i, "impulse": i * 0.5}, timestamp_us=i * 10)
    client.send_msg("evt.note", "plain text", timestamp_us=100)
    client.send_msg("evt.blob", b"\x00\xffraw", timestamp_us=200)

    assert _wait_for(lambda: len(client.get_msgs("evt.collision", 0, 10**6)) == 5)
    msgs = client.get_msgs("evt.collision", 0, 10**6)
    assert msgs[0] == (0, {"id": 1000, "impulse": 0.0})
    assert msgs[4][1]["id"] == 1004

    # range (inclusive stop, per msg-log semantics) + limit
    subset = client.get_msgs("evt.collision", 10, 35)
    assert [t for t, _ in subset] == [10, 20, 30]
    limited = client.get_msgs("evt.collision", 0, 10**6, limit=2)
    assert len(limited) == 2

    # str decodes via JSON only when valid JSON; plain text comes back as bytes
    ((_, note),) = client.get_msgs("evt.note", 0, 10**6)
    assert note == b"plain text"
    ((_, blob),) = client.get_msgs("evt.blob", 0, 10**6, raw=True)
    assert blob == b"\x00\xffraw"


def test_msg_stream_live(client):
    client.send_msg("evt.live", {"seq": -1}, timestamp_us=1)  # create the log
    received = []
    with client.msg_stream("evt.live") as stream:
        time.sleep(0.3)  # let the subscription attach

        def pump():
            for i in range(10):
                client.send_msg("evt.live", {"seq": i}, timestamp_us=1000 + i)
                time.sleep(0.02)

        t = threading.Thread(target=pump)
        t.start()
        deadline = time.time() + 5.0
        for ts, payload in stream:
            received.append((ts, payload))
            if payload.get("seq") == 9 or time.time() > deadline:
                break
        t.join()
    assert received, "live msg stream should deliver messages"
    assert received[-1][1] == {"seq": 9}
    assert received[-1][0] == 1009


# ── stream(): live + fixed-rate replay ───────────────────────────────────────
def test_stream_live_rows(client):
    writer = client.table_writer({"stm.a": edb.f64[2], "stm.b": edb.i32})
    # Register the components (and their schemas) before subscribing.
    writer.write(timestamp_us=500, values={"stm.a": [0, 0], "stm.b": 0})

    rows = []
    with client.stream(["stm.a", "stm.b"]) as stream:
        time.sleep(0.5)  # let the subscription settle
        for i in range(1, 21):
            writer.write_nowait(timestamp_us=1000 + i * 100, values={"stm.a": [i, -i], "stm.b": i})
            time.sleep(0.01)
        deadline = time.time() + 5.0
        for row in stream:
            rows.append(row)
            if row.timestamp_us >= 1000 + 20 * 100 or time.time() > deadline:
                break
    assert rows, "live stream should deliver rows"
    last = rows[-1]
    assert set(last.values) == {"stm.a", "stm.b"}
    assert last["stm.b"].dtype == np.int32
    # timestamps never go backwards on a live stream of a single writer
    ts = [r.timestamp_us for r in rows]
    assert ts == sorted(ts)
    writer.close()


def test_stream_fixed_rate_replay(client):
    writer = client.table_writer({"rep.v": edb.f64})
    n = 30
    for i in range(n):
        writer.write(timestamp_us=1000 + i * 1000, values={"rep.v": float(i)})
    assert _wait_for(lambda: len(client.time_series("rep.v", 0, 10**6)[0]) == n)

    rows = []
    with client.stream("rep.v", rate_hz=500, start="earliest") as stream:
        deadline = time.time() + 10.0
        for row in stream:
            rows.append(row)
            if row["rep.v"] >= n - 1 or time.time() > deadline:
                break
    assert rows, "replay should deliver rows"
    values = [float(r["rep.v"]) for r in rows]
    # replay advances monotonically through the recorded values from earliest
    assert values == sorted(values)
    assert values[-1] == n - 1
    assert values[0] <= 1.0, f"replay should start near the earliest sample, got {values[0]}"


def test_stream_row_per_component_timestamps(client):
    """Multi-rate rows expose each component's own sample time; the row
    timestamp is the newest of them."""
    fast_w = client.table_writer({"mr.fast": edb.f64})
    slow_w = client.table_writer({"mr.slow": edb.f64})
    slow_w.write(timestamp_us=1_000, values={"mr.slow": 0.0})
    fast_w.write(timestamp_us=1_500, values={"mr.fast": 0.0})

    with client.stream(["mr.fast", "mr.slow"]) as stream:
        time.sleep(0.5)  # let the subscription settle
        # Only the fast component advances; the slow sample stays at t=1000.
        for i in range(1, 21):
            fast_w.write_nowait(timestamp_us=1_500 + i * 1_000, values={"mr.fast": float(i)})
            time.sleep(0.01)
        deadline = time.time() + 5.0
        row = None
        for candidate in stream:
            if "mr.fast" in candidate and "mr.slow" in candidate:
                row = candidate
                if candidate.timestamps["mr.fast"] > candidate.timestamps["mr.slow"]:
                    break
            if time.time() > deadline:
                break
    assert row is not None, "stream should deliver rows with both components"
    assert row.timestamp_us == max(row.timestamps.values())
    assert row.timestamps["mr.slow"] == 1_000
    assert row.timestamps["mr.fast"] > row.timestamps["mr.slow"]
    fast_w.close()
    slow_w.close()


def test_stream_close_ends_iteration(client):
    writer = client.table_writer({"stc.v": edb.f64})
    writer.write(timestamp_us=1, values={"stc.v": 0.0})
    stream = client.stream("stc.v")
    stream.close()
    with pytest.raises(StopIteration):
        next(stream)
    writer.close()


# ── writer observability & queue policies ────────────────────────────────────
def test_queue_policy_validation(client):
    with pytest.raises(ValueError, match="queue policy"):
        client.table_writer({"qp.v": edb.f64}, queue="drop")


def test_oversized_field_rejected_at_construction(client):
    # 8192 f64s = 65536 bytes; vtable offsets/lengths are u16 and would wrap.
    with pytest.raises(ValueError, match="limited to"):
        client.table_writer({"big.v": edb.f64[8192]})
    # Just under the limit (with the 8-byte timestamp) still works.
    writer = client.table_writer({"big.ok": edb.f64[8190]})
    writer.close()


def test_drop_oldest_policy_sheds_on_overflow():
    # Nothing listening: every row stays queued, so overflow exercises the
    # policy (exact shed order is covered by Rust unit tests in writer.rs).
    client = edb.Client.connect(f"127.0.0.1:{next(_port)}")
    writer = client.table_writer({"q.v": edb.f64}, queue="drop-oldest", maxlen=8)
    for i in range(64):
        writer.write_nowait(timestamp_us=i, values={"q.v": float(i)})
    assert writer.dropped > 0
    writer.close()
    client.close()


def test_writer_state_and_schema_conflict(server):
    with edb.Client.connect(server.addr) as client:
        a = client.table_writer({"conf.x": edb.f64[3]})
        a.write(timestamp_us=1, values={"conf.x": [1, 2, 3]})
        assert a.state() == "Connected"
        assert _wait_for(lambda: len(client.time_series("conf.x", 0, 100)[0]) == 1)

        # Re-register the same component with a different shape: the DB
        # rejects the vtable and the error surfaces on the writer. last_error
        # holds the *most recent* rejection, which is either the schema
        # mismatch itself or the follow-on "vtable not found" for the row
        # that referenced the rejected vtable.
        b = client.table_writer({"conf.x": edb.f64[2]})
        b.write(timestamp_us=2, values={"conf.x": [1, 2]})
        assert _wait_for(lambda: b.last_error is not None)
        assert "schema mismatch" in b.last_error or "vtable not found" in b.last_error
        assert a.last_error is None
        a.close()
        b.close()


def test_multiprocess_writers_share_db(server):
    """Two OS processes with independent (randomized) vtable ids must be able
    to write different components to one database without clobbering each
    other's registration."""
    script = textwrap.dedent(
        """
        import sys
        import elodin.db as edb

        addr, name = sys.argv[1], sys.argv[2]
        client = edb.Client.connect(addr)
        writer = client.table_writer({name: edb.f64[2]})
        for i in range(20):
            writer.write(timestamp_us=1000 + i, values={name: [i, -i]})
        writer.close()
        client.close()
        """
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, server.addr, f"mp.proc{i}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for i in range(2)
    ]
    for p in procs:
        _, stderr = p.communicate(timeout=60)
        assert p.returncode == 0, stderr.decode()

    with edb.Client.connect(server.addr) as client:
        for i in range(2):
            name = f"mp.proc{i}"
            assert _wait_for(lambda n=name: len(client.time_series(n, 0, 10**6)[0]) == 20)
            _, vals = client.time_series(name, 0, 10**6)
            np.testing.assert_array_equal(vals[5], [5.0, -5.0])
