"""Acceptance tests for `elodin.db` (see the AGP feature request document).

1. write/read snippet works against an embedded Server.start
2. write_nowait under a dead DB is cheap and never raises
3. batched writer: one shared-timestamp Table row per write
4. 1M-sample time_series read is fast
5. element_names metadata round-trips (Editor axis labels)
6. all prim types + shapes round-trip
7. data is queryable via SQL (Arrow IPC -> pyarrow)
"""

import itertools
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
def test_time_series_1m_read(client):
    writer = client.table_writer({"perf.v": edb.f64}, maxlen=65536)
    n = 1_000_000
    for i in range(n):
        writer.write(timestamp_us=i, values={"perf.v": float(i)})
    assert _wait_for(lambda: len(client.time_series("perf.v", n - 10, n)[0]) == 10, timeout_s=30)
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
