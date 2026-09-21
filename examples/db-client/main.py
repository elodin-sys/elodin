#!/usr/bin/env python3
"""elodin.db showcase: fly a virtual Crazyflie from a plain Python process.

This example is NOT an Elodin simulation. It demonstrates the standalone
`elodin.db` client: an embedded database server, multi-rate batched writers,
a live read-stream that publishes derived telemetry, message-log events, and
the full read API (latest / time_series / sql / msgs) — all from one script.

Run from the repository root (after `just install` in the nix devshell):

    uv run python examples/db-client/main.py

The Elodin Editor opens on a schematic where `crazyflie.glb` flies a figure-8
with graphs of every written signal. Closing the Editor stops the writers and
prints a read-back summary. Headless variant (no Editor, fixed duration):

    uv run python examples/db-client/main.py --no-editor --duration 5
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# Imported eagerly (Client.sql uses it lazily) so the shutdown summary never
# performs a first-time import after a potentially hours-long Editor session.
import pyarrow  # noqa: F401

import elodin.db as edb

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
SCHEMATIC = EXAMPLE_DIR / "schematic.kdl"

STATE_RATE_HZ = 100.0  # world_pos + IMU
STATUS_DIV = 10  # status writer runs every N state ticks (10 Hz)

# Figure-8 (lissajous) flight path, sized for the ~10 cm crazyflie model.
RADIUS_X = 0.8  # m
RADIUS_Y = 0.5  # m
ALT_BASE = 0.6  # m
ALT_AMP = 0.2  # m
LAP_S = 8.0  # seconds per figure-8 lap


def flight_state(t: float):
    """Analytic figure-8 pose + body rates at time ``t`` (seconds)."""
    w = 2.0 * math.pi / LAP_S
    x = RADIUS_X * math.sin(w * t)
    y = RADIUS_Y * math.sin(2.0 * w * t)
    z = ALT_BASE + ALT_AMP * math.sin(0.5 * w * t)
    vx = RADIUS_X * w * math.cos(w * t)
    vy = RADIUS_Y * 2.0 * w * math.cos(2.0 * w * t)
    ax = -RADIUS_X * w * w * math.sin(w * t)
    ay = -RADIUS_Y * (2.0 * w) ** 2 * math.sin(2.0 * w * t)
    az = -ALT_AMP * (0.5 * w) ** 2 * math.sin(0.5 * w * t)

    # Yaw follows the velocity vector; bank into the lateral acceleration.
    yaw = math.atan2(vy, vx)
    speed = math.hypot(vx, vy)
    lat_acc = (vx * ay - vy * ax) / speed if speed > 1e-6 else 0.0
    roll = max(-0.6, min(0.6, 0.15 * lat_acc))
    pitch = max(-0.4, min(0.4, -0.08 * (vx * ax + vy * ay) / max(speed, 1e-6)))

    # Intrinsic ZYX euler -> scalar-last quaternion [qx, qy, qz, qw].
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    world_pos = [qx, qy, qz, qw, x, y, z]
    accel = [ax, ay, az + 9.81]  # specific force: gravity shows up on +z
    gyro = [roll * w, pitch * w, w * math.copysign(1.0, math.cos(w * t))]
    return world_pos, accel, gyro, speed


def flight_loop(client: edb.Client, stop: threading.Event):
    """100 Hz state writer + 10 Hz status writer + per-lap events."""
    state_w = client.table_writer(
        {
            "drone.world_pos": edb.f64[7].labeled("q0", "q1", "q2", "q3", "x", "y", "z"),
            "drone.imu.accel": edb.f64[3].labeled("x", "y", "z"),
            "drone.imu.gyro": edb.f64[3].labeled("p", "q", "r"),
            "drone.propeller_angle": edb.f64[4].labeled("p0", "p1", "p2", "p3"),
        },
        queue="drop-oldest",
        maxlen=2048,
    )
    status_w = client.table_writer(
        {
            "drone.motor.rpm": edb.f32[4].labeled("m1", "m2", "m3", "m4"),
            "drone.battery.voltage": edb.f64,
            "drone.status.armed": edb.bool_,
            "drone.status.mode": edb.i32,
        }
    )

    t0 = time.time()
    period = 1.0 / STATE_RATE_HZ
    tick = 0
    lap = 0
    voltage = 4.2
    # Quad-X spin directions (M1/M3 CW -> negative, M2/M4 CCW -> positive),
    # matching the crazyflie-edu example.
    prop_dir = (-1.0, 1.0, -1.0, 1.0)
    prop_angle = [0.0, 0.0, 0.0, 0.0]
    rpm = [0.0, 0.0, 0.0, 0.0]
    try:
        while not stop.is_set():
            t = time.time() - t0
            t_us = time.time_ns() // 1_000
            world_pos, accel, gyro, speed = flight_state(t)

            base_rpm = 12_000.0 + 3_000.0 * speed
            rpm = [base_rpm + 120.0 * math.sin(t * 7.0 + k) for k in range(4)]
            # Integrate the animation angle from RPM (rpm * 6 = deg/s), scaled
            # way down so ~200 rev/s aliased at 100 Hz still reads as a spin.
            for k in range(4):
                prop_angle[k] += rpm[k] * 6.0 * 0.01 * period * prop_dir[k]
                prop_angle[k] = (prop_angle[k] + 180.0) % 360.0 - 180.0

            # Hot path: never blocks, sheds per drop-oldest if the DB stalls.
            state_w.write_nowait(
                timestamp_us=t_us,
                values={
                    "drone.world_pos": world_pos,
                    "drone.imu.accel": accel,
                    "drone.imu.gyro": gyro,
                    "drone.propeller_angle": prop_angle,
                },
            )

            if tick % STATUS_DIV == 0:
                voltage = max(3.0, 4.2 - 0.005 * t)
                status_w.write(
                    timestamp_us=t_us,
                    values={
                        "drone.motor.rpm": rpm,
                        "drone.battery.voltage": voltage,
                        "drone.status.armed": True,
                        "drone.status.mode": 2 if speed > 0.5 else 1,  # 2=cruise, 1=hover
                    },
                )

            completed_laps = int(t / LAP_S)
            if completed_laps > lap:
                lap = completed_laps
                client.send_msg(
                    "drone.events",
                    {"event": "lap", "lap": lap, "battery_v": round(voltage, 3)},
                    timestamp_us=t_us,
                )
                print(f"[flight] lap {lap} complete (battery {voltage:.2f} V)")

            tick += 1
            sleep_for = t0 + tick * period - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        drops = state_w.dropped
        state_w.close()
        status_w.close()
        if drops:
            print(f"[flight] state writer shed {drops} rows")


def derived_loop(client: edb.Client, stop: threading.Event):
    """Consume the live world_pos stream and publish ground speed back."""
    speed_w = client.table_writer({"drone.nav.speed": edb.f64})
    prev = None
    try:
        with client.stream("drone.world_pos") as rows:
            for row in rows:
                if stop.is_set():
                    break
                pos = row["drone.world_pos"][4:6]
                if prev is not None:
                    dt_us = row.timestamp_us - prev[0]
                    if dt_us > 0:
                        speed = float(np.linalg.norm(pos - prev[1]) / (dt_us * 1e-6))
                        speed_w.write_nowait(
                            timestamp_us=row.timestamp_us,
                            values={"drone.nav.speed": speed},
                        )
                prev = (row.timestamp_us, pos.copy())
    finally:
        speed_w.close()


def print_summary(client: edb.Client, t_start_us: int):
    """Read everything back: discovery, latest, time_series, SQL, msgs."""
    now_us = time.time_ns() // 1_000
    print("\n=== read-back summary " + "=" * 40)

    infos = client.components()
    print(f"components ({len(infos)}):")
    for name in sorted(infos):
        info = infos[name]
        labels = f" [{','.join(info.element_names)}]" if info.element_names else ""
        print(f"  {name:24s} {info.prim_type}{list(info.shape)}{labels}")

    print(f"earliest_timestamp: {client.earliest_timestamp()} us")

    sample = client.latest("drone.world_pos")
    if sample is not None:
        x, y, z = sample.values[4:]
        print(f"latest world_pos:   ({x:+.2f}, {y:+.2f}, {z:+.2f}) m at t={sample.timestamp_us}")

    for name in ("drone.world_pos", "drone.imu.accel", "drone.battery.voltage", "drone.nav.speed"):
        try:
            ts, _ = client.time_series(name, t_start_us, now_us)
            print(f"time_series {name}: {len(ts)} samples")
        except RuntimeError as e:
            print(f"time_series {name}: unavailable ({e})")

    table_name = edb.sql_table_name("drone.battery.voltage")
    table = client.sql(f"SELECT COUNT(*) AS n, AVG({table_name}) AS mean_v FROM {table_name}")
    print(f"sql over {table_name}: {table.to_pylist()}")

    try:
        msgs = client.get_msgs("drone.events", 0, now_us)
        print(f"events ({len(msgs)}):", [payload for _, payload in msgs[-5:]])
    except RuntimeError:
        # The msg log is created on first send_msg — no lap completed yet.
        print("events: none yet (a lap takes", f"{LAP_S:.0f}s)")
    print("=" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--addr", default="127.0.0.1:2240", help="bind address for the DB")
    parser.add_argument("--no-editor", action="store_true", help="run headless (no Editor)")
    parser.add_argument(
        "--duration", type=float, default=30.0, help="headless run length in seconds"
    )
    args = parser.parse_args()

    # The server creates the database at a fresh path (an existing directory
    # is treated as a database to open).
    db_path = str(Path(tempfile.mkdtemp(prefix="elodin-db-client-")) / "db")
    print(f"[db] serving {db_path} on {args.addr}")
    t_start_us = time.time_ns() // 1_000

    with edb.Server.start(db_path, args.addr), edb.Client.connect(args.addr) as client:
        stop = threading.Event()
        threads = [
            threading.Thread(target=flight_loop, args=(client, stop), daemon=True),
            threading.Thread(target=derived_loop, args=(client, stop), daemon=True),
        ]
        for thread in threads:
            thread.start()

        # Prime the latest-value subscription so `latest()` has data both for
        # the live status line below and for the shutdown summary.
        client.latest("drone.world_pos")

        try:
            if args.no_editor:
                print(f"[db] writing telemetry for {args.duration:.0f}s (headless)")
                deadline = time.time() + args.duration
                while time.time() < deadline:
                    time.sleep(min(2.0, max(0.0, deadline - time.time())))
                    sample = client.latest("drone.world_pos")
                    if sample is not None:
                        x, y, z = sample.values[4:]
                        print(f"[latest] pos=({x:+.2f}, {y:+.2f}, {z:+.2f}) m")
            else:
                if shutil.which("elodin") is None:
                    print(
                        "error: `elodin` binary not found on PATH — run `just install` "
                        "in the nix devshell first (or use --no-editor)",
                        file=sys.stderr,
                    )
                    return 1
                print("[editor] launching; close the window to stop the demo")
                subprocess.run(
                    ["elodin", "editor", args.addr, "--kdl", str(SCHEMATIC)],
                    env={**os.environ, "ELODIN_ASSETS_DIR": str(REPO_ROOT / "assets")},
                    check=False,
                )
        except KeyboardInterrupt:
            print("\n[demo] interrupted")
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=5.0)

        print_summary(client, t_start_us)
    return 0


if __name__ == "__main__":
    sys.exit(main())
