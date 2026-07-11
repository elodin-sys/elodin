"""Serving-mode probes for the elodin-db QA plan (`elodin.db` Python client).

Usage (inside `nix develop`, from repo root, against a running elodin-db):
  uv run python .cursor/skills/qa-test-plan/elodin-db/serve_probe.py replay <addr> <component> <rate_hz> <rows>
  uv run python .cursor/skills/qa-test-plan/elodin-db/serve_probe.py live <addr>

replay — subscribes to <component> as a fixed-rate replay from the earliest
timestamp (client.stream(rate_hz=..., start="earliest")) and prints wall-clock
pacing vs. the expected rows/rate_hz duration. A correct replay paces the
wall clock to the requested rate regardless of how fast the data was recorded.

live — writes 10 fresh `qa.heartbeat` samples via client.send (registers the
component on first write), reads them back with time_series, and prints both
counts. Proves the write -> store -> read round trip on a live server.
"""

import sys
import time

import elodin.db as edb

mode, addr = sys.argv[1], sys.argv[2]

if mode == "replay":
    component, rate_hz, rows = sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
    with edb.Client.connect(addr) as c:
        stamps = []
        t0 = time.monotonic()
        for row in c.stream([component], rate_hz=rate_hz, start="earliest"):
            stamps.append(row.timestamp_us)
            if len(stamps) >= rows:
                break
        wall_s = time.monotonic() - t0
        span_s = (stamps[-1] - stamps[0]) / 1e6
        expect_s = rows / rate_hz
        print(
            f"REPLAY rows={rows} rate_hz={rate_hz} wall_s={wall_s:.2f} "
            f"expect_s={expect_s:.2f} data_span_s={span_s:.2f}"
        )
elif mode == "live":
    with edb.Client.connect(addr) as c:
        t0 = time.time_ns() // 1000
        for i in range(10):
            c.send("qa.heartbeat", float(i), time.time_ns() // 1000)
            time.sleep(0.1)
        time.sleep(0.3)
        ts, vals = c.time_series("qa.heartbeat", t0, time.time_ns() // 1000)
        last = vals[-1]
        try:
            last = float(last[0])
        except (TypeError, IndexError):
            last = float(last)
        print(f"LIVE wrote=10 readback={len(ts)} last={last}")
else:
    raise SystemExit(f"unknown mode {mode!r} (expected 'replay' or 'live')")
