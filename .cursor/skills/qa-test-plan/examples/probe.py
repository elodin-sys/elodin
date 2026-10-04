"""Reusable live-DB probe for QA validation.

Usage: uv run python /tmp/qa-probe.py <addr> <component> [<component> ...]
Connects to a running elodin-db, lists components, and for each requested
component prints sample count over full history and per-element first/last/min/max.
"""

import sys
import time

import elodin.db as edb

addr = sys.argv[1]
wanted = sys.argv[2:]

with edb.Client.connect(addr) as c:
    infos = c.components()
    names = sorted(infos)
    print("N_COMPONENTS", len(names))
    print("COMPONENTS", ",".join(names))
    now = time.time_ns() // 1000
    t0 = c.earliest_timestamp()
    print("EARLIEST_US", t0)
    for name in wanted:
        if name.startswith("msg:"):
            mname = name[4:]
            try:
                msgs = c.get_msgs(mname, 0, now)
                print(f"MSG {mname} COUNT {len(msgs)}")
                if msgs:
                    tail = msgs[-3:]
                    print(f"MSG {mname} TAIL {[str(p)[:60] for _, p in tail]}")
            except Exception as e:  # noqa: BLE001
                print(f"MSG {mname} ERR {e}")
            continue
        if name not in infos:
            print(f"MISSING {name}")
            continue
        try:
            ts, vals = c.time_series(name, t0, now)
        except Exception as e:  # noqa: BLE001
            print(f"ERR {name} {e}")
            continue
        n = len(ts)
        if n == 0:
            print(f"{name} SAMPLES 0")
            continue

        def _row(v):
            try:
                return [float(x) for x in v]
            except TypeError:
                return [float(v)]

        rows = [_row(v) for v in vals]
        first = [round(x, 4) for x in rows[0]]
        last = [round(x, 4) for x in rows[-1]]
        cols = list(zip(*rows))
        mins = [round(min(col), 4) for col in cols]
        maxs = [round(max(col), 4) for col in cols]
        print(f"{name} SAMPLES {n}")
        print(f"{name} FIRST {first}")
        print(f"{name} LAST {last}")
        print(f"{name} MIN {mins}")
        print(f"{name} MAX {maxs}")
