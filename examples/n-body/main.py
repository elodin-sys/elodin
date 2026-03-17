#!/usr/bin/env uv run

from __future__ import annotations

import os
import shutil
from pathlib import Path

from sim import (
    DB_NAME_ENV,
    DEFAULT_DB_PATH,
    SIMULATION_RATE_HZ,
    TELEMETRY_RATE_HZ,
    build_system,
    build_world,
    get_default_max_ticks,
    make_truth_post_step,
)

MAX_TICKS_ENV = "ELODIN_NBODY_MAX_TICKS"
TRUTH_START_TIMESTAMP_US = 1_577_836_800_000_000  # 2020-01-01T00:00:00Z

w = build_world()
sys = build_system()
post_step = make_truth_post_step()
default_max_ticks = get_default_max_ticks()
max_ticks = int(os.environ.get(MAX_TICKS_ENV, str(default_max_ticks)))
db_path = Path(os.environ.get(DB_NAME_ENV, DEFAULT_DB_PATH))

# This example always starts from a clean DB to avoid timestamp conflicts
# when rerun with the same DB path.
if db_path.exists():
    if db_path.is_dir():
        shutil.rmtree(db_path)
    else:
        db_path.unlink()

w.run(
    sys,
    simulation_rate=SIMULATION_RATE_HZ,
    telemetry_rate=TELEMETRY_RATE_HZ,
    start_timestamp=TRUTH_START_TIMESTAMP_US,
    max_ticks=max_ticks,
    post_step=post_step,
    db_path=str(db_path),
)
