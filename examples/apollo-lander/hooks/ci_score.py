"""CI post-run smoke: pass when the truncated sim produced a result artifact."""

from __future__ import annotations

import json
from pathlib import Path


def post_run(ctx):
    result_path = Path(ctx.run_dir) / "result.json"
    if not result_path.exists():
        raise RuntimeError(f"CI smoke: missing {result_path}")
    result = json.loads(result_path.read_text())
    traj_rmse = result.get("traj_rmse")
    if traj_rmse is None:
        raise RuntimeError(f"CI smoke: {result_path} has no traj_rmse")
    return {
        "pass": True,
        "smoke": True,
        "landed": bool(result.get("landed", False)),
        "traj_rmse_m": traj_rmse,
    }
