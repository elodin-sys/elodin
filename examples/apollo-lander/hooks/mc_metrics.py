"""Shared Monte Carlo metric helpers for the Apollo lander hooks and scripts.

Imported by `score.py`, `report.py`, `ci_gate.py`, `ci_score.py`, and
`calibrate.py` so the soft-landing criteria, trajectory RMSE selection, and
run-pass semantics live in exactly one place. Standard library only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Soft-landing acceptance criteria (mirrors the sim's own check in main.py).
SOFT_HORIZONTAL_SPEED_MPS = 1.0
SOFT_VERTICAL_SPEED_MPS = 3.0
UPRIGHT_DOT_MIN = 0.94


def read_json(path: Path) -> dict:
    """Best-effort JSON read; returns {} when the file is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except OSError:
        return {}


def to_float(value, default: float | None = None) -> float | None:
    """Parse a finite float, or return `default` for missing/NaN/inf values."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def soft_landing(post: dict, result: dict) -> bool:
    """Whether a run met the soft-landing criteria.

    Trusts the `post_run` hook's `soft_landing`/`pass` verdict when present,
    otherwise recomputes it from the raw `result.json` kinematics so callers
    work even before scoring has run.
    """
    if "soft_landing" in post:
        return bool(post["soft_landing"])
    if post.get("pass") is not None:
        return bool(post["pass"])
    touchdown_speed = to_float(result.get("touchdown_speed"))
    horizontal_speed = to_float(result.get("horizontal_speed"))
    fuel_remaining = to_float(result.get("fuel_remaining"))
    upright_dot = to_float(result.get("upright_dot"))
    if touchdown_speed is None or horizontal_speed is None:
        return False
    return (
        bool(result.get("landed", False))
        and touchdown_speed <= SOFT_VERTICAL_SPEED_MPS
        and horizontal_speed <= SOFT_HORIZONTAL_SPEED_MPS
        and upright_dot is not None
        and upright_dot >= UPRIGHT_DOT_MIN
        and fuel_remaining is not None
        and fuel_remaining > 0.0
    )


def traj_rmse(post: dict, result: dict) -> float | None:
    """Altitude-tracking RMSE, preferring the scored value over the raw one."""
    rmse = to_float(post.get("traj_rmse_m"))
    if rmse is None:
        rmse = to_float(result.get("traj_rmse"))
    return rmse


def run_passed(row: dict) -> bool:
    """Whether a `results.csv` row counts as a pass.

    Uses the explicit `passed` column when the runner emits it, falling back to
    the process `status` for older campaign outputs.
    """
    passed = row.get("passed")
    if passed is not None and passed != "":
        return str(passed).strip().lower() in {"true", "1", "yes"}
    return row.get("status", "") == "ok"
