"""End-to-end closed-loop mission tests (Phase 5-7 exit criteria).

Runs the real SITL stack — compiled plant + Rust FSW over UDP — as a
subprocess and asserts on the emitted result line. Slower than the unit
tests (~1 min each); CI runs the Monte Carlo smoke script instead.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).parent
REPO_ROOT = EXAMPLE_DIR.parent.parent


def _run_mission(max_ticks: int | None = None, timeout_s: float = 900.0) -> dict:
    env = os.environ.copy()
    env.pop("ELODIN_MONTE_CARLO_CONTEXT", None)
    if max_ticks is not None:
        env["ELODIN_FALCON9_MAX_TICKS"] = str(max_ticks)
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE_DIR / "main.py"), "run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("[falcon9] result:"):
            return ast.literal_eval(line.split("result:", 1)[1].strip())
    raise AssertionError(
        f"no result line.\nstdout tail: {proc.stdout[-2000:]}\nstderr tail: {proc.stderr[-2000:]}"
    )


@pytest.mark.slow
def test_ascent_hits_meco_bands():
    """Phase 5 exit: MECO within the recorded CRS-12 bands."""
    result = _run_mission(max_ticks=170_000)
    meco_t = result.get("phase_4_t_s")
    assert meco_t is not None, f"never reached MECO: {result}"
    assert abs(meco_t - 147.0) <= 8.0, f"MECO at {meco_t}s (recorded 147)"
    # MECO gate is speed-commanded at 1656 m/s: reaching phase 4 is the check.


@pytest.mark.slow
def test_full_mission_lands_at_lz1():
    """Phase 6/7 exit: full autonomous RTLS run, all four burns + purges,
    engines-out relight budget respected, soft-ish touchdown near LZ-1
    (pre-calibration bands; the campaign tightens these to parity)."""
    result = _run_mission()
    assert result["landed"], f"did not land: {result}"
    assert result["final_phase"] == 11.0
    # All burns happened, in order, with a purge after each cutoff.
    for pid in (4, 5, 6, 7, 8, 9, 10, 11):
        assert f"phase_{pid}_t_s" in result, f"phase {pid} missing: {result}"
    assert result["purge_events"] == 4
    # Event times against the recorded mission (loose pre-calibration bands).
    assert abs(result["phase_4_t_s"] - 147.0) <= 10.0  # MECO
    assert abs(result["phase_8_t_s"] - 370.0) <= 20.0  # entry start
    assert abs(result["phase_10_t_s"] - 433.0) <= 15.0  # landing start
    assert abs(result["touchdown_t_s"] - 466.0) <= 20.0
    # Touchdown state (pre-calibration): intact, upright-ish, near LZ-1.
    assert result["touchdown_vertical_mps"] <= 10.0
    assert result["touchdown_lateral_mps"] <= 20.0
    assert result["touchdown_pos_err_m"] <= 3_000.0
    assert result["prop_remaining_kg"] > 0.0
    assert result["nitrogen_remaining_kg"] > 0.0
    # Physics plausibility: descent q-bar peak in the retropropulsion band.
    assert 40.0 <= result["qbar_peak_descent_kpa"] <= 120.0
