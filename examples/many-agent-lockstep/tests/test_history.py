"""End-to-end runs of the `lockstep-history` mode, with the Python hook."""

import json
import os
import subprocess
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1]


def run(path: Path, **env: str) -> tuple[int, dict, dict]:
    """Runs 120 unpaced ticks of 2 agents; returns the exit code, the receipt and the result."""
    path.mkdir(parents=True)
    summary, result = path / "summary.json", path / "result.json"
    completed = subprocess.run(
        ["elodin", "run", str(EXAMPLE / "main.py")],
        env={
            **os.environ,
            "ELODIN_MANY_AGENT_MODE": "lockstep-history",
            "ELODIN_MANY_AGENT_HOOK": "python",
            "ELODIN_MANY_AGENT_AGENTS": "2",
            "ELODIN_MANY_AGENT_MAX_TICKS": "120",
            "ELODIN_MANY_AGENT_REALTIME": "0",
            "ELODIN_SIM_WARMUP_TICKS": "0",
            "ELODIN_DB_PATH": str(path / "db"),
            "ELODIN_SIM_SUMMARY_JSON": str(summary),
            "ELODIN_MANY_AGENT_RESULT_JSON": str(result),
            **env,
        },
        capture_output=True,
        timeout=300,
        check=False,
    )
    output = json.loads(result.read_text()) if result.is_file() else {}
    return completed.returncode, json.loads(summary.read_text()), output


def test_capture_evidence_is_the_same_every_run(tmp_path):
    runs = [run(tmp_path / str(index)) for index in range(2)]
    for code, receipt, result in runs:
        assert code == 0, receipt["failure"]
        # Frames 1 to 10 are due within 120 ticks, and the producer drops frame 7.
        assert (result["frames"]["found"], result["frames"]["missing"]) == (9, 1)
        assert receipt["sync"]["commits"] == "synchronous"
        assert receipt["callbacks"]["deferred_rows"] == 3 * 10
    assert runs[0][2]["evidence_sha256"] == runs[1][2]["evidence_sha256"]


def test_a_producer_that_stops_fails_the_run(tmp_path):
    code, receipt, _ = run(
        tmp_path / "run",
        ELODIN_MANY_AGENT_PRODUCER_FRAMES="3",
        ELODIN_MANY_AGENT_PRODUCER_TIMEOUT_S="3",
    )
    assert code != 0
    assert receipt["failed_hook"] == {"index": 1, "name": "history_step", "committed": False}
    assert "never handled source time 40000" in receipt["failure"]
