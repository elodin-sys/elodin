"""End-to-end runs of the `lockstep-history` mode, with Elodin's render server and the Python hook."""

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import elodin.db as edb
import numpy as np

EXAMPLE = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start(path: Path, ticks: int, **env: str) -> tuple[subprocess.Popen, str]:
    """Starts a paced run of 2 agents on a DB address of its own; returns it and the address."""
    address = f"127.0.0.1:{free_port()}"
    process = subprocess.Popen(
        ["elodin", "run", str(EXAMPLE / "main.py"), "--addr", address],
        env={
            **os.environ,
            "ELODIN_MANY_AGENT_HOOK": "python",
            "ELODIN_MANY_AGENT_AGENTS": "2",
            "ELODIN_MANY_AGENT_MAX_TICKS": str(ticks),
            "ELODIN_SIM_WARMUP_TICKS": "0",
            "ELODIN_DB_PATH": str(path / "db"),
            "ELODIN_SIM_SUMMARY_JSON": str(path / "summary.json"),
            "ELODIN_MANY_AGENT_RESULT_JSON": str(path / "result.json"),
            **env,
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, address


def has_frames(address: str) -> bool:
    try:
        with edb.Client.connect(address) as client:
            return bool(client.get_msgs("camera.overhead", 0, 2**62, limit=1, raw=True))
    except RuntimeError:
        return False  # The simulation does not serve its DB yet.


def test_camera_frames_are_paired_with_the_truth_at_their_capture_time(tmp_path):
    process, _ = start(tmp_path, 12_000)
    assert process.wait(timeout=300) == 0
    receipt = json.loads((tmp_path / "summary.json").read_text())
    frames = json.loads((tmp_path / "result.json").read_text())["frames"]
    assert frames["found"] > 0
    assert receipt["sync"]["commits"] == "synchronous"
    assert receipt["callbacks"]["deferred_batches"] == frames["found"] + frames["missing"]

    with (
        edb.Server.start(str(tmp_path / "db"), f"127.0.0.1:{free_port()}") as server,
        edb.Client.connect(server.addr) as client,
    ):
        # Skip each component's initial sample, recorded at time 0.
        _, times = client.time_series("camera.capture_time", 1, 2**62)
        _, truths = client.time_series("camera.capture_truth", 1, 2**62)
        assert len(times) == len(truths) == frames["found"] + frames["missing"]
        for capture_time, truth in zip(times[:, 0].astype(int), truths):
            _, position = client.time_series("agent_000.position", capture_time, capture_time + 1)
            np.testing.assert_array_equal(position[0], truth)


def test_a_camera_that_stops_fails_the_run(tmp_path):
    process, address = start(tmp_path, 60_000, ELODIN_MANY_AGENT_CAMERA_TIMEOUT_S="3")
    try:
        deadline = time.monotonic() + 120
        while not has_frames(address):
            assert time.monotonic() < deadline, "the camera never sent a frame"
            time.sleep(0.5)
        for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                args = cmdline.read_bytes().split(b"\0")
            except OSError:
                continue
            if b"render-server" in args and address.encode() in args:
                os.kill(int(cmdline.parent.name), signal.SIGKILL)
        assert process.wait(timeout=120) != 0
    finally:
        process.kill()
    receipt = json.loads((tmp_path / "summary.json").read_text())
    assert receipt["failed_hook"] == {"index": 1, "name": "history_step", "committed": False}
    assert "the sensor camera stopped sending frames" in receipt["failure"]
