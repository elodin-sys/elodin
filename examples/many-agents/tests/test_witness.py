"""Witnesses of unpaced `lockstep` runs of 2 agents: repeats, the one-bit fault, replay, and the
completion race with its causal-token fix."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from benchmark import HOOK_LIBRARY, free_ports

EXAMPLE = Path(__file__).resolve().parents[1]
FAULT = {"ELODIN_MANY_AGENT_FAULT_TICK": "300", "ELODIN_MANY_AGENT_FAULT_AGENT": "1"}


@pytest.fixture(scope="module", autouse=True)
def built():
    for crate in ("controller", "hook"):
        subprocess.run([sys.executable, str(EXAMPLE / crate / "build.py")], check=True)


def env(path: Path, **extra: str) -> dict[str, str]:
    return {
        **os.environ,
        "ELODIN_MANY_AGENT_MODE": "lockstep",
        "ELODIN_MANY_AGENT_AGENTS": "2",
        "ELODIN_MANY_AGENT_MAX_TICKS": "600",
        "ELODIN_MANY_AGENT_REALTIME": "0",
        "ELODIN_MANY_AGENT_PORT_BASE": str(free_ports(4, socket.SOCK_DGRAM)),
        "ELODIN_DB_PATH": str(path / "db"),
        "ELODIN_SIM_SUMMARY_JSON": str(path / "summary.json"),
        "ELODIN_MANY_AGENT_RESULT_JSON": str(path / "result.json"),
        "ELODIN_MANY_AGENT_WITNESS_PATH": str(path / "run.witness"),
        **extra,
    }


def run(tmp_path: Path, name: str, **extra: str) -> dict:
    """Runs the example; returns the receipt's witness section and the final-state digest."""
    path = tmp_path / name
    path.mkdir()
    address = f"127.0.0.1:{free_ports(3)}"
    process = subprocess.run(
        ["elodin", "run", str(EXAMPLE / "main.py"), "--addr", address],
        env=env(path, **extra),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    witness = json.loads((path / "summary.json").read_text())["witness"]
    witness["sha256"] = json.loads((path / "result.json").read_text())["sha256"]
    return witness


def determinism(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["elodin", "determinism", *args], capture_output=True, text=True)


def located(mismatch: dict) -> tuple:
    keys = ("tick", "stage", "hook", "kind", "record", "entity", "row", "byte")
    return tuple(mismatch[key] for key in keys)


def test_repeats_and_both_hooks_record_the_same_witness(tmp_path):
    native = [run(tmp_path, f"native-{repeat}") for repeat in range(2)]
    python = run(tmp_path, "python", ELODIN_MANY_AGENT_HOOK="python")
    assert native[0]["root"] == native[1]["root"] == python["root"]
    assert native[0]["sha256"] == python["sha256"]
    assert native[0]["ticks"] == 600


def test_a_one_bit_fault_is_located_and_replayed(tmp_path):
    run(tmp_path, "baseline", ELODIN_MANY_AGENT_WITNESS_RAW="1")
    baseline = str(tmp_path / "baseline/run.witness")
    faulted = run(
        tmp_path,
        "faulted",
        ELODIN_MANY_AGENT_FAULT_MODE="bit-flip",
        ELODIN_MANY_AGENT_FAULT_BYTE="3",
        ELODIN_MANY_AGENT_WITNESS_REFERENCE=baseline,
        **FAULT,
    )
    expected = (
        300,
        "after_hook:0",
        "many_agent_step",
        "channel",
        "controller.response",
        "agent_001",
        1,
        3,
    )
    assert located(faulted["first_mismatch"]) == expected
    assert faulted["raw_ticks"] == 33

    compared = determinism("compare", baseline, faulted["path"], "--stop-at-first")
    assert compared.returncode == 1, compared.stderr
    assert "  channel: controller.response\n  entity: agent_001\n  row: 1\n  byte: 3" in (
        compared.stdout
    )

    replay = ["--library", str(HOOK_LIBRARY), "--symbol", "many_agent_step"]
    reproduced = determinism("replay-hook", baseline, *replay)
    assert reproduced.returncode == 0, reproduced.stdout + reproduced.stderr
    assert reproduced.stdout.startswith("reproduced: 600 ticks")
    replayed = determinism("replay-hook", faulted["path"], *replay)
    assert replayed.returncode == 1, replayed.stdout + replayed.stderr
    assert compared.stdout.split("\n  left")[0] in replayed.stdout

    # The Python hook reproduces the native hook's recording too.
    path = tmp_path / "python-replay"
    path.mkdir()
    python = subprocess.run(
        [sys.executable, str(EXAMPLE / "main.py")],
        env=env(path, ELODIN_MANY_AGENT_HOOK="python", ELODIN_MANY_AGENT_REPLAY=baseline),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert python.returncode == 0, python.stdout + python.stderr
    assert json.loads((path / "result.json").read_text())["ticks"] == 600


def test_a_completion_race_differs_at_the_response(tmp_path):
    race = {"ELODIN_MANY_AGENT_FAULT_MODE": "completion-race", **FAULT}
    reference = run(tmp_path, "race-0", ELODIN_MANY_AGENT_WITNESS_RAW="1", **race)["path"]
    for repeat in range(1, 5):
        witness = run(
            tmp_path, f"race-{repeat}", ELODIN_MANY_AGENT_WITNESS_REFERENCE=reference, **race
        )
        if witness["first_mismatch"]:
            break
    mismatch = witness["first_mismatch"]
    assert mismatch, "the race never changed a response"
    # The ingress record comes first, so the state sent that tick was equal.
    assert (mismatch["stage"], mismatch["record"], mismatch["entity"]) == (
        "after_hook:0",
        "controller.response",
        "agent_001",
    )
    assert mismatch["tick"] >= 300


def test_a_causal_token_repeats_exactly(tmp_path):
    token = {"ELODIN_MANY_AGENT_FAULT_MODE": "causal-token", **FAULT}
    first = run(tmp_path, "token-0", **token)
    for repeat in range(1, 3):
        witness = run(
            tmp_path, f"token-{repeat}", ELODIN_MANY_AGENT_WITNESS_REFERENCE=first["path"], **token
        )
        assert witness["first_mismatch"] is None
        assert witness["root"] == first["root"]
