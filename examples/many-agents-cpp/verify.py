"""Prove the C++ program reproduces the many-agents Elodin run, and repeats exactly.

    python3 examples/many-agents-cpp/verify.py --bin DIR [--agents N] [--ticks N]

`DIR` holds `many_agents_cpp` and `many_agents_fsw`, built for `--agents` agents. The script runs
the original `examples/many-agents` in `lockstep` mode with a witness, then:

1. the same Elodin run with the C++ flight software in place of the Rust controllers;
2. the C++ host, flight software in-process, then as one process per agent;
3. the C++ host again, and with one worker thread;
4. the C++ host with a one-bit fault, which the comparison must locate exactly;
5. Elodin's own native hook, replayed from the C++ witness;
6. the C++ trajectories against the positions, velocities and commands Elodin recorded.

It prints one line per check, plots the trajectories, and exits 1 if any check fails.
"""

import argparse
import json
import os
import random
import socket
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

import elodin.determinism as determinism

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = ROOT / "examples" / "many-agents"
CONTROLLER = ORIGINAL / "controller/build/release/many-agent-controller"
HOOK = (
    ORIGINAL
    / "hook/build/release"
    / ("libmany_agent_hook.dylib" if sys.platform == "darwin" else "libmany_agent_hook.so")
)
FAULT = {"tick": 300, "agent": 1, "byte": 3}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--bin", type=Path, required=True, help="directory of the C++ programs")
parser.add_argument("--agents", type=int, default=4)
parser.add_argument("--ticks", type=int, default=600)
parser.add_argument("--out", type=Path, default=HERE / "build" / "verify")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
FAULT["tick"] = min(FAULT["tick"], args.ticks - 1)
FAULT["agent"] = min(FAULT["agent"], args.agents - 1)
failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{f': {detail}' if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def free_port() -> int:
    """A free port P with P + 1 and P + 2 free for the DB's asset and gRPC servers."""
    while True:
        port = random.randrange(20000, 32000)
        try:
            for offset in range(3):
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", port + offset))
        except OSError:
            continue
        return port


def elodin(name: str, controller: Path, port_base: int) -> dict:
    """Runs examples/many-agents in lockstep with a raw witness; returns its receipt's witness."""
    path = args.out / name
    env = {
        **os.environ,
        "ELODIN_MANY_AGENT_MODE": "lockstep",
        "ELODIN_MANY_AGENT_AGENTS": str(args.agents),
        "ELODIN_MANY_AGENT_MAX_TICKS": str(args.ticks),
        "ELODIN_MANY_AGENT_REALTIME": "0",
        "ELODIN_MANY_AGENT_PORT_BASE": str(port_base),
        # The run starts as the controllers do; this only waits longer for their first replies.
        "ELODIN_MANY_AGENT_TIMEOUT_S": "10",
        "ELODIN_MANY_AGENT_CONTROLLER_BIN": str(controller),
        "ELODIN_DB_PATH": str(path.with_suffix(".db")),
        "ELODIN_SIM_SUMMARY_JSON": str(path.with_suffix(".json")),
        "ELODIN_MANY_AGENT_WITNESS_PATH": str(path.with_suffix(".witness")),
        "ELODIN_MANY_AGENT_WITNESS_RAW": "1",
    }
    command = [sys.executable, str(ORIGINAL / "main.py"), "run", f"127.0.0.1:{free_port()}"]
    with path.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"the Elodin run {name} failed; see {path.with_suffix('.log')}")
    return json.loads(path.with_suffix(".json").read_text())["witness"]


def cpp(name: str, *options: str) -> Path:
    """Runs the C++ host with a raw witness; returns the witness path."""
    witness = args.out / f"{name}.witness"
    command = [
        str(args.bin / "many_agents_cpp"),
        "--ticks",
        str(args.ticks),
        "--realtime",
        "0",
        "--witness",
        str(witness),
        "--witness-raw",
        "--receipt",
        str(args.out / f"{name}.json"),
        *options,
    ]
    with (args.out / f"{name}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"the C++ run {name} failed; see {args.out / name}.log")
    return witness


def compare(left: Path, right: Path) -> dict:
    return determinism.compare(left, right, stop_at_first=True)


def same(name: str, left: Path, right: Path) -> None:
    result = compare(left, right)
    mismatch = result["first_mismatch"]
    detail = f"{result['ticks']} ticks identical" if result["identical"] else ""
    if mismatch:
        detail = determinism.describe(mismatch).replace("\n", ";")
    check(name, result["identical"], detail)


def read_witness(path: Path) -> tuple[dict, list[list[bytes]]]:
    """The header and every tick's raw records of a witness kept raw (libs/nox-py/src/witness)."""
    data = path.read_bytes()
    (header_len,) = struct.unpack_from("<I", data, 12)
    header = json.loads(data[16 : 16 + header_len])
    records = header["schema"]["records"]
    hashed = sum(record["kind"] != "external" for record in records)
    at, ticks = 16 + header_len, []
    while data[at] == 1:
        at += 17 + 32 * hashed
        raw, at = data[at], at + 1
        if not raw:
            sys.exit(f"{path} did not keep tick {len(ticks)} raw")
        tick = []
        for _ in records:
            (size,) = struct.unpack_from("<I", data, at)
            tick.append(data[at + 4 : at + 4 + size])
            at += 4 + size
        ticks.append(tick)
    return header, ticks


def series(path: Path, stage: str, name: str, width: int) -> np.ndarray:
    """One record of every tick, as (ticks, rows, width)."""
    header, ticks = read_witness(path)
    stages = header["schema"]["stages"]
    index = next(
        i
        for i, record in enumerate(header["schema"]["records"])
        if stages[record["stage"]] == stage and record["name"] == name
    )
    return np.stack([np.frombuffer(tick[index], "<f8").reshape(-1, width) for tick in ticks])


print(f"many-agents, lockstep, {args.agents} agents, {args.ticks} ticks, 8 workers")
baseline_receipt = elodin("elodin", CONTROLLER, 17000)
baseline = Path(baseline_receipt["path"])

fsw = elodin("elodin-cpp-fsw", args.bin / "many_agents_fsw", 17100)
check(
    "Elodin with the C++ flight software records the root of Elodin with the Rust controllers",
    fsw["root"] == baseline_receipt["root"],
    baseline_receipt["root"][:16],
)

in_process = cpp("cpp-in-process", "--csv", str(args.out / "cpp.csv"))
same("C++ host, flight software in-process, against Elodin", baseline, in_process)
processes = cpp("cpp-processes", "--fsw", "processes", "--port-base", "17200")
same("C++ host, flight software as processes, against Elodin", baseline, processes)

repeat = cpp("cpp-repeat")
same("C++ host repeats exactly", in_process, repeat)
serial = cpp("cpp-one-worker", "--workers", "1")
same("C++ host with one worker thread records the same witness", in_process, serial)

faulted = cpp(
    "cpp-fault",
    "--fault-tick",
    str(FAULT["tick"]),
    "--fault-agent",
    str(FAULT["agent"]),
    "--fault-byte",
    str(FAULT["byte"]),
)
mismatch = compare(baseline, faulted)["first_mismatch"] or {}
located = (mismatch.get("tick"), mismatch.get("record"), mismatch.get("entity"))
located += (mismatch.get("row"), mismatch.get("byte"))
expected = (FAULT["tick"], "controller.response", f"agent_{FAULT['agent']:03d}")
expected += (FAULT["agent"], FAULT["byte"])
check(
    "a one-bit fault is located at its tick, record, agent and byte",
    located == expected,
    determinism.describe(mismatch).replace("\n", ";") if mismatch else "no mismatch found",
)

replay = determinism.replay_hook(in_process, HOOK, "many_agent_step")
check(
    "Elodin's native hook replays the C++ witness byte for byte",
    replay["first_mismatch"] is None and replay["ticks"] == args.ticks,
    f"{replay['ticks']} ticks, {replay['records']} records, {replay['bytes']} bytes",
)

table = np.loadtxt(args.out / "cpp.csv", delimiter=",", skiprows=1)
native = table[:, 2:].reshape(args.ticks, args.agents, 9)
recorded = np.concatenate(
    [
        series(baseline, "after_systems", "exchange.position", 3),
        series(baseline, "after_systems", "exchange.velocity", 3),
        series(baseline, "after_hook:0", "exchange.command", 3),
    ],
    axis=2,
)
difference = np.abs(native - recorded).max(axis=(1, 2))
check(
    "C++ positions, velocities and commands equal Elodin's at every tick",
    np.array_equal(native, recorded),
    f"largest difference {np.nanmax(difference):.3g}",
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

objects = series(baseline, "after_systems", "objects.object_position", 3)
figure, (tracks, error) = plt.subplots(1, 2, figsize=(12, 5.5))
for agent in range(args.agents):
    line = tracks.plot(*recorded[:, agent, :2].T, linewidth=3, alpha=0.35)[0]
    tracks.plot(*native[:, agent, :2].T, "--", color=line.get_color(), linewidth=1)
for row in range(objects.shape[1]):
    tracks.plot(*objects[:, row, :2].T, ":", color="orange", linewidth=1)
tracks.set_title("agents: Elodin (wide) and C++ (dashed); objects dotted")
tracks.set_xlabel("x (m)")
tracks.set_ylabel("y (m)")
tracks.axis("equal")
error.plot(np.arange(args.ticks), difference)
error.set_title("largest |C++ - Elodin| over agents and fields")
error.set_xlabel("tick")
figure.tight_layout()
plot = args.out / "trajectories.png"
figure.savefig(plot, dpi=110)
print(f"saved {plot}")

if failures:
    sys.exit(f"{len(failures)} check(s) failed")
print("all checks passed")
