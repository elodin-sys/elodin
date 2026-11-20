#!/usr/bin/env uv run

from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import socket
import sys
import time
from pathlib import Path

import elodin as el
import numpy as np

import schematic
from protocol import (
    COMMAND_DTYPE,
    COMMAND_SIZE,
    RESPONSE_SIZE,
    STATE_SIZE,
    decode_responses,
    encode_states,
    state_frames,
)
from sim import CAPTURE_SHAPES, RATE_HZ, build


ROOT = Path(__file__).resolve().parent
HOOK_LIBRARY = (
    ROOT
    / "hook/build/release"
    / ("libmany_agent_hook.dylib" if sys.platform == "darwin" else "libmany_agent_hook.so")
)
AGENT_COUNT = int(os.environ.get("ELODIN_MANY_AGENT_AGENTS", "24"))
OBJECT_COUNT = int(os.environ.get("ELODIN_MANY_AGENT_OBJECTS", str(AGENT_COUNT)))
MAX_TICKS = int(os.environ.get("ELODIN_MANY_AGENT_MAX_TICKS", "60000"))
MODE = os.environ.get("ELODIN_MANY_AGENT_MODE", "lockstep-history")
HOOK = os.environ.get("ELODIN_MANY_AGENT_HOOK", "native")
PORT_BASE = int(os.environ.get("ELODIN_MANY_AGENT_PORT_BASE", "13000"))
TIMEOUT_S = float(os.environ.get("ELODIN_MANY_AGENT_TIMEOUT_S", "0.25"))
REAL_TIME = os.environ.get("ELODIN_MANY_AGENT_REALTIME", "1") != "0"
WORKERS = int(os.environ.get("ELODIN_MANY_AGENT_WORKERS", "8"))
HOOK_WORKERS = int(os.environ.get("ELODIN_MANY_AGENT_HOOK_WORKERS", "2"))
DB_PATH = Path(os.environ.get("ELODIN_DB_PATH", "dbs/many-agents"))
RESULT_PATH = os.environ.get("ELODIN_MANY_AGENT_RESULT_JSON")
CAMERA_TIMEOUT_S = float(os.environ.get("ELODIN_MANY_AGENT_CAMERA_TIMEOUT_S", "10"))
WITNESS_PATH = os.environ.get("ELODIN_MANY_AGENT_WITNESS_PATH")
WITNESS_RAW = os.environ.get("ELODIN_MANY_AGENT_WITNESS_RAW", "0")
WITNESS_REFERENCE = os.environ.get("ELODIN_MANY_AGENT_WITNESS_REFERENCE")
REPLAY = os.environ.get("ELODIN_MANY_AGENT_REPLAY")
FAULT_MODE = os.environ.get("ELODIN_MANY_AGENT_FAULT_MODE", "")
FAULT_TICK = int(os.environ.get("ELODIN_MANY_AGENT_FAULT_TICK", "4096"))
FAULT_AGENT = int(os.environ.get("ELODIN_MANY_AGENT_FAULT_AGENT", "7"))
FAULT_BYTE = int(os.environ.get("ELODIN_MANY_AGENT_FAULT_BYTE", "3"))
FAULT_MASK = int(os.environ.get("ELODIN_MANY_AGENT_FAULT_MASK", "1"))
LOCALHOST = socket.gethostbyname("localhost")
TICK_US = round(1e6 / RATE_HZ)
CAMERA_FPS = 25
FRAME_US = round(1e6 / CAMERA_FPS)
# A frame is due two frame periods after it was captured.
LATENCY_US = 2 * FRAME_US

if MODE not in {"runtime", "local", "lockstep", "lockstep-history"}:
    raise ValueError("ELODIN_MANY_AGENT_MODE must be runtime, local, lockstep or lockstep-history")
LOCKSTEP = MODE in {"lockstep", "lockstep-history"}
HISTORY = MODE == "lockstep-history"
if HOOK not in {"native", "python"}:
    raise ValueError("ELODIN_MANY_AGENT_HOOK must be native or python")
if PORT_BASE < 1024 or PORT_BASE + AGENT_COUNT * 2 > 65535:
    raise ValueError("configured port range is invalid")
if FAULT_MODE not in {"", "bit-flip", "completion-race", "causal-token"}:
    raise ValueError(
        "ELODIN_MANY_AGENT_FAULT_MODE must be bit-flip, completion-race or causal-token"
    )


def agent_name(index: int) -> str:
    return f"agent_{index:03d}"


def object_name(index: int) -> str:
    return f"object_{index:03d}"


AGENT_FIELDS = ("position", "velocity", "sensor_frame", "diagnostic_bank")
OBJECT_FIELDS = ("object_position", "object_velocity", "object_signal")


def write_result(tick: int, values: dict[str, np.ndarray], commands: np.ndarray, **extra) -> None:
    if not RESULT_PATH:
        return
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(np.asarray(values[name], dtype="<f8").tobytes())
    digest.update(np.asarray(commands, dtype="<f8").tobytes())
    path = Path(RESULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tick": tick,
                "agent_count": AGENT_COUNT,
                "object_count": OBJECT_COUNT,
                "sha256": digest.hexdigest(),
                **extra,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class ControllerBridge:
    """Sends every agent's state to its controller and waits for all of their replies."""

    def __init__(self, states: np.ndarray) -> None:
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.replies = np.zeros(AGENT_COUNT, dtype=COMMAND_DTYPE)
        packets = memoryview(states.view(np.uint8))
        replies = memoryview(self.replies.view(np.uint8))
        self.sends = [
            (packets[index * STATE_SIZE :][:STATE_SIZE], (LOCALHOST, PORT_BASE + index * 2))
            for index in range(AGENT_COUNT)
        ]
        # epoll on Linux, kqueue on macOS.
        self.poll = selectors.DefaultSelector()
        for index in range(AGENT_COUNT):
            command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            command_socket.bind((LOCALHOST, PORT_BASE + index * 2 + 1))
            command_socket.setblocking(False)
            reply = replies[index * COMMAND_SIZE :][:COMMAND_SIZE]
            self.poll.register(command_socket, selectors.EVENT_READ, (index, reply))

    def exchange(self) -> np.ndarray:
        sendto = self.state_socket.sendto
        for packet, address in self.sends:
            sendto(packet, address)
        pending = set(range(AGENT_COUNT))
        while pending:
            events = self.poll.select(TIMEOUT_S)
            if not events:
                raise RuntimeError("controller response timeout")
            for key, _ in events:
                index, reply = key.data
                key.fileobj.recv_into(reply)
                pending.discard(index)
        return self.replies


world, system, agents, objects, camera = build(AGENT_COUNT, OBJECT_COUNT)
world.schematic(schematic.build(AGENT_COUNT, OBJECT_COUNT, camera=HISTORY))
bridge: ControllerBridge | None = None
states = state_frames(AGENT_COUNT)

# A replay reads the recorded replies, so it needs no controllers.
if LOCKSTEP and not REPLAY:
    controller_binary = Path(
        os.environ.get(
            "ELODIN_MANY_AGENT_CONTROLLER_BIN",
            ROOT / "controller/build/release/many-agent-controller",
        )
    ).resolve()
    if not controller_binary.is_file():
        raise RuntimeError("controller binary missing; build examples/many-agents/controller")
    for index in range(AGENT_COUNT):
        world.recipe(
            el.s10.PyRecipe.process(
                name=f"many-agent-controller-{index:03d}",
                cmd=str(controller_binary),
                args=[
                    str(PORT_BASE + index * 2),
                    str(PORT_BASE + index * 2 + 1),
                    str(index),
                ],
                cwd=str(ROOT),
                ready=el.s10.Ready.delay(100),
                ready_timeout="30s",
            )
        )
if HISTORY:
    # Looks straight down on the scene; Elodin's render server records its frames in the DB.
    world.sensor_camera(
        entity=camera,
        name="overhead",
        width=320,
        height=240,
        fov=60.0,
        rot_offset=[0.0, 90.0, 0.0],
        fps=CAMERA_FPS,
        create_frustum=True,
        frustums_color=[1.0, 0.85, 0.0, 1.0],
        frustums_face_alpha=0.03,
    )


observed = [objects[index % OBJECT_COUNT] for index in range(AGENT_COUNT)]
exchange = world.prepare_batch(
    {
        **{field: world.prepare_component(field, agents) for field in AGENT_FIELDS},
        **{field: world.prepare_component(field, observed) for field in OBJECT_FIELDS},
        "command": world.prepare_component("command", agents, access="write"),
    }
)
objects_state = world.prepare_batch(
    {field: world.prepare_component(field, objects) for field in OBJECT_FIELDS}
)
witness = world.prepare_witness(
    {"exchange": exchange, "objects": objects_state},
    channels={
        "controller.ingress": (exchange, STATE_SIZE),
        "controller.response": (exchange, RESPONSE_SIZE),
    },
    # The replies carry the controllers' service times, so they are kept for replay, not compared.
    external={"controller.reply": (exchange, COMMAND_SIZE)},
    # The history hook's evidence depends on when the render server delivers frames.
    stages=["after_systems", "after_hook:0", "after_hook:2", "after_commit"] if HISTORY else None,
)
if HISTORY:
    frames = world.prepare_message_history(schematic.CAMERA)
    agent_truth = world.prepare_component_history("agent_000.position")
    capture = world.prepare_timestamped_writer(
        {field: f"camera.capture_{field}" for field in CAPTURE_SHAPES}
    )
frame_counts = {"found": 0, "missing": 0, "late": 0}
# The capture time of the camera's first frame and of the next one due, the frame last counted
# late, and when the camera was last seen sending frames.
camera_state = {"first": None, "next": None, "late": None, "seen": None}


def history_step(tick: int, ctx: el.WorldStepContext) -> None:
    """Looks up each camera frame by its exact capture time once it is due, pairs it with agent
    0's position at that time, and appends that evidence after the commit."""
    state, now = camera_state, time.monotonic()
    state["seen"] = state["seen"] or now
    if state["next"] is None:
        # The first frame sets the camera's schedule; look for it once per frame period.
        newest = (
            None
            if tick % (FRAME_US // TICK_US)
            else ctx.read_at(frames, ctx.timestamp, "at_or_before")
        )
        if newest is None:
            if now - state["seen"] > CAMERA_TIMEOUT_S:
                raise RuntimeError("the sensor camera never sent a frame")
            return
        state.update(first=newest[0], next=newest[0], seen=now)
    due = state["next"]
    if due + LATENCY_US > ctx.timestamp:
        return
    found = ctx.read_at(frames, due, "exact")
    if found is None and ctx.read_at(frames, ctx.timestamp, "at_or_before")[0] <= due:
        # Not delivered yet: the render server is behind, or has stopped.
        if now - state["seen"] > CAMERA_TIMEOUT_S:
            raise RuntimeError("the sensor camera stopped sending frames")
        if state["late"] != due:
            state["late"] = due
            frame_counts["late"] += 1
        return
    # A newer frame means the render server skipped this one.
    state["seen"] = now
    frame_counts["missing" if found is None else "found"] += 1
    ctx.defer_write(
        capture,
        ctx.timestamp + 1,
        {
            "frame": (due - state["first"]) // FRAME_US,
            "time": due,
            "disposition": 1.0 if found is None else 0.0,
            "truth": ctx.read_at(agent_truth, due, "at_or_before")[1],
        },
    )
    state["next"] = due + FRAME_US


def finish(tick: int, ctx: el.WorldStepContext, view, commands: np.ndarray) -> None:
    """At the last tick, writes the result and stops the controllers."""
    if tick < MAX_TICKS - 1:
        return
    values = {
        f"{agent_name(index)}.{field}": view[field][index]
        for field in AGENT_FIELDS
        for index in range(AGENT_COUNT)
    }
    objects_view = ctx.batch(objects_state)
    values |= {
        f"{object_name(index)}.{field}": objects_view[field][index]
        for field in OBJECT_FIELDS
        for index in range(OBJECT_COUNT)
    }
    write_result(tick, values, commands, **({"frames": frame_counts} if HISTORY else {}))
    ctx.stop_recipes()


def exchange_step(tick: int, ctx: el.WorldStepContext) -> None:
    """The Python hook: gets this tick's commands and writes them into the world. It witnesses
    the state it sends, the replies, and the responses it takes from them; a replay supplies the
    recorded replies."""
    global bridge
    with ctx.batch(exchange) as view:
        if not LOCKSTEP:
            view.command[...] = np.clip(
                1.4 * (view.object_position - view.position)
                + 0.45 * (view.object_velocity - view.velocity),
                -12.0,
                12.0,
            )
            return
        encode_states(
            states,
            tick=tick,
            agent_position=view.position,
            agent_velocity=view.velocity,
            object_position=view.object_position,
            object_velocity=view.object_velocity,
            diagnostics=np.concatenate(
                [view.sensor_frame, view.diagnostic_bank, view.object_signal], axis=1
            ),
        )
        ctx.witness_bytes("controller.ingress", states)
        replies = ctx.witness_replay("controller.reply")
        if replies is None:
            bridge = bridge or ControllerBridge(states)
            replies = bridge.exchange()
            ctx.witness_bytes("controller.reply", replies)
        else:
            replies = replies.view(COMMAND_DTYPE)[:, 0]
        responses = decode_responses(replies, tick)
        if FAULT_MODE == "bit-flip" and tick == FAULT_TICK:
            responses.view(np.uint8)[FAULT_AGENT, FAULT_BYTE] ^= FAULT_MASK
        ctx.witness_bytes("controller.response", responses)
        view.command[...] = responses[:, :3]


def finish_step(tick: int, ctx: el.WorldStepContext) -> None:
    """Runs `finish` with the commands an earlier hook wrote."""
    if tick >= MAX_TICKS - 1:
        with ctx.batch(exchange) as view:
            finish(tick, ctx, view, view.command)


if MODE == "runtime":
    pre_commit = None
else:
    if LOCKSTEP and HOOK == "native":
        hook_library = Path(
            os.environ.get(
                "ELODIN_MANY_AGENT_HOOK_LIB",
                HOOK_LIBRARY,
            )
        ).resolve()
        if not hook_library.is_file():
            raise RuntimeError("hook library missing; build examples/many-agents/hook")
        exchange_hook = el.NativeStepHook.load(
            str(hook_library), input=exchange, symbol="many_agent_step", workers=HOOK_WORKERS
        )
    else:
        exchange_hook = exchange_step
    # One ordered boundary: each hook sees the world writes of those before it.
    pre_commit = [exchange_hook, *([history_step] if HISTORY else []), finish_step]


if REPLAY:
    replay = world.replay_pre_commit(REPLAY, pre_commit)
    if RESULT_PATH:
        Path(RESULT_PATH).write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
    mismatch = replay["first_mismatch"]
    print(el.determinism.describe(mismatch) if mismatch else f"reproduced {replay['ticks']} ticks")
    raise SystemExit(1 if mismatch else 0)


def witness_raw(value: str) -> bool | tuple[int, int]:
    """`1` keeps every tick's exact bytes, `START:STOP` those ticks', `0` none."""
    if ":" in value:
        start, stop = value.split(":")
        return int(start), int(stop)
    return value == "1"


if DB_PATH.exists():
    if DB_PATH.is_dir():
        shutil.rmtree(DB_PATH)
    else:
        DB_PATH.unlink()

world.run(
    system,
    simulation_rate=RATE_HZ,
    telemetry_rate=RATE_HZ,
    generate_real_time=REAL_TIME,
    max_ticks=MAX_TICKS,
    db_path=str(DB_PATH),
    pre_commit=pre_commit,
    interactive=False,
    start_timestamp=0,
    workers=WORKERS,
    determinism_trace=el.DeterminismTrace(
        WITNESS_PATH, witness, raw=witness_raw(WITNESS_RAW), reference=WITNESS_REFERENCE
    )
    if WITNESS_PATH
    else None,
)
