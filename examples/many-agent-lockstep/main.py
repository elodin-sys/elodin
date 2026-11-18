#!/usr/bin/env uv run

from __future__ import annotations

import hashlib
import json
import os
import select
import shutil
import socket
from pathlib import Path

import elodin as el
import numpy as np

from protocol import (
    COMMAND_DTYPE,
    COMMAND_SIZE,
    STATE_SIZE,
    decode_commands,
    encode_states,
    state_frames,
)
from sim import RATE_HZ, build


ROOT = Path(__file__).resolve().parent
AGENT_COUNT = int(os.environ.get("ELODIN_MANY_AGENT_AGENTS", "24"))
OBJECT_COUNT = int(os.environ.get("ELODIN_MANY_AGENT_OBJECTS", str(AGENT_COUNT)))
MAX_TICKS = int(os.environ.get("ELODIN_MANY_AGENT_MAX_TICKS", "10000"))
MODE = os.environ.get("ELODIN_MANY_AGENT_MODE", "lockstep")
HOOK = os.environ.get("ELODIN_MANY_AGENT_HOOK", "native")
PORT_BASE = int(os.environ.get("ELODIN_MANY_AGENT_PORT_BASE", "13000"))
TIMEOUT_S = float(os.environ.get("ELODIN_MANY_AGENT_TIMEOUT_S", "0.25"))
REAL_TIME = os.environ.get("ELODIN_MANY_AGENT_REALTIME", "1") != "0"
WORKERS = int(os.environ.get("ELODIN_MANY_AGENT_WORKERS", "8"))
HOOK_WORKERS = int(os.environ.get("ELODIN_MANY_AGENT_HOOK_WORKERS", "2"))
DB_PATH = Path(os.environ.get("ELODIN_DB_PATH", "dbs/many-agent-lockstep"))
RESULT_PATH = os.environ.get("ELODIN_MANY_AGENT_RESULT_JSON")
LOCALHOST = socket.gethostbyname("localhost")

if MODE not in {"runtime", "local", "lockstep"}:
    raise ValueError("ELODIN_MANY_AGENT_MODE must be runtime, local, or lockstep")
if HOOK not in {"native", "python"}:
    raise ValueError("ELODIN_MANY_AGENT_HOOK must be native or python")
if PORT_BASE < 1024 or PORT_BASE + AGENT_COUNT * 2 > 65535:
    raise ValueError("configured port range is invalid")


def agent_name(index: int) -> str:
    return f"agent_{index:03d}"


def object_name(index: int) -> str:
    return f"object_{index:03d}"


AGENT_FIELDS = ("position", "velocity", "sensor_frame", "diagnostic_bank")
OBJECT_FIELDS = ("object_position", "object_velocity", "object_signal")


def write_result(tick: int, values: dict[str, np.ndarray], commands: np.ndarray) -> None:
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class ControllerBridge:
    """Sends every agent's state to its controller and waits for all of their commands."""

    def __init__(self) -> None:
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.frames = state_frames(AGENT_COUNT)
        self.replies = np.zeros(AGENT_COUNT, dtype=COMMAND_DTYPE)
        packets = memoryview(self.frames.view(np.uint8))
        replies = memoryview(self.replies.view(np.uint8))
        self.sends = [
            (packets[index * STATE_SIZE :][:STATE_SIZE], (LOCALHOST, PORT_BASE + index * 2))
            for index in range(AGENT_COUNT)
        ]
        self.receives = {}
        self.poll = select.epoll()
        for index in range(AGENT_COUNT):
            command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            command_socket.bind((LOCALHOST, PORT_BASE + index * 2 + 1))
            command_socket.setblocking(False)
            reply = replies[index * COMMAND_SIZE :][:COMMAND_SIZE]
            self.receives[command_socket.fileno()] = (index, command_socket, reply)
            self.poll.register(command_socket, select.EPOLLIN)

    def exchange(self, tick: int, view) -> np.ndarray:
        encode_states(
            self.frames,
            tick=tick,
            agent_position=view.position,
            agent_velocity=view.velocity,
            object_position=view.object_position,
            object_velocity=view.object_velocity,
            diagnostics=np.concatenate(
                [view.sensor_frame, view.diagnostic_bank, view.object_signal], axis=1
            ),
        )
        sendto = self.state_socket.sendto
        for packet, address in self.sends:
            sendto(packet, address)
        pending = set(range(AGENT_COUNT))
        while pending:
            events = self.poll.poll(TIMEOUT_S)
            if not events:
                raise RuntimeError("controller response timeout")
            for fd, _ in events:
                index, command_socket, reply = self.receives[fd]
                command_socket.recv_into(reply)
                pending.discard(index)
        return decode_commands(self.replies, tick)


world, system, agents, objects = build(AGENT_COUNT, OBJECT_COUNT)
bridge: ControllerBridge | None = None

if MODE == "lockstep":
    controller_binary = Path(
        os.environ.get(
            "ELODIN_MANY_AGENT_CONTROLLER_BIN",
            ROOT / "controller/build/release/many-agent-controller",
        )
    ).resolve()
    if not controller_binary.is_file():
        raise RuntimeError(
            "controller binary missing; build examples/many-agent-lockstep/controller"
        )
    for index in range(AGENT_COUNT):
        world.recipe(
            el.s10.PyRecipe.process(
                name=f"many-agent-controller-{index:03d}",
                cmd=str(controller_binary),
                args=[
                    str(PORT_BASE + index * 2),
                    str(PORT_BASE + index * 2 + 1),
                ],
                cwd=str(ROOT),
                ready=el.s10.Ready.delay(100),
                ready_timeout="30s",
            )
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
    write_result(tick, values, commands)
    ctx.stop_recipes()


def exchange_step(tick: int, ctx: el.WorldStepContext) -> None:
    """The Python hook: gets this tick's commands and writes them into the world."""
    global bridge
    with ctx.batch(exchange) as view:
        if MODE == "lockstep":
            if bridge is None:
                bridge = ControllerBridge()
            commands = bridge.exchange(tick, view)
        else:
            commands = np.clip(
                1.4 * (view.object_position - view.position)
                + 0.45 * (view.object_velocity - view.velocity),
                -12.0,
                12.0,
            )
        view.command[...] = commands
        finish(tick, ctx, view, commands)


def finish_step(tick: int, ctx: el.WorldStepContext) -> None:
    """Runs `finish` after the native hook has written the commands."""
    if tick >= MAX_TICKS - 1:
        with ctx.batch(exchange) as view:
            finish(tick, ctx, view, view.command)


if MODE == "runtime":
    pre_commit = None
elif MODE == "lockstep" and HOOK == "native":
    hook_library = Path(
        os.environ.get(
            "ELODIN_MANY_AGENT_HOOK_LIB",
            ROOT / "hook/build/release/libmany_agent_hook.so",
        )
    ).resolve()
    if not hook_library.is_file():
        raise RuntimeError("hook library missing; build examples/many-agent-lockstep/hook")
    hook = el.NativeStepHook.load(
        str(hook_library), input=exchange, symbol="many_agent_step", workers=HOOK_WORKERS
    )
    pre_commit = [hook, finish_step]
else:
    pre_commit = exchange_step


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
)
