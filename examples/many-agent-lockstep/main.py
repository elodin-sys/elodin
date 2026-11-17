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
    decode_command,
    decode_commands,
    encode_state_into,
    encode_states,
    state_frames,
)
from sim import RATE_HZ, build


ROOT = Path(__file__).resolve().parent
AGENT_COUNT = int(os.environ.get("ELODIN_MANY_AGENT_AGENTS", "24"))
OBJECT_COUNT = int(os.environ.get("ELODIN_MANY_AGENT_OBJECTS", str(AGENT_COUNT)))
MAX_TICKS = int(os.environ.get("ELODIN_MANY_AGENT_MAX_TICKS", "10000"))
MODE = os.environ.get("ELODIN_MANY_AGENT_MODE", "lockstep")
IO_MODE = os.environ.get("ELODIN_MANY_AGENT_IO", "native")
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
if IO_MODE not in {"native", "prepared", "batch", "scalar"}:
    raise ValueError("ELODIN_MANY_AGENT_IO must be native, prepared, batch or scalar")
if PORT_BASE < 1024 or PORT_BASE + AGENT_COUNT * 2 > 65535:
    raise ValueError("configured port range is invalid")


def agent_name(index: int) -> str:
    return f"agent_{index:03d}"


def object_name(index: int) -> str:
    return f"object_{index:03d}"


AGENT_FIELDS = ("position", "velocity", "sensor_frame", "diagnostic_bank")
OBJECT_FIELDS = ("object_position", "object_velocity", "object_signal")
READS = tuple(
    f"{agent_name(index)}.{field}" for index in range(AGENT_COUNT) for field in AGENT_FIELDS
) + tuple(
    f"{object_name(index)}.{field}" for index in range(OBJECT_COUNT) for field in OBJECT_FIELDS
)


class LockstepBridge:
    def __init__(self) -> None:
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_sockets = []
        self.command_by_socket = {}
        self.state_buffers = [bytearray(STATE_SIZE) for _ in range(AGENT_COUNT)]
        for index in range(AGENT_COUNT):
            command_port = PORT_BASE + index * 2 + 1
            command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            command_socket.bind((LOCALHOST, command_port))
            command_socket.setblocking(False)
            self.command_sockets.append(command_socket)
            self.command_by_socket[command_socket] = index

    def exchange(self, tick: int, values: dict[str, np.ndarray]) -> np.ndarray:
        for index in range(AGENT_COUNT):
            agent = agent_name(index)
            observed = object_name(index % OBJECT_COUNT)
            diagnostics = np.concatenate(
                [
                    values[f"{agent}.sensor_frame"],
                    values[f"{agent}.diagnostic_bank"],
                    values[f"{observed}.object_signal"],
                ]
            )
            packet = encode_state_into(
                self.state_buffers[index],
                tick=tick,
                agent_position=values[f"{agent}.position"],
                agent_velocity=values[f"{agent}.velocity"],
                object_position=values[f"{observed}.object_position"],
                object_velocity=values[f"{observed}.object_velocity"],
                diagnostics=diagnostics,
            )
            self.state_socket.sendto(
                packet,
                (LOCALHOST, PORT_BASE + index * 2),
            )

        commands = np.zeros((AGENT_COUNT, 3), dtype=np.float64)
        pending = set(self.command_sockets)
        while pending:
            readable, _, exceptional = select.select(
                tuple(pending),
                (),
                tuple(pending),
                TIMEOUT_S,
            )
            if exceptional or not readable:
                raise RuntimeError("controller response timeout")
            for command_socket in readable:
                packet, _ = command_socket.recvfrom(COMMAND_SIZE)
                index = self.command_by_socket[command_socket]
                commands[index] = decode_command(packet, tick)
                pending.remove(command_socket)
        return commands


def local_commands(values: dict[str, np.ndarray]) -> np.ndarray:
    commands = np.zeros((AGENT_COUNT, 3), dtype=np.float64)
    for index in range(AGENT_COUNT):
        agent = agent_name(index)
        observed = object_name(index % OBJECT_COUNT)
        position_error = values[f"{observed}.object_position"] - values[f"{agent}.position"]
        velocity_error = values[f"{observed}.object_velocity"] - values[f"{agent}.velocity"]
        commands[index] = np.clip(
            1.4 * position_error + 0.45 * velocity_error,
            -12.0,
            12.0,
        )
    return commands


def read_values(ctx: el.StepContext) -> dict[str, np.ndarray]:
    if IO_MODE == "batch":
        return ctx.component_batch_operation(reads=READS)
    return {name: ctx.read_component(name) for name in READS}


def write_commands(
    ctx: el.StepContext,
    commands: np.ndarray,
) -> None:
    writes = {f"{agent_name(index)}.command": commands[index] for index in range(AGENT_COUNT)}
    if IO_MODE == "batch":
        ctx.component_batch_operation(writes=writes)
    else:
        for name, value in writes.items():
            ctx.write_component(name, value)


def write_result(tick: int, values: dict[str, np.ndarray], commands: np.ndarray) -> None:
    if not RESULT_PATH:
        return
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(np.asarray(values[name], dtype="<f8").tobytes())
    digest.update(np.asarray(commands, dtype="<f8").tobytes())
    agent_digests = []
    for index in range(AGENT_COUNT):
        agent_digest = hashlib.sha256()
        agent = agent_name(index)
        for field in AGENT_FIELDS:
            agent_digest.update(np.asarray(values[f"{agent}.{field}"], dtype="<f8").tobytes())
        agent_digest.update(np.asarray(commands[index], dtype="<f8").tobytes())
        agent_digests.append(agent_digest.hexdigest())
    object_digests = []
    for index in range(OBJECT_COUNT):
        object_digest = hashlib.sha256()
        observed = object_name(index)
        for field in OBJECT_FIELDS:
            object_digest.update(np.asarray(values[f"{observed}.{field}"], dtype="<f8").tobytes())
        object_digests.append(object_digest.hexdigest())
    path = Path(RESULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "elodin.many_agent.result.v1",
                "tick": tick,
                "agent_count": AGENT_COUNT,
                "object_count": OBJECT_COUNT,
                "sha256": digest.hexdigest(),
                "agent_sha256": agent_digests,
                "object_sha256": object_digests,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class PreparedBridge:
    """`LockstepBridge` over batch views: every frame is encoded and decoded at once."""

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
bridge: LockstepBridge | PreparedBridge | None = None

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


def post_step(tick: int, ctx: el.StepContext) -> None:
    global bridge
    values = read_values(ctx)
    if MODE == "lockstep":
        if bridge is None:
            bridge = LockstepBridge()
        commands = bridge.exchange(tick, values)
    else:
        commands = local_commands(values)
    write_commands(ctx, commands)
    if tick >= MAX_TICKS - 1:
        write_result(tick, values, commands)
        ctx.stop_recipes()


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
    global bridge
    with ctx.batch(exchange) as view:
        if MODE == "lockstep":
            if bridge is None:
                bridge = PreparedBridge()
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
    """`finish` after the native hook, which wrote the commands."""
    if tick >= MAX_TICKS - 1:
        with ctx.batch(exchange) as view:
            finish(tick, ctx, view, view.command)


if MODE == "runtime" or IO_MODE in {"batch", "scalar"}:
    pre_commit = None
elif MODE == "lockstep" and IO_MODE == "native":
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
    post_step=post_step if MODE != "runtime" and IO_MODE in {"batch", "scalar"} else None,
    pre_commit=pre_commit,
    interactive=False,
    start_timestamp=0,
    workers=WORKERS,
)
