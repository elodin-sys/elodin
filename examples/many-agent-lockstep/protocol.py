"""Fixed-width loopback protocol for the many-agent lockstep example."""

from __future__ import annotations

import struct

import numpy as np


STATE_MAGIC = b"MALS"
COMMAND_MAGIC = b"MALC"
VERSION = 1
HEADER = struct.Struct("<4sHHQ")
STATE_DOUBLES = 126
COMMAND_DOUBLES = 11
STATE_SIZE = HEADER.size + STATE_DOUBLES * 8
COMMAND_SIZE = HEADER.size + COMMAND_DOUBLES * 8
HEADER_FIELDS = [("magic", "S4"), ("version", "<u2"), ("flags", "<u2"), ("tick", "<u8")]
# One packet per row, laid out as `HEADER` followed by the doubles.
STATE_DTYPE = np.dtype(HEADER_FIELDS + [("values", "<f8", (STATE_DOUBLES,))])
COMMAND_DTYPE = np.dtype(HEADER_FIELDS + [("values", "<f8", (COMMAND_DOUBLES,))])


def _vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must contain {size} values")
    return vector


def encode_state_into(
    buffer: bytearray,
    *,
    tick: int,
    agent_position,
    agent_velocity,
    object_position,
    object_velocity,
    diagnostics=(),
) -> memoryview:
    if len(buffer) != STATE_SIZE:
        raise ValueError(f"state buffer must be {STATE_SIZE} bytes")
    HEADER.pack_into(buffer, 0, STATE_MAGIC, VERSION, 0, int(tick))
    values = np.frombuffer(buffer, dtype="<f8", offset=HEADER.size)
    values.fill(0.0)
    values[0:3] = _vector(agent_position, 3, "agent_position")
    values[3:6] = _vector(agent_velocity, 3, "agent_velocity")
    values[6:9] = _vector(object_position, 3, "object_position")
    values[9:12] = _vector(object_velocity, 3, "object_velocity")
    diagnostic_values = np.asarray(diagnostics, dtype=np.float64).reshape(-1)
    count = min(diagnostic_values.size, STATE_DOUBLES - 12)
    values[12 : 12 + count] = diagnostic_values[:count]
    return memoryview(buffer)


def state_frames(count: int) -> np.ndarray:
    frames = np.zeros(count, dtype=STATE_DTYPE)
    frames["magic"] = STATE_MAGIC
    frames["version"] = VERSION
    return frames


def encode_states(
    frames: np.ndarray,
    *,
    tick: int,
    agent_position,
    agent_velocity,
    object_position,
    object_velocity,
    diagnostics,
) -> None:
    """`encode_state_into` for every row of `frames` at once."""
    frames["tick"] = tick
    values = frames["values"]
    values[:, 0:3] = agent_position
    values[:, 3:6] = agent_velocity
    values[:, 6:9] = object_position
    values[:, 9:12] = object_velocity
    end = 12 + diagnostics.shape[1]
    values[:, 12:end] = diagnostics
    values[:, end:] = 0.0


def decode_commands(replies: np.ndarray, expected_tick: int) -> np.ndarray:
    """`decode_command` for every row of `replies` at once."""
    if (
        np.any(replies["magic"] != COMMAND_MAGIC)
        or np.any(replies["version"] != VERSION)
        or np.any(replies["flags"] != 0)
        or np.any(replies["tick"] != expected_tick)
    ):
        raise ValueError("invalid command packet header")
    return replies["values"][:, :3].copy()


def decode_state(packet: bytes | bytearray | memoryview) -> tuple[int, np.ndarray]:
    if len(packet) != STATE_SIZE:
        raise ValueError(f"state packet must be {STATE_SIZE} bytes")
    magic, version, flags, tick = HEADER.unpack_from(packet)
    if magic != STATE_MAGIC or version != VERSION or flags != 0:
        raise ValueError("invalid state packet header")
    values = np.frombuffer(packet, dtype="<f8", offset=HEADER.size).copy()
    return tick, values


def encode_command(tick: int, command) -> bytes:
    values = np.zeros(COMMAND_DOUBLES, dtype="<f8")
    values[:3] = _vector(command, 3, "command")
    return HEADER.pack(COMMAND_MAGIC, VERSION, 0, int(tick)) + values.tobytes()


def decode_command(packet: bytes | bytearray | memoryview, expected_tick: int) -> np.ndarray:
    if len(packet) != COMMAND_SIZE:
        raise ValueError(f"command packet must be {COMMAND_SIZE} bytes")
    magic, version, flags, tick = HEADER.unpack_from(packet)
    if magic != COMMAND_MAGIC or version != VERSION or flags != 0 or tick != expected_tick:
        raise ValueError("invalid command packet header")
    return np.frombuffer(packet, dtype="<f8", offset=HEADER.size, count=3).copy()
