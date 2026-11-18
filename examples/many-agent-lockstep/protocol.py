"""Fixed-size packets exchanged with the controller processes and the history producer."""

from __future__ import annotations

import numpy as np


STATE_MAGIC = b"MALS"
COMMAND_MAGIC = b"MALC"
VERSION = 1
STATE_DOUBLES = 126
COMMAND_DOUBLES = 11
HEADER_FIELDS = [("magic", "S4"), ("version", "<u2"), ("flags", "<u2"), ("tick", "<u8")]
# One packet per row: the 16-byte header, then the doubles.
STATE_DTYPE = np.dtype(HEADER_FIELDS + [("values", "<f8", (STATE_DOUBLES,))])
COMMAND_DTYPE = np.dtype(HEADER_FIELDS + [("values", "<f8", (COMMAND_DOUBLES,))])
STATE_SIZE = STATE_DTYPE.itemsize
COMMAND_SIZE = COMMAND_DTYPE.itemsize
# A history-producer frame: what it saw of agent 0 and its object at the frame's source time.
FRAME_DTYPE = np.dtype(
    [
        ("frame", "<f8"),
        ("source_time", "<f8"),
        ("agent_position", "<f8", (3,)),
        ("object_position", "<f8", (3,)),
    ]
)


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
    """Fills the state packet of every agent, one per row of `frames`."""
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
    """Returns the command in every reply, after checking that each answers `expected_tick`."""
    if (
        np.any(replies["magic"] != COMMAND_MAGIC)
        or np.any(replies["version"] != VERSION)
        or np.any(replies["flags"] != 0)
        or np.any(replies["tick"] != expected_tick)
    ):
        raise ValueError("invalid command packet header")
    return replies["values"][:, :3].copy()
