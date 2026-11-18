import numpy as np
import pytest

from protocol import (
    COMMAND_DTYPE,
    COMMAND_MAGIC,
    COMMAND_SIZE,
    STATE_SIZE,
    VERSION,
    decode_commands,
    encode_states,
    state_frames,
)


def test_state_packets_have_the_layout_controllers_read():
    rows = np.arange(3 * 76, dtype=np.float64).reshape(3, 76) * 0.25
    frames = state_frames(3)
    encode_states(
        frames,
        tick=9,
        agent_position=rows[:, 0:3],
        agent_velocity=rows[:, 3:6],
        object_position=rows[:, 6:9],
        object_velocity=rows[:, 9:12],
        diagnostics=rows[:, 12:],
    )

    assert STATE_SIZE == 1024
    for row, frame in zip(rows, frames):
        packet = frame.tobytes()
        assert packet[:8] == b"MALS\x01\x00\x00\x00"
        assert int.from_bytes(packet[8:16], "little") == 9
        values = np.frombuffer(packet, dtype="<f8", offset=16)
        np.testing.assert_array_equal(values[:76], row)
        assert not values[76:].any()


def test_commands_must_answer_the_current_tick():
    replies = np.zeros(2, dtype=COMMAND_DTYPE)
    replies["magic"] = COMMAND_MAGIC
    replies["version"] = VERSION
    replies["tick"] = 23
    replies["values"][:, :3] = [[1.5, -2.5, 3.5], [4.0, 5.0, -6.0]]

    assert COMMAND_SIZE == 104
    np.testing.assert_array_equal(
        decode_commands(replies, 23), [[1.5, -2.5, 3.5], [4.0, 5.0, -6.0]]
    )
    with pytest.raises(ValueError, match="header"):
        decode_commands(replies, 24)
