import numpy as np
import pytest

from protocol import (
    COMMAND_DTYPE,
    COMMAND_SIZE,
    STATE_SIZE,
    decode_command,
    decode_commands,
    decode_state,
    encode_command,
    encode_state_into,
    encode_states,
    state_frames,
)


def test_state_round_trip_has_fixed_width():
    buffer = bytearray(STATE_SIZE)
    packet = encode_state_into(
        buffer,
        tick=17,
        agent_position=[1.0, 2.0, 3.0],
        agent_velocity=[4.0, 5.0, 6.0],
        object_position=[7.0, 8.0, 9.0],
        object_velocity=[10.0, 11.0, 12.0],
        diagnostics=np.arange(20, dtype=np.float64),
    )

    tick, values = decode_state(packet)

    assert len(packet) == STATE_SIZE == 1024
    assert tick == 17
    np.testing.assert_array_equal(values[:12], np.arange(1.0, 13.0))
    np.testing.assert_array_equal(values[12:32], np.arange(20, dtype=np.float64))


def test_command_round_trip_has_fixed_width():
    packet = encode_command(23, [1.5, -2.5, 3.5])

    assert len(packet) == COMMAND_SIZE == 104
    np.testing.assert_array_equal(
        decode_command(packet, 23),
        [1.5, -2.5, 3.5],
    )


def test_batched_packets_match_single_packets():
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
    for row, frame in zip(rows, frames):
        expected = encode_state_into(
            bytearray(STATE_SIZE),
            tick=9,
            agent_position=row[0:3],
            agent_velocity=row[3:6],
            object_position=row[6:9],
            object_velocity=row[9:12],
            diagnostics=row[12:],
        )
        assert frame.tobytes() == bytes(expected)

    replies = np.frombuffer(
        b"".join(encode_command(9, row[:3]) for row in rows), dtype=COMMAND_DTYPE
    )
    np.testing.assert_array_equal(decode_commands(replies, 9), rows[:, :3])
    with pytest.raises(ValueError, match="header"):
        decode_commands(replies, 10)


def test_protocol_rejects_wrong_size_and_stale_tick():
    with pytest.raises(ValueError, match="state packet"):
        decode_state(b"short")
    with pytest.raises(ValueError, match="command packet"):
        decode_command(b"short", 1)
    with pytest.raises(ValueError, match="header"):
        decode_command(encode_command(2, [0.0, 0.0, 0.0]), 3)
