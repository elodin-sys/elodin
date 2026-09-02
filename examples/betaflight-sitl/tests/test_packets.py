"""Tests for the Betaflight SITL UDP packet boundary."""

import struct

import numpy as np
import pytest

from comms import FDMPacket, RCPacket, ServoPacket, ServoPacketRaw


def test_servo_packet_golden_native_motor_order() -> None:
    """The normalized motor packet is four little-endian floats in BF order."""
    # Native Betaflight order: back-right, front-right, back-left, front-left.
    motors = np.array([0.125, 0.25, 0.5, 1.0])
    golden = bytes.fromhex("0000003e 0000803e 0000003f 0000803f")

    assert ServoPacket(motor_speed=motors).pack() == golden

    decoded = ServoPacket.from_bytes(golden)
    np.testing.assert_array_equal(decoded.motor_speed, motors)


def test_rc_packet_golden_channel_order() -> None:
    """The RC packet keeps its timestamp and 16 channels in AETR/AUX order."""
    # Roll, pitch, throttle, yaw, AUX1, followed by centered unused channels.
    channels = np.array(
        [1600, 1400, 1000, 1550, 1800, *([1500] * 11)], dtype=np.uint16
    )
    golden = bytes.fromhex(
        "000000000000f83f "  # timestamp: 1.5 as a little-endian double
        "4006 7805 e803 0e06 0807 "
        "dc05 dc05 dc05 dc05 dc05 dc05 dc05 dc05 dc05 dc05 dc05"
    )

    assert RCPacket(timestamp=1.5, channels=channels).pack() == golden

    decoded = RCPacket.from_bytes(golden)
    assert decoded.timestamp == 1.5
    np.testing.assert_array_equal(decoded.channels, channels)


def test_fdm_packet_field_layout_and_round_trip() -> None:
    """The FDM packet contains 18 little-endian doubles in protocol order."""
    packet = FDMPacket(
        timestamp=1.0,
        imu_angular_velocity_rpy=np.array([2.0, 3.0, 4.0]),
        imu_linear_acceleration_xyz=np.array([5.0, 6.0, 7.0]),
        imu_orientation_quat=np.array([8.0, 9.0, 10.0, 11.0]),
        velocity_xyz=np.array([12.0, 13.0, 14.0]),
        position_xyz=np.array([15.0, 16.0, 17.0]),
        pressure=18.0,
    )

    packed = packet.pack()
    assert len(packed) == 144
    assert struct.unpack("<18d", packed) == tuple(float(value) for value in range(1, 19))

    decoded = FDMPacket.from_bytes(packed)
    assert decoded.timestamp == packet.timestamp
    np.testing.assert_array_equal(
        decoded.imu_angular_velocity_rpy, packet.imu_angular_velocity_rpy
    )
    np.testing.assert_array_equal(
        decoded.imu_linear_acceleration_xyz, packet.imu_linear_acceleration_xyz
    )
    np.testing.assert_array_equal(decoded.imu_orientation_quat, packet.imu_orientation_quat)
    np.testing.assert_array_equal(decoded.velocity_xyz, packet.velocity_xyz)
    np.testing.assert_array_equal(decoded.position_xyz, packet.position_xyz)
    assert decoded.pressure == packet.pressure


def test_raw_servo_packet_field_layout_and_round_trip() -> None:
    """The raw packet has an aligned count followed by 16 PWM floats."""
    pwm_output = np.arange(1000.0, 1160.0, 10.0)
    packet = ServoPacketRaw(motor_count=4, pwm_output=pwm_output)

    packed = packet.pack()
    assert len(packed) == 68
    assert struct.unpack("<Hxx16f", packed) == (4, *pwm_output)

    decoded = ServoPacketRaw.from_bytes(packed)
    assert decoded.motor_count == 4
    np.testing.assert_array_equal(decoded.pwm_output, pwm_output)


@pytest.mark.parametrize(
    ("packet_type", "valid_size"),
    [
        (FDMPacket, 144),
        (RCPacket, 40),
        (ServoPacket, 16),
        (ServoPacketRaw, 68),
    ],
)
def test_packet_decoder_rejects_short_datagram(packet_type, valid_size: int) -> None:
    """A truncated UDP datagram must not be interpreted as a complete packet."""
    with pytest.raises(
        ValueError, match=rf"Data too short: {valid_size - 1} < {valid_size}"
    ):
        packet_type.from_bytes(bytes(valid_size - 1))
