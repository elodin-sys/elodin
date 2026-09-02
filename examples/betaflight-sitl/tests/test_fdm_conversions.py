"""Golden tests for Elodin-to-Betaflight FDM conversion."""

import numpy as np
import pytest

from comms import build_fdm_from_components


def test_level_rest_fdm_conversion() -> None:
    """A level stationary FLU body produces the canonical resting FDM packet."""
    packet = build_fdm_from_components(
        world_pos=np.array([0.0, 0.0, 0.0, 1.0, 12.0, -4.0, 2.0]),
        world_vel=np.array([0.0, 0.0, 0.0, 1.0, -2.0, 3.0]),
        accel=np.array([0.0, 0.0, 9.80665]),
        gyro=np.zeros(3),
        timestamp=0.125,
    )

    assert packet.timestamp == 0.125
    np.testing.assert_array_equal(packet.imu_angular_velocity_rpy, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        packet.imu_linear_acceleration_xyz, [0.0, 0.0, -9.80665]
    )
    np.testing.assert_array_equal(packet.imu_orientation_quat, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(packet.velocity_xyz, [1.0, -2.0, 3.0])
    np.testing.assert_array_equal(packet.position_xyz, [12.0, -4.0, 2.0])
    assert packet.pressure == 101301.0


@pytest.mark.parametrize(
    ("gyro_flu", "expected_gyro_frd"),
    [
        pytest.param([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], id="roll"),
        pytest.param([0.0, 1.0, 0.0], [0.0, -1.0, 0.0], id="pitch"),
        pytest.param([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], id="yaw"),
    ],
)
def test_canonical_body_rate_fdm_conversion(gyro_flu, expected_gyro_frd) -> None:
    """Pure positive FLU body rates have fixed signs in the FRD sensor frame."""
    packet = build_fdm_from_components(
        world_pos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        world_vel=np.zeros(6),
        accel=np.array([0.0, 0.0, 9.80665]),
        gyro=np.array(gyro_flu),
        timestamp=0.0,
    )

    np.testing.assert_array_equal(packet.imu_angular_velocity_rpy, expected_gyro_frd)


def test_fdm_quaternion_scalar_order_and_gazebo_signs() -> None:
    """FDM converts xyzw to wxyz and negates the Gazebo bridge Y/Z terms."""
    packet = build_fdm_from_components(
        world_pos=np.array([0.1, -0.2, 0.3, 0.9, 4.0, -5.0, 6.0]),
        world_vel=np.zeros(6),
        accel=np.zeros(3),
        gyro=np.zeros(3),
        timestamp=0.0,
    )

    np.testing.assert_array_equal(packet.imu_orientation_quat, [0.9, 0.1, 0.2, -0.3])
