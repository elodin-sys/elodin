"""Consistency tests for the native Betaflight Quad-X motor layout."""

import numpy as np
import pytest

from config import DEFAULT_CONFIG


def test_native_motor_order_positions_and_spins() -> None:
    """Config indices follow Betaflight's native BR, FR, BL, FL order."""
    config = DEFAULT_CONFIG
    diagonal = config.arm_length / np.sqrt(2.0)
    expected_positions = np.array(
        [
            [-diagonal, -diagonal, 0.0],  # BR
            [diagonal, -diagonal, 0.0],  # FR
            [-diagonal, diagonal, 0.0],  # BL
            [diagonal, diagonal, 0.0],  # FL
        ]
    )

    np.testing.assert_allclose(config.motor_positions, expected_positions)
    np.testing.assert_allclose(
        np.linalg.norm(config.motor_positions, axis=1), config.arm_length
    )
    np.testing.assert_array_equal(config.motor_spin_directions, [-1.0, 1.0, 1.0, -1.0])
    np.testing.assert_array_equal(
        config.motor_thrust_directions,
        np.tile([0.0, 0.0, 1.0], (4, 1)),
    )


@pytest.mark.parametrize(
    ("motor_index", "expected_torque_signs"),
    [
        pytest.param(0, [-1.0, 1.0, -1.0], id="back-right"),
        pytest.param(1, [-1.0, -1.0, 1.0], id="front-right"),
        pytest.param(2, [1.0, 1.0, 1.0], id="back-left"),
        pytest.param(3, [1.0, -1.0, -1.0], id="front-left"),
    ],
)
def test_individual_motor_torque_signs(
    motor_index: int, expected_torque_signs
) -> None:
    """Each indexed motor produces the expected roll, pitch, and yaw signs."""
    config = DEFAULT_CONFIG
    torque = config.motor_torque_axes[motor_index].copy()
    torque[2] += config.motor_torque_coeff * config.motor_spin_directions[motor_index]

    np.testing.assert_array_equal(np.sign(torque), expected_torque_signs)


def test_equal_motor_thrust_cancels_body_torque() -> None:
    """A symmetric command produces thrust without roll, pitch, or yaw torque."""
    config = DEFAULT_CONFIG
    equal_thrust = np.ones(4)

    thrust_torque = np.sum(config.motor_torque_axes * equal_thrust[:, None], axis=0)
    yaw_torque = config.motor_torque_coeff * np.dot(
        equal_thrust, config.motor_spin_directions
    )
    total_torque = thrust_torque + np.array([0.0, 0.0, yaw_torque])

    np.testing.assert_allclose(total_torque, np.zeros(3), atol=1e-15)
