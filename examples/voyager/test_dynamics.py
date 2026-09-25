import jax.numpy as jnp
import numpy as np
from dynamics import heliocentric_relative_acceleration, state_error


def test_heliocentric_relative_acceleration():
    # planet at +3, probe at +5, mu=9: direct=-2.25, sun=+1
    np.testing.assert_allclose(
        heliocentric_relative_acceleration(
            jnp.array([5.0, 0.0, 0.0]), jnp.array([3.0, 0.0, 0.0]), 9.0
        ),
        [-3.25, 0.0, 0.0],
    )

    # source at the origin has no indirect term
    np.testing.assert_allclose(
        heliocentric_relative_acceleration(jnp.array([2.0, 0.0, 0.0]), jnp.zeros(3), 16.0),
        [-4.0, 0.0, 0.0],
    )

    probe = jnp.array([4.0, 0.0, 0.0])
    np.testing.assert_allclose(
        heliocentric_relative_acceleration(probe, jnp.array([2.0, 0.0, 0.0]), 8.0)
        + heliocentric_relative_acceleration(probe, jnp.array([-2.0, 0.0, 0.0]), 18.0),
        [0.0, 0.0, 0.0],
    )


def test_state_error_is_zero_for_matching_states():
    position = np.array([1.0e9, -2.0e9, 3.0e9])
    velocity = np.array([12000.0, -4000.0, 250.0])

    position_error_km, velocity_error_mps = state_error(
        position, velocity, position, velocity
    )

    assert position_error_km == 0.0
    assert velocity_error_mps == 0.0


def test_state_error_uses_km_for_position_and_mps_for_velocity():
    position_error_km, velocity_error_mps = state_error(
        np.array([3.0, 4.0, 0.0]),
        np.array([0.3, 0.4, 0.0]),
        np.zeros(3),
        np.zeros(3),
    )

    np.testing.assert_allclose(position_error_km, 0.005)
    np.testing.assert_allclose(velocity_error_mps, 0.5)


def test_state_error_only_depends_on_the_difference():
    offset_position = np.array([1000.0, -2000.0, 2000.0])
    offset_velocity = np.array([1.0, -2.0, 2.0])
    origin = np.array([8.0e10, -2.0e10, 4.0e10])
    base_velocity = np.array([10000.0, 5000.0, -3000.0])

    shifted = state_error(
        origin + offset_position,
        base_velocity + offset_velocity,
        origin,
        base_velocity,
    )

    np.testing.assert_allclose(shifted, (3.0, 3.0))
