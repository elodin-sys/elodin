import jax.numpy as jnp
import numpy as np
from dynamics import heliocentric_relative_acceleration


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
