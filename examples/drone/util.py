import math
import jax
from jax import numpy as jnp


def motor_thrust_axes(motor_angles: jax.Array) -> jax.Array:
    def motor_position(theta: float) -> jax.Array:
        return jnp.array([math.cos(theta), 0.0, math.sin(theta)])

    thrust_dir = jnp.array([0.0, 1.0, 0.0])
    return jnp.array(
        [jnp.cross(-motor_position(theta), thrust_dir) for theta in motor_angles]
    )
