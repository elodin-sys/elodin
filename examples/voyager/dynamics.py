import jax.numpy as jnp
import numpy as np


def heliocentric_relative_acceleration(probe_position, source_position, mu):
    """Direct pull on the probe minus that source's acceleration of the Sun."""
    to_probe = source_position - probe_position
    direct = mu * to_probe / jnp.linalg.norm(to_probe) ** 3
    r = jnp.linalg.norm(source_position)
    sun = jnp.where(r > 0.0, mu * source_position / r**3, 0.0)
    return direct - sun


def state_error(simulated_position_m, simulated_velocity_mps, truth_position_m, truth_velocity_mps):
    """Return position error in km and velocity error in m/s."""
    position_delta_m = np.asarray(simulated_position_m, dtype=np.float64) - np.asarray(
        truth_position_m, dtype=np.float64
    )
    velocity_delta_mps = np.asarray(simulated_velocity_mps, dtype=np.float64) - np.asarray(
        truth_velocity_mps, dtype=np.float64
    )

    return (
        float(np.linalg.norm(position_delta_m) / 1000.0),
        float(np.linalg.norm(velocity_delta_mps)),
    )
