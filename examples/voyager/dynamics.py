import jax.numpy as jnp


def heliocentric_relative_acceleration(probe_position, source_position, mu):
    """Direct pull on the probe minus that source's acceleration of the Sun."""
    to_probe = source_position - probe_position
    direct = mu * to_probe / jnp.linalg.norm(to_probe) ** 3
    r = jnp.linalg.norm(source_position)
    sun = jnp.where(r > 0.0, mu * source_position / r**3, 0.0)
    return direct - sun
