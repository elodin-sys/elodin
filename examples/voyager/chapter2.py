import elodin as el
import jax.numpy as jnp

from chapter_sim import JupiterPosition, jupiter_gm, run, sun_gm
from dynamics import heliocentric_relative_acceleration


@el.map
def gravity(pos: el.WorldPos, inertia: el.Inertia, jupiter: JupiterPosition) -> el.Force:
    probe = pos.linear()
    sun_pull = -sun_gm * probe / jnp.linalg.norm(probe) ** 3
    # chapter 2 also accounts for Jupiter pulling on the Sun
    jupiter_pull = heliocentric_relative_acceleration(probe, jupiter, jupiter_gm)
    return el.SpatialForce(linear=inertia.mass() * (sun_pull + jupiter_pull))


if __name__ == "__main__":
    run(gravity, 2)
