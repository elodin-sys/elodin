import elodin as el
import jax.numpy as jnp

from chapter_sim import JupiterPosition, jupiter_gm, run, sun_gm


@el.map
def gravity(pos: el.WorldPos, inertia: el.Inertia, jupiter: JupiterPosition) -> el.Force:
    probe = pos.linear()
    sun_pull = -sun_gm * probe / jnp.linalg.norm(probe) ** 3
    to_jupiter = jupiter - probe
    jupiter_pull = jupiter_gm * to_jupiter / jnp.linalg.norm(to_jupiter) ** 3
    return el.SpatialForce(linear=inertia.mass() * (sun_pull + jupiter_pull))


if __name__ == "__main__":
    run(gravity, 1)
