#!/usr/bin/env uv run
"""Compare equivalent Cholesky and direct covariance ellipsoids."""

import elodin as el
import jax.numpy as jnp
import numpy as np

CHOLESKY = np.array(
    [
        [1.2, 0.0, 0.0],
        [0.4, 0.8, 0.0],
        [-0.2, 0.3, 0.5],
    ]
)
COVARIANCE = CHOLESKY @ CHOLESKY.T


def pack_cholesky(matrix: np.ndarray) -> tuple[float, ...]:
    indices = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
    return tuple(round(float(matrix[row, col]), 12) for row, col in indices)


def pack_covariance(matrix: np.ndarray) -> tuple[float, ...]:
    indices = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    return tuple(round(float(matrix[row, col]), 12) for row, col in indices)


def body(x: float) -> el.Body:
    return el.Body(
        world_pos=el.SpatialTransform(linear=jnp.array([x, 0.0, 0.0])),
        inertia=el.SpatialInertia(mass=1.0),
    )


def world() -> el.World:
    world = el.World()
    world.spawn(body(-4.0), name="cholesky")
    world.spawn(body(4.0), name="covariance")

    cholesky = pack_cholesky(CHOLESKY)
    covariance = pack_covariance(COVARIANCE)
    world.schematic(
        f"""
        coordinate frame=ENU
        hsplit {{
            viewport name="Cholesky: P = LLᵀ" pos="(0,0,0,1, 0,-6,4)" look_at="cholesky.world_pos" far=30.0 show_grid=#true active=#true
            viewport frame=ECEF name="Direct covariance: P" pos="(0,0,0,1, 8,-6,4)" look_at="covariance.world_pos" far=30.0 show_grid=#true active=#true
        }}

        object_3d cholesky.world_pos {{
            ellipsoid error_covariance_cholesky="{cholesky}" error_confidence_interval=70.0 show_grid=#true {{
                color 255 152 0 72
                grid_color 255 193 7 220
            }}
        }}
        object_3d frame=ECEF covariance.world_pos {{
            ellipsoid error_covariance="{covariance}" error_confidence_interval=70.0 show_grid=#true {{
                color 3 169 244 72
                grid_color 0 188 212 220
            }}
        }}
        """,
        "covariance-ellipsoids.kdl",
    )
    return world


@el.map
def no_force(force: el.Force) -> el.Force:
    return force


if __name__ == "__main__":
    world().run(
        el.six_dof(sys=no_force),
        simulation_rate=60.0,
        generate_real_time=True,
    )
