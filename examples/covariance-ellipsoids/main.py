#!/usr/bin/env uv run
"""Compare equivalent Cholesky and direct covariance ellipsoids."""

import math
import typing as ty

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

SIM_RATE = 60.0
CYCLE_SECONDS = 8.0

CholeskyFactor = ty.Annotated[
    jax.Array,
    el.Component(
        "cholesky_factor",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
    ),
]
ErrorCovariance = ty.Annotated[
    jax.Array,
    el.Component(
        "error_covariance",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
    ),
]


@el.dataclass
class CholeskyData(el.Archetype):
    cholesky_factor: CholeskyFactor


@el.dataclass
class CovarianceData(el.Archetype):
    error_covariance: ErrorCovariance


def covariance_at(t: float) -> tuple[np.ndarray, np.ndarray]:
    phase = 2.0 * math.pi * t / CYCLE_SECONDS
    cholesky = np.array(
        [
            [1.2 + 0.3 * math.sin(phase), 0.0, 0.0],
            [0.4 * math.sin(phase * 0.7), 0.8 + 0.2 * math.cos(phase), 0.0],
            [
                -0.25 * math.cos(phase * 0.8),
                0.3 * math.sin(phase * 1.3),
                0.5 + 0.15 * math.sin(phase + 0.5),
            ],
        ]
    )
    return cholesky, cholesky @ cholesky.T


CHOLESKY, COVARIANCE = covariance_at(0.0)


def pack_cholesky(matrix: np.ndarray) -> np.ndarray:
    indices = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
    return np.array([matrix[row, col] for row, col in indices])


def pack_covariance(matrix: np.ndarray) -> np.ndarray:
    indices = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    return np.array([matrix[row, col] for row, col in indices])


def body(x: float) -> el.Body:
    return el.Body(
        world_pos=el.SpatialTransform(linear=jnp.array([x, 0.0, 0.0])),
        inertia=el.SpatialInertia(mass=1.0),
    )


def world() -> el.World:
    world = el.World()
    world.spawn(
        [
            body(-4.0),
            CholeskyData(cholesky_factor=jnp.array(pack_cholesky(CHOLESKY))),
        ],
        name="cholesky",
    )
    world.spawn(
        [
            body(4.0),
            CovarianceData(error_covariance=jnp.array(pack_covariance(COVARIANCE))),
        ],
        name="covariance",
    )

    world.schematic(
        """
        coordinate frame=ENU
        hsplit {
            viewport name="Cholesky: P = LLᵀ" pos="(0,0,0,1, 0,-6,4)" look_at="cholesky.world_pos" far=30.0 show_grid=#true active=#true
            viewport frame=ENU name="Direct covariance: P" pos="(0,0,0,1, 8,-6,4)" look_at="covariance.world_pos" far=30.0 show_grid=#true active=#true
        }

        object_3d cholesky.world_pos {
            ellipsoid error_covariance_cholesky="cholesky.cholesky_factor" error_confidence_interval=70.0 show_grid=#true {
                color 255 152 0 72
                grid_color 255 193 7 220
            }
        }
        object_3d frame=ENU covariance.world_pos {
            ellipsoid error_covariance="covariance.error_covariance" error_confidence_interval=70.0 show_grid=#true {
                color 3 169 244 72
                grid_color 0 188 212 220
            }
        }
        """,
        "covariance-ellipsoids.kdl",
    )
    return world


@el.map
def no_force(force: el.Force) -> el.Force:
    return force


def post_step(tick: int, ctx: el.StepContext) -> None:
    cholesky, covariance = covariance_at(tick / SIM_RATE)
    ctx.write_component("cholesky.cholesky_factor", pack_cholesky(cholesky))
    ctx.write_component("covariance.error_covariance", pack_covariance(covariance))


if __name__ == "__main__":
    world().run(
        el.six_dof(sys=no_force),
        simulation_rate=SIM_RATE,
        generate_real_time=True,
        post_step=post_step,
    )
