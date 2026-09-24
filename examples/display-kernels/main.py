#!/usr/bin/env uv run
"""Driven craft whose packed covariance is visualized by display kernels."""

import math
import typing as ty

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

from schematic import build as build_schematic

SIM_RATE = 60.0
PATH_RADIUS = 3.0
CYCLE_SECONDS = 8.0


ErrorCovariance = ty.Annotated[
    jax.Array,
    el.Component(
        "error_covariance",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
        metadata={"element_names": "p00,p10,p20,p11,p21,p22"},
    ),
]


@el.dataclass
class CovarianceData(el.Archetype):
    error_covariance: ErrorCovariance


def pose_at(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = 2.0 * math.pi * t / CYCLE_SECONDS
    pos = np.array(
        [PATH_RADIUS * math.cos(phase), PATH_RADIUS * math.sin(phase), 1.0],
        dtype=np.float64,
    )
    omega = 2.0 * math.pi / CYCLE_SECONDS
    vel = np.array(
        [-PATH_RADIUS * omega * math.sin(phase), PATH_RADIUS * omega * math.cos(phase), 0.0],
        dtype=np.float64,
    )
    heading = math.atan2(vel[1], vel[0])
    half = 0.5 * heading
    quat = np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)
    return pos, vel, quat


def covariance_at(t: float, vel: np.ndarray) -> np.ndarray:
    speed = float(np.linalg.norm(vel))
    heading = math.atan2(vel[1], vel[0])
    breathe = 0.5 * (1.0 + math.sin(4.0 * math.pi * t / CYCLE_SECONDS))
    var_along = 0.20 + 0.18 * speed + 0.12 * breathe
    var_cross = 0.08
    var_vert = var_along / 9.0
    c, s = math.cos(heading), math.sin(heading)
    p00 = var_along * c * c + var_cross * s * s
    p11 = var_along * s * s + var_cross * c * c
    p10 = (var_along - var_cross) * c * s
    return np.array([p00, p10, 0.0, p11, 0.0, var_vert], dtype=np.float64)


def world() -> el.World:
    pos, vel, quat = pose_at(0.0)
    cov0 = covariance_at(0.0, vel)
    world = el.World()
    world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    angular=el.Quaternion(jnp.array(quat)),
                    linear=jnp.array(pos),
                ),
                inertia=el.SpatialInertia(mass=1.0),
            ),
            CovarianceData(error_covariance=jnp.array(cov0)),
        ],
        name="craft",
    )
    world.schematic(build_schematic(), "display-kernels.kdl")
    return world


@el.map
def no_force(force: el.Force) -> el.Force:
    return force


def post_step(tick: int, ctx: el.StepContext) -> None:
    t = tick / SIM_RATE
    pos, vel, quat = pose_at(t)
    ctx.write_component(
        "craft.world_pos",
        np.array([*quat, *pos], dtype=np.float64),
    )
    ctx.write_component("craft.error_covariance", covariance_at(t, vel))


if __name__ == "__main__":
    world().run(
        el.six_dof(sys=no_force),
        simulation_rate=SIM_RATE,
        generate_real_time=True,
        post_step=post_step,
    )
