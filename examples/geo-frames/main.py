#!/usr/bin/env uv run

import math

import elodin as el
import jax.numpy as jnp
import numpy as np

from schematic import build as build_schematic

SIM_RATE = 60.0

LAT_DEG = 34.72
LON_DEG = -86.64
ALT_M = 180.5
WGS84_A_M = 6_378_137.0
WGS84_E2 = 6.6943799901413165e-3
WGS84_B_M = WGS84_A_M * math.sqrt(1.0 - WGS84_E2)
CUBE_SEPARATION_M = 1_500_000.0
ORBIT_RADIUS_M = WGS84_A_M + 1_200_000.0
ORBIT_PERIOD_S = 20.0
SPIN_RATE_RAD_S = math.radians(10.0)
ECEF_MARKERS = (
    ("ecef_equator_x_pos", (WGS84_A_M, 0.0, 0.0)),
    ("ecef_equator_y_pos", (0.0, WGS84_A_M, 0.0)),
    ("ecef_equator_x_neg", (-WGS84_A_M, 0.0, 0.0)),
    ("ecef_equator_y_neg", (0.0, -WGS84_A_M, 0.0)),
    ("ecef_north_pole", (0.0, 0.0, WGS84_B_M)),
    ("ecef_south_pole", (0.0, 0.0, -WGS84_B_M)),
)


def _ecef_from_enu(east: float, north: float, up: float) -> jnp.ndarray:
    lat = math.radians(LAT_DEG)
    lon = math.radians(LON_DEG)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # WGS84_E2 = 0.0

    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    origin = jnp.array(
        [
            (n + ALT_M) * cos_lat * cos_lon,
            (n + ALT_M) * cos_lat * sin_lon,
            (n * (1.0 - WGS84_E2) + ALT_M) * sin_lat,
        ]
    )
    delta = jnp.array(
        [
            -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up,
            cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up,
            cos_lat * north + sin_lat * up,
        ]
    )
    return origin + delta


def _body(pos: jnp.ndarray, angular_vel: jnp.ndarray | None = None) -> el.Body:
    if angular_vel is None:
        angular_vel = jnp.zeros(3)
    return el.Body(
        world_pos=el.SpatialTransform(linear=pos),
        world_vel=el.SpatialMotion(angular=angular_vel),
        inertia=el.SpatialInertia(mass=1.0),
    )


def world() -> el.World:
    world = el.World()
    y_axis_spin = jnp.array([0.0, SPIN_RATE_RAD_S, 0.0])

    world.spawn(_body(jnp.array([0.0, 0.0, 0.0]), y_axis_spin), name="ned_origin")
    world.spawn(
        _body(jnp.array([CUBE_SEPARATION_M, 0.0, 0.0]), y_axis_spin),
        name="enu_far_east",
    )
    world.spawn(
        _body(_ecef_from_enu(0.0, 0.0, CUBE_SEPARATION_M), y_axis_spin),
        name="ecef_far_up",
    )
    for name, pos in ECEF_MARKERS:
        world.spawn(_body(jnp.array(pos)), name=name)
    world.spawn(_body(jnp.array([0.0, 0.0, 0.0])), name="earth")
    world.spawn(_body(jnp.array([ORBIT_RADIUS_M, 0.0, 0.0])), name="ecef_orbit_line")

    world.schematic(build_schematic(), "geo-frames.kdl")
    return world


@el.map
def no_force(f: el.Force) -> el.Force:
    return f


def system() -> el.System:
    return el.six_dof(sys=no_force)


def post_step(tick: int, ctx: el.StepContext) -> None:
    angle = 2.0 * math.pi * (tick / SIM_RATE) / ORBIT_PERIOD_S
    pos = np.array(
        [
            ORBIT_RADIUS_M * math.cos(angle),
            ORBIT_RADIUS_M * math.sin(angle),
            0.0,
        ],
        dtype=np.float64,
    )
    ctx.write_component(
        "ecef_orbit_line.world_pos",
        np.array([0.0, 0.0, 0.0, 1.0, pos[0], pos[1], pos[2]], dtype=np.float64),
    )


if __name__ == "__main__":
    world().run(system(), simulation_rate=SIM_RATE, max_ticks=1200, post_step=post_step)
