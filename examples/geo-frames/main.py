#!/usr/bin/env uv run

import math

import elodin as el
import jax.numpy as jnp
import numpy as np

SIM_RATE = 60.0

LAT_DEG = 34.72
LON_DEG = -86.64
ALT_M = 180.5
WGS84_A_M = 6_378_137.0
WGS84_E2 = 6.6943799901413165e-3
WGS84_B_M = WGS84_A_M * math.sqrt(1.0 - WGS84_E2)
CUBE_SIZE_M = 500_000.0
CUBE_SEPARATION_M = 1_500_000.0
ASSET_MESH_SCALE_M = 1_000_000.0
AXIS_ARROW_SCALE_M = 1_000_000.0
AXIS_ARROW_THICKNESS = 2500.0
ORBIT_RADIUS_M = WGS84_A_M + 1_200_000.0
ORBIT_PERIOD_S = 20.0
SPIN_RATE_RAD_S = math.radians(10.0)
PURPLE = "156 39 176"
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


def _ecef_marker_objects() -> str:
    return "\n".join(
        f"""
        object_3d frame="ECEF" {name}.world_pos {{
            box x={CUBE_SIZE_M} y={CUBE_SIZE_M} z={CUBE_SIZE_M} {{
                color {PURPLE}
            }}
            icon builtin="location_on" size=48 {{
                color {PURPLE}
            }}
        }}
        object_3d frame="ECEF" {name}.world_pos {{
            glb path="compass.glb" scale={ASSET_MESH_SCALE_M}
        }}""".rstrip()
        for name, _ in ECEF_MARKERS
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

    world.schematic(
        f"""
        coordinate frame=NED lat={LAT_DEG} lon={LON_DEG} alt={ALT_M}
        hsplit {{
            tabs {{
                viewport name=Frames frame="NED" pos="(0,0,0,1, 4000000,4000000,-3000000)" look_at="(0,0,0,1, 0,0,0)" far=15000000.0 hdr=#true show_grid=#false active=#true
                inspector
                hierarchy
            }}
        }}

        object_3d frame="ECEF" earth.world_pos {{
            glb path="earth.glb"
            icon builtin="public" size=64 {{
                color 255 255 255
            }}
        }}
        {_ecef_marker_objects()}
        object_3d frame="NED" ned_origin.world_pos {{
            box x={CUBE_SIZE_M} y={CUBE_SIZE_M} z={CUBE_SIZE_M} {{
                color 244 67 54
            }}
            icon builtin="my_location" size=56 {{
                color 244 67 54
            }}
        }}
        object_3d frame="NED" ned_origin.world_pos {{
            glb path="compass.glb" scale={ASSET_MESH_SCALE_M}
        }}
        object_3d frame="ENU" enu_far_east.world_pos {{
            box x={CUBE_SIZE_M} y={CUBE_SIZE_M} z={CUBE_SIZE_M} {{
                color 33 150 243
            }}
            icon builtin="explore" size=56 {{
                color 33 150 243
            }}
        }}
        object_3d frame="ENU" enu_far_east.world_pos {{
            glb path="compass.glb" scale={ASSET_MESH_SCALE_M}
        }}
        object_3d frame="ECEF" ecef_far_up.world_pos {{
            box x={CUBE_SIZE_M} y={CUBE_SIZE_M} z={CUBE_SIZE_M} {{
                color 76 175 80
            }}
            icon builtin="gps_fixed" size=56 {{
                color 76 175 80
            }}
        }}
        object_3d frame="ECEF" ecef_far_up.world_pos {{
            glb path="compass.glb" scale={ASSET_MESH_SCALE_M}
        }}
        object_3d frame="ECEF" ecef_orbit_line.world_pos {{
            sphere radius={CUBE_SIZE_M * 0.25} {{
                color cyan
            }}
            icon builtin="satellite_alt" size=48 {{
                color cyan
            }}
        }}

        line_3d frame="NED" ned_origin.world_pos line_width=2.0 {{
            color 244 67 54
        }}
        line_3d frame="ENU" enu_far_east.world_pos line_width=2.0 {{
            color 33 150 243
        }}
        line_3d frame="ECEF" ecef_far_up.world_pos line_width=2.0 {{
            color 76 175 80
        }}
        line_3d frame="ECEF" ecef_orbit_line.world_pos line_width=4.0 perspective=#false {{
            color cyan
        }}

        vector_arrow frame="NED" "(0, 1, 0)" origin="ned_origin.world_pos" scale={AXIS_ARROW_SCALE_M} normalize=#true arrow_thickness={AXIS_ARROW_THICKNESS} label_position=0.0 name="NED Y-axis" show_name=#true body_frame=#true {{
            color 244 67 54
        }}
        vector_arrow frame="ENU" "(0, 1, 0)" origin="enu_far_east.world_pos" scale={AXIS_ARROW_SCALE_M} normalize=#true arrow_thickness={AXIS_ARROW_THICKNESS} label_position=0.0 name="ENU Y-axis" show_name=#true body_frame=#true {{
            color 33 150 243
        }}
        vector_arrow frame="ECEF" "(0, 1, 0)" origin="ecef_far_up.world_pos" scale={AXIS_ARROW_SCALE_M} normalize=#true arrow_thickness={AXIS_ARROW_THICKNESS} label_position=0.0 name="ECEF Y-axis" show_name=#true body_frame=#true {{
            color 76 175 80
        }}
        """,
        "geo-frames.kdl",
    )
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
