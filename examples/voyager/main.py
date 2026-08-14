import os
import typing as ty

import elodin as el
import jax
from jax import numpy as jnp
from jax.numpy import linalg as la
import spiceypy as spice
import numpy as np
from pathlib import Path

from dynamics import (
    gravity_source_entity_names,
    heliocentric_relative_acceleration,
)

# SIM_TIME_STEP = 1.0 / 120.0
SIM_TIME_STEP = 3600.0
# SIM_TIME_STEP = 86400.0
# Set the gravitational constant for Newton's law of universal gravitation
SIMULATION_RATE_HZ = 1 / SIM_TIME_STEP
G = 6.6743e-11
DEFAULT_DB_PATH = "dbs/voyager"
DB_PATH_ENV = "DB_PATH"
MAX_TICKS_ENV = "MAX_TICKS"
DYNAMICS_CHAPTER_ENV = "VOYAGER_DYNAMICS_CHAPTER"

SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"
SPICE_KERNELS = [
    SPICE_DIR / "naif0012.tls",
    SPICE_DIR / "de440.bsp",
    SPICE_DIR / "Voyager_1.a54206u_V0.2_merged.bsp",
    SPICE_DIR / "Voyager_2.m05016u.merged.bsp",
]

for kernel in SPICE_KERNELS:
    spice.furnsh(str(kernel))

start_time_et = spice.utc2et("1978-01-01T00:00:00")
start_time_epoch_us = 252_452_400_000_000

PLANETS = [
    {
        "spice_name": "MERCURY BARYCENTER",
        "entity_name": "mercury",
        "radius": 2000000000.0,
        "color": "white",
        "trail_color": "white 220",
        "mass": 3.3011e23,
    },
    {
        "spice_name": "VENUS BARYCENTER",
        "entity_name": "venus",
        "radius": 3000000000.0,
        "color": "peach",
        "trail_color": "peach 220",
        "mass": 4.8675e24,
    },
    {
        "spice_name": "EARTH",
        "entity_name": "earth",
        "radius": 6000000000.0,
        "color": "hyperblue",
        "trail_color": "hyperblue 220",
        "mass": 5.97219e24,
    },
    {
        "spice_name": "MARS BARYCENTER",
        "entity_name": "mars",
        "radius": 4000000000.0,
        "color": "red",
        "trail_color": "red 220",
        "mass": 6.4171e23,
    },
    {
        "spice_name": "JUPITER BARYCENTER",
        "entity_name": "jupiter",
        "radius": 12000000000.0,
        "color": "orange",
        "trail_color": "orange 220",
        "mass": 1.898125e27,
    },
    {
        "spice_name": "SATURN BARYCENTER",
        "entity_name": "saturn",
        "radius": 10000000000.0,
        "color": "yolk",
        "trail_color": "yolk 220",
        "mass": 5.6834e26,
    },
    {
        "spice_name": "URANUS BARYCENTER",
        "entity_name": "uranus",
        "radius": 8000000000.0,
        "color": "cyan",
        "trail_color": "cyan 220",
        "mass": 8.6813e25,
    },
    {
        "spice_name": "NEPTUNE BARYCENTER",
        "entity_name": "neptune",
        "radius": 8000000000.0,
        "color": "blue",
        "trail_color": "blue 220",
        "mass": 1.02413e26,
    },
]
PROBE_RADIUS = 4000000000.0
PROBES = [
    {
        "spice_name": "VOYAGER 1",
        "entity_name": "voyager1",
        "radius": PROBE_RADIUS,
        "color": "red",
        "trail_color": "red 235",
        "mass": 825.0,
    },
    {
        "spice_name": "VOYAGER 2",
        "entity_name": "voyager2",
        "radius": PROBE_RADIUS,
        "color": "red",
        "trail_color": "red 235",
        "mass": 825.0,
    },
]
TRUTH_PROBES = [
    {
        "spice_name": "VOYAGER 1",
        "entity_name": "voyager1_truth",
        "radius": PROBE_RADIUS,
        "color": "green",
        "trail_color": "green 235",
        "mass": 825.0,
    },
    {
        "spice_name": "VOYAGER 2",
        "entity_name": "voyager2_truth",
        "radius": PROBE_RADIUS,
        "color": "green",
        "trail_color": "green 235",
        "mass": 825.0,
    },
]
PositionErrorKm = ty.Annotated[
    jax.Array,
    el.Component(
        "position_error_km",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"external_control": "true"},
    ),
]
VelocityErrorMps = ty.Annotated[
    jax.Array,
    el.Component(
        "velocity_error_mps",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"external_control": "true"},
    ),
]

EPHEMERIS_BODIES = PLANETS
DISPLAY_BODIES = PLANETS + PROBES + TRUTH_PROBES
SUN_MASS = 1.9885e30


w = el.World()

sun = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([0.0, 0.0, 0.0])),
            inertia=el.Inertia(SUN_MASS),
        ),
    ],
    name="Sun",
)

body_entity_ids = {"Sun": sun}

for body in EPHEMERIS_BODIES + PROBES + TRUTH_PROBES:
    init_state, _ = spice.spkezr(body["spice_name"], start_time_et, "ECLIPJ2000", "NONE", "SUN")

    init_pos = jnp.array(init_state[:3]) * 1000.0
    init_vel = jnp.array(init_state[3:]) * 1000.0

    print(body["spice_name"])
    print(init_pos)
    print(init_vel)

    components = [
        el.Body(
            world_pos=el.WorldPos(linear=init_pos),
            world_vel=el.WorldVel(linear=init_vel),
            inertia=el.Inertia(body["mass"]),
        ),
    ]
    if body in PROBES:
        components.extend(
            [
                el.C(PositionErrorKm, jnp.array([0.0], dtype=jnp.float64)),
                el.C(VelocityErrorMps, jnp.array([0.0], dtype=jnp.float64)),
            ]
        )

    body_entity_ids[body["entity_name"]] = w.spawn(
        components,
        name=body["entity_name"],
    )


def pre_step(tick: int, ctx: el.StepContext):
    current_time_et = start_time_et + tick * SIM_TIME_STEP

    for body in EPHEMERIS_BODIES + TRUTH_PROBES:
        state, _ = spice.spkezr(body["spice_name"], current_time_et, "ECLIPJ2000", "NONE", "SUN")
        pos_m = np.asarray(state[:3], dtype=np.float64) * 1000.0
        vel_ms = np.asarray(state[3:], dtype=np.float64) * 1000.0

        ctx.write_component(
            f"{body['entity_name']}.world_pos",
            np.array([0.0, 0.0, 0.0, 1.0, pos_m[0], pos_m[1], pos_m[2]], dtype=np.float64),
        )
        ctx.write_component(
            f"{body['entity_name']}.world_vel",
            np.array([0.0, 0.0, 0.0, vel_ms[0], vel_ms[1], vel_ms[2]], dtype=np.float64),
        )


def post_step(tick: int, ctx: el.StepContext) -> None:
    """Record numerical divergence from SPICE at the completed tick epoch."""
    current_time_et = start_time_et + (tick + 1) * SIM_TIME_STEP

    for probe in PROBES:
        simulated_pos = np.asarray(
            ctx.read_component(f"{probe['entity_name']}.world_pos"),
            dtype=np.float64,
        )[4:7]
        simulated_vel = np.asarray(
            ctx.read_component(f"{probe['entity_name']}.world_vel"),
            dtype=np.float64,
        )[3:6]

        truth_state, _ = spice.spkezr(
            probe["spice_name"], current_time_et, "ECLIPJ2000", "NONE", "SUN"
        )
        truth_pos = np.asarray(truth_state[:3], dtype=np.float64) * 1000.0
        truth_vel = np.asarray(truth_state[3:], dtype=np.float64) * 1000.0

        position_error_km = np.linalg.norm(simulated_pos - truth_pos) / 1000.0
        velocity_error_mps = np.linalg.norm(simulated_vel - truth_vel)

        ctx.write_component(
            f"{probe['entity_name']}.position_error_km",
            np.array([position_error_km], dtype=np.float64),
        )
        ctx.write_component(
            f"{probe['entity_name']}.velocity_error_mps",
            np.array([velocity_error_mps], dtype=np.float64),
        )


GravityEdge = el.Annotated[el.Edge, el.Component("gravity_edge", el.ComponentType.Edge)]


@el.dataclass
class GravityConstraint(el.Archetype):
    a: GravityEdge

    def __init__(self, a: el.EntityId, b: el.EntityId):
        self.a = GravityEdge(a, b)


@el.system
def gravity(
    graph: el.GraphQuery[GravityEdge],
    query: el.Query[el.WorldPos, el.Inertia],
) -> el.Query[el.Force]:
    def gravity_fn(force, a_pos, a_inertia, b_pos, b_inertia):
        r = a_pos.linear() - b_pos.linear()
        m = a_inertia.mass()
        M = b_inertia.mass()
        norm = la.norm(r)
        f = G * M * m * r / (norm * norm * norm)
        return el.Force(linear=force.force() - f)

    return graph.edge_fold(
        left_query=query,
        right_query=query,
        return_type=el.Force,
        init_value=el.Force(),
        fold_fn=gravity_fn,
    )


@el.system
def heliocentric_gravity(
    graph: el.GraphQuery[GravityEdge],
    query: el.Query[el.WorldPos, el.Inertia],
) -> el.Query[el.Force]:
    """Chapter 2 gravity in a nonrotating frame with a Sun-centered origin."""

    def gravity_fn(force, probe_pos, probe_inertia, source_pos, source_inertia):
        probe_mass = probe_inertia.mass()
        gravitational_parameter = G * source_inertia.mass()

        relative_acceleration = heliocentric_relative_acceleration(
            probe_pos.linear(), source_pos.linear(), gravitational_parameter
        )

        return el.Force(linear=force.force() + probe_mass * relative_acceleration)

    return graph.edge_fold(
        left_query=query,
        right_query=query,
        return_type=el.Force,
        init_value=el.Force(),
        fold_fn=gravity_fn,
    )


for probe in PROBES:
    probe_id = body_entity_ids[probe["entity_name"]]
    for source_name in gravity_source_entity_names(PLANETS):
        w.spawn(
            GravityConstraint(probe_id, body_entity_ids[source_name]),
            name=f"{probe['entity_name']} -> {source_name}",
        )

body_objects = "\n".join(
    f"""    object_3d {body["entity_name"]}.world_pos {{
        sphere radius={body["radius"]} {{
            color {body["color"]}
        }}
    }}
    line_3d {body["entity_name"]}.world_pos line_width=1.0 perspective=#false {{
        color {body["trail_color"]}
    }}"""
    for body in DISPLAY_BODIES
)

w.schematic(
    """
    hsplit {{
        tabs share=0.2 {{
            hierarchy
        }}
        tabs share=0.6 {{
            //viewport name=Viewport pos="(0,0,0,0,0,0,100)" look_at="(0,0,0,0,0,0,0)" hdr=#true
            viewport name=Viewport pos="(0,0,0,0, 0,0,2000000000000.0)" look_at="(0,0,0,0, 0,0,0)" fov=45.0 near=1000000.0

            graph "voyager1.position_error_km" name="Voyager 1 position error (km)"
            graph "voyager2.position_error_km" name="Voyager 2 position error (km)"
        }}
        tabs share=0.2 {{
            inspector
            graph "voyager1.velocity_error_mps" name="Voyager 1 velocity error (m/s)"
            graph "voyager2.velocity_error_mps" name="Voyager 2 velocity error (m/s)"
        }}
    }}
    object_3d sun.world_pos {{
        sphere radius=40000000000.0 emissivity=0.25 {{
            color yellow
        }}
    }}
{body_objects}
""".format(body_objects=body_objects)
)

dynamics_chapter = os.environ.get(DYNAMICS_CHAPTER_ENV, "1")
if dynamics_chapter not in ("1", "2"):
    raise ValueError(f"{DYNAMICS_CHAPTER_ENV} must be '1' or '2'")

gravity_system = gravity if dynamics_chapter == "1" else heliocentric_gravity
sys = el.six_dof(sys=gravity_system)
db_path = Path(os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH))
max_ticks_env = os.environ.get(MAX_TICKS_ENV)
max_ticks = int(max_ticks_env) if max_ticks_env is not None else None

# sim = w.run(sys, SIM_TIME_STEP, run_time_step=1 / 120.0, pre_step=pre_step)
sim = w.run(
    sys,
    simulation_rate=SIMULATION_RATE_HZ,
    pre_step=pre_step,
    post_step=post_step,
    max_ticks=max_ticks,
    start_timestamp=start_time_epoch_us,
    db_path=str(db_path),
    interactive=False,
)
