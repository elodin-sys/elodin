import os
import typing as ty
from pathlib import Path

import elodin as el
import jax
from jax import numpy as jnp
import numpy as np
import spiceypy as spice

from simulation import (
    PLANETS,
    build_world,
    chapter_gravity_system,
    make_ephemeris_pre_step,
)

# SIM_TIME_STEP = 1.0 / 120.0
SIM_TIME_STEP = 3600.0
# SIM_TIME_STEP = 86400.0
SIMULATION_RATE_HZ = 1 / SIM_TIME_STEP
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


def probe_components(body: dict) -> list:
    if body not in PROBES:
        return []
    return [
        el.C(PositionErrorKm, jnp.array([0.0], dtype=jnp.float64)),
        el.C(VelocityErrorMps, jnp.array([0.0], dtype=jnp.float64)),
    ]


w, body_entity_ids = build_world(
    start_time_et,
    probes=PROBES,
    truth_probes=TRUTH_PROBES,
    extra_components=probe_components,
    log_initial_states=True,
)

pre_step = make_ephemeris_pre_step(
    start_time_et,
    SIM_TIME_STEP,
    EPHEMERIS_BODIES + TRUTH_PROBES,
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
    timeline follow_latest=#true
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
gravity_system = chapter_gravity_system(dynamics_chapter)
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
