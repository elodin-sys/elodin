#!/usr/bin/env python3
"""BDX RC jet — provisional parametric 6-DOF model in a rotating ECEF world.

Aircraft data comes from the vendored open-air package (analysis-correlated;
see model/elodin_package/provenance.md) plus a logged class-D fallback set.
The world is WGS84 ECEF anchored over the Mojave RC field: editor
viewports are drone-like (aircraft + Mojave mesh) while cinematic RGB and
Boson+ 640-style LWIR cameras are rendered by the sibling render-server.

Usage:
    elodin editor examples/rc-jet/main.py     # 3D visualization + RC control

The RC controller starts automatically (gamepad or WASD/Q/E/arrow keys).
Scenario selection: ELODIN_RC_JET_SCENARIO=demo|validation (default demo).
"""

import math
import os
from dataclasses import field
from pathlib import Path

import elodin as el
import jax.numpy as jnp
import numpy as np

import bdx_model
from class_d_fallbacks import FALLBACKS
from frames import enu_basis, level_attitude_ecef, quaternion_xyzw_from_matrix
from scenario import Numerics, Scenario, load_scenario
from sim import build_system, make_jet


@el.dataclass
class StaticMarker(el.Archetype):
    """A static visual marker with no physics (no inertia, force, velocity)."""

    world_pos: el.WorldPos = field(default_factory=el.SpatialTransform)


def setup_world(
    scenario: Scenario, numerics: Numerics
) -> tuple[el.World, el.EntityId, el.EntityId]:
    world = el.World()
    init = scenario.initial

    site = scenario.site
    lat = math.radians(site.lat_deg)
    lon = math.radians(site.lon_deg)
    basis = np.asarray(enu_basis(lat, lon))
    heading = math.radians(scenario.heading_deg)
    forward = math.sin(heading) * basis[0] + math.cos(heading) * basis[1]

    jet = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    angular=el.Quaternion(jnp.asarray(init.quat_xyzw)),
                    linear=jnp.asarray(init.pos_ecef),
                ),
                world_vel=el.SpatialMotion(linear=jnp.asarray(init.vel_ecef), angular=jnp.zeros(3)),
                inertia=el.SpatialInertia(
                    mass=MODEL.mass.operating_empty_mass_kg + init.fuel_kg,
                    inertia=np.array(
                        [
                            FALLBACKS.inertia.ixx_kg_m2,
                            FALLBACKS.inertia.iyy_kg_m2,
                            FALLBACKS.inertia.izz_kg_m2,
                        ]
                    ),
                ),
            ),
            make_jet(scenario),
        ],
        name="bdx",
    )

    # Target drone ~9 s ahead; local-level attitude so the mesh is not ECEF-tilted.
    target_pos = init.pos_ecef + 350.0 * forward + 5.0 * basis[2]
    target_att = quaternion_xyzw_from_matrix(level_attitude_ecef(lat, lon, scenario.heading_deg))
    target = world.spawn(
        StaticMarker(
            world_pos=el.SpatialTransform(
                angular=el.Quaternion(jnp.asarray(target_att)),
                linear=jnp.asarray(target_pos),
            ),
        ),
        name="target",
    )
    world.thermal_tag(target, temperature_c=18.0, emissivity=0.92)

    world.sensor_camera(
        entity=jet,
        name="fpv_cam",
        width=640,
        height=480,
        fov=90.0,
        fps=30.0,
        near=0.1,
        # Far plane must cover the terrain and horizon at altitude. Frustum
        # visualization shares this far plane, so keep create_frustum off.
        far=100_000.0,
        pos_offset=[1.2, 0.0, 0.1],
        rot_offset=[0.0, 0.0, 0.0],
        create_frustum=False,
        cinematic=True,
        ev100=13.5,
        environment={
            "sun": {
                "illuminance": 100000.0,
                "shadows": True,
                "direction": (-0.870, -0.488, -0.070),
            },
            "ambient_scale": 0.05,
            "earth": True,
        },
    )
    world.sensor_camera(
        entity=jet,
        name="ir_cam",
        camera_model="boson640p",
        lens_hfov=18.0,
        effect="lwir",
        near=0.1,
        far=100_000.0,
        pos_offset=[1.2, 0.0, 0.1],
        rot_offset=[0.0, 0.0, 0.0],
        create_frustum=False,
    )

    schematic_path = Path(__file__).with_name("bdx.kdl")
    world.schematic(schematic_path.read_text(), schematic_path.name)
    return world, jet, target


MODEL = bdx_model.load()
NUMERICS = Numerics()
SCENARIO = load_scenario(MODEL, FALLBACKS)

world, jet, target = setup_world(SCENARIO, NUMERICS)
sim_system = build_system(MODEL, FALLBACKS, SCENARIO, NUMERICS)

print("BDX RC Jet Simulation")
print("=====================")
print(
    f"Model: {MODEL.model_id} ({MODEL.credibility}), pipeline run "
    f"{MODEL.provenance['pipeline_run_id']}"
)
print(f"Scenario: {SCENARIO.name} at {SCENARIO.site.name}")
print(
    f"  location: {SCENARIO.site.format_latlon()}, "
    f"field elevation {SCENARIO.site.field_elevation_m:.0f} m"
)
print(
    f"  altitude: {SCENARIO.altitude_m:.0f} m MSL, TAS {SCENARIO.tas_mps:.1f} m/s, "
    f"heading {SCENARIO.heading_deg:.0f} deg"
)
print(
    f"  trim: alpha {math.degrees(SCENARIO.initial.alpha_rad):.2f} deg, "
    f"elevator {math.degrees(SCENARIO.initial.elevator_rad):.2f} deg, "
    f"throttle {SCENARIO.initial.throttle:.3f}"
)
print(
    f"  mass: {MODEL.mass.operating_empty_mass_kg + SCENARIO.initial.fuel_kg:.2f} kg "
    f"({SCENARIO.initial.fuel_kg:.2f} kg fuel)"
)
print(f"Time step: {NUMERICS.dt:.6f} s ({1 / NUMERICS.dt:.0f} Hz)")
print()

# RC controller runs alongside the simulation (gamepad or keyboard).
controller_path = Path(__file__).parent / "controller"
controller_host = os.environ.get("ELODIN_RC_JET_CONTROLLER_HOST")
controller_args = ["--host", controller_host] if controller_host else None
controller_binary = os.environ.get("ELODIN_RC_JET_CONTROLLER_BIN")
if controller_binary:
    binary = Path(controller_binary).resolve()
    if not binary.is_file():
        raise RuntimeError(f"RC controller binary not found: {binary}")
    controller = el.s10.PyRecipe.process(
        name="controller",
        cmd=str(binary),
        args=controller_args,
        ready=el.s10.Ready.delay(100),
        ready_timeout="120s",
    )
else:
    controller = el.s10.PyRecipe.cargo(
        name="controller",
        path=str(controller_path),
        args=controller_args,
        ready=el.s10.Ready.delay(100),
        ready_timeout="120s",
    )
world.recipe(controller)

world.run(
    sim_system,
    simulation_rate=1.0 / NUMERICS.dt,
    generate_real_time=True,
    max_ticks=int(os.environ.get("ELODIN_MAX_TICKS", NUMERICS.total_ticks)),
    db_path=os.environ.get("ELODIN_DB_PATH"),
    interactive=os.environ.get("ELODIN_NON_INTERACTIVE") != "1",
)
