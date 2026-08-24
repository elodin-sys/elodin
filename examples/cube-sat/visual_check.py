#!/usr/bin/env uv run
"""Parked OreSat LEO day/night visual checks."""

import os
from dataclasses import dataclass
from datetime import datetime
from math import cos, radians, sin
from typing import Annotated

import elodin as el
import jax
import jax.numpy as np

# Same SoCal LEO slot as falcon9 visual_check night-sky (solar noon 20:16Z).
LAT_DEG = 34.05
LON_DEG = -124.0
EARTH_RADIUS_M = 6378.1e3
ALTITUDE_M = 400e3
RADIUS_M = EARTH_RADIUS_M + ALTITUDE_M

SIM_HZ = 120.0
SIM_DT = 1.0 / SIM_HZ
T_S = 4.0
DELAY_S = 12.0

SCENARIO = os.environ.get("ELODIN_CUBESAT_SCENARIO", "night")
SCENARIOS = {
    # Night → day (20 Mar 2026, noon 20:16Z)
    "night": "2026-03-20T10:21:00+00:00",
    "dawn-limb": "2026-03-20T12:39:00+00:00",
    "sunrise": "2026-03-20T13:10:00+00:00",
    "morning": "2026-03-20T14:30:00+00:00",
    "day": "2026-03-20T17:00:00+00:00",
    # Day → night (mirrored around noon)
    "afternoon": "2026-03-20T23:32:00+00:00",
    "dusk": "2026-03-21T02:02:00+00:00",
    "sunset": "2026-03-21T03:22:00+00:00",
    "sunset-limb": "2026-03-21T03:53:00+00:00",
    "night-am": "2026-03-21T06:11:00+00:00",
}
if SCENARIO not in SCENARIOS:
    raise SystemExit(f"unknown ELODIN_CUBESAT_SCENARIO={SCENARIO!r}; choose {list(SCENARIOS)}")

os.environ.setdefault("ELODIN_SCREENSHOT", f"/tmp/cubesat-{SCENARIO}.png")
os.environ.setdefault("ELODIN_SCREENSHOT_DELAY", str(DELAY_S))
os.environ.setdefault("ELODIN_SCREENSHOT_EXIT", "1")

_lat, _lon = radians(LAT_DEG), radians(LON_DEG)
_cl, _sl = cos(_lat), sin(_lat)
_co, _so = cos(_lon), sin(_lon)
r0 = np.array([RADIUS_M * _cl * _co, RADIUS_M * _cl * _so, RADIUS_M * _sl])
east = np.array([-_so, _co, 0.0])
up = np.array([_cl * _co, _cl * _so, _sl])
cam = -east * 20.0 + up * 6.0
look = east * 2.0

START_TIMESTAMP_US = int(datetime.fromisoformat(SCENARIOS[SCENARIO]).timestamp() * 1_000_000)

SatMarker = Annotated[
    jax.Array, el.Component("cs_viz_sat", el.ComponentType(el.PrimitiveType.F64, (1,)))
]


@dataclass
class SatTag(el.Archetype):
    marker: SatMarker


@el.system
def park(
    tick: el.Query[el.SimulationTick],
    sats: el.Query[SatMarker],
) -> el.Query[el.WorldPos]:
    _ = tick[0]
    pose = el.SpatialTransform(linear=r0)
    return sats.map(el.WorldPos, lambda _m: pose)


w = el.World()
w.spawn(
    [
        SatTag(np.array([1.0])),
        el.Body(
            world_pos=el.SpatialTransform(linear=r0),
            world_vel=el.SpatialMotion(),
            inertia=el.SpatialInertia(1.0),
        ),
    ],
    name="OreSat",
    id="ore_sat",
)

w.schematic(
    f"""
    coordinate frame=ECEF
    environment {{
        sun illuminance=100000.0 shadows=#true
        ambient scale=0.05
        earth
    }}
    tabs {{
        viewport name=Viewport pos="ore_sat.world_pos.translate_world({float(cam[0]):.4f}, {float(cam[1]):.4f}, {float(cam[2]):.4f})" look_at="ore_sat.world_pos.translate_world({float(look[0]):.4f}, {float(look[1]):.4f}, {float(look[2]):.4f})" up="({float(up[0]):.5f}, {float(up[1]):.5f}, {float(up[2]):.5f})" near=0.5 cinematic=#true show_grid=#false active=#true
    }}
    object_3d ore_sat.world_pos {{
        glb path="oresat-low.glb"
    }}
""",
    "cube-sat-visual-check.kdl",
)

print(
    f"[cube-sat visual_check] scenario={SCENARIO} utc={SCENARIOS[SCENARIO]} → {os.environ['ELODIN_SCREENSHOT']}"
)

w.run(
    system=park,
    simulation_rate=SIM_HZ,
    generate_real_time=True,
    max_ticks=int((T_S + 5.0) * SIM_HZ),
    optimize=True,
    interactive=False,
    start_timestamp=START_TIMESTAMP_US,
    log_level="warn",
)
