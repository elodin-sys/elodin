#!/usr/bin/env uv run
"""Lean kinematic visual checks for falcon9 cinematic effects (no FSW).

  # Editor reads ELODIN_SCREENSHOT_* at process start — export delay with the launch:
  ELODIN_VIZCHECK_SCENARIO=plume-close ELODIN_SCREENSHOT_DELAY=10 \\
    elodin editor examples/falcon9/visual_check.py
  ELODIN_VIZCHECK_SCENARIO=rcs-flip ELODIN_SCREENSHOT_DELAY=8 \\
    elodin editor examples/falcon9/visual_check.py
  ELODIN_VIZCHECK_SCENARIO=barge ELODIN_SCREENSHOT_DELAY=10 \\
    elodin editor examples/falcon9/visual_check.py

Screenshots default to /tmp/f9-<scenario>.png via the editor screenshot plugin.
Each scenario starts at mission time t0 so the interesting frame is early.

NOTE: no `from __future__ import annotations` — `@el.system` introspects real
annotation objects (same constraint as sim.py).
"""

import math
import os
import sys
import typing as ty
from pathlib import Path

# Allow `uv run python examples/falcon9/visual_check.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import elodin as el
import jax
import jax.numpy as jnp

from constants import (
    LZ1_ALT_M,
    LZ1_LAT_DEG,
    LZ1_LON_DEG,
    PAD_ALT_M,
    PAD_LAT_DEG,
    PAD_LON_DEG,
    START_TIMESTAMP_US,
)
from frames import ellipsoid_up, geodetic_to_ecef, ned_basis
from reference import build_reference
from sim import StaticSceneObject, surface_attitude, upright_attitude

SIM_HZ = 120.0
SIM_DT = 1.0 / SIM_HZ
SCENARIO = os.environ.get("ELODIN_VIZCHECK_SCENARIO", "plume-close")
REF = build_reference("crs12")

# t0 = mission-time offset (s); delay = editor wall seconds before screenshot.
SCENARIOS = {
    "plume-close": {"t0": 22.0, "t_s": 8.0, "delay": 10.0},
    "rcs-flip": {"t0": 36.0, "t_s": 4.0, "delay": 8.0},
    "barge": {"t0": 0.0, "t_s": 8.0, "delay": 10.0},
}
if SCENARIO not in SCENARIOS:
    raise SystemExit(f"unknown ELODIN_VIZCHECK_SCENARIO={SCENARIO!r}; choose {list(SCENARIOS)}")

_cfg = SCENARIOS[SCENARIO]
T0 = float(_cfg["t0"])
os.environ.setdefault("ELODIN_SCREENSHOT", f"/tmp/f9-{SCENARIO}.png")
os.environ.setdefault("ELODIN_SCREENSHOT_DELAY", str(_cfg["delay"]))
os.environ.setdefault("ELODIN_SCREENSHOT_EXIT", "1")

ThrustViz = ty.Annotated[
    jax.Array, el.Component("thrust_viz", el.ComponentType(el.PrimitiveType.F64, (1,)))
]
SmokeViz = ty.Annotated[
    jax.Array, el.Component("smoke_viz", el.ComponentType(el.PrimitiveType.F64, (1,)))
]
PadSmokeViz = ty.Annotated[
    jax.Array, el.Component("pad_smoke_viz", el.ComponentType(el.PrimitiveType.F64, (1,)))
]
LandingSmokeViz = ty.Annotated[
    jax.Array,
    el.Component("landing_smoke_viz", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
RcsLevels = ty.Annotated[
    jax.Array, el.Component("rcs_levels", el.ComponentType(el.PrimitiveType.F64, (8,)))
]
BoosterMarker = ty.Annotated[
    jax.Array, el.Component("vizcheck_booster", el.ComponentType(el.PrimitiveType.F64, (1,)))
]


def pad_ecef() -> jnp.ndarray:
    return geodetic_to_ecef(math.radians(PAD_LAT_DEG), math.radians(PAD_LON_DEG), PAD_ALT_M)


def lz1_ecef() -> jnp.ndarray:
    return geodetic_to_ecef(math.radians(LZ1_LAT_DEG), math.radians(LZ1_LON_DEG), LZ1_ALT_M)


_PAD_NED = ned_basis(math.radians(PAD_LAT_DEG), math.radians(PAD_LON_DEG))
PAD_UP = ellipsoid_up(math.radians(PAD_LAT_DEG), math.radians(PAD_LON_DEG))
PAD_TRACK = (_PAD_NED[0] + _PAD_NED[1]) / jnp.linalg.norm(_PAD_NED[0] + _PAD_NED[1])
LZ1_UP = ellipsoid_up(math.radians(LZ1_LAT_DEG), math.radians(LZ1_LON_DEG))


def engines_toward_deck_attitude() -> el.Quaternion:
    """Nose (+X) along local-up so Merlin nozzles (−X) face the barge deck."""
    target = jnp.asarray(LZ1_UP)
    x = jnp.array([1.0, 0.0, 0.0])
    axis = jnp.cross(x, target)
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-12)
    angle = jnp.arccos(jnp.clip(jnp.dot(x, target), -1.0, 1.0))
    return el.Quaternion.from_axis_angle(axis, angle)


ref_t = jnp.asarray(REF.time_s)
ref_alt = jnp.asarray(REF.altitude_m)
ref_dr = jnp.asarray(REF.downrange_m)
pad = pad_ecef()
lz1 = lz1_ecef()
up = jnp.asarray(PAD_UP)
track = jnp.asarray(PAD_TRACK)
ascent_att = upright_attitude()
landing_att = engines_toward_deck_attitude()


def make_advance():
    out_tys = (el.WorldPos, ThrustViz, SmokeViz, PadSmokeViz, LandingSmokeViz, RcsLevels)

    if SCENARIO == "barge":

        @el.system
        def advance_barge(
            tick: el.Query[el.SimulationTick],
            boosters: el.Query[BoosterMarker],
        ) -> el.Query[
            el.WorldPos, ThrustViz, SmokeViz, PadSmokeViz, LandingSmokeViz, RcsLevels
        ]:
            _ = tick[0]
            # ~35 m above deck; keep landing smoke low so the barge stays visible.
            r = lz1 + jnp.asarray(LZ1_UP) * 35.0
            pose = el.SpatialTransform(angular=landing_att, linear=r)
            vals = (
                pose,
                jnp.array([0.12]),
                jnp.array([0.05]),
                jnp.array([0.0]),
                jnp.array([0.08]),
                jnp.zeros(8),
            )
            return boosters.map(out_tys, lambda _m: vals)

        return advance_barge

    dim_for_rcs = 1.0 if SCENARIO == "rcs-flip" else 0.0

    @el.system
    def advance_ascent(
        tick: el.Query[el.SimulationTick],
        boosters: el.Query[BoosterMarker],
    ) -> el.Query[el.WorldPos, ThrustViz, SmokeViz, PadSmokeViz, LandingSmokeViz, RcsLevels]:
        t = T0 + tick[0] * SIM_DT
        alt = jnp.interp(t, ref_t, ref_alt)
        dr = jnp.interp(t, ref_t, ref_dr)
        r = pad + up * alt + track * dr
        thrust = jnp.where(t < 147.0, 1.0, jnp.where(t < 400.0, 0.0, 0.12))
        rcs_on = (t >= 36.0) & (t <= 39.5)
        thrust = jnp.where((dim_for_rcs > 0.5) & rcs_on, 0.12, thrust)
        density = jnp.clip(jnp.exp(-alt / 8500.0), 0.0, 1.0)
        smoke = thrust * density
        pad_smoke = thrust * jnp.sqrt(
            jnp.clip(1.0 - jnp.linalg.norm(r - pad) / 300.0, 0.0, 1.0)
        )
        land_smoke = thrust * jnp.sqrt(
            jnp.clip(1.0 - jnp.linalg.norm(r - lz1) / 300.0, 0.0, 1.0)
        )
        pulse = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        # Only the rcs-flip scenario pulses jets (plume-close keeps them dark).
        rcs = jnp.where((dim_for_rcs > 0.5) & rcs_on, pulse, jnp.zeros(8))
        pose = el.SpatialTransform(angular=ascent_att, linear=r)
        vals = (
            pose,
            jnp.array([thrust]),
            jnp.array([smoke]),
            jnp.array([pad_smoke]),
            jnp.array([land_smoke]),
            rcs,
        )
        return boosters.map(out_tys, lambda _m: vals)

    return advance_ascent


world = el.WorldBuilder()
pad_att = surface_attitude(PAD_LAT_DEG, PAD_LON_DEG)
lz1_att = surface_attitude(LZ1_LAT_DEG, LZ1_LON_DEG)
pad_up = jnp.asarray(PAD_UP)
lz1_up = jnp.asarray(LZ1_UP)

init_att = landing_att if SCENARIO == "barge" else ascent_att
if SCENARIO == "barge":
    init_r = lz1 + lz1_up * 35.0
else:
    init_alt = float(jnp.interp(T0, ref_t, ref_alt))
    init_dr = float(jnp.interp(T0, ref_t, ref_dr))
    init_r = pad + up * init_alt + track * init_dr
world.spawn(
    [
        StaticSceneObject(el.WorldPos(angular=init_att, linear=init_r)),
        el.C(ThrustViz, jnp.array([0.0])),
        el.C(SmokeViz, jnp.array([0.0])),
        el.C(PadSmokeViz, jnp.array([0.0])),
        el.C(LandingSmokeViz, jnp.array([0.0])),
        el.C(RcsLevels, jnp.zeros(8)),
        el.C(BoosterMarker, jnp.array([1.0])),
    ],
    name="booster",
)
world.spawn(StaticSceneObject(el.WorldPos(linear=jnp.zeros(3))), name="earth")
world.spawn(StaticSceneObject(el.WorldPos(angular=pad_att, linear=pad)), name="pad")
world.spawn(
    StaticSceneObject(el.WorldPos(angular=pad_att, linear=pad - pad_up * 2.0)),
    name="ground",
)
world.spawn(StaticSceneObject(el.WorldPos(angular=lz1_att, linear=lz1)), name="lz1")
world.spawn(
    StaticSceneObject(el.WorldPos(angular=lz1_att, linear=lz1 - lz1_up * 2.0)),
    name="lz1_ground",
)

kdl_path = Path(__file__).with_name("visual_check.kdl")
chase_active = SCENARIO != "barge"
kdl = (
    kdl_path.read_text()
    .replace("__CHASE_ACTIVE__", "active=#true" if chase_active else "")
    .replace("__LANDING_ACTIVE__", "active=#true" if not chase_active else "")
)
world.schematic(kdl, kdl_path.name)

print(f"[visual_check] scenario={SCENARIO} → {os.environ['ELODIN_SCREENSHOT']}")

world.run(
    make_advance(),
    simulation_rate=SIM_HZ,
    generate_real_time=True,
    max_ticks=int((_cfg["t_s"] + 5.0) * SIM_HZ),
    optimize=True,
    interactive=False,
    start_timestamp=START_TIMESTAMP_US,
    log_level="warn",
)
