"""Repro for f32 quantization in the editor's plot/line_3d display path (#760).

A ball flies a smooth, slow climbing arc near a real launch site in ECEF
coordinates (~6.37e6 m magnitude — the worst case for f32). The recorded
trajectory is f64 and perfectly smooth, yet the ECEF `line_3d` trail
staircases on screen: the display path casts each sample to f32 at ingestion,
and at that magnitude one f32 ULP is ~0.25-0.5 m.

The same motion is recorded a second time in a local ENU component (numbers of
tens of metres) and drawn as a second `line_3d` right next to the ECEF one:

    green  trail  = ball.pos_ecef  (large numbers)  -> staircases
    magenta trail = ball.pos_enu   (small numbers)  -> stays smooth

Identical f64 signal, identical rendering; only the coordinate magnitude at the
f32 cast differs. That isolates the quantization to the display-side cast, not
the sim, the DB, or the line's first-point subtraction (which happens after the
cast and therefore cannot recover the lost bits).

Noise is OFF by default so nothing in the data can be mistaken for the effect.
Set DEMO_NOISE_SIGMA > 0 to overlay a realistic Ornstein-Uhlenbeck nav wander.

Record then replay (from the repo root, inside `nix develop`):

    uv run python examples/f32-quant-repro/main.py run     # writes ./db
    elodin-db run 127.0.0.1:2241 examples/f32-quant-repro/db --log-level warn &
    elodin editor 127.0.0.1:2241 --replay

Zoom on the ball during the arc: the green (ECEF) trail steps; the magenta
(ENU) trail is smooth.
"""

import math
import os
import typing

from dataclasses import field

import elodin as el
import jax
from jax import numpy as jnp
from jax import random

SIM_TIME_STEP = 1.0 / 100.0
MAX_TICKS = 6000  # 60 s

# Trajectory phases (seconds).
DWELL = 15.0
FLY = 30.0
HELIX_RADIUS = 30.0
HELIX_PERIOD = 60.0  # half a turn over the flight -> ~3.1 m/s
CLIMB_RATE = 1.5

# Ornstein-Uhlenbeck noise: stationary sigma and correlation time. The per-tick
# update is n' = a*n + s*N(0,1) with a = exp(-dt/tau) and s = sigma*sqrt(1-a^2),
# which keeps the stationary std exactly at sigma. Default sigma is 0 so the
# recorded curve is a clean f64 ground truth: any staircase on the displayed
# ECEF trail is then display-path f32 truncation, not data.
NOISE_SIGMA = float(os.environ.get("DEMO_NOISE_SIGMA", "0.0"))
NOISE_TAU = 0.15
NOISE_A = math.exp(-SIM_TIME_STEP / NOISE_TAU)
NOISE_STEP = NOISE_SIGMA * math.sqrt(1.0 - NOISE_A * NOISE_A)

# Arbitrary surface point (WGS84 lat/lon/alt). Any location works — the repro
# only needs a large ECEF magnitude (~6.4e6 m) to expose the f32 cast.
LAT, LON, ALT = 35.0, -117.5, 600.0


def _wgs84_ecef(lat_deg: float, lon_deg: float, alt: float) -> tuple:
    a, e2 = 6378137.0, 6.69437999014e-3
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    x = (n + alt) * math.cos(lat) * math.cos(lon)
    y = (n + alt) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - e2) + alt) * math.sin(lat)
    return x, y, z


def _enu_to_ecef_matrix(lat_deg: float, lon_deg: float):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    # Columns are the ENU basis vectors expressed in ECEF.
    return jnp.array(
        [
            [-so, -sl * co, cl * co],
            [co, -sl * so, cl * so],
            [0.0, cl, sl],
        ]
    )


BASE_ECEF = jnp.array(_wgs84_ecef(LAT, LON, ALT))
ENU2ECEF = _enu_to_ecef_matrix(LAT, LON)

SimT = typing.Annotated[
    jax.Array,
    el.Component("sim_t", el.ComponentType(el.PrimitiveType.F64, ())),
]

PosEcef = typing.Annotated[
    jax.Array,
    el.Component(
        "pos_ecef",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

NavNoise = typing.Annotated[
    jax.Array,
    el.Component(
        "nav_noise",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

# Same motion expressed in local ENU (small numbers). Rendering it next to the
# ECEF trail isolates the f32 magnitude effect: identical f64 signal, only the
# coordinate magnitude at the f32 cast differs.
PosEnu = typing.Annotated[
    jax.Array,
    el.Component(
        "pos_enu",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]


@el.dataclass
class BallData(el.Archetype):
    sim_t: SimT = field(default_factory=lambda: jnp.float64(0.0))
    nav_noise: NavNoise = field(default_factory=lambda: jnp.zeros(3))
    pos_ecef: PosEcef = field(default_factory=lambda: jnp.array(BASE_ECEF))
    pos_enu: PosEnu = field(default_factory=lambda: jnp.array([0.0, 0.0, 2.0]))


@el.map
def advance_time(t: SimT) -> SimT:
    return t + SIM_TIME_STEP


@el.map
def wander_noise(t: SimT, n: NavNoise) -> NavNoise:
    # OU step: correlated wander, deterministic per tick index.
    tick = jnp.int64(jnp.round(t / SIM_TIME_STEP))
    return NOISE_A * n + NOISE_STEP * random.normal(random.key(tick), shape=(3,))


def _enu_pos(t: jax.Array, n: jax.Array) -> jax.Array:
    # Dwell -> climbing arc -> dwell, defined in local ENU.
    tt = jnp.clip(t - DWELL, 0.0, FLY)
    ang = 2.0 * jnp.pi * tt / HELIX_PERIOD
    enu = jnp.array(
        [
            HELIX_RADIUS * jnp.sin(ang),
            HELIX_RADIUS * (1.0 - jnp.cos(ang)),
            2.0 + CLIMB_RATE * tt,
        ]
    )
    return enu + n


@el.map
def move_ball(t: SimT, n: NavNoise, _p: PosEcef) -> PosEcef:
    return BASE_ECEF + ENU2ECEF @ _enu_pos(t, n)


@el.map
def move_ball_enu(t: SimT, n: NavNoise, _p: PosEnu) -> PosEnu:
    return _enu_pos(t, n)


def world() -> el.World:
    w = el.World()
    w.spawn(BallData(), name="ball")
    w.schematic(
        f"""
        coordinate frame=ENU lat={LAT} lon={LON} alt={ALT}
        hsplit name="f32 quantization — ECEF (green) vs ENU (magenta)" share=1.0 {{
            viewport frame=ECEF name="RAW camera  (smoothing=0)" near=0.1 smoothing=0.0 pos="(0,0,0,0,6,6,-4) + (0,0,0,1, ball.pos_ecef[0], ball.pos_ecef[1], ball.pos_ecef[2])" look_at="(0,0,0,1, ball.pos_ecef[0], ball.pos_ecef[1], ball.pos_ecef[2])" show_grid=#true show_view_cube=#true
            viewport frame=ECEF name="SMOOTHED camera  (smoothing=1)" near=0.1 smoothing=1.0 pos="(0,0,0,0,6,6,-4) + (0,0,0,1, ball.pos_ecef[0], ball.pos_ecef[1], ball.pos_ecef[2])" look_at="(0,0,0,1, ball.pos_ecef[0], ball.pos_ecef[1], ball.pos_ecef[2])" show_grid=#true show_view_cube=#true
        }}
        object_3d frame=ECEF "(0,0,0,1, ball.pos_ecef[0], ball.pos_ecef[1], ball.pos_ecef[2])" {{
            sphere radius=0.4 {{
                color orange
            }}
        }}
        line_3d frame=ECEF "(0,0,0,1, ball.pos_ecef[0], ball.pos_ecef[1], ball.pos_ecef[2])" perspective=#false smoothing=0.0 {{
            color 0 255 0
            future_color 0 180 0
        }}
        line_3d frame=ENU "(0,0,0,1, ball.pos_enu[0] + 1.5, ball.pos_enu[1], ball.pos_enu[2])" perspective=#false smoothing=0.0 {{
            color 255 0 255
            future_color 160 0 160
        }}
    """,
        "main.kdl",
    )
    return w


def system() -> el.System:
    return advance_time | wander_noise | move_ball | move_ball_enu


if __name__ == "__main__":
    db_path = os.environ.get(
        "DEMO_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
    )
    world().run(
        system(),
        simulation_rate=1.0 / SIM_TIME_STEP,
        max_ticks=MAX_TICKS,
        # db_path=db_path,
    )
