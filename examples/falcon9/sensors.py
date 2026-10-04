"""Sensor models and the webcast display model (WHITEPAPER 12).

The FSW flies exclusively on these outputs. Patterns follow the proven
1000 Hz production architecture: IMU at the physics rate, slower sensors on
timer-accumulator + hold with sample counters (the FSW detects freshness by
counter change), deterministic noise via fold_in(key, sample_tick).

NOTE: no `from __future__ import annotations` (el.map introspects real
annotation objects).
"""

import math
import typing as ty

import jax
import jax.numpy as jnp
import jax.random as rng
from constants import (
    ALTIMETER_RATE_HZ,
    GPS_RATE_HZ,
    OMEGA_EARTH_RADPS,
    SIM_RATE_HZ,
)

import elodin as el

SIM_TIME_STEP = 1.0 / SIM_RATE_HZ
GPS_DT_S = 1.0 / GPS_RATE_HZ
RADAR_DT_S = 1.0 / ALTIMETER_RATE_HZ

# Error model (EST; dispersed by the campaign).
IMU_ACCEL_SIGMA = 0.02  # m/s^2
IMU_GYRO_SIGMA = 1.0e-3  # rad/s
GPS_POS_SIGMA = 1.5  # m
GPS_VEL_SIGMA = 0.05  # m/s
PRESSURE_SIGMA_PA = 1.0e3
RADAR_MAX_RANGE_M = 500.0  # LR-D1 class
RADAR_FOV_COS = math.cos(math.radians(35.0))
RADAR_SIGMA_M = 0.15
# GPS blackout during high-plasma retropropulsion (WHITEPAPER 12.2).
BLACKOUT_MACH_MIN = 2.5
BLACKOUT_THRUST_MIN_N = 1.0e5

DISPLAY_SPEED_STEP = 1000.0 / 3600.0  # 1 km/h, as broadcast
DISPLAY_ALT_STEP = 100.0  # 0.1 km

OMEGA_E_VEC = jnp.array([0.0, 0.0, OMEGA_EARTH_RADPS])
_BASE_KEY = rng.key(20170814)

# --- Components -----------------------------------------------------------------
SensorTick = ty.Annotated[
    jax.Array,
    el.Component("sensor_tick", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
ImuAccel = ty.Annotated[
    jax.Array,
    el.Component("imu_accel", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
ImuGyro = ty.Annotated[
    jax.Array,
    el.Component("imu_gyro", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
GpsTimer = ty.Annotated[
    jax.Array,
    el.Component("gps_timer", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
GpsPos = ty.Annotated[
    jax.Array,
    el.Component("gps_pos", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
GpsVel = ty.Annotated[
    jax.Array,
    el.Component("gps_vel", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
GpsCount = ty.Annotated[
    jax.Array,
    el.Component("gps_count", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
RadarTimer = ty.Annotated[
    jax.Array,
    el.Component("radar_timer", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
RadarRange = ty.Annotated[
    jax.Array,
    el.Component("radar_range", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
RadarCount = ty.Annotated[
    jax.Array,
    el.Component("radar_count", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
PressureMeas = ty.Annotated[
    jax.Array,
    el.Component("pressure_meas", el.ComponentType(el.PrimitiveType.F64, (4,))),
]
DisplaySpeed = ty.Annotated[
    jax.Array,
    el.Component("display_speed", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
DisplayAlt = ty.Annotated[
    jax.Array,
    el.Component("display_alt", el.ComponentType(el.PrimitiveType.F64, (1,))),
]


def _noise(count, salt: int, shape, sigma):
    key = rng.fold_in(rng.fold_in(_BASE_KEY, salt), count.astype(jnp.int32))
    return sigma * rng.normal(key, shape=shape, dtype=jnp.float64)


def sensor_components() -> list:
    return [
        el.C(SensorTick, jnp.array([0.0])),
        el.C(ImuAccel, jnp.zeros(3)),
        el.C(ImuGyro, jnp.zeros(3)),
        el.C(GpsTimer, jnp.array([0.0])),
        el.C(GpsPos, jnp.zeros(3)),
        el.C(GpsVel, jnp.zeros(3)),
        el.C(GpsCount, jnp.array([0.0])),
        el.C(RadarTimer, jnp.array([0.0])),
        el.C(RadarRange, jnp.array([-1.0])),
        el.C(RadarCount, jnp.array([0.0])),
        el.C(PressureMeas, jnp.zeros(4)),
        el.C(DisplaySpeed, jnp.array([0.0])),
        el.C(DisplayAlt, jnp.array([0.0])),
    ]
