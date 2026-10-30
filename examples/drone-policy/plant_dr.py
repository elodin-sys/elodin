"""Per-entity plant parameters for the drone: motor lag, thrust/torque scale, drag, wind and a
constant disturbance, read inside the effectors instead of trace-time constants."""

import functools
import typing as ty
from dataclasses import dataclass, field

import _drone_path  # noqa: F401  (examples/drone on sys.path)
import elodin as el
import jax
import jax.numpy as jnp
import motors
import numpy as np
import sim
from config import Config

# drone_params layout
THRUST_SCALE = slice(0, 4)
KQ_SCALE = 4
MOTOR_TAU = 5
DRAG_COEFF = 6
WIND = slice(7, 10)
FORCE = slice(10, 13)
TORQUE = slice(13, 16)
N_PARAMS = 16

NOMINAL_PARAMS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1] + [0.0] * 9)

DroneParams = ty.Annotated[
    jax.Array,
    el.Component(
        "drone_params",
        el.ComponentType(el.PrimitiveType.F64, (N_PARAMS,)),
        metadata={
            "transient": "true",
            "element_names": "ts1,ts2,ts3,ts4,kq,tau,c,wx,wy,wz,fx,fy,fz,tx,ty,tz",
        },
    ),
]


@dataclass
class Plant(el.Archetype):
    drone_params: DroneParams = field(default_factory=lambda: jnp.array(NOMINAL_PARAMS))


def pack(thrust_scale, kq_scale, motor_tau, drag_coeff, wind, force, torque) -> jax.Array:
    scalars = jnp.stack([kq_scale, motor_tau, drag_coeff])
    return jnp.concatenate([thrust_scale, scalars, wind, force, torque]).astype(jnp.float64)


@functools.cache
def motor_constants() -> tuple[float, float]:
    """(kT, kQ) from the thrust-curve CSV, fitted exactly as sim.motor_thrust_response does."""
    _, thrust_ref, torque_ref, rpm_ref = Config.GLOBAL.thrust_curve()
    kt = np.linalg.lstsq(rpm_ref[:, np.newaxis] ** 2, thrust_ref, rcond=None)[0][0]
    kq = np.linalg.lstsq(rpm_ref[:, np.newaxis] ** 2, torque_ref, rcond=None)[0][0]
    return float(kt), float(kq)


class HoverTrim(ty.NamedTuple):
    quat: np.ndarray  # [4] xyzw zero-yaw attitude that turns the net thrust vertical
    axis: np.ndarray  # [3] body direction of the net thrust
    rpm: np.ndarray  # [4] motor rpm, |net thrust| = stock m g


@functools.cache
def hover_trim() -> HoverTrim:
    """Torque-free hover of the stock vehicle from the motor geometry: the thrust split with zero
    net body torque (moments r x d plus the reaction torques along d, as in sim.body_thrust) and
    the attitude that makes its net thrust vertical. The Talon's rear motors are canted 3.75 deg
    back, so the trim is a 1.89 deg nose-down pitch."""
    cfg = Config.GLOBAL
    kt, kq = motor_constants()
    d = np.asarray(cfg.motor_thrust_directions)
    yaw_factor = cfg.frame.motor_matrix[2]
    moments = np.cross(cfg.motor_positions, d) + (kq / kt) * yaw_factor[:, None] * d
    share = np.linalg.svd(moments.T)[2][-1]
    share = share / share.sum()
    f = share @ d
    axis = f / np.linalg.norm(f)
    rot = np.cross(axis, [0.0, 0.0, 1.0])
    s = np.linalg.norm(rot)
    half = 0.5 * np.arctan2(s, axis[2])
    quat = np.append(rot / max(s, 1e-300) * np.sin(half), np.cos(half))
    rpm = np.sqrt(share * cfg.mass * 9.81 / (np.linalg.norm(f) * kt))  # sim.gravity
    return HoverTrim(quat, axis, rpm)


@el.map
def motor_response_dr(
    pwm: motors.MotorPwm, prev_rpm: motors.MotorRpm, params: DroneParams
) -> tuple[sim.Thrust, sim.Torque, motors.MotorRpm]:
    dt = Config.GLOBAL.fast_loop_time_step
    pwm_ref, _, _, rpm_ref = Config.GLOBAL.thrust_curve()
    _, _, yaw_factor, _ = Config.GLOBAL.frame.motor_matrix
    thrust_constant, torque_constant = motor_constants()

    scale = params[THRUST_SCALE]
    alpha = dt / (dt + params[MOTOR_TAU])
    rpm = jnp.interp(pwm, pwm_ref, rpm_ref)
    rpm = prev_rpm + alpha * (rpm - prev_rpm)

    thrust = rpm**2 * thrust_constant * scale
    torque = rpm**2 * torque_constant * yaw_factor * (scale * params[KQ_SCALE])
    return thrust, torque, rpm


@el.map
def drag_wind(v: el.WorldVel, params: DroneParams) -> sim.BodyDrag:
    rel_v = params[WIND] - v.linear()
    rel_v_norm = jnp.linalg.norm(rel_v)
    return params[DRAG_COEFF] * rel_v * rel_v_norm


@el.map
def disturbance(params: DroneParams, f: el.Force) -> el.Force:
    return f + el.SpatialForce(linear=params[FORCE], torque=params[TORQUE])


effectors = (
    sim.gravity
    | drag_wind
    | motor_response_dr
    | sim.body_thrust
    | disturbance
    | sim.apply_body_forces
)
