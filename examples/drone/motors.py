import elodin as el
import typing as ty
from dataclasses import field, dataclass
import jax
import jax.numpy as jnp
import numpy as np
import math

from numpy._typing import NDArray

import params

THROTTLE_RPY_MIX = 0.5
MAX_PWM_THROTTLE = 1000.0  # 1000us


class MotorMatrix:
    _roll_factor: NDArray[np.float64]
    _pitch_factor: NDArray[np.float64]
    _yaw_factor: NDArray[np.float64]
    _throttle_factor: NDArray[np.float64]
    _angles: NDArray[np.float64]

    def __init__(self):
        # X-frame configuration
        self._angles = math.pi * np.array([0.75, 0.25, -0.25, -0.75])
        self._roll_factor = np.sin(self._angles)
        self._pitch_factor = np.sin(self._angles - jnp.pi / 2)
        self._yaw_factor = np.array([1.0, -1.0, 1.0, -1.0])
        self._throttle_factor = np.ones(4)
        self.normalize_factors()

    # scale factors to [-0.5, 0.5] for each axis
    def normalize_factors(self):
        roll_fac = np.max(np.abs(self._roll_factor))
        pitch_fac = np.max(np.abs(self._pitch_factor))
        yaw_fac = np.max(np.abs(self._yaw_factor))

        self._roll_factor /= 2 * roll_fac
        self._pitch_factor /= 2 * pitch_fac
        self._yaw_factor /= 2 * yaw_fac


MotorInput = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_input",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "roll,pitch,yaw,throttle"},
    ),
]

MotorThrust = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_thrust",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "m0,m1,m2,m3"},
    ),
]

MotorActuator = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_actuator",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "m0,m1,m2,m3"},
    ),
]


MotorPwm = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_pwm",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "m0,m1,m2,m3"},
    ),
]


@dataclass
class Motors(el.Archetype):
    motor_input: MotorInput = field(default_factory=lambda: jnp.zeros(4))
    motor_thrust: MotorThrust = field(default_factory=lambda: jnp.zeros(4))
    motor_actuator: MotorActuator = field(default_factory=lambda: jnp.zeros(4))
    motor_pwm: MotorPwm = field(default_factory=lambda: jnp.zeros(4))


@el.map
def motor_input_to_thrust(inputs: MotorInput) -> MotorThrust:
    motors = MotorMatrix()
    # roll input range: -1 to 1
    # pitch input range: -1 to 1
    # yaw input range: -1 to 1
    # throttle input range: 0 to 1
    roll, pitch, yaw, throttle = inputs
    throttle_avg_max = (
        THROTTLE_RPY_MIX * params.MOT_THST_HOVER + (1 - THROTTLE_RPY_MIX) * throttle
    )
    # allows for raising throttle above pilot input but still below hover throttle
    throttle_avg_max = jnp.clip(throttle_avg_max, throttle, 1.0)
    # throttle providing maximum roll, pitch and yaw range
    throttle_best_rpy = jnp.min(jnp.array([0.5, throttle_avg_max]))

    # clamp yaw to fit in the remaining range after pitch and roll
    out = roll * motors._roll_factor + pitch * motors._pitch_factor
    room = out + throttle_best_rpy
    room = jnp.where(jnp.positive(yaw * motors._yaw_factor), 1.0 - room, room)
    yaw_allowed = jnp.min(jnp.clip(room, 0.0) / jnp.abs(motors._yaw_factor))
    yaw = jnp.clip(yaw, -yaw_allowed, yaw_allowed)
    out += yaw * motors._yaw_factor

    rpy_low = jnp.min(out)
    rpy_high = jnp.max(out)
    rpy_scale = 1.0
    rpy_scale = jax.lax.cond(
        rpy_high - rpy_low > 1.0,
        lambda _: 1.0 / (rpy_high - rpy_low),
        lambda _: rpy_scale,
        operand=None,
    )
    rpy_scale = jax.lax.cond(
        throttle_avg_max + rpy_low < 0.0,
        lambda _: jnp.min(jnp.array([rpy_scale, -throttle_avg_max / rpy_low])),
        lambda _: rpy_scale,
        operand=None,
    )
    rpy_low *= rpy_scale
    rpy_high *= rpy_scale
    throttle_best_rpy = -rpy_low
    thr_adj = throttle - throttle_best_rpy
    thr_adj = jnp.where(rpy_scale < 1.0, jnp.float64(0.0), thr_adj)
    thr_adj = jnp.clip(thr_adj, 0.0, 1.0 - (throttle_best_rpy + rpy_high))

    out = (throttle_best_rpy + thr_adj) * motors._throttle_factor + out * rpy_scale
    return out


@el.map
def motor_thrust_to_actuator(linear_throttle: MotorThrust) -> MotorActuator:
    def apply_thrust_curve_scaling(y: jax.Array) -> jax.Array:
        # y = ax^2 + bx + c, b = (1-a), c = 0
        # y = ax^2 + (1-a)x
        # ax^2 + (1-a)x - y = 0
        # x = (-b + sqrt(b^2 - 4ac)) / 2a, where a = a, b = (1-a), c = -y
        # x = (-(1-a) + sqrt((1-a)^2 + 4ay)) / 2a
        # x = (a - 1 + sqrt((1-a)^2 + 4ay)) / 2a
        a = params.MOT_THST_EXPO
        b = 1 - a
        return (-b + jnp.sqrt(b**2 + 4 * a * y)) / (2 * a)

    linear_throttle = jnp.clip(linear_throttle, 0.0, 1.0)
    # scale throttle from [0, 1] to [MIN_THROTTLE, MAX_THROTTLE]
    scaled_throttle = (
        apply_thrust_curve_scaling(linear_throttle)
        * (params.MOT_SPIN_MAX - params.MOT_SPIN_MIN)
        + params.MOT_SPIN_MIN
    )
    return scaled_throttle


@el.map
def motor_actuator_to_pwm(actuator: MotorActuator) -> MotorPwm:
    return actuator * MAX_PWM_THROTTLE


output = motor_input_to_thrust | motor_thrust_to_actuator | motor_actuator_to_pwm
