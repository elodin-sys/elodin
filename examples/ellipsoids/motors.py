import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp
import params
from config import Config

THROTTLE_RPY_MIX = 0.5
MAX_PWM_THROTTLE = 1000.0  # 1000us


MotorInput = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_input",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "roll,pitch,yaw,throttle"},
    ),
]

MotorPwm = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_pwm",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "m1,m2,m3,m4"},
    ),
]

MotorRpm = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_rpm",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "m1,m2,m3,m4"},
    ),
]


@dataclass
class Motors(el.Archetype):
    motor_input: MotorInput = field(default_factory=lambda: jnp.zeros(4))
    motor_pwm: MotorPwm = field(default_factory=lambda: jnp.zeros(4))
    motor_rpm: MotorRpm = field(default_factory=lambda: jnp.zeros(4))


@el.map
def motor_input_to_pwm(inputs: MotorInput) -> MotorPwm:
    mot_thst_hover = Config.GLOBAL.control.motor_thrust_hover
    roll_factor, pitch_factor, yaw_factor, throttle_factor = Config.GLOBAL.frame.motor_matrix
    # roll input range: -1 to 1
    # pitch input range: -1 to 1
    # yaw input range: -1 to 1
    # throttle input range: 0 to 1
    roll, pitch, yaw, throttle = inputs
    throttle_avg_max = THROTTLE_RPY_MIX * mot_thst_hover + (1 - THROTTLE_RPY_MIX) * throttle
    # allows for raising throttle above pilot input but still below hover throttle
    throttle_avg_max = jnp.clip(throttle_avg_max, throttle, 1.0)
    # throttle providing maximum roll, pitch and yaw range
    throttle_best_rpy = jnp.min(jnp.array([0.5, throttle_avg_max]))

    # clamp yaw to fit in the remaining range after pitch and roll
    out = roll * roll_factor + pitch * pitch_factor
    room = out + throttle_best_rpy
    room = jnp.where(jnp.positive(yaw * yaw_factor), 1.0 - room, room)
    yaw_allowed = jnp.min(jnp.clip(room, 0.0) / jnp.abs(yaw_factor))
    yaw = jnp.clip(yaw, -yaw_allowed, yaw_allowed)
    out += yaw * yaw_factor

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

    linear_throttle = (throttle_best_rpy + thr_adj) * throttle_factor + out * rpy_scale

    def apply_thrust_curve_scaling(y: jax.Array) -> jax.Array:
        # y = ax^2 + bx + c, where b = (1-a), c = 0
        # ax^2 + bx - y = 0
        # ax^2 + bx + c = 0, where a = a, b = (1-a), c = -y
        # x = (-b + sqrt(b^2 - 4ac)) / 2a
        a = Config.GLOBAL.control.motor_thrust_exponent
        b = 1 - a
        c = -y
        return (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)

    linear_throttle = jnp.clip(linear_throttle, 0.0, 1.0)
    # scale throttle from [0, 1] to [MIN_THROTTLE, MAX_THROTTLE]
    actuator = apply_thrust_curve_scaling(linear_throttle)

    return actuator * (params.MOT_PWM_THST_MAX - params.MOT_PWM_THST_MIN) + params.MOT_PWM_THST_MIN


output = motor_input_to_pwm
