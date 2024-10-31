import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp
import motors

BodyAngVel = ty.Annotated[
    jax.Array,
    el.Component(
        "body_ang_vel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]
MotorAngVel = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_ang_vel",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "m1,m2,m3,m4"},
    ),
]


@dataclass
class Telemetry(el.Archetype):
    body_ang_vel: BodyAngVel = field(default_factory=lambda: jnp.zeros(3))
    motor_ang_vel: MotorAngVel = field(default_factory=lambda: jnp.zeros(4))


@el.map
def body_ang_vel(p: el.WorldPos, v: el.WorldVel) -> BodyAngVel:
    return p.angular().inverse() @ v.angular()


@el.map
def motor_ang_vel(motor_rpm: motors.MotorRpm) -> MotorAngVel:
    return motor_rpm * 2 * jnp.pi / 60


telemetry = body_ang_vel | motor_ang_vel
