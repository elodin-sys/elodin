import typing as ty
from dataclasses import dataclass, field

import control
import elodin as el
import jax
import jax.numpy as jnp
import mekf
import motors
import params
import sensors
from config import Config

BodyThrust = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "body_thrust",
        metadata={
            "priority": 200,
            "element_names": "τx,τy,τz,x,y,z",
        },
    ),
]
BodyDrag = ty.Annotated[
    jax.Array,
    el.Component(
        "body_drag",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

Thrust = ty.Annotated[
    jax.Array,
    el.Component(
        "thrust",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"priority": 98},
    ),
]
Torque = ty.Annotated[
    jax.Array,
    el.Component(
        "torque",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"priority": 97},
    ),
]


@dataclass
class Drone(el.Archetype):
    body_thrust: BodyThrust = field(default_factory=lambda: el.SpatialForce())
    body_drag: BodyDrag = field(default_factory=lambda: jnp.zeros(3))
    thrust: Thrust = field(default_factory=lambda: jnp.zeros(4))
    torque: Torque = field(default_factory=lambda: jnp.zeros(4))


@el.map
def motor_thrust_response(
    pwm: motors.MotorPwm, prev_thrust: Thrust, prev_torque: Torque, prev_rpm: motors.MotorRpm
) -> tuple[Thrust, Torque, motors.MotorRpm]:
    dt = Config.GLOBAL.fast_loop_time_step
    pwm_ref, thrust_ref, torque_ref, rpm_ref = Config.GLOBAL.thrust_curve()
    _, _, yaw_factor, _ = Config.GLOBAL.frame.motor_matrix

    thrust = jnp.interp(pwm, pwm_ref, thrust_ref)
    torque = jnp.interp(pwm, pwm_ref, torque_ref) * yaw_factor
    rpm = jnp.interp(pwm, pwm_ref, rpm_ref)

    alpha = dt / (dt + params.MOT_TIME_CONST)
    thrust = prev_thrust + alpha * (thrust - prev_thrust)
    torque = prev_torque + alpha * (torque - prev_torque)
    rpm = prev_rpm + alpha * (rpm - prev_rpm)
    return thrust, torque, rpm


@el.map
def body_thrust(thrust: Thrust, torque: Torque) -> BodyThrust:
    config = Config.GLOBAL
    thrust_dir = config.motor_thrust_directions
    torque_dir = config.motor_torque_axes
    body_thrust = el.SpatialForce(linear=jnp.sum(thrust_dir * thrust[:, None], axis=0))
    yaw_torque = el.SpatialForce(torque=jnp.sum(thrust_dir * torque[:, None], axis=0))
    # additional torque from differential thrust:
    pitch_roll_torque = el.SpatialForce(torque=jnp.sum(torque_dir * thrust[:, None], axis=0))
    return body_thrust + yaw_torque + pitch_roll_torque


@el.map
def drag(v: el.WorldVel) -> BodyDrag:
    rel_v = -v.linear()
    rel_v_norm = jnp.linalg.norm(rel_v)
    return 0.2 * 0.5 * rel_v * rel_v_norm


@el.map
def apply_body_forces(
    thrust: BodyThrust, drag: BodyDrag, pos: el.WorldPos, f: el.Force
) -> el.Force:
    return f + el.SpatialForce(linear=drag) + pos.angular() @ thrust


@el.map
def gravity(inertia: el.Inertia, f: el.Force) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())


def world() -> el.World:
    world = el.World()
    drone = world.spawn(
        [
            el.Body(
                world_pos=Config.GLOBAL.spatial_transform,
                inertia=Config.GLOBAL.spatial_inertia,
            ),
            world.glb(Config.GLOBAL.drone_glb),
            Drone(),
            motors.Motors(),
            sensors.IMU(),
            control.AttitudeController(),
            mekf.MEKF(),
        ],
        name="Drone",
    )
    world.spawn(
        el.Panel.hsplit(
            el.Panel.vsplit(
                el.Panel.viewport(
                    track_entity=drone,
                    track_rotation=False,
                    active=True,
                    show_grid=True,
                    pos=[-3.0, -0.5, 0.5],
                    looking_at=[0.0, 0.0, 0.5],
                ),
            ),
            el.Panel.vsplit(
                el.Panel.graph(el.GraphEntity(drone, control.AngleDesired)),
                el.Panel.graph(
                    el.GraphEntity(
                        drone,
                        *el.Component.index(el.WorldPos)[:4],
                        control.AttitudeTarget,
                    )
                ),
                el.Panel.graph(el.GraphEntity(drone, control.AngVelSetpoint)),
            ),
            active=True,
        ),
        name="Viewport",
    )
    world.spawn(
        el.Panel.hsplit(
            el.Panel.vsplit(
                el.Panel.graph(el.GraphEntity(drone, motors.MotorInput)),
                el.Panel.graph(el.GraphEntity(drone, motors.MotorPwm)),
                el.Panel.graph(el.GraphEntity(drone, motors.MotorRpm)),
            ),
            el.Panel.vsplit(
                el.Panel.graph(el.GraphEntity(drone, Thrust)),
            ),
        ),
        name="Motor Panel",
    )
    world.spawn(
        el.Panel.hsplit(
            el.Panel.vsplit(
                el.Panel.graph(el.GraphEntity(drone, control.RatePIDState)),
            ),
            el.Panel.vsplit(
                el.Panel.graph(
                    el.GraphEntity(
                        drone,
                        sensors.Gyro,
                        control.AngVelSetpoint,
                    ),
                    name="Drone: rate_control",
                ),
            ),
        ),
        name="Rate Control Panel",
    )
    return world


def inner_loop(run_count: int, system: el.System) -> el.System:
    out_sys = system
    for _ in range(run_count - 1):
        out_sys = out_sys | system
    return out_sys


def system() -> el.System:
    non_effectors = (
        control.attitude_flight_plan
        | control.update_target_attitude
        | control.attitude_control
        | control.rate_pid_state
        | control.rate_control
        | motors.output
    )
    effectors = gravity | drag | motor_thrust_response | body_thrust | apply_body_forces

    INNER_RUN_COUNT = round(Config.GLOBAL.dt / Config.GLOBAL.fast_loop_time_step)
    assert INNER_RUN_COUNT == 3

    inner = inner_loop(
        INNER_RUN_COUNT,
        el.six_dof(
            Config.GLOBAL.fast_loop_time_step,
            effectors,
            integrator=el.Integrator.SemiImplicit,
        )
        | sensors.imu,
    )
    return non_effectors | inner
