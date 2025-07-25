import typing as ty
from dataclasses import dataclass, field

import control
import elodin as el
import jax
import jax.numpy as jnp
import mekf
import motors
import numpy as np
import params
import sensors
import telemetry
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
    thrust_constant = np.linalg.lstsq(rpm_ref[:, np.newaxis] ** 2, thrust_ref, rcond=None)[0][0]
    torque_constant = np.linalg.lstsq(rpm_ref[:, np.newaxis] ** 2, torque_ref, rcond=None)[0][0]

    alpha = dt / (dt + params.MOT_TIME_CONST)
    rpm = jnp.interp(pwm, pwm_ref, rpm_ref)
    rpm = prev_rpm + alpha * (rpm - prev_rpm)

    thrust = rpm**2 * thrust_constant
    torque = rpm**2 * torque_constant * yaw_factor
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


def world() -> tuple[el.World, el.EntityId]:
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
            telemetry.Telemetry(),
        ],
        name="drone",
    )
    world.spawn(
        el.Panel.hsplit(
            el.Panel.vsplit(
                el.Panel.viewport(
                    pos="drone.world_pos + (0,0,0,0,2,2,2)",
                    look_at="drone.world_pos",
                    active=True,
                    show_grid=True,
                ),
            ),
            el.Panel.vsplit(
                el.Panel.graph("drone.angle_desired"),
                el.Panel.graph(
                    "drone.world_pos.q0, drone.world_pos.q1, drone.world_pos.q2, drone.world_pos.q3, drone.attitude_target",
                ),
                el.Panel.graph("drone.ang_vel_setpoint"),
            ),
            active=True,
        ),
        name="viewport",
    )
    world.spawn(
        el.Panel.hsplit(
            el.Panel.vsplit(
                el.Panel.graph("drone.motor_input"),
                el.Panel.graph("drone.motor_pwm"),
                el.Panel.graph("drone.motor_rpm"),
            ),
            el.Panel.vsplit(
                el.Panel.graph("drone.thrust"),
            ),
        ),
        name="Motor Panel",
    )
    world.spawn(
        el.Panel.hsplit(
            el.Panel.vsplit(
                el.Panel.graph("drone.rate_pid_state"),
            ),
            el.Panel.vsplit(
                el.Panel.graph(
                    "drone.gyro, drone.ang_vel_setpoint",
                    name="Drone: rate_control",
                ),
            ),
        ),
        name="Rate Control Panel",
    )
    world.spawn(
        el.Panel.hsplit(
            el.Panel.vsplit(
                el.Panel.graph("drone.gyro"),
                el.Panel.graph("drone.accel"),
                el.Panel.graph("drone.magnetometer"),
            ),
        ),
        name="Sensor Panel",
    )
    return world, drone


def inner_loop(run_count: int, system: el.System) -> el.System:
    out_sys = system
    for _ in range(run_count - 1):
        out_sys = out_sys | system
    return out_sys


def system(only_rate_control: bool = False) -> el.System:
    non_effectors = (
        control.attitude_flight_plan
        | control.update_target_attitude
        | control.attitude_control
        | control.rate_pid_state
        | control.rate_control
        | motors.output
    )
    if only_rate_control:
        non_effectors = (
            control.rate_flight_plan | control.rate_pid_state | control.rate_control | motors.output
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
        | sensors.imu
        | telemetry.telemetry,
    )
    return non_effectors | inner
