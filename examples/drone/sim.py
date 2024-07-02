import elodin as el
from dataclasses import field, dataclass
import jax
import typing as ty
import jax.numpy as jnp
from functools import partial
import math
import os

import params
import util
import sensors
import mekf
import control
import motors

TIME_STEP = 1.0 / params.SCHED_LOOP_RATE

# The thrust + current information is from https://www.rcgroups.com/forums/showthread.php?2376436-Multistar-Elite-2216-920kv-3S-and-4S-Prop-Data.
# The KV rating is from the product page: http://www.hexsoon.com/en/product/product-57-913.html.
# Most of this data is kind of garbage, but it's a start.
KV = 920.0  # Motor KV rating
opt_current = 1.36  # Current at optimal thrust efficiency (A)
opt_thrust = 155.0 * 9.81 / 1000  # Thrust at optimal thrust efficiency (N)
max_thrust = 990.0 * 9.81 / 1000  # Maximum thrust (N)

# Motor indexes:
#
#  0   1
#   \ /
#    X
#   / \
#  3   2

# Motor spin directions (1 for CW, -1 for CCW)
motor_spin_dir = jnp.array([1.0, -1.0, 1.0, -1.0])
# Angle of each motor in the quadcopter frame (rad)
quad_motor_angles = math.pi * jnp.array([0.75, 0.25, -0.25, -0.75])
# The axis of rotation for the torque produced by each motor
motor_torque_axes = util.motor_torque_axes(quad_motor_angles)
# Motor distances from the center of mass (m)
motor_dist = 0.24 * jnp.ones(4)

# Calculate torque-thrust ratio at optimal thrust efficiency
KT = 60.0 / (2 * jnp.pi * KV)  # Motor torque coefficient
opt_torque = KT * opt_current  # Torque at optimal thrust efficiency
torque_ratio = opt_torque / opt_thrust  # Torque-thrust ratio


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
class Globals(el.Archetype):
    time: el.Time = field(default_factory=lambda: jnp.float64(0.0))


@dataclass
class Drone(el.Archetype):
    body_thrust: BodyThrust = field(default_factory=lambda: el.SpatialForce.zero())
    body_drag: BodyDrag = field(default_factory=lambda: jnp.zeros(3))
    thrust: Thrust = field(default_factory=lambda: jnp.zeros(4))
    torque: Torque = field(default_factory=lambda: jnp.zeros(4))


@el.map
def motor_thrust_response(
    time: el.Time, pwm: motors.MotorPwm, prev_thrust: Thrust
) -> Thrust:
    poly_coefs = jnp.array([params.MOT_THST_EXPO, 1 - params.MOT_THST_EXPO, 0.0])
    remove_thrust_curve_scaling = partial(jnp.polyval, poly_coefs)
    thrust = remove_thrust_curve_scaling(pwm / motors.MAX_PWM_THROTTLE) * max_thrust
    dt = params.FAST_LOOP_TIME_STEP
    alpha = dt / (dt + params.MOT_TIME_CONST)
    return prev_thrust + alpha * (thrust - prev_thrust)


@el.map
def motor_torque(thrust: Thrust) -> Torque:
    return thrust * torque_ratio


@el.map
def body_thrust(thrust: Thrust, torque: Torque) -> BodyThrust:
    yaw_torque_sum = jnp.sum(torque * motor_spin_dir)
    body_thrust = el.SpatialForce.from_linear(jnp.array([0.0, 0.0, jnp.sum(thrust)]))
    yaw_torque = el.SpatialForce.from_torque(jnp.array([0.0, 0.0, yaw_torque_sum]))
    # additional torque from differential thrust:
    pitch_roll_torque = el.SpatialForce.from_torque(
        jnp.sum(motor_torque_axes * thrust[:, None] * motor_dist[:, None], axis=0)
    )
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
    return (
        f + el.SpatialForce.from_linear(drag) + el.SpatialForce(pos.angular() @ thrust)
    )


@el.map
def gravity(inertia: el.Inertia, f: el.Force) -> el.Force:
    return f + el.SpatialForce.from_linear(
        jnp.array([0.0, 0.0, -9.81]) * inertia.mass()
    )


def world() -> el.World:
    world = el.World()
    drone = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform.from_linear(jnp.array([0.0, 0.0, 2.0]))
                + el.SpatialTransform.from_axis_angle(
                    jnp.array([0.0, 0.0, 1.0]), jnp.pi * 0.0 / 4.0
                ),
                inertia=el.SpatialInertia(1.0, inertia=jnp.array([0.1, 0.1, 0.2])),
            ),
            world.glb(os.path.abspath("./examples/drone/edu-450-v2-drone.glb")),
            Drone(),
            motors.Motors(),
            sensors.IMU(),
            control.AttitudeController(),
            mekf.MEKF(),
            Globals(),
        ],
        name="Drone",
    )
    world.spawn(
        el.Panel.hsplit(
            [
                el.Panel.vsplit(
                    [
                        el.Panel.viewport(
                            track_entity=drone,
                            track_rotation=False,
                            active=True,
                            show_grid=True,
                            pos=[-0.5, -3.0, 0.5],
                            looking_at=[0.0, 0.0, 0.5],
                        ),
                        el.Panel.graph(
                            [
                                el.GraphEntity(
                                    drone,
                                    [
                                        el.Component.index(mekf.AttEstError),
                                    ],
                                )
                            ]
                        ),
                    ]
                ),
                el.Panel.vsplit(
                    [
                        el.Panel.graph(
                            [
                                el.GraphEntity(
                                    drone,
                                    [
                                        el.Component.index(motors.MotorInput),
                                    ],
                                )
                            ]
                        ),
                        el.Panel.graph(
                            [
                                el.GraphEntity(
                                    drone,
                                    [
                                        el.Component.index(Thrust),
                                    ],
                                )
                            ]
                        ),
                        el.Panel.graph(
                            [
                                el.GraphEntity(
                                    drone,
                                    [
                                        el.Component.index(control.AngVelSetpoint),
                                        el.Component.index(control.EulerRateTarget),
                                        el.Component.index(sensors.Gyro),
                                    ],
                                )
                            ]
                        ),
                    ]
                ),
            ],
            active=True,
        ),
        name="Viewport",
    )
    return world


def inner_loop(run_count: int, system: el.System) -> el.System:
    out_sys = system
    for _ in range(run_count - 1):
        out_sys = out_sys | system
    return out_sys


def system() -> el.System:
    non_effectors = (
        mekf.update_filter
        | mekf.att_est_error
        | control.attitude_flight_plan
        | control.update_target_attitude
        | control.attitude_control
        | control.rate_pid_state
        | control.rate_control
        | motors.output
    )
    effectors = (
        gravity
        | drag
        | motor_thrust_response
        | motor_torque
        | body_thrust
        | apply_body_forces
    )

    INNER_RUN_COUNT = round(TIME_STEP / params.FAST_LOOP_TIME_STEP)
    assert INNER_RUN_COUNT == 3

    inner = inner_loop(
        INNER_RUN_COUNT,
        el.advance_time(params.FAST_LOOP_TIME_STEP)
        | el.six_dof(
            params.FAST_LOOP_TIME_STEP, effectors, integrator=el.Integrator.SemiImplicit
        )
        | sensors.imu,
    )
    return non_effectors | inner
