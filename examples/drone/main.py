import elodin as el
from dataclasses import field, dataclass
import jax
import typing as ty
import jax.numpy as jnp
from functools import partial
import math
import util
import os

# Corresponds to ArduPilot's SCHED_LOOP_RATE parameter.
# The default SCHED_LOOP_RATE for a QuadPlane is to 300 (Hz).
# https://ardupilot.org/plane/docs/quadplane-parameters.html
TIME_STEP = 1.0 / 300.0

# The thrust + current information is from https://www.rcgroups.com/forums/showthread.php?2376436-Multistar-Elite-2216-920kv-3S-and-4S-Prop-Data.
# The KV rating is from the product page: http://www.hexsoon.com/en/product/product-57-913.html.
# Most of this data is kind of garbage, but it's a start.
KV = 920.0  # Motor KV rating
max_voltage = 3.7 * 4  # 4S LiPo
max_throttle = 1000.0  # 1000us
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
motor_torque_axes = util.motor_thrust_axes(quad_motor_angles)
# Motor distances from the center of mass (m)
motor_dist = 0.24 * jnp.ones(4)

# Motor thrust scaling to compensate for the non-linear relationship between thrust and throttle.
# 0.65 is the default Ardupilot value: https://ardupilot.org/copter/docs/motor-thrust-scaling.html
motor_thrust_exp = 0.65
# Point at which the motors start to spin expressed as a number from 0 to 1 in the entire output range. Should be lower than MOT_SPIN_MIN.
# https://ardupilot.org/copter/docs/parameters.html#mot-spin-arm-motor-spin-armed
motor_spin_arm = 0.10
# Point at which the thrust starts expressed as a number from 0 to 1 in the entire output range. Should be higher than MOT_SPIN_ARM.
# https://ardupilot.org/copter/docs/parameters.html#mot-spin-min
motor_spin_min = 0.15
# Point at which the thrust saturates expressed as a number from 0 to 1 in the entire output range.
# https://ardupilot.org/copter/docs/parameters.html#mot-spin-max-motor-spin-maximum
motor_spin_max = 0.95

# Time to reach 95% of the final value / 3 (s)
settle_time = 0.03

# Calculate torque-thrust ratio at optimal thrust efficiency
KT = 60.0 / (2 * jnp.pi * KV)  # Motor torque coefficient
opt_torque = KT * opt_current  # Torque at optimal thrust efficiency
torque_ratio = opt_torque / opt_thrust  # Torque-thrust ratio

scale_thrust = partial(
    jnp.polyval, jnp.array([motor_thrust_exp, 1 - motor_thrust_exp, 0.0])
)


BodyThrust = ty.Annotated[
    el.SpatialForce,
    el.Component("body_thrust", metadata={"priority": 200}),
]

Throttle = ty.Annotated[
    jax.Array,
    el.Component(
        "throttle",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"priority": 100},
    ),
]
ExpectedThrust = ty.Annotated[
    jax.Array,
    el.Component(
        "expected_thrust",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"priority": 99},
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
    throttle: Throttle = field(default_factory=lambda: jnp.zeros(4))
    expected_thrust: ExpectedThrust = field(default_factory=lambda: jnp.zeros(4))
    thrust: Thrust = field(default_factory=lambda: jnp.zeros(4))
    torque: Torque = field(default_factory=lambda: jnp.zeros(4))


@el.map
def expected_motor_thrust(throttle: Throttle) -> ExpectedThrust:
    scaled_throttle = jnp.clip(throttle / max_throttle, motor_spin_min, motor_spin_max)
    scaled_thrust = scale_thrust(scaled_throttle)
    return scaled_thrust * max_thrust


@el.map
def motor_thrust_response(expected_thrust: ExpectedThrust) -> Thrust:
    # Assume instantaneous response for now.
    return expected_thrust


@el.map
def motor_torque(thrust: Thrust) -> Torque:
    return thrust * torque_ratio


@el.map
def body_thrust(thrust: Thrust, torque: Torque) -> BodyThrust:
    yaw_torque_sum = jnp.sum(torque * motor_spin_dir)
    body_thrust = el.SpatialForce.from_linear(jnp.array([0.0, jnp.sum(thrust), 0.0]))
    yaw_torque = el.SpatialForce.from_torque(jnp.array([0.0, yaw_torque_sum, 0.0]))
    # additional torque from differential thrust:
    pitch_roll_torque = el.SpatialForce.from_torque(
        jnp.sum(motor_torque_axes * thrust[:, None] * motor_dist[:, None], axis=0)
    )
    return body_thrust + yaw_torque + pitch_roll_torque


@el.map
def apply_body_forces(thrust: BodyThrust, pos: el.WorldPos, f: el.Force) -> el.Force:
    return f + el.SpatialForce(pos.angular() @ thrust)


@el.map
def flight_plan(time: el.Time, throttle: Throttle) -> Throttle:
    points = jnp.array(
        [
            [0.50, 0.50, 0.50, 0.50],
            [0.50, 0.50, 0.50, 0.50],
            [0.50, 0.50, 0.55, 0.55],
            [0.55, 0.55, 0.50, 0.50],
            [0.55, 0.55, 0.50, 0.50],
            [0.50, 0.50, 0.55, 0.55],
            [0.42, 0.42, 0.42, 0.42],
            [0.50, 0.50, 0.50, 0.50],
            [0.65, 0.35, 0.65, 0.35],
            [0.65, 0.35, 0.65, 0.35],
            [0.52, 0.50, 0.50, 0.50],
            [0.50, 0.50, 0.52, 0.50],
            [0.40, 0.40, 0.42, 0.40],
            [0.42, 0.40, 0.40, 0.40],
            [0.42, 0.42, 0.42, 0.42],
        ]
    )
    return jax.lax.cond(
        time < len(points),
        lambda _: points[time.astype(jnp.int32)] * 1000.0,
        lambda _: throttle,
        operand=None,
    )


@el.map
def gravity(inertia: el.Inertia, f: el.Force) -> el.Force:
    return f + el.SpatialForce.from_linear(
        jnp.array([0.0, -9.81, 0.0]) * inertia.mass()
    )


world = el.World()
drone = world.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform.from_linear(jnp.array([0.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(1.0, inertia=jnp.array([0.1, 0.4, 0.1])),
        ),
        world.glb(os.path.abspath("./examples/drone/edu-450-v2-drone.glb")),
        Drone(),
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
                        pos=[-2.0, 1.0, -2.0],
                        looking_at=[0.0, 0.5, 0.0],
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
                                    el.Component.index(Throttle),
                                ],
                            )
                        ]
                    ),
                    el.Panel.graph(
                        [
                            el.GraphEntity(
                                drone,
                                [
                                    el.Component.index(ExpectedThrust),
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
                                    el.Component.index(Torque),
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

non_effectors = (
    el.advance_time(TIME_STEP)
    | flight_plan
    | expected_motor_thrust
    | motor_thrust_response
    | motor_torque
    | body_thrust
)
effectors = gravity | apply_body_forces
sys = non_effectors | el.six_dof(TIME_STEP, effectors)
world.run(sys, time_step=TIME_STEP)
