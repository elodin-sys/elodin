import elodin as el
from dataclasses import field, dataclass
import jax
import typing as ty
import jax.numpy as jnp
from functools import partial

import params
import sensors
import mekf
import control
import motors
from config import Config

# The thrust + current information is from https://www.rcgroups.com/forums/showthread.php?2376436-Multistar-Elite-2216-920kv-3S-and-4S-Prop-Data.
# The KV rating is from the product page: http://www.hexsoon.com/en/product/product-57-913.html.
# Most of this data is kind of garbage, but it's a start.
KV = 920.0  # Motor KV rating
opt_current = 1.36  # Current at optimal thrust efficiency (A)
opt_thrust = 155.0 * 9.81 / 1000  # Thrust at optimal thrust efficiency (N)
max_thrust = 990.0 * 9.81 / 1000  # Maximum thrust (N)

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
class Drone(el.Archetype):
    body_thrust: BodyThrust = field(default_factory=lambda: el.SpatialForce.zero())
    body_drag: BodyDrag = field(default_factory=lambda: jnp.zeros(3))
    thrust: Thrust = field(default_factory=lambda: jnp.zeros(4))
    torque: Torque = field(default_factory=lambda: jnp.zeros(4))


@el.map
def motor_thrust_response(pwm: motors.MotorPwm, prev_thrust: Thrust) -> Thrust:
    poly_coefs = jnp.array([params.MOT_THST_EXPO, 1 - params.MOT_THST_EXPO, 0.0])
    remove_thrust_curve_scaling = partial(jnp.polyval, poly_coefs)
    thrust = remove_thrust_curve_scaling(pwm / motors.MAX_PWM_THROTTLE) * max_thrust
    dt = Config.GLOBAL.fast_loop_time_step
    alpha = dt / (dt + params.MOT_TIME_CONST)
    return prev_thrust + alpha * (thrust - prev_thrust)


@el.map
def motor_torque(thrust: Thrust) -> Torque:
    return thrust * torque_ratio


@el.map
def body_thrust(thrust: Thrust, torque: Torque) -> BodyThrust:
    yaw_torque_sum = jnp.sum(torque * Config.GLOBAL.motor_spin_dir)
    body_thrust = el.SpatialForce(linear=jnp.array([0.0, 0.0, jnp.sum(thrust)]))
    yaw_torque = el.SpatialForce(torque=jnp.array([0.0, 0.0, yaw_torque_sum]))
    # additional torque from differential thrust:
    pitch_roll_torque = el.SpatialForce(
        torque=jnp.sum(
            Config.GLOBAL.motor_torque_axes * thrust[:, None],
            axis=0,
        )
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
                    pos=[-0.5, -3.0, 0.5],
                    looking_at=[0.0, 0.0, 0.5],
                ),
                el.Panel.graph(el.GraphEntity(drone, mekf.AttEstError)),
            ),
            el.Panel.vsplit(
                el.Panel.graph(el.GraphEntity(drone, motors.MotorInput)),
                el.Panel.graph(el.GraphEntity(drone, Thrust)),
                el.Panel.graph(
                    el.GraphEntity(
                        drone,
                        control.AngVelSetpoint,
                        control.EulerRateTarget,
                        sensors.Gyro,
                    ),
                    name="Drone: rate_control",
                ),
            ),
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
