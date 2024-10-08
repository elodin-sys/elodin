import typing as ty
from dataclasses import dataclass, field

import elodin as el
import filter
import jax
import jax.numpy as jnp
import motors
import params
import sensors
import util
from config import Config

AC_ATTITUDE_THRUST_ERROR_ANGLE = 30.0 * jnp.pi / 180.0  # 30 degrees

AngleDesired = ty.Annotated[
    jax.Array,
    el.Component(
        "angle_desired",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        # roll and pitch are angles (rad), yaw is rate (rad/s)
        metadata={"priority": 300, "element_names": "r,p,y"},
    ),
]
AttitudeTarget = ty.Annotated[el.Quaternion, el.Component("attitude_target")]
AngVelSetpoint = ty.Annotated[
    jax.Array,
    el.Component(
        "ang_vel_setpoint",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": 299, "element_names": "r,p,y"},
    ),
]
EulerRateTarget = ty.Annotated[
    jax.Array,
    el.Component(
        "euler_rate_target",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": 298, "element_names": "r,p,y"},
    ),
]
RatePIDState = ty.Annotated[
    jax.Array,
    el.Component(
        "rate_pid_state",
        el.ComponentType(el.PrimitiveType.F64, (3, 3)),
        metadata={"element_names": "e.r,e.p,e.y,i.r,i.p,i.y,d.r,d.p,d.y"},
    ),
]


# proportional controller for smoothing out the target rate
def shape_euler_rate(
    target_rate: jax.Array,
    desired_rate: jax.Array,
    accel_max: float,
    dt: float,
    time_constant: float,
):
    error_rate = desired_rate - target_rate
    p = 1.0 / max(time_constant, 0.01)
    # the clip here is to prevent over-correction in the last time step
    correction_rate = jnp.clip(
        error_rate * p,
        -jnp.abs(error_rate) / dt,
        jnp.abs(error_rate) / dt,
    )
    desired_rate = target_rate + correction_rate * dt
    # limit acceleration
    delta_rate_max = accel_max * dt
    desired_rate = jnp.clip(
        desired_rate,
        target_rate - delta_rate_max,
        target_rate + delta_rate_max,
    )
    return desired_rate


# proportional controller for angle error to target rate
def shape_angle(
    error_angle: jax.Array,
    target_rate: jax.Array,
    accel_max: float,
    dt: float,
    time_constant: float,
) -> jax.Array:
    p = 1.0 / max(time_constant, 0.01)
    linear_dist = accel_max / p**2
    # enforce P and 2nd-order limit
    correction_rate = jnp.where(
        jnp.abs(error_angle) > linear_dist,
        jnp.sign(error_angle) * jnp.sqrt(2 * accel_max * (jnp.sign(error_angle) * error_angle - linear_dist / 2.0)),
        error_angle * p,
    )  # fmt: skip
    # the clip here is to prevent over-correction in the last time step
    desired_ang_rate = jnp.clip(
        correction_rate,
        -jnp.abs(error_angle) / dt,
        jnp.abs(error_angle) / dt,
    )
    return shape_euler_rate(target_rate, desired_ang_rate, accel_max, dt, 0.0)


# convert angular acceleration limits (in body frame) to euler acceleration limits
def angular_to_euler_accel_limit(att: el.Quaternion, ang_rate: jax.Array) -> jax.Array:
    x_rate, y_rate, z_rate = ang_rate
    phi, theta, _ = util.quat_to_euler(att)
    sin_phi = jnp.clip(jnp.abs(jnp.sin(phi)), 0.1, 1.0)
    cos_phi = jnp.clip(jnp.abs(jnp.cos(phi)), 0.1, 1.0)
    sin_theta = jnp.clip(jnp.abs(jnp.sin(theta)), 0.1, 1.0)
    cos_theta = jnp.clip(jnp.abs(jnp.cos(theta)), 0.1, 1.0)

    roll_rate = x_rate
    pitch_rate = jnp.min(jnp.array([y_rate / cos_phi, z_rate / sin_phi]))
    yaw_rate = jnp.min(
        jnp.array(
            [
                jnp.min(jnp.array([x_rate / sin_theta, y_rate / (sin_phi * cos_theta)])),
                z_rate / (cos_phi * cos_theta),
            ]
        )
    )
    return jnp.array([roll_rate, pitch_rate, yaw_rate])


# Calculate two ordered rotations to move att_body to att_target
# The first rotation corrects the thrust vector and the second rotation corrects the heading vector.
def thrust_vector_rotation_angles(
    att_target: el.Quaternion, att_body: el.Quaternion
) -> tuple[jax.Array, jax.Array]:
    thrust_up = jnp.array([0.0, 0.0, 1.0])
    att_target_thrust = att_target @ thrust_up
    att_body_thrust = att_body @ thrust_up
    # thrust_angle = jnp.arccos(jnp.clip(jnp.dot(thrust_up, att_body_thrust), -1.0, 1.0))
    thrust_error_angle = jnp.arccos(
        jnp.clip(jnp.dot(att_body_thrust, att_target_thrust), -1.0, 1.0)
    )
    thrust_vec_axis = jnp.cross(att_body_thrust, att_target_thrust)
    thrust_vec_len = jnp.linalg.norm(thrust_vec_axis)
    thrust_vec_axis = jax.lax.cond(
        jnp.min(jnp.array([thrust_vec_len, thrust_error_angle])) > 1e-6,
        lambda _: thrust_vec_axis / thrust_vec_len,
        lambda _: thrust_up,
        operand=None,
    )
    thrust_vec_axis = att_body.inverse() @ thrust_vec_axis
    thrust_correction = jax.lax.cond(
        thrust_error_angle > 1e-6,
        lambda _: el.Quaternion.from_axis_angle(thrust_vec_axis, thrust_error_angle),
        lambda _: el.Quaternion.identity(),
        operand=None,
    )
    att_error_x, att_error_y, _ = util.quat_to_axis_angle(thrust_correction)

    # heading correction
    heading_correction = thrust_correction.inverse() * att_body.inverse() * att_target
    _, _, att_error_z = util.quat_to_axis_angle(heading_correction)

    att_error = jnp.array([att_error_x, att_error_y, att_error_z])

    # TODO: limit yaw error to the maximum that would saturate the output when yaw rate is zero
    return att_error, thrust_error_angle


@dataclass
class AttitudeController(el.Archetype):
    angle_desired: AngleDesired = field(default_factory=lambda: jnp.zeros(3))
    att_target: AttitudeTarget = field(default_factory=lambda: el.Quaternion.identity())
    ang_vel_sp: AngVelSetpoint = field(default_factory=lambda: jnp.zeros(3))
    euler_rate_target: EulerRateTarget = field(default_factory=lambda: jnp.zeros(3))
    rate_pid_state: RatePIDState = field(default_factory=lambda: jnp.zeros((3, 3)))


@el.map
def rate_pid_state(state: RatePIDState, target: AngVelSetpoint, gyro: sensors.Gyro) -> RatePIDState:
    dt = Config.GLOBAL.dt
    e_filter = filter.LPF(
        jnp.array(
            [
                params.ATC_RAT_RLL_FLTE,
                params.ATC_RAT_PIT_FLTE,
                params.ATC_RAT_YAW_FLTE,
            ]
        ),
        1.0 / dt,
    )
    d_filter = filter.LPF(
        jnp.array(
            [
                params.ATC_RAT_RLL_FLTD,
                params.ATC_RAT_PIT_FLTD,
                params.ATC_RAT_YAW_FLTD,
            ]
        ),
        1.0 / dt,
    )

    e_prev, i_prev, d_prev = state

    e = target - gyro
    e = e_filter.apply(e_prev, e)
    i = i_prev + (e * dt)
    d = (e - e_prev) / dt
    d = d_filter.apply(d_prev, d)

    return jnp.array([e, i, d])


@el.map
def rate_control(state: RatePIDState) -> motors.MotorInput:
    # set throttle to hover + 5% for maneuvering
    hover = Config.GLOBAL.control.motor_thrust_hover + 0.05
    mv = jnp.sum(state * Config.GLOBAL.control.rate_pid_gains, axis=0)
    roll_mv, pitch_mv, yaw_mv = mv
    return jnp.array([roll_mv, pitch_mv, yaw_mv, hover])


@el.map
def update_target_attitude(
    angle_desired: AngleDesired,
    att_target: AttitudeTarget,
    euler_rate_target: EulerRateTarget,
) -> tuple[AttitudeTarget, EulerRateTarget]:
    dt = Config.GLOBAL.dt
    atc_input_tc = Config.GLOBAL.control.attitude_control_input_tc
    y_rate_tc = Config.GLOBAL.control.pilot_yaw_rate_tc

    roll_desired, pitch_desired, yaw_rate_desired = angle_desired
    roll_target, pitch_target, yaw_target = util.quat_to_euler(att_target)
    roll_rate_target, pitch_rate_target, yaw_rate_target = euler_rate_target
    ang_accel_limit = jnp.array(
        [params.ATC_ACCEL_R_MAX, params.ATC_ACCEL_P_MAX, params.ATC_ACCEL_Y_MAX]
    )
    # convert from centi-degree/s^2 to rad/s^2
    ang_accel_limit_rad = ang_accel_limit * 0.01 * jnp.pi / 180
    # convert from angular acceleration to euler axis acceleration
    euler_accel_limit = angular_to_euler_accel_limit(att_target, ang_accel_limit_rad)
    roll_accel_limit, pitch_accel_limit, yaw_accel_limit = euler_accel_limit

    roll_rate_target = shape_angle(
        util.normalize_angle(roll_desired - roll_target),
        roll_rate_target,
        roll_accel_limit,
        dt,
        atc_input_tc,
    )
    pitch_rate_target = shape_angle(
        util.normalize_angle(pitch_desired - pitch_target),
        pitch_rate_target,
        pitch_accel_limit,
        dt,
        atc_input_tc,
    )
    yaw_rate_target = shape_euler_rate(
        yaw_rate_target, yaw_rate_desired, yaw_accel_limit, dt, y_rate_tc
    )
    euler_rate_target = jnp.array([roll_rate_target, pitch_rate_target, yaw_rate_target])
    ang_vel_target = jnp.nan_to_num(util.euler_to_angular_rate(att_target, euler_rate_target))
    att_target = att_target * util.quat_from_axis_angle(ang_vel_target * dt)
    return att_target, euler_rate_target


@el.map
def attitude_control(
    pos: el.WorldPos,
    gyro: sensors.Gyro,
    att_target: AttitudeTarget,
    euler_rate_target: EulerRateTarget,
    prev_ang_vel_sp: AngVelSetpoint,
) -> AngVelSetpoint:
    att_body = pos.angular()
    att_target = att_target * Config.GLOBAL.attitude
    target_to_body_rotation = att_body.inverse() * att_target
    ang_vel_target = jnp.nan_to_num(util.euler_to_angular_rate(att_target, euler_rate_target))
    ang_vel_body_feedforward = target_to_body_rotation @ ang_vel_target
    att_error, thrust_error_angle = thrust_vector_rotation_angles(att_target, att_body)
    ang_vel_body = att_error * Config.GLOBAL.control.angle_p_gains

    def feedforward(ang_vel_body, ang_vel_body_feedforward, thrust_error_angle, gyro):
        feedforward_scalar = (
            1.0
            - (thrust_error_angle - AC_ATTITUDE_THRUST_ERROR_ANGLE) / AC_ATTITUDE_THRUST_ERROR_ANGLE
        )
        ang_vel_body.at[0].set(ang_vel_body[0] + ang_vel_body_feedforward[0] * feedforward_scalar)
        ang_vel_body.at[1].set(ang_vel_body[1] + ang_vel_body_feedforward[1] * feedforward_scalar)
        ang_vel_body.at[2].set(ang_vel_body[2] + ang_vel_body_feedforward[2])
        ang_vel_body.at[2].set(
            gyro[2] * (1.0 - feedforward_scalar) + ang_vel_body[2] * feedforward_scalar
        )
        return ang_vel_body

    ang_vel_body = jax.lax.cond(
        thrust_error_angle > AC_ATTITUDE_THRUST_ERROR_ANGLE * 2.0,
        lambda _: ang_vel_body.at[2].set(gyro[2]),
        lambda _: jax.lax.cond(
            thrust_error_angle > AC_ATTITUDE_THRUST_ERROR_ANGLE,
            lambda _: feedforward(ang_vel_body, ang_vel_body_feedforward, thrust_error_angle, gyro),
            lambda _: ang_vel_body + ang_vel_body_feedforward,
            operand=None,
        ),
        operand=None,
    )

    t_filter = filter.LPF(
        jnp.array(
            [
                params.ATC_RAT_RLL_FLTT,
                params.ATC_RAT_PIT_FLTT,
                params.ATC_RAT_YAW_FLTT,
            ]
        ),
        Config.GLOBAL.dt,
    )
    ang_vel_body = t_filter.apply(prev_ang_vel_sp, ang_vel_body)
    return ang_vel_body


@el.system
def attitude_flight_plan(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    angle: el.Query[AngleDesired],
) -> el.Query[AngleDesired]:
    pitch_test_points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    roll_test_points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    yaw_test_points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, -0.2],
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 0.0],
        ]
    )
    combined_test_points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.4, 0.0],
            [-0.3, 0.4, 0.0],
            [0.1, 0.1, 0.0],
            [0.3, -0.4, 0.0],
        ]
    )
    points = jnp.concatenate(
        [
            combined_test_points,
            pitch_test_points,
            roll_test_points,
            yaw_test_points,
        ]
    )
    time = tick[0] * dt[0]
    point = points[time.astype(jnp.int32)]
    return angle.map(AngleDesired, lambda _: point)


@el.system
def rate_flight_plan(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    target: el.Query[AngVelSetpoint],
) -> el.Query[AngVelSetpoint]:
    pitch_test_points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, -0.7, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    roll_test_points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    yaw_test_points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.1],
            [0.0, 0.0, -0.1],
            [0.0, 0.0, -0.3],
            [0.0, 0.0, -0.3],
            [0.0, 0.0, 0.0],
        ]
    )
    points = jnp.concatenate([pitch_test_points, roll_test_points, yaw_test_points])
    time = tick[0] * dt[0]
    point = points[time.astype(jnp.int32)]
    return target.map(AngVelSetpoint, lambda _: point)
