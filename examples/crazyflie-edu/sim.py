"""
Crazyflie Physics Simulation

6-DOF rigid body dynamics with motor thrust, drag, and gravity.
Based on the Elodin drone example, adapted for Crazyflie 2.1 parameters.
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

from config import CrazyflieConfig as Config

# =============================================================================
# Component Definitions
# =============================================================================

BodyThrust = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "body_thrust",
        metadata={
            "priority": 200,
            "element_names": "τx,τy,τz,fx,fy,fz",
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
        metadata={"priority": 98, "element_names": "m1,m2,m3,m4"},
    ),
]

Torque = ty.Annotated[
    jax.Array,
    el.Component(
        "torque",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"priority": 97, "element_names": "m1,m2,m3,m4"},
    ),
]

MotorRpm = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_rpm",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"priority": 96, "element_names": "m1,m2,m3,m4"},
    ),
]

MotorPwm = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_pwm",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"priority": 95, "element_names": "m1,m2,m3,m4", "external_control": "true"},
    ),
]


# Thrust visualization vectors (point downward, scaled 0.001-0.1)
ThrustVizM1 = ty.Annotated[
    jax.Array,
    el.Component("thrust_viz_m1", el.ComponentType(el.PrimitiveType.F64, (3,))),
]

ThrustVizM2 = ty.Annotated[
    jax.Array,
    el.Component("thrust_viz_m2", el.ComponentType(el.PrimitiveType.F64, (3,))),
]

ThrustVizM3 = ty.Annotated[
    jax.Array,
    el.Component("thrust_viz_m3", el.ComponentType(el.PrimitiveType.F64, (3,))),
]

ThrustVizM4 = ty.Annotated[
    jax.Array,
    el.Component("thrust_viz_m4", el.ComponentType(el.PrimitiveType.F64, (3,))),
]

# =============================================================================
# Archetypes
# =============================================================================


@dataclass
class CrazyflieDrone(el.Archetype):
    """Archetype containing all drone-specific components."""

    body_thrust: BodyThrust = field(default_factory=lambda: el.SpatialForce())
    body_drag: BodyDrag = field(default_factory=lambda: jnp.zeros(3))
    thrust: Thrust = field(default_factory=lambda: jnp.zeros(4))
    torque: Torque = field(default_factory=lambda: jnp.zeros(4))
    motor_rpm: MotorRpm = field(default_factory=lambda: jnp.zeros(4))
    motor_pwm: MotorPwm = field(default_factory=lambda: jnp.zeros(4))
    # Thrust visualization vectors (point downward from each rotor)
    thrust_viz_m1: ThrustVizM1 = field(default_factory=lambda: jnp.array([0.0, 0.0, -0.001]))
    thrust_viz_m2: ThrustVizM2 = field(default_factory=lambda: jnp.array([0.0, 0.0, -0.001]))
    thrust_viz_m3: ThrustVizM3 = field(default_factory=lambda: jnp.array([0.0, 0.0, -0.001]))
    thrust_viz_m4: ThrustVizM4 = field(default_factory=lambda: jnp.array([0.0, 0.0, -0.001]))


# =============================================================================
# Physics Systems
# =============================================================================


@el.map
def motor_dynamics(
    pwm: MotorPwm,
    prev_rpm: MotorRpm,
) -> tuple[MotorRpm, Thrust, Torque]:
    """
    Simulate motor response to PWM commands.

    Models:
    1. PWM to target RPM (using affine mapping)
    2. First-order motor response (time constant)
    3. RPM to thrust (quadratic relationship)
    4. RPM to torque (quadratic relationship)
    
    Note: Safety/arming logic is handled in user_code.py, not here.
    The simulation applies whatever PWM values are provided.
    """
    config = Config.GLOBAL
    dt = config.fast_loop_time_step

    # PWM threshold - below this, motors are considered OFF
    # The linear fit (rpm = a + b*pwm) is only valid above a minimum PWM
    pwm_min_threshold = 5.0
    
    # Convert PWM to target RPM
    # rpm = a + b * pwm (only when PWM is above threshold)
    target_rpm = jnp.where(
        pwm > pwm_min_threshold,
        config.pwm_to_rpm_a + config.pwm_to_rpm_b * pwm,
        jnp.zeros(4)  # Motors OFF below threshold
    )

    # Clamp to valid range (motors don't spin backwards)
    target_rpm = jnp.maximum(target_rpm, 0.0)

    # Apply motor time constant (first-order response)
    alpha = dt / (dt + config.motor_time_constant)
    rpm = prev_rpm + alpha * (target_rpm - prev_rpm)

    # Convert RPM to rad/s
    omega = rpm * 2.0 * jnp.pi / 60.0

    # Calculate thrust: F = k * omega^2
    thrust = config.thrust_constant * omega**2

    # Calculate torque: tau = kt * omega^2
    # Sign depends on motor rotation direction (handled in body_thrust)
    torque = config.torque_constant * omega**2

    return rpm, thrust, torque


@el.map
def body_thrust(thrust: Thrust, torque: Torque) -> BodyThrust:
    """
    Compute total force and torque on the body from motor thrusts.

    Combines:
    1. Linear thrust (all motors push up)
    2. Yaw torque (reaction torque from spinning motors)
    3. Roll/pitch torque (differential thrust)
    """
    config = Config.GLOBAL

    thrust_dir = config.motor_thrust_directions
    torque_axes = config.motor_torque_axes

    # Total linear thrust (sum of all motor thrusts in their directions)
    linear_thrust = jnp.sum(thrust_dir * thrust[:, None], axis=0)

    # Yaw torque from motor reaction
    # CW motors (M1, M3) produce CCW torque, CCW motors (M2, M4) produce CW torque
    _, _, yaw_factor, _ = config.frame.motor_matrix
    yaw_torque_vec = jnp.sum(thrust_dir * (torque * yaw_factor)[:, None], axis=0)

    # Roll/pitch torque from differential thrust
    roll_pitch_torque = jnp.sum(torque_axes * thrust[:, None], axis=0)

    total_torque = yaw_torque_vec + roll_pitch_torque

    return el.SpatialForce(linear=linear_thrust, torque=total_torque)


@el.map
def drag(v: el.WorldVel) -> BodyDrag:
    """
    Simple aerodynamic drag model.

    Uses a linear drag model: F_drag = -k * v
    """
    config = Config.GLOBAL
    return -config.drag_coefficient * v.linear()


@el.map
def apply_body_forces(
    thrust: BodyThrust, drag: BodyDrag, pos: el.WorldPos, f: el.Force
) -> el.Force:
    """
    Apply thrust and drag forces to the body.

    Transforms body-frame forces to world frame using current orientation.
    """
    # Rotate body thrust to world frame
    world_thrust = pos.angular() @ thrust
    # Add drag (already in world frame)
    drag_force = el.SpatialForce(linear=drag)
    return f + world_thrust + drag_force


@el.map
def gravity(inertia: el.Inertia, f: el.Force) -> el.Force:
    """Apply gravitational force."""
    g = 9.81
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -g]) * inertia.mass())


@el.map
def ground_constraint(pos: el.WorldPos, vel: el.WorldVel) -> tuple[el.WorldPos, el.WorldVel]:
    """
    Simple ground constraint to prevent the drone from falling through the floor.

    Also adds angular damping when on ground to prevent tipping.
    """
    # Get current position
    z = pos.linear()[2]

    # Ground level
    ground_z = 0.01  # Small offset for visual

    # If below ground, push back up and zero vertical velocity
    new_z = jnp.maximum(z, ground_z)
    new_pos_linear = pos.linear().at[2].set(new_z)

    # Zero out downward velocity when on ground
    on_ground = z <= ground_z
    new_vel_linear = jnp.where(
        on_ground & (vel.linear()[2] < 0),
        vel.linear().at[2].set(0.0),
        vel.linear(),
    )

    # Add angular damping when on ground (prevents tipping)
    damping = 0.95
    new_vel_angular = jnp.where(
        on_ground,
        vel.angular() * (1.0 - damping),
        vel.angular(),
    )

    new_pos = el.SpatialTransform(linear=new_pos_linear, angular=pos.angular())
    new_vel = el.SpatialMotion(linear=new_vel_linear, angular=new_vel_angular)

    return new_pos, new_vel


@el.map
def thrust_visualization(
    thrust: Thrust,
) -> tuple[ThrustVizM1, ThrustVizM2, ThrustVizM3, ThrustVizM4]:
    """
    Compute thrust visualization vectors for each motor.

    Outputs downward-pointing vectors with length proportional to thrust.
    Length is normalized to 0.001-0.1 range for visualization.
    """
    # Normalize thrust to visualization height (0.001 to 0.1)
    max_thrust = 0.1  # ~100mN, well above hover thrust per motor
    min_height = 0.001
    max_height = 0.1

    def normalize(t: jax.Array) -> jax.Array:
        normalized = jnp.clip(t / max_thrust, 0.0, 1.0)
        height = min_height + normalized * (max_height - min_height)
        return jnp.array([0.0, 0.0, -height])  # Point DOWN

    return (
        normalize(thrust[0]),
        normalize(thrust[1]),
        normalize(thrust[2]),
        normalize(thrust[3]),
    )


# =============================================================================
# System Composition
# =============================================================================


def create_effector_system() -> el.System:
    """Create the physics effector system (forces and torques)."""
    return motor_dynamics | body_thrust | drag | apply_body_forces | gravity


def create_physics_system() -> el.System:
    """
    Create the complete physics system.

    Combines 6-DOF integration with ground constraints.
    """
    config = Config.GLOBAL

    effectors = create_effector_system()

    # 6-DOF integration with semi-implicit Euler
    physics = el.six_dof(
        config.fast_loop_time_step,
        effectors,
        integrator=el.Integrator.SemiImplicit,
    )

    # Add ground constraint
    return physics | ground_constraint


# =============================================================================
# Utility Functions for Powertrain Identification (Lab 2)
# =============================================================================


def pwm_from_speed(desired_speed_rad_s: float) -> int:
    """
    Convert desired motor speed (rad/s) to PWM command.

    This is the INVERSE of the motor model.
    Students will implement this in Lab 2.

    Args:
        desired_speed_rad_s: Desired motor angular velocity in rad/s

    Returns:
        PWM command (0-255)
    """
    config = Config.GLOBAL

    # Convert rad/s to RPM
    desired_rpm = desired_speed_rad_s * 60.0 / (2.0 * np.pi)

    # Invert the PWM-to-RPM mapping: rpm = a + b * pwm
    # pwm = (rpm - a) / b
    pwm = (desired_rpm - config.pwm_to_rpm_a) / config.pwm_to_rpm_b

    # Clamp to valid range
    return int(np.clip(pwm, config.pwm_min, config.pwm_max))


def speed_from_force(desired_force_n: float) -> float:
    """
    Convert desired thrust force (N) to motor speed (rad/s).

    This is the INVERSE of the thrust model.
    Students will implement this in Lab 2.

    Args:
        desired_force_n: Desired thrust force in Newtons

    Returns:
        Motor angular velocity in rad/s
    """
    config = Config.GLOBAL

    if desired_force_n <= 0:
        return 0.0

    # Invert thrust model: F = k * omega^2
    # omega = sqrt(F / k)
    omega = np.sqrt(desired_force_n / config.thrust_constant)

    return float(omega)


def force_from_speed(speed_rad_s: float) -> float:
    """
    Convert motor speed (rad/s) to thrust force (N).

    Uses the quadratic thrust model: F = k * omega^2

    Args:
        speed_rad_s: Motor angular velocity in rad/s

    Returns:
        Thrust force in Newtons
    """
    config = Config.GLOBAL
    return config.thrust_constant * speed_rad_s**2

