"""
Physics Simulation for Betaflight SITL

This module implements the drone physics simulation that interfaces with
Betaflight SITL. Unlike the main drone example, this simulation does NOT
include any control systems - all control comes from Betaflight.

The simulation:
1. Receives motor commands from Betaflight (normalized 0-1)
2. Converts to thrust forces and torques
3. Integrates rigid body dynamics
4. Generates sensor outputs (IMU, position, velocity)

Physics Model:
- 6-DOF rigid body with thrust from 4 motors
- Quadratic drag in linear and angular motion
- Ground collision (simple constraint)
- Motor dynamics (first-order response)
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp

from config import DroneConfig


# --- Component Type Definitions ---

# Motor commands from Betaflight (normalized 0-1)
# Marked as external_control so Betaflight can write to it via Elodin-DB
#
# Motor order after SITL's Gazebo remapping (see sitl.c pwmCompleteMotorUpdate):
#   Index 0: Front Right (FR) - CCW  (originally BF Motor 1)
#   Index 1: Back Left (BL) - CCW    (originally BF Motor 2)
#   Index 2: Front Left (FL) - CW    (originally BF Motor 3)
#   Index 3: Back Right (BR) - CW    (originally BF Motor 0)
#
# See config.py for motor positions and spin directions.
MotorCommand = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_command",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={
            "element_names": "FR,BL,FL,BR",  # After SITL Gazebo remapping
            "priority": 100,
            "external_control": "true",  # Allows external writes from Betaflight bridge
        },
    ),
]

# Current motor thrust state (for dynamics)
# Same motor order as MotorCommand: FR(0), BL(1), FL(2), BR(3)
MotorThrust = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_thrust",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "FR,BL,FL,BR", "priority": 99},
    ),
]

# Body frame thrust force (for visualization)
BodyThrust = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "body_thrust",
        metadata={"priority": 98, "element_names": "τx,τy,τz,fx,fy,fz"},
    ),
]

# Drag force (for visualization)
BodyDrag = ty.Annotated[
    jax.Array,
    el.Component(
        "body_drag",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "fx,fy,fz"},
    ),
]

# Simulation time component
SimTime = ty.Annotated[
    jax.Array,
    el.Component(
        "sim_time",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"priority": 200},
    ),
]


@dataclass
class Drone(el.Archetype):
    """
    Drone archetype with physics state components.

    This archetype is spawned for each simulated drone entity.
    """

    motor_command: MotorCommand = field(default_factory=lambda: jnp.zeros(4))
    motor_thrust: MotorThrust = field(default_factory=lambda: jnp.zeros(4))
    body_thrust: BodyThrust = field(default_factory=lambda: el.SpatialForce())
    body_drag: BodyDrag = field(default_factory=lambda: jnp.zeros(3))
    sim_time: SimTime = field(default_factory=lambda: jnp.zeros(1))


# --- Physics Systems ---


def create_motor_dynamics(config: DroneConfig):
    """
    Create motor dynamics system.

    Motors have first-order response dynamics:
        thrust' = (commanded - thrust) / time_constant
    """
    dt = config.sim_time_step
    tau = config.motor_time_constant
    max_thrust = config.motor_max_thrust
    alpha = dt / (dt + tau)  # First-order filter coefficient

    @el.map
    def motor_dynamics(cmd: MotorCommand, thrust: MotorThrust) -> MotorThrust:
        """Update motor thrust based on commanded values."""
        # Clamp commands to valid range
        cmd_clamped = jnp.clip(cmd, 0.0, 1.0)

        # Convert normalized command to thrust target
        target_thrust = cmd_clamped * max_thrust

        # First-order low-pass filter for motor response
        new_thrust = thrust + alpha * (target_thrust - thrust)

        return new_thrust

    return motor_dynamics


def create_body_thrust_system(config: DroneConfig):
    """
    Create system to compute body-frame thrust and torques.

    Each motor produces:
    - Thrust force in body Z direction
    - Torque from thrust offset (roll/pitch)
    - Reaction torque from spin (yaw)
    """
    motor_positions = jnp.array(config.motor_positions)
    thrust_directions = jnp.array(config.motor_thrust_directions)
    spin_directions = jnp.array(config.motor_spin_directions)
    torque_coeff = config.motor_torque_coeff

    # Compute torque arms (cross product of position and thrust direction)
    torque_arms = jnp.cross(motor_positions, thrust_directions)

    @el.map
    def compute_body_thrust(thrust: MotorThrust) -> BodyThrust:
        """Compute total body-frame force and torque from motors."""
        # Linear force: sum of all motor thrusts in their directions
        total_force = jnp.sum(thrust[:, None] * thrust_directions, axis=0)

        # Torque from differential thrust (roll/pitch)
        diff_torque = jnp.sum(thrust[:, None] * torque_arms, axis=0)

        # Yaw torque from motor spin (reaction torque)
        yaw_torque = jnp.sum(thrust * spin_directions) * torque_coeff

        # Combine torques
        total_torque = diff_torque + jnp.array([0.0, 0.0, yaw_torque])

        return el.SpatialForce(linear=total_force, torque=total_torque)

    return compute_body_thrust


def create_drag_system(config: DroneConfig):
    """
    Create aerodynamic drag system.

    Drag is modeled as quadratic:
        F_drag = -0.5 * rho * Cd * A * |v| * v

    Simplified to linear coefficient times v * |v|
    """
    linear_drag = jnp.array(config.linear_drag)

    @el.map
    def compute_drag(vel: el.WorldVel) -> BodyDrag:
        """Compute drag force from velocity."""
        v = vel.linear()
        v_mag = jnp.linalg.norm(v)

        # Quadratic drag: F = -k * |v| * v
        drag_force = -linear_drag * v_mag * v

        return drag_force

    return compute_drag


def create_apply_forces_system(config: DroneConfig):
    """
    Create system to apply all forces to the body.

    Combines:
    - Motor thrust (body frame, rotated to world)
    - Drag (world frame)
    - Gravity (world frame)
    """
    gravity_vec = jnp.array([0.0, 0.0, -config.gravity])
    angular_drag = jnp.array(config.angular_drag)

    @el.map
    def apply_forces(
        thrust: BodyThrust,
        drag: BodyDrag,
        pos: el.WorldPos,
        vel: el.WorldVel,
        inertia: el.Inertia,
        force: el.Force,
    ) -> el.Force:
        """Apply all forces to the body."""
        # Rotate body thrust to world frame
        quat = pos.angular()
        world_thrust = quat @ thrust

        # Gravity force
        gravity_force = el.SpatialForce(linear=gravity_vec * inertia.mass())

        # Linear drag
        drag_force = el.SpatialForce(linear=drag)

        # Angular drag (damping on rotation)
        omega = vel.angular()
        omega_mag = jnp.linalg.norm(omega)
        angular_drag_torque = -angular_drag * omega_mag * omega
        angular_drag_force = el.SpatialForce(torque=angular_drag_torque)

        # Sum all forces
        return force + world_thrust + gravity_force + drag_force + angular_drag_force

    return apply_forces


def create_ground_constraint_system(config: DroneConfig):
    """
    Create ground collision constraint with friction.

    Prevents the drone from going below ground level.
    When on/near ground, applies angular damping to simulate ground contact
    friction that prevents tipping. The damping gradually decreases with
    altitude to provide a smooth transition from ground to flight.
    """
    ground_level = config.ground_level
    # Ground contact angular damping factor (0-1, higher = more damping)
    # 0.95 means 95% of angular velocity is removed each timestep when on ground
    max_damping = 0.95
    # Height at which damping starts (on ground)
    damping_start = ground_level + 0.01
    # Height at which damping ends (in flight) - gradual transition over 0.5m
    damping_end = ground_level + 0.5

    @el.map
    def ground_constraint(pos: el.WorldPos, vel: el.WorldVel) -> tuple[el.WorldPos, el.WorldVel]:
        """Apply ground constraint with gradual angular damping."""
        p = pos.linear()
        v = vel.linear()
        omega = vel.angular()

        # If below ground, clamp position and zero downward velocity
        below_ground = p[2] < ground_level
        new_z = jnp.where(below_ground, ground_level, p[2])
        new_vz = jnp.where(below_ground & (v[2] < 0), 0.0, v[2])

        # Gradual damping transition based on altitude
        # damping_factor goes from max_damping at ground to 0 at damping_end
        damping_ratio = jnp.clip((damping_end - p[2]) / (damping_end - damping_start), 0.0, 1.0)
        damping_factor = max_damping * damping_ratio

        # Apply damping: omega * (1 - damping_factor)
        # At ground: omega * 0.05 (95% removed)
        # At 0.5m: omega * 1.0 (no damping)
        new_omega = omega * (1.0 - damping_factor)

        new_pos = el.SpatialTransform(
            linear=jnp.array([p[0], p[1], new_z]),
            angular=pos.angular(),
        )
        new_vel = el.SpatialMotion(
            linear=jnp.array([v[0], v[1], new_vz]),
            angular=new_omega,
        )

        return new_pos, new_vel

    return ground_constraint


def create_time_update_system(config: DroneConfig):
    """Create system to track simulation time."""
    dt = config.sim_time_step

    @el.map
    def update_time(t: SimTime) -> SimTime:
        """Increment simulation time."""
        return t + dt

    return update_time


# --- World and System Construction ---


def create_world(config: DroneConfig) -> tuple[el.World, el.EntityId]:
    """
    Create the simulation world with a drone entity.

    Args:
        config: Drone configuration

    Returns:
        Tuple of (world, drone_entity_id)
    """
    world = el.World()

    # Initial state from config
    initial_pos = el.SpatialTransform(
        linear=jnp.array(config.initial_position),
        angular=el.Quaternion(jnp.array(config.initial_quaternion)),
    )
    initial_vel = el.SpatialMotion(
        linear=jnp.array(config.initial_velocity),
        angular=jnp.array(config.initial_angular_velocity),
    )
    inertia = el.SpatialInertia(
        mass=config.mass,
        inertia=jnp.array(config.inertia_diagonal),
    )

    # Spawn drone entity
    drone = world.spawn(
        [
            el.Body(
                world_pos=initial_pos,
                world_vel=initial_vel,
                inertia=inertia,
            ),
            Drone(sim_time=jnp.array([0.0])),
        ],
        name="drone",
    )

    return world, drone


def create_physics_system(config: DroneConfig) -> el.System:
    """
    Create the complete physics system for the simulation.

    Args:
        config: Drone configuration

    Returns:
        Combined physics system
    """
    # Create individual systems
    motor_dynamics = create_motor_dynamics(config)
    body_thrust = create_body_thrust_system(config)
    drag = create_drag_system(config)
    apply_forces = create_apply_forces_system(config)
    ground = create_ground_constraint_system(config)
    time_update = create_time_update_system(config)

    # Effector systems (applied before integration)
    effectors = motor_dynamics | body_thrust | drag | apply_forces

    # 6-DOF integrator with effectors
    physics = el.six_dof(
        config.sim_time_step,
        effectors,
        integrator=el.Integrator.SemiImplicit,
    )

    # Post-integration systems
    post_systems = ground | time_update

    return physics | post_systems


# --- Helper Functions ---

# Note: For SITL integration, we use the world.run() callback mechanism
# rather than direct component access. See main.py for integration example.


if __name__ == "__main__":
    # Quick test of the physics compilation
    from config import DEFAULT_CONFIG
    import polars as pl

    config = DEFAULT_CONFIG
    config.simulation_time = 0.2  # Short test
    config.set_as_global()

    print("Testing physics simulation...")
    print(f"Config: {config.mass}kg, dt={config.sim_time_step * 1000}ms")
    print(f"Hover throttle: {config.hover_throttle:.1%}")

    # Test 1: Free fall (motors off)
    print("\n--- Free fall test (motors off) ---")
    world, drone = create_world(config)
    system = create_physics_system(config)
    exec = world.build(system)

    # Run 200 ticks (200ms at 1kHz)
    exec.run(200)

    # Get history data
    df = exec.history(["drone.world_pos", "drone.world_vel", "drone.sim_time"])
    print(f"Simulated {len(df)} ticks")

    # world_pos is [qx, qy, qz, qw, x, y, z] (Elodin scalar-last format) - extract using polars
    df_expanded = df.select(
        pl.col("drone.world_pos").arr.get(6).alias("z"),
        pl.col("drone.world_vel").arr.get(2).alias("vz"),
    )

    final_z = df_expanded[-1]["z"]
    final_vz = df_expanded[-1]["vz"]

    print(f"Final z: {final_z} m (started at {config.initial_position[2]:.3f} m)")
    print(f"Final vz: {final_vz} m/s")
    print(f"Expected fall: {0.5 * 9.81 * 0.2**2:.3f} m")

    # Check if z changed (indicates gravity is working)
    initial_z = df_expanded[0]["z"]
    delta_z = initial_z - final_z
    print(f"Actual fall: {delta_z} m")

    # Test 2: Check physics system builds correctly
    print("\n--- System compilation test ---")
    print("All physics systems compiled successfully!")

    print("\nPhysics test complete!")
