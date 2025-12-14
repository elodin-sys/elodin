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
import numpy as np

from config import DroneConfig


# --- Component Type Definitions ---

# Motor commands from Betaflight (normalized 0-1)
MotorCommand = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_command",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "m0,m1,m2,m3", "priority": 100},
    ),
]

# Current motor thrust state (for dynamics)
MotorThrust = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_thrust",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "t0,t1,t2,t3", "priority": 99},
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
    Create simple ground collision constraint.
    
    Prevents the drone from going below ground level.
    """
    ground_level = config.ground_level
    
    @el.map
    def ground_constraint(pos: el.WorldPos, vel: el.WorldVel) -> tuple[el.WorldPos, el.WorldVel]:
        """Apply ground constraint."""
        p = pos.linear()
        v = vel.linear()
        
        # If below ground, clamp position and zero downward velocity
        below_ground = p[2] < ground_level
        
        new_z = jnp.where(below_ground, ground_level, p[2])
        new_vz = jnp.where(below_ground & (v[2] < 0), 0.0, v[2])
        
        new_pos = el.SpatialTransform(
            linear=jnp.array([p[0], p[1], new_z]),
            angular=pos.angular(),
        )
        new_vel = el.SpatialMotion(
            linear=jnp.array([v[0], v[1], new_vz]),
            angular=vel.angular(),
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
    
    # Editor schematic for visualization
    world.schematic(
        """
        tabs {
            hsplit name = "Viewport" {
                viewport name=Viewport pos="drone.world_pos + (0,0,0,0, 3,3,3)" look_at="drone.world_pos" show_grid=#true active=#true
                vsplit share=0.4 {
                    graph "drone.motor_command" name="Motor Commands"
                    graph "drone.motor_thrust" name="Motor Thrust"
                    graph "drone.world_pos.linear" name="Position"
                }
            }
            vsplit name="State" {
                graph "drone.world_vel.linear" name="Velocity"
                graph "drone.world_vel.angular" name="Angular Velocity"
                graph "drone.world_pos.angular" name="Attitude (Quat)"
            }
        }
        """,
        "betaflight-sitl.kdl",
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
    effectors = (
        motor_dynamics
        | body_thrust
        | drag
        | apply_forces
    )
    
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
    print(f"Config: {config.mass}kg, dt={config.sim_time_step*1000}ms")
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
    
    # world_pos is [qw, qx, qy, qz, x, y, z] - extract using polars
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
