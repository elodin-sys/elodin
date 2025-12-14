"""
Sensor Simulation for Betaflight SITL

This module simulates the IMU (Inertial Measurement Unit) and other sensors
that provide data to Betaflight's flight controller algorithms.

Sensor Models:
- Accelerometer: Measures linear acceleration in body frame
- Gyroscope: Measures angular velocity in body frame
- (Optional) Barometer: Measures altitude via pressure
- (Optional) GPS: Provides position and velocity

Coordinate Systems:
- Elodin: ENU (East-North-Up) with X=forward, Y=left, Z=up
- Betaflight: NED (North-East-Down) for sensor data
- Conversion handled in comms.py send_state method
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

from config import DroneConfig


# --- Sensor Component Types ---

# IMU accelerometer reading in body frame [ax, ay, az] m/s^2
Accel = ty.Annotated[
    jax.Array,
    el.Component(
        "accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 150},
    ),
]

# IMU gyroscope reading in body frame [wx, wy, wz] rad/s
Gyro = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 151},
    ),
]

# Barometer altitude reading in meters
Baro = ty.Annotated[
    jax.Array,
    el.Component(
        "baro",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"priority": 152},
    ),
]

# Body-frame linear velocity (for reference/debugging)
BodyVel = ty.Annotated[
    jax.Array,
    el.Component(
        "body_vel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 153},
    ),
]


@dataclass
class IMU(el.Archetype):
    """
    IMU sensor archetype.
    
    Stores computed sensor values for the drone entity.
    """
    accel: Accel = field(default_factory=lambda: jnp.zeros(3))
    gyro: Gyro = field(default_factory=lambda: jnp.zeros(3))
    baro: Baro = field(default_factory=lambda: jnp.zeros(1))
    body_vel: BodyVel = field(default_factory=lambda: jnp.zeros(3))


# --- Sensor Computation Systems ---

def create_imu_system(config: DroneConfig, add_noise: bool = False):
    """
    Create the IMU sensor computation system.
    
    The accelerometer measures:
    - Specific force = total_force / mass (excluding gravity in free fall)
    - In body frame
    
    The gyroscope measures:
    - Angular velocity in body frame
    
    Args:
        config: Drone configuration
        add_noise: Whether to add realistic sensor noise
        
    Returns:
        System that computes IMU readings
    """
    gravity = config.gravity
    
    # Noise parameters (typical MEMS IMU)
    accel_noise_std = 0.05 if add_noise else 0.0  # m/s^2
    gyro_noise_std = 0.001 if add_noise else 0.0  # rad/s
    
    @el.map
    def compute_imu(
        pos: el.WorldPos,
        vel: el.WorldVel,
        force: el.Force,
        inertia: el.Inertia,
    ) -> tuple[Accel, Gyro, BodyVel]:
        """
        Compute IMU sensor readings from physics state.
        
        The accelerometer measures specific force, which is the total force
        minus gravity, divided by mass. When in free fall, the accelerometer
        reads zero. When sitting on the ground, it reads +g upward (1g).
        """
        # Get orientation quaternion for frame transformation
        quat = pos.angular()
        quat_inv = quat.inverse()
        
        # --- Accelerometer ---
        # Total acceleration from forces (in world frame)
        mass = inertia.mass()
        total_accel_world = force.linear() / mass
        
        # Gravity vector in world frame (ENU: +Z is up)
        gravity_world = jnp.array([0.0, 0.0, -gravity])
        
        # Specific force in world frame (what accelerometer measures)
        # specific_force = total_accel - gravity
        # Note: When hovering, total_accel = 0, so specific_force = +g (upward)
        # When in free fall, total_accel = -g, so specific_force = 0
        specific_force_world = total_accel_world - gravity_world
        
        # Transform to body frame
        accel_body = quat_inv @ specific_force_world
        
        # --- Gyroscope ---
        # Angular velocity is already in body frame in Elodin
        gyro_body = vel.angular()
        
        # --- Body frame velocity (for reference) ---
        vel_world = vel.linear()
        body_vel = quat_inv @ vel_world
        
        return accel_body, gyro_body, body_vel
    
    return compute_imu


def create_baro_system(config: DroneConfig, add_noise: bool = False):
    """
    Create barometer sensor system.
    
    Simulates barometric altitude measurement based on height.
    """
    # Barometer noise (typical for consumer barometers)
    baro_noise_std = 0.3 if add_noise else 0.0  # meters
    
    @el.map
    def compute_baro(pos: el.WorldPos) -> Baro:
        """Compute barometer reading from altitude."""
        # Simple model: altitude = z position
        altitude = pos.linear()[2]
        return jnp.array([altitude])
    
    return compute_baro


def create_sensor_system(config: DroneConfig, add_noise: bool = False) -> el.System:
    """
    Create the complete sensor system.
    
    Args:
        config: Drone configuration
        add_noise: Whether to add sensor noise
        
    Returns:
        Combined sensor system
    """
    imu = create_imu_system(config, add_noise)
    baro = create_baro_system(config, add_noise)
    
    return imu | baro


# --- Sensor Data Extraction ---

def extract_sensor_data(
    world_pos: np.ndarray,
    world_vel: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    baro: np.ndarray,
) -> dict:
    """
    Extract and format sensor data for Betaflight communication.
    
    This function takes raw sensor values and prepares them for
    transmission to Betaflight SITL via the UDP bridge.
    
    Args:
        world_pos: Position [qw, qx, qy, qz, x, y, z]
        world_vel: Velocity [vx, vy, vz, wx, wy, wz]
        accel: Accelerometer [ax, ay, az] in body frame
        gyro: Gyroscope [wx, wy, wz] in body frame
        baro: Barometer [altitude]
        
    Returns:
        Dictionary with formatted sensor data for FDM packet
    """
    # Extract quaternion (Elodin format: [qw, qx, qy, qz])
    quat = world_pos[:4]
    position = world_pos[4:7]
    velocity = world_vel[:3]
    
    # Barometric pressure from altitude
    # Standard atmosphere: P = P0 * (1 - L*h/T0)^(g*M/(R*L))
    # Simplified: P ≈ 101325 - 12 * h (good for low altitudes)
    altitude = baro[0] if len(baro) > 0 else position[2]
    pressure = 101325.0 - 12.0 * altitude
    
    return {
        "angular_velocity": gyro,        # Body frame, rad/s
        "linear_acceleration": accel,     # Body frame, m/s^2
        "orientation_quat": quat,         # [w, x, y, z]
        "velocity": velocity,             # World frame, m/s (ENU)
        "position": position,             # World frame, m (ENU)
        "pressure": pressure,             # Pascals
    }


if __name__ == "__main__":
    # Test sensor computation
    from config import DEFAULT_CONFIG
    import polars as pl
    from sim import create_world, create_physics_system
    
    print("Testing sensor simulation...")
    
    config = DEFAULT_CONFIG
    config.simulation_time = 0.2
    config.set_as_global()
    
    # Create world with sensors
    world = el.World()
    
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
    
    from sim import Drone
    
    drone = world.spawn(
        [
            el.Body(
                world_pos=initial_pos,
                world_vel=initial_vel,
                inertia=inertia,
            ),
            Drone(),
            IMU(),
        ],
        name="drone",
    )
    
    # Create physics + sensor system
    physics = create_physics_system(config)
    sensors = create_sensor_system(config, add_noise=False)
    full_system = physics | sensors
    
    # Run simulation
    exec = world.build(full_system)
    exec.run(200)
    
    # Get sensor history
    df = exec.history(["drone.accel", "drone.gyro", "drone.baro"])
    print(f"Simulated {len(df)} ticks")
    
    # Check accelerometer reading at start (should be ~+g since on ground)
    first_accel = df[0]["drone.accel"].to_list()
    last_accel = df[-1]["drone.accel"].to_list()
    
    print(f"\nAccelerometer (body frame):")
    print(f"  First reading: {first_accel}")
    print(f"  Last reading:  {last_accel}")
    print(f"  Expected at rest: [0, 0, +{config.gravity:.2f}] m/s^2")
    
    # Check gyro (should be ~0 for stable drone)
    last_gyro = df[-1]["drone.gyro"].to_list()
    print(f"\nGyroscope (body frame):")
    print(f"  Last reading: {last_gyro}")
    print(f"  Expected at rest: [0, 0, 0] rad/s")
    
    # Check barometer
    last_baro = df[-1]["drone.baro"].to_list()
    print(f"\nBarometer:")
    print(f"  Last reading: {last_baro} m")
    
    print("\nSensor test complete!")
