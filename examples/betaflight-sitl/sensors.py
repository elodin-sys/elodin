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
from comms import FDMPacket

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
        world_pos: Position [qx, qy, qz, qw, x, y, z] (Elodin scalar-last format)
        world_vel: Velocity [wx, wy, wz, vx, vy, vz] (angular first in Elodin)
        accel: Accelerometer [ax, ay, az] in body frame
        gyro: Gyroscope [wx, wy, wz] in body frame
        baro: Barometer [altitude]
        
    Returns:
        Dictionary with formatted sensor data for FDM packet
    """
    # Extract quaternion from Elodin format [qx, qy, qz, qw] and convert to [qw, qx, qy, qz]
    quat_xyzw = world_pos[:4]
    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
    position = world_pos[4:7]
    # Elodin stores world_vel as [wx, wy, wz, vx, vy, vz] (angular first)
    velocity = world_vel[3:6]

    # Barometric pressure from altitude
    # Standard atmosphere: P = P0 * (1 - L*h/T0)^(g*M/(R*L))
    # Simplified: P ≈ 101325 - 12 * h (good for low altitudes)
    altitude = baro[0] if len(baro) > 0 else position[2]
    pressure = 101325.0 - 12.0 * altitude

    return {
        "angular_velocity": gyro,  # Body frame, rad/s
        "linear_acceleration": accel,  # Body frame, m/s^2
        "orientation_quat": quat,  # [w, x, y, z]
        "velocity": velocity,  # World frame, m/s (ENU)
        "position": position,  # World frame, m (ENU)
        "pressure": pressure,  # Pascals
    }


def build_fdm_from_components(
    world_pos: np.ndarray,
    world_vel: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    timestamp: float,
    gravity: float = 9.80665,
) -> FDMPacket:
    """
    Build an FDM packet directly from Elodin component data.

    This is the primary function for extracting sensor data from the simulation
    and packaging it for Betaflight. It handles:
    - Quaternion extraction from world_pos (Elodin scalar-last to Betaflight scalar-first)
    - Velocity extraction from world_vel
    - ENU to NED coordinate conversion for gyro/accel
    - Pressure calculation from altitude

    Args:
        world_pos: Position array [qx, qy, qz, qw, x, y, z] (Elodin scalar-last format)
        world_vel: Velocity array [wx, wy, wz, vx, vy, vz] (angular first in Elodin)
        accel: Accelerometer [ax, ay, az] in body frame (ENU)
        gyro: Gyroscope [wx, wy, wz] in body frame (ENU)
        timestamp: Simulation time in seconds
        gravity: Gravity constant (for reference)

    Returns:
        FDMPacket ready for transmission to Betaflight
    """
    # Import here to avoid circular dependency
    from comms import FDMPacket

    # Extract quaternion from Elodin format [qx, qy, qz, qw] and convert to [qw, qx, qy, qz]
    quat_xyzw = np.array(world_pos[:4])
    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]

    # Extract position [x, y, z] (ENU)
    position = np.array(world_pos[4:7])

    # Elodin stores world_vel as [wx, wy, wz, vx, vy, vz]
    angular_vel = np.array(world_vel[:3])
    linear_vel = np.array(world_vel[3:6])

    # Use provided sensor readings or compute from velocity
    accel_enu = np.array(accel) if accel is not None else np.array([0.0, 0.0, gravity])
    gyro_enu = np.array(gyro) if gyro is not None else angular_vel

    # Convert from Elodin ENU body frame to Betaflight NED body frame
    # ENU to NED: x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu
    accel_ned = np.array(
        [
            accel_enu[1],  # ENU-Y -> NED-X (forward)
            accel_enu[0],  # ENU-X -> NED-Y (right)
            -accel_enu[2],  # ENU-Z -> NED-Z (down, inverted)
        ]
    )

    gyro_ned = np.array(
        [
            gyro_enu[1],  # pitch rate
            gyro_enu[0],  # roll rate
            -gyro_enu[2],  # yaw rate (inverted)
        ]
    )

    # Calculate pressure from altitude (simplified atmosphere model)
    altitude = position[2]
    pressure = 101325.0 - 12.0 * altitude

    return FDMPacket(
        timestamp=timestamp,
        imu_angular_velocity_rpy=gyro_ned,
        imu_linear_acceleration_xyz=accel_ned,
        imu_orientation_quat=quat,
        velocity_xyz=linear_vel,  # ENU, which Betaflight expects for GPS
        position_xyz=position,  # ENU
        pressure=pressure,
    )


def extract_from_history(df, tick: int = -1) -> dict:
    """
    Extract sensor data from an Elodin history DataFrame.

    This is useful for post-processing or when accessing data via exec.history().

    Args:
        df: Polars DataFrame from exec.history()
        tick: Which tick to extract (-1 for latest)

    Returns:
        Dictionary with component arrays
    """

    row = df[tick]

    result = {}

    # Try to extract common components
    for col in [
        "drone.world_pos",
        "drone.world_vel",
        "drone.accel",
        "drone.gyro",
        "drone.baro",
        "drone.motor_command",
    ]:
        if col in df.columns:
            result[col.split(".")[-1]] = np.array(row[col].to_list())

    return result


class SensorDataBuffer:
    """
    Buffer for accumulating sensor data during simulation.

    This class provides a convenient way to store and retrieve sensor data
    for use in the post_step callback. It's updated during simulation and
    read by the Betaflight bridge.
    """

    def __init__(self):
        # Elodin format: [qx, qy, qz, qw, x, y, z] - identity quaternion is [0,0,0,1]
        self.world_pos = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.world_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.accel = np.array([0.0, 0.0, 9.80665])  # 1g upward at rest
        self.gyro = np.array([0.0, 0.0, 0.0])
        self.baro = np.array([0.0])
        self.timestamp = 0.0

    def update(
        self,
        world_pos: np.ndarray = None,
        world_vel: np.ndarray = None,
        accel: np.ndarray = None,
        gyro: np.ndarray = None,
        baro: np.ndarray = None,
        timestamp: float = None,
    ):
        """Update sensor buffer with new values."""
        if world_pos is not None:
            self.world_pos = np.array(world_pos)
        if world_vel is not None:
            self.world_vel = np.array(world_vel)
        if accel is not None:
            self.accel = np.array(accel)
        if gyro is not None:
            self.gyro = np.array(gyro)
        if baro is not None:
            self.baro = np.array(baro)
        if timestamp is not None:
            self.timestamp = timestamp

    def build_fdm(self) -> FDMPacket:
        """Build FDM packet from current buffer state."""
        return build_fdm_from_components(
            self.world_pos,
            self.world_vel,
            self.accel,
            self.gyro,
            self.timestamp,
        )


if __name__ == "__main__":
    # Test sensor computation
    from config import DEFAULT_CONFIG
    from sim import create_physics_system

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

    print("\nAccelerometer (body frame):")
    print(f"  First reading: {first_accel}")
    print(f"  Last reading:  {last_accel}")
    print(f"  Expected at rest: [0, 0, +{config.gravity:.2f}] m/s^2")

    # Check gyro (should be ~0 for stable drone)
    last_gyro = df[-1]["drone.gyro"].to_list()
    print("\nGyroscope (body frame):")
    print(f"  Last reading: {last_gyro}")
    print("  Expected at rest: [0, 0, 0] rad/s")

    # Check barometer
    last_baro = df[-1]["drone.baro"].to_list()
    print("\nBarometer:")
    print(f"  Last reading: {last_baro} m")

    print("\nSensor test complete!")
