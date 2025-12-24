"""
Sensor Simulation for Betaflight SITL

This module simulates the IMU (Inertial Measurement Unit) and other sensors
that provide data to Betaflight's flight controller algorithms.

Sensor Models:
- Accelerometer: Measures linear acceleration in body frame
- Gyroscope: Measures angular velocity in body frame
- (Optional) Barometer: Measures altitude via pressure
- (Optional) GPS: Provides position and velocity

Noise Model (from proven drone example):
- Gaussian measurement noise on each sample
- Bias drift (random walk) for gyroscope

Note: No filtering is applied here - Betaflight handles its own gyro/accel
filtering. This is a Software-In-The-Loop test, so we want to send realistic
noisy sensor data and let Betaflight process it.

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
import jax.random as rng
import numpy as np

from config import DroneConfig
from comms import FDMPacket

# --- Noise Model ---


class Noise:
    """
    Sensor noise model with Gaussian noise and bias drift.

    Uses JAX random keys seeded deterministically via tick counter
    for reproducible noise across simulation runs.
    """

    def __init__(
        self,
        seed: int,
        device: int,
        noise_covariance: float,
        bias_drift_covariance: float,
    ):
        """
        Initialize noise model.

        Args:
            seed: Random seed for reproducibility
            device: Device index (for different noise streams per sensor)
            noise_covariance: Variance of measurement noise
            bias_drift_covariance: Variance of bias drift per timestep
        """
        self.noise_covariance = noise_covariance
        self.bias_drift_covariance = bias_drift_covariance
        self.key = rng.fold_in(rng.key(seed), device)

    def drift_bias(self, bias: jax.Array, tick: jax.Array, dt: float) -> jax.Array:
        """Apply random walk to bias (bias drift over time)."""
        # Fold in tick, then fold in 0 to differentiate from sample() stream
        key = rng.fold_in(rng.fold_in(self.key, tick), 0)
        std_dev = jnp.sqrt(self.bias_drift_covariance)
        drift = std_dev * rng.normal(key, shape=bias.shape, dtype=bias.dtype) * dt
        return bias + drift

    def sample(self, m: jax.Array, bias: jax.Array, tick: jax.Array) -> jax.Array:
        """Add measurement noise and bias to a measurement."""
        # Fold in tick, then fold in 1 to differentiate from drift_bias() stream
        key = rng.fold_in(rng.fold_in(self.key, tick), 1)
        std_dev = jnp.sqrt(self.noise_covariance)
        noise = std_dev * rng.normal(key, shape=m.shape, dtype=m.dtype)
        return m + noise + bias


# Noise instances - tuned for Betaflight SITL testing
#
# Note: Betaflight's attitude estimator is sensitive to noise during the
# bootgrace/calibration period. High noise causes attitude drift and
# motor imbalance at liftoff.
#
# Noise sweep results:
#   - 1e-8: Perfectly stable, motors balanced
#   - 1e-7: Slightly shaky but stable hover (recommended for SITL)
#   - 1e-6: Unstable, drone flips
#
# Production sensors (drone example) use noise_cov=0.001 (~1.8 deg/s std).
# SITL requires lower noise (1e-7) for stable lockstep simulation.
gyro_noise = Noise(0, 0, 1e-7, 1e-7)  # Gyro noise + bias drift
accel_noise = Noise(0, 1, 1e-7, 0.0)  # Accel noise (no drift)
baro_noise = Noise(0, 2, 0.001, 0.0)  # ~0.03m std dev


# Initial gyro bias (set to zero for SITL - avoids consistent drift direction)
# In a real sensor, this would be calibrated out during Betaflight's startup
init_gyro_bias = jnp.array([0.0, 0.0, 0.0])


# --- Sensor Component Types ---

# Sensor tick counter for deterministic RNG
SensorTick = ty.Annotated[jax.Array, el.Component("sensor_tick", el.ComponentType.U64)]

# IMU accelerometer reading in body frame [ax, ay, az] m/s^2
Accel = ty.Annotated[
    jax.Array,
    el.Component(
        "accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 150},
    ),
]

# Accelerometer bias [bx, by, bz] m/s^2
AccelBias = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
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

# Gyroscope bias [bx, by, bz] rad/s
GyroBias = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
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
    IMU sensor archetype with noise state.

    Stores computed sensor values and bias state for the drone entity.
    No filtering state - Betaflight handles its own filtering.
    """

    sensor_tick: SensorTick = field(default_factory=lambda: jnp.array(0, dtype=jnp.uint64))
    gyro: Gyro = field(default_factory=lambda: jnp.zeros(3))
    gyro_bias: GyroBias = field(default_factory=lambda: jnp.array(init_gyro_bias))
    accel: Accel = field(default_factory=lambda: jnp.zeros(3))
    accel_bias: AccelBias = field(default_factory=lambda: jnp.zeros(3))
    baro: Baro = field(default_factory=lambda: jnp.zeros(1))
    body_vel: BodyVel = field(default_factory=lambda: jnp.zeros(3))


# --- Sensor Computation Systems ---


@el.map
def advance_sensor_tick(tick: SensorTick) -> SensorTick:
    """Advance the sensor tick counter for deterministic RNG."""
    return tick + 1


def create_gyro_bias_drift_system(config: DroneConfig):
    """Create system to drift the gyro bias over time."""
    dt = config.sim_time_step

    @el.map
    def update_gyro_bias(tick: SensorTick, bias: GyroBias) -> GyroBias:
        """Apply random walk to gyro bias."""
        if config.sensor_noise:
            return gyro_noise.drift_bias(bias, tick, dt)
        return bias

    return update_gyro_bias


def create_gyro_system(config: DroneConfig):
    """
    Create the gyroscope sensor computation system.

    The gyroscope measures angular velocity in body frame.
    Applies noise and bias when enabled. No filtering - Betaflight handles that.
    """

    @el.map
    def compute_gyro(
        tick: SensorTick,
        pos: el.WorldPos,
        vel: el.WorldVel,
        bias: GyroBias,
    ) -> Gyro:
        """Compute gyroscope reading from physics state."""
        # Angular velocity is already in body frame in Elodin
        body_v = pos.angular().inverse() @ vel.angular()

        # Add noise and bias if enabled
        if config.sensor_noise:
            body_v = gyro_noise.sample(body_v, bias, tick)

        return body_v

    return compute_gyro


def create_accel_system(config: DroneConfig):
    """
    Create the accelerometer sensor computation system.

    The accelerometer measures specific force (acceleration minus gravity)
    in body frame. Applies noise and bias when enabled.
    No filtering - Betaflight handles that.

    Detects ground contact to correctly report +g when at rest (the ground
    constraint zeros velocity but doesn't add normal force to the Force component).
    """
    gravity = config.gravity
    ground_level = config.ground_level

    @el.map
    def compute_accel(
        tick: SensorTick,
        pos: el.WorldPos,
        vel: el.WorldVel,
        force: el.Force,
        inertia: el.Inertia,
        bias: AccelBias,
    ) -> Accel:
        """
        Compute accelerometer reading from physics state.

        The accelerometer measures specific force, which is the total force
        minus gravity, divided by mass. When in free fall, the accelerometer
        reads zero. When sitting on the ground, it reads +g upward (1g).

        Detects ground contact (z ≤ ground_level and not moving up) to correctly
        show +g when at rest, since the ground constraint zeros velocity but
        doesn't add a normal force.
        """
        # Get orientation quaternion for frame transformation
        quat = pos.angular()
        quat_inv = quat.inverse()

        # Check if on ground: position at ground level and not moving upward
        z = pos.linear()[2]
        vz = vel.linear()[2]
        on_ground = (z <= ground_level + 0.01) & (vz <= 0.01)

        # Compute acceleration from forces
        mass = inertia.mass()
        total_accel_from_force = force.linear() / mass

        # Gravity vector in world frame (ENU: +Z is up)
        gravity_world = jnp.array([0.0, 0.0, -gravity])

        # When on ground, effective acceleration is 0 (velocity is clamped)
        # Otherwise use force-based acceleration
        total_accel_world = jnp.where(
            on_ground,
            jnp.zeros(3),  # On ground: no acceleration (constrained)
            total_accel_from_force,  # In air: normal physics
        )

        # Specific force in world frame (what accelerometer measures)
        # specific_force = total_accel - gravity
        # When on ground: total_accel = 0, so specific_force = 0 - (-g) = +g ✓
        # When in free fall: total_accel = -g, so specific_force = -g - (-g) = 0 ✓
        specific_force_world = total_accel_world - gravity_world

        # Transform to body frame
        body_a = quat_inv @ specific_force_world

        # Add noise and bias if enabled
        if config.sensor_noise:
            body_a = accel_noise.sample(body_a, bias, tick)

        return body_a

    return compute_accel


def create_body_vel_system(config: DroneConfig):
    """Create system to compute body-frame velocity (for reference/debugging)."""

    @el.map
    def compute_body_vel(pos: el.WorldPos, vel: el.WorldVel) -> BodyVel:
        """Transform world velocity to body frame."""
        quat_inv = pos.angular().inverse()
        return quat_inv @ vel.linear()

    return compute_body_vel


def create_baro_system(config: DroneConfig):
    """
    Create barometer sensor system.

    Simulates barometric altitude measurement based on height.
    Applies noise when enabled (~0.3m std dev typical for consumer barometers).
    """

    @el.map
    def compute_baro(tick: SensorTick, pos: el.WorldPos) -> Baro:
        """Compute barometer reading from altitude."""
        # Simple model: altitude = z position
        altitude = pos.linear()[2]
        baro_reading = jnp.array([altitude])

        # Add noise if enabled
        if config.sensor_noise:
            # Use zero bias since barometers don't typically have significant drift
            baro_reading = baro_noise.sample(baro_reading, jnp.zeros(1), tick)

        return baro_reading

    return compute_baro


def create_imu_system(config: DroneConfig) -> el.System:
    """
    Create the complete IMU sensor system with noise model.

    Combines:
    - Sensor tick advancement
    - Gyro bias drift
    - Gyroscope computation with noise
    - Accelerometer computation with noise
    - Body velocity computation

    Note: No filtering is applied - Betaflight handles its own filtering.
    This sends realistic noisy sensor data for SITL testing.

    Args:
        config: Drone configuration

    Returns:
        Combined IMU system
    """
    return (
        advance_sensor_tick
        | create_gyro_bias_drift_system(config)
        | create_gyro_system(config)
        | create_accel_system(config)
        | create_body_vel_system(config)
    )


def create_sensor_system(config: DroneConfig) -> el.System:
    """
    Create the complete sensor system.

    Args:
        config: Drone configuration

    Returns:
        Combined sensor system
    """
    imu = create_imu_system(config)
    baro = create_baro_system(config)

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
    linear_vel = np.array(world_vel[3:6])

    # Use provided sensor readings or compute from velocity
    accel_enu = np.array(accel) if accel is not None else np.array([0.0, 0.0, gravity])
    gyro_enu = np.array(gyro) if gyro is not None else np.array(world_vel[:3])

    # Convert from Elodin FLU body frame to Betaflight FRD body frame
    #
    # Betaflight SITL (sitl.c) applies internal sign conversions to incoming data:
    #   accel: negates all axes (-X, -Y, -Z)
    #   gyro:  keeps X, negates Y and Z (X, -Y, -Z)
    #
    # We pre-compensate so that AFTER BF's conversion, correct FRD values result.
    #
    # FLU→FRD conversion (conceptually):
    #   FRD_x = FLU_x   (forward stays forward)
    #   FRD_y = -FLU_y  (right = -left)
    #   FRD_z = -FLU_z  (down = -up)
    #
    # Accelerometer: We want [FLU_x, -FLU_y, -FLU_z] after BF negates all.
    #   Send [-FLU_x, FLU_y, FLU_z] → BF gets [FLU_x, -FLU_y, -FLU_z] ✓
    accel_ned = np.array(
        [
            -accel_enu[0],  # BF: -(-X) = X
            accel_enu[1],  # BF: -Y
            accel_enu[2],  # BF: -Z
        ]
    )

    # Gyroscope: We want [FLU_x, -FLU_y, -FLU_z] after BF's (X, -Y, -Z).
    #   Note: Elodin pitch sign is inverted, so we negate Y before conversion.
    #   Send [FLU_x, -FLU_y, FLU_z] → BF gets [FLU_x, FLU_y, -FLU_z]
    #   But with pitch already negated: [FLU_x, FLU_y, -FLU_z] (correct FRD)
    gyro_ned = np.array(
        [
            gyro_enu[0],  # Roll: correct sign
            -gyro_enu[1],  # Pitch: negate (Elodin pitch is inverted)
            gyro_enu[2],  # Yaw: BF negates to get -Z
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

    print("Testing sensor simulation with noise model (no filtering)...")

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
    sensors = create_sensor_system(config)
    full_system = physics | sensors

    # Run simulation
    exec = world.build(full_system)
    exec.run(200)

    # Get sensor history
    df = exec.history(["drone.accel", "drone.gyro", "drone.baro", "drone.gyro_bias"])
    print(f"Simulated {len(df)} ticks")

    # Check accelerometer reading at start (should be ~+g since on ground)
    first_accel = df[0]["drone.accel"].to_list()
    last_accel = df[-1]["drone.accel"].to_list()

    print("\nAccelerometer (body frame):")
    print(f"  First reading: {first_accel}")
    print(f"  Last reading:  {last_accel}")
    print(f"  Expected at rest: [0, 0, +{config.gravity:.2f}] m/s^2")

    # Check gyro (should be ~0 for stable drone, with noise)
    last_gyro = df[-1]["drone.gyro"].to_list()
    print("\nGyroscope (body frame):")
    print(f"  Last reading: {last_gyro}")
    print("  Expected at rest: [~0, ~0, ~0] rad/s (with noise)")

    # Check gyro bias drift
    first_bias = df[0]["drone.gyro_bias"].to_list()
    last_bias = df[-1]["drone.gyro_bias"].to_list()
    print("\nGyro Bias:")
    print(f"  Initial: {first_bias}")
    print(f"  Final:   {last_bias}")

    # Check barometer
    last_baro = df[-1]["drone.baro"].to_list()
    print("\nBarometer:")
    print(f"  Last reading: {last_baro} m")

    print(f"\nSensor noise enabled: {config.sensor_noise}")
    print("No filtering applied - Betaflight handles its own filtering.")
    print("Sensor test complete!")
