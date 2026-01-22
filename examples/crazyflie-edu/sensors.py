"""
Crazyflie Sensor Simulation

Simulates the BMI088 IMU (accelerometer and gyroscope) with realistic noise models.
The Crazyflie 2.1 uses a BMI088 6-axis IMU.
"""

import sys
import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp
import jax.random as rng

from config import CrazyflieConfig as Config

# =============================================================================
# HITL Mode Detection
# =============================================================================

# In HITL mode, sensor components need external_control=true so that
# post_step callbacks can write real sensor data from the Crazyflie hardware.
# In SITL mode, the sensor simulation systems write to these components.
HITL_MODE = "--hitl" in sys.argv

# Build metadata dict conditionally
_gyro_metadata: dict[str, str] = {"priority": "90", "element_names": "x,y,z"}
_accel_metadata: dict[str, str] = {"priority": "89", "element_names": "x,y,z"}

if HITL_MODE:
    _gyro_metadata["external_control"] = "true"
    _accel_metadata["external_control"] = "true"

# =============================================================================
# Component Definitions
# =============================================================================

SensorTick = ty.Annotated[
    jax.Array,
    el.Component("sensor_tick", el.ComponentType.U64),
]

Gyro = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata=_gyro_metadata,
    ),
]

GyroBias = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

Accel = ty.Annotated[
    jax.Array,
    el.Component(
        "accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata=_accel_metadata,
    ),
]

AccelBias = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

# Noise enable flag (for Lab 1 comparison, 1.0 = enabled, 0.0 = disabled)
NoiseEnabled = ty.Annotated[
    jax.Array,
    el.Component("noise_enabled", el.ComponentType.F64),
]


# =============================================================================
# Noise Model
# =============================================================================


class SensorNoise:
    """
    Noise model for IMU sensors.

    Models:
    - Gaussian white noise (measurement noise)
    - Bias drift (random walk)
    """

    def __init__(
        self,
        seed: int,
        device: int,
        noise_std: float,
        bias_drift_std: float,
    ):
        """
        Initialize noise model.

        Args:
            seed: Random seed
            device: Device ID for unique random streams
            noise_std: Standard deviation of measurement noise
            bias_drift_std: Standard deviation of bias drift per second
        """
        self.noise_std = noise_std
        self.bias_drift_std = bias_drift_std
        self.key = rng.fold_in(rng.key(seed), device)

    def drift_bias(self, bias: jax.Array, tick: SensorTick, dt: float) -> jax.Array:
        """Update bias with random walk drift."""
        key = rng.fold_in(self.key, tick)
        drift = (
            self.bias_drift_std * rng.normal(key, shape=bias.shape, dtype=bias.dtype) * jnp.sqrt(dt)
        )
        return bias + drift

    def sample(
        self, measurement: jax.Array, bias: jax.Array, tick: SensorTick, noise_enabled: NoiseEnabled
    ) -> jax.Array:
        """Add noise and bias to measurement."""
        key = rng.fold_in(self.key, tick + 1000000)  # Different key than drift
        noise = self.noise_std * rng.normal(key, shape=measurement.shape, dtype=measurement.dtype)

        # Only add noise if enabled
        noisy = jnp.where(noise_enabled, measurement + noise + bias, measurement)
        return noisy


# Noise model instances
# Based on BMI088 datasheet characteristics
gyro_noise = SensorNoise(
    seed=42,
    device=0,
    noise_std=0.0017,  # ~0.1 deg/s RMS
    bias_drift_std=0.0001,  # Small drift
)

accel_noise = SensorNoise(
    seed=42,
    device=1,
    noise_std=0.01,  # ~10 mg RMS (in g units)
    bias_drift_std=0.0,  # No drift for accelerometer
)

# Initial biases (small random offsets)
init_gyro_bias = jnp.array([0.001, -0.0005, 0.0008])  # rad/s
init_accel_bias = jnp.array([0.005, -0.003, 0.002])  # g


# =============================================================================
# Archetypes
# =============================================================================


@dataclass
class IMU(el.Archetype):
    """IMU sensor archetype with gyro and accelerometer."""

    sensor_tick: SensorTick = field(default_factory=lambda: jnp.array(0, dtype=jnp.uint64))
    gyro: Gyro = field(default_factory=lambda: jnp.zeros(3))
    gyro_bias: GyroBias = field(default_factory=lambda: init_gyro_bias)
    accel: Accel = field(default_factory=lambda: jnp.array([0.0, 0.0, 1.0]))  # 1g up at rest
    accel_bias: AccelBias = field(default_factory=lambda: init_accel_bias)
    noise_enabled: NoiseEnabled = field(default_factory=lambda: jnp.array(1.0))


# =============================================================================
# Sensor Systems
# =============================================================================


@el.map
def advance_sensor_tick(tick: SensorTick) -> SensorTick:
    """Advance the sensor tick counter."""
    return tick + 1


@el.map
def update_gyro_bias(tick: SensorTick, bias: GyroBias, noise_enabled: NoiseEnabled) -> GyroBias:
    """Update gyro bias with random walk drift."""
    config = Config.get_global()
    dt = config.fast_loop_time_step

    # Only drift if noise is enabled
    new_bias = gyro_noise.drift_bias(bias, tick, dt)
    return jnp.where(noise_enabled, new_bias, bias)


@el.map
def gyro_sensor(
    tick: SensorTick,
    pos: el.WorldPos,
    vel: el.WorldVel,
    bias: GyroBias,
    noise_enabled: NoiseEnabled,
) -> Gyro:
    """
    Simulate gyroscope measurement.

    The gyroscope measures angular velocity in the body frame.
    Output is in rad/s.
    """
    # Get angular velocity in body frame
    # WorldVel.angular() is in world frame, need to rotate to body
    body_angular_vel = pos.angular().inverse() @ vel.angular()

    # Add noise if enabled
    if Config.get_global().sensor_noise:
        body_angular_vel = gyro_noise.sample(body_angular_vel, bias, tick, noise_enabled)

    return body_angular_vel


@el.map
def accel_sensor(
    tick: SensorTick,
    pos: el.WorldPos,
    accel: el.WorldAccel,
    bias: AccelBias,
    noise_enabled: NoiseEnabled,
) -> Accel:
    """
    Simulate accelerometer measurement.

    The accelerometer measures specific force (acceleration minus gravity)
    in the body frame. Output is in g (Earth gravity units).

    At rest on the ground, the accelerometer reads [0, 0, 1] g
    (measuring the normal force that opposes gravity).
    """
    # Gravity vector in world frame (pointing down)
    g = 9.81
    gravity_world = jnp.array([0.0, 0.0, -g])

    # Specific force in world frame = acceleration - gravity
    specific_force_world = accel.linear() - gravity_world

    # Rotate to body frame
    body_specific_force = pos.angular().inverse() @ specific_force_world

    # Convert to g units
    body_accel_g = body_specific_force / g

    # Add noise if enabled
    if Config.get_global().sensor_noise:
        body_accel_g = accel_noise.sample(body_accel_g, bias, tick, noise_enabled)

    return body_accel_g


# =============================================================================
# System Composition
# =============================================================================


def create_imu_system() -> el.System:
    """Create the complete IMU sensor system."""
    return advance_sensor_tick | update_gyro_bias | gyro_sensor | accel_sensor


# Convenience alias
imu = create_imu_system()
