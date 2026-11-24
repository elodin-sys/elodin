"""Deterministic 6-DoF trajectories for navigator tests.

Exact gyro and specific force come from the scripted motion. Sensor noise uses
the same ``sensors.Noise`` streams as the simulator. Corners come from
``synthetic_vision`` and never from ``navigation.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from course import Gate
from sensors import accel_noise, baro_noise, gyro_noise, mag_noise
from synthetic_vision import SyntheticNoise, synthetic_sequence
from vision import FPV_INTRINSICS, FPV_MOUNT, GateCandidate

_G = np.array([0.0, 0.0, -9.81])
_NORTH = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True, slots=True)
class Trajectory:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    yaw: np.ndarray
    gyro: np.ndarray
    specific_force: np.ndarray
    barometer: np.ndarray
    magnetometer: np.ndarray
    candidates: list[GateCandidate | None]
    gate: Gate


def straight_run(
    *,
    duration_s: float = 6.0,
    dt_s: float = 0.002,
    speed_mps: float = 1.0,
    rest_s: float = 0.5,
    seed: int = 1,
    gate_center: tuple[float, float, float] = (10.0, 0.0, 1.5),
    gate_yaw: float = 0.0,
    pixel_sigma: float = 0.0,
    dropout: float = 0.0,
    frame_dt_s: float = 1.0 / 30.0,
    depth_bias_m: float = 0.0,
) -> Trajectory:
    """Rest on the pad, then fly east at constant speed toward one upright gate."""

    count = int(duration_s / dt_s)
    time = np.arange(count, dtype=np.float64) * dt_s
    position = np.zeros((count, 3), dtype=np.float64)
    velocity = np.zeros((count, 3), dtype=np.float64)
    yaw = np.zeros(count, dtype=np.float64)
    # Half-cosine ramp: finite acceleration, ends at the cruise speed and cruise height.
    ramp_s = 0.8
    cruise_z = 1.5
    for index, stamp in enumerate(time):
        position[index], velocity[index] = _rest_then_cruise(
            float(stamp), rest_s, ramp_s, speed_mps, 0.1, cruise_z
        )
    gyro_true = np.zeros((count, 3), dtype=np.float64)
    force_true = np.zeros((count, 3), dtype=np.float64)
    acceleration = np.gradient(velocity, dt_s, axis=0)
    gyro = np.zeros_like(gyro_true)
    force = np.zeros_like(force_true)
    mag = np.zeros_like(force_true)
    baro = np.zeros(count, dtype=np.float64)
    for index in range(count):
        rotation = _yaw_rotation(float(yaw[index]))
        force_true[index] = rotation.T @ (acceleration[index] - _G)
        gyro[index] = np.asarray(
            gyro_noise.sample(jnp.asarray(gyro_true[index]), jnp.zeros(3), jnp.asarray(index))
        )
        force[index] = np.asarray(
            accel_noise.sample(jnp.asarray(force_true[index]), jnp.zeros(3), jnp.asarray(index))
        )
        mag[index] = np.asarray(
            mag_noise.sample(jnp.asarray(rotation.T @ _NORTH), jnp.zeros(3), jnp.asarray(index))
        )
        baro[index] = float(
            baro_noise.sample(jnp.asarray([position[index, 2]]), jnp.zeros(1), jnp.asarray(index))[
                0
            ]
        )
    gate = Gate(0, gate_center, gate_yaw, 2.5)
    frame_index = np.arange(0, count, max(1, int(round(frame_dt_s / dt_s))))
    poses = [_pose(float(yaw[i]), tuple(position[i])) for i in frame_index]
    samples = synthetic_sequence(
        gate,
        poses,
        time[frame_index],
        SyntheticNoise(pixel_sigma=pixel_sigma, dropout_probability=dropout, seed=seed),
        intrinsics=FPV_INTRINSICS,
        mount=FPV_MOUNT,
    )
    if depth_bias_m != 0.0:
        samples = [_bias_range(sample, depth_bias_m) for sample in samples]
    candidates: list[GateCandidate | None] = [None] * count
    for index, sample in zip(frame_index, samples, strict=True):
        candidates[int(index)] = sample
    return Trajectory(time, position, velocity, yaw, gyro, force, baro, mag, candidates, gate)


def _rest_then_cruise(
    stamp: float, rest_s: float, ramp_s: float, speed: float, z0: float, z1: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if stamp <= rest_s or ramp_s <= 0.0:
        return (0.0, 0.0, z0), (0.0, 0.0, 0.0)
    tau = stamp - rest_s
    if tau >= ramp_s:
        distance = speed * ramp_s / 2.0 + speed * (tau - ramp_s)
        return (distance, 0.0, z1), (speed, 0.0, 0.0)
    phase = math.pi * tau / ramp_s
    forward = speed * (tau / 2.0 - ramp_s / (2.0 * math.pi) * math.sin(phase))
    forward_rate = speed * (1.0 - math.cos(phase)) / 2.0
    # Sine velocity bump: start and end the climb at zero vertical speed.
    climb = (z1 - z0) * (1.0 - math.cos(phase)) / 2.0
    climb_rate = (z1 - z0) * math.pi / (2.0 * ramp_s) * math.sin(phase)
    return (forward, 0.0, z0 + climb), (forward_rate, 0.0, climb_rate)


def _yaw_rotation(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _pose(yaw: float, position: tuple[float, float, float]) -> list[float]:
    half = yaw * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half), *position]


def _bias_range(sample: GateCandidate | None, bias_m: float) -> GateCandidate | None:
    if sample is None or sample.pose is None:
        return sample
    direction = np.asarray(sample.pose.position_body, dtype=np.float64)
    distance = float(np.linalg.norm(direction))
    if distance < 1e-6:
        return sample
    shifted = direction * ((distance + bias_m) / distance)
    pose = type(sample.pose)(
        sample.pose.rotation_cam_gate,
        sample.pose.translation_cam,
        shifted,
        sample.pose.normal_body,
        sample.pose.bearing_body,
        distance + bias_m,
    )
    return GateCandidate(
        sample.sample_time,
        sample.corners_px,
        sample.confidence,
        pose,
        sample.reprojection_error_px,
        sample.corner_order_ambiguous,
    )


def silence_jax() -> None:
    """Importing sensors builds a JAX mesh. Tests do not need its logs."""

    jax.config.update("jax_platforms", "cpu")
