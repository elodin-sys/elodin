"""Truth-based gate projection for tests and offline fixtures.

Production guidance must not import this module. It needs the simulator's gate
pose and the drone's world pose, which vision mode does not receive.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from course import Gate
from vision import (
    FPV_INTRINSICS,
    FPV_MOUNT,
    CameraIntrinsics,
    CameraMount,
    GateCandidate,
    candidate_from_corners,
    gate_object_corners,
)


def _world_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Body-to-world rotation for a scalar-last quaternion."""

    x, y, z, w = (float(value) for value in quaternion_xyzw)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_gate_corners(
    gate: Gate,
    drone_world_pos: Sequence[float] | np.ndarray,
    intrinsics: CameraIntrinsics = FPV_INTRINSICS,
    mount: CameraMount = FPV_MOUNT,
) -> np.ndarray | None:
    """Project the inner corners. ``None`` if any corner is not fully visible.

    ``drone_world_pos`` is ``[qx, qy, qz, qw, x, y, z]`` in ENU, matching
    ``world_pos``. Partial visibility is not represented.
    """

    pose = np.asarray(drone_world_pos, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("drone world pose must be seven finite values")
    rotation_world_body = _world_rotation(pose[:4])
    origin = pose[4:]
    pixels = []
    for corner in gate_object_corners(gate.inner_size):
        world = np.asarray(gate.local_to_world(corner), dtype=np.float64)
        body = rotation_world_body.T @ (world - origin)
        camera = mount.body_to_camera(body, point=True)
        if camera[2] <= 1e-8:
            return None
        u = intrinsics.fx * camera[0] / camera[2] + intrinsics.cx
        v = intrinsics.fy * camera[1] / camera[2] + intrinsics.cy
        if u < 0.0 or v < 0.0 or u > intrinsics.width or v > intrinsics.height:
            return None
        pixels.append((u, v))
    return np.asarray(pixels, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SyntheticNoise:
    """Deterministic corner noise. ``seed`` fixes the whole sequence."""

    pixel_sigma: float = 0.0
    dropout_probability: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.pixel_sigma < 0.0 or not np.isfinite(self.pixel_sigma):
            raise ValueError("pixel sigma must be finite and non-negative")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout probability must be in [0, 1]")


def synthetic_sequence(
    gate: Gate,
    poses: Sequence[Sequence[float]],
    sample_times: Sequence[float],
    noise: SyntheticNoise,
    *,
    intrinsics: CameraIntrinsics = FPV_INTRINSICS,
    mount: CameraMount = FPV_MOUNT,
    confidence: float = 1.0,
) -> list[GateCandidate | None]:
    """Project one candidate per pose. Dropped or invisible samples are ``None``."""

    if len(poses) != len(sample_times):
        raise ValueError("poses and sample times must have the same length")
    rng = np.random.default_rng(noise.seed)
    samples: list[GateCandidate | None] = []
    for pose, sample_time in zip(poses, sample_times, strict=True):
        if float(rng.random()) < noise.dropout_probability:
            samples.append(None)
            continue
        corners = project_gate_corners(gate, pose, intrinsics, mount)
        if corners is None:
            samples.append(None)
            continue
        if noise.pixel_sigma > 0.0:
            corners = corners + rng.normal(0.0, noise.pixel_sigma, corners.shape)
        samples.append(
            candidate_from_corners(
                float(sample_time),
                corners,
                confidence,
                gate.inner_size,
                intrinsics,
                mount,
            )
        )
    return samples


def _pose(yaw: float, position: tuple[float, float, float]) -> list[float]:
    half = yaw * 0.5
    return [0.0, 0.0, float(np.sin(half)), float(np.cos(half)), *position]


def fixture_records(gate: Gate) -> list[dict[str, object]]:
    """Canonical noise-free poses used by the offline fixture command."""

    centered_z = gate.center[2] - float(FPV_MOUNT.position_body[2])
    poses = {
        "centered": _pose(0.0, (0.0, 0.0, centered_z)),
        "translated_left": _pose(0.0, (0.0, 2.0, centered_z)),
        "translated_down": _pose(0.0, (0.0, 0.0, centered_z - 0.5)),
        "yawed": _pose(np.deg2rad(20.0), (0.0, 0.0, centered_z)),
        "range_5m": _pose(0.0, (gate.center[0] - 5.0 - 0.08, 0.0, centered_z)),
    }
    records = []
    for name, pose in poses.items():
        corners = project_gate_corners(gate, pose)
        if corners is None:
            raise RuntimeError(f"canonical fixture {name} is not fully visible")
        candidate = candidate_from_corners(
            0.0, corners, 1.0, gate.inner_size, FPV_INTRINSICS, FPV_MOUNT
        )
        if candidate.pose is None:
            raise RuntimeError(f"canonical fixture {name} has no pose")
        records.append(
            {
                "name": name,
                "drone_world_pos": pose,
                "corners_px": corners.tolist(),
                "bearing_body": candidate.pose.bearing_body.tolist(),
                "range_m": candidate.pose.range_m,
            }
        )
    return records


if __name__ == "__main__":
    from course import course_from_name

    print(json.dumps(fixture_records(course_from_name("single").gates[0]), indent=2))
