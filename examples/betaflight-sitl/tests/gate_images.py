"""Test-only rasterizer for orange gate rings.

NumPy only: OpenCV stays inside ``gate_detector.py``. Images are RGBA in the
FPV camera's continuous pixel coordinates, with 4× supersampling. A pixel is
orange where its samples fall inside the projected outer square and outside
the inner opening. Coordinates may lie outside the image, so clipped gates can
be drawn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from course import GATE_BAR_THICKNESS_M, Gate
from synthetic_vision import _world_rotation
from vision import FPV_INTRINSICS, FPV_MOUNT, CameraIntrinsics, CameraMount

ORANGE_RGB = np.array([255.0, 128.0, 0.0], dtype=np.float64)
BACKGROUND_RGB = np.array([20.0, 24.0, 28.0], dtype=np.float64)
CORE_RANGES_M = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
CORE_BEARINGS_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)


def square_corners(half_extent: float) -> np.ndarray:
    """Gate-frame TL, TR, BR, BL corners of a square opening."""

    half = float(half_extent)
    return np.array(
        [
            [0.0, half, half],
            [0.0, -half, half],
            [0.0, -half, -half],
            [0.0, half, -half],
        ],
        dtype=np.float64,
    )


def project_points(
    gate: Gate,
    local_points: np.ndarray,
    drone_world_pos: Sequence[float] | np.ndarray,
    intrinsics: CameraIntrinsics = FPV_INTRINSICS,
    mount: CameraMount = FPV_MOUNT,
) -> np.ndarray | None:
    """Project gate-local points. ``None`` if any point is behind the camera.

    Unlike ``project_gate_corners``, points outside the image are kept.
    """

    pose = np.asarray(drone_world_pos, dtype=np.float64)
    rotation = _world_rotation(pose[:4])
    origin = pose[4:]
    pixels = []
    for local in local_points:
        world = np.asarray(gate.local_to_world(local), dtype=np.float64)
        body = rotation.T @ (world - origin)
        camera = mount.body_to_camera(body, point=True)
        if camera[2] <= 1e-8:
            return None
        pixels.append(
            (
                intrinsics.fx * camera[0] / camera[2] + intrinsics.cx,
                intrinsics.fy * camera[1] / camera[2] + intrinsics.cy,
            )
        )
    return np.asarray(pixels, dtype=np.float64)


def render_gates(
    gates: Sequence[Gate],
    drone_world_pos: Sequence[float] | np.ndarray,
    *,
    thickness: float = GATE_BAR_THICKNESS_M,
    intrinsics: CameraIntrinsics = FPV_INTRINSICS,
    mount: CameraMount = FPV_MOUNT,
    samples: int = 4,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Draw farther gates first. Returns the image and each inner-corner quad."""

    pose = np.asarray(drone_world_pos, dtype=np.float64)
    order = sorted(
        gates,
        key=lambda gate: float(
            np.linalg.norm(np.asarray(gate.center, dtype=np.float64) - pose[4:])
        ),
        reverse=True,
    )
    color = np.empty((intrinsics.height, intrinsics.width, 3), dtype=np.float64)
    color[...] = BACKGROUND_RGB
    inners: dict[int, np.ndarray] = {}
    for gate in order:
        outer = project_points(
            gate, square_corners(gate.inner_size * 0.5 + thickness), pose, intrinsics, mount
        )
        inner = project_points(gate, square_corners(gate.inner_size * 0.5), pose, intrinsics, mount)
        if outer is None or inner is None:
            continue
        inners[gate.index] = inner
        coverage = _coverage(outer, inner, intrinsics.width, intrinsics.height, samples)
        color = color * (1.0 - coverage[..., None]) + ORANGE_RGB * coverage[..., None]
    image = np.empty((intrinsics.height, intrinsics.width, 4), dtype=np.uint8)
    image[..., :3] = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    image[..., 3] = 255
    return image, [inners[gate.index] for gate in gates if gate.index in inners]


def level_pose(position: Sequence[float], yaw: float = 0.0) -> np.ndarray:
    """Scalar-last identity attitude, optional yaw, at an ENU position."""

    half = yaw * 0.5
    return np.array([0.0, 0.0, math.sin(half), math.cos(half), *position], dtype=np.float64)


def camera_pose_for_range(
    gate: Gate,
    range_m: float,
    *,
    bearing_deg: float = 0.0,
    mount: CameraMount = FPV_MOUNT,
) -> np.ndarray:
    """Level body pose whose camera looks at ``gate`` from ``range_m``.

    ``bearing_deg`` shifts the drone sideways. Positive bearing is gate-right,
    which is body +Y when the drone faces the gate along world +X and the gate
    yaw is zero.
    """

    forward = np.asarray(gate.normal, dtype=np.float64)
    lateral = np.asarray(gate.lateral, dtype=np.float64)
    mount_position = np.asarray(mount.position_body, dtype=np.float64)
    center = np.asarray(gate.center, dtype=np.float64)
    offset = range_m * math.tan(math.radians(bearing_deg))
    camera = center - forward * range_m - lateral * offset
    yaw = gate.yaw
    cosine, sine = math.cos(yaw), math.sin(yaw)
    mount_world = np.array(
        [
            cosine * mount_position[0] - sine * mount_position[1],
            sine * mount_position[0] + cosine * mount_position[1],
            mount_position[2],
        ]
    )
    return level_pose(camera - mount_world, yaw)


def _coverage(
    outer: np.ndarray, inner: np.ndarray, width: int, height: int, samples: int
) -> np.ndarray:
    offsets = (np.arange(samples) + 0.5) / samples
    y_coord = (np.arange(height)[:, None] + offsets[None, :])[:, :, None, None]
    x_coord = (np.arange(width)[:, None] + offsets[None, :])[None, None, :, :]
    points_y = np.broadcast_to(y_coord, (height, samples, width, samples)).reshape(-1)
    points_x = np.broadcast_to(x_coord, (height, samples, width, samples)).reshape(-1)
    points = np.column_stack((points_x, points_y))
    inside = _inside_convex(points, outer) & ~_inside_convex(points, inner)
    return inside.reshape(height, samples, width, samples).mean(axis=(1, 3))


def _inside_convex(points: np.ndarray, quad: np.ndarray) -> np.ndarray:
    centroid = quad.mean(axis=0)
    edge = quad[1] - quad[0]
    winding = (edge[0] * (centroid[1] - quad[0, 1])) - (edge[1] * (centroid[0] - quad[0, 0]))
    sign = 1.0 if winding >= 0.0 else -1.0
    mask = np.ones(points.shape[0], dtype=bool)
    for index in range(4):
        start = quad[index]
        stop = quad[(index + 1) % 4]
        span = stop - start
        cross = span[0] * (points[:, 1] - start[1]) - span[1] * (points[:, 0] - start[0])
        mask &= cross * sign >= -1e-6
    return mask
