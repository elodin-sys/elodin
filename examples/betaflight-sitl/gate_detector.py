"""Classical orange-gate detector. OpenCV types do not leave this module.

The detector reads one RGBA frame and returns Section 7.7 candidates. It does
not track, fly, or read simulator truth. Call it only on fresh 30 Hz frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from controls import GuidanceUpdate
from vision import (
    FPV_INTRINSICS,
    FPV_MOUNT,
    CameraIntrinsics,
    CameraMount,
    GateCandidate,
    candidate_from_corners,
)

# (outer² − inner²) / inner² for 0.2 m bars on a 2.5 m opening.
_FRONTAL_RING_RATIO = (2.9**2 - 2.5**2) / 2.5**2


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Thresholds for the saturated-orange gate ring.

    Hue is OpenCV's 0–180 scale. ``(255, 128, 0)`` is hue 15. Edge points are
    moved from the contour pixel to the orange-to-hole crossing
    (``pixel_center_offset`` is the half-pixel center of that pixel).
    ``corner_inset_px`` is an extra shift toward the hole after that crossing.
    """

    hue_min: int = 5
    hue_max: int = 25
    sat_min: int = 120
    val_min: int = 50
    close_kernel: int = 3
    min_hole_area_px: float = 200.0
    border_margin_px: float = 2.0
    ring_ratio_min: float = 0.15
    ring_ratio_max: float = 0.8
    frontal_ring_ratio: float = _FRONTAL_RING_RATIO
    max_reproj_px: float = 1.5
    min_range_m: float = 1.0
    max_range_m: float = 40.0
    approx_epsilon_ratio: float = 0.02
    pixel_center_offset: float = 0.5
    corner_inset_px: float = 0.0

    def __post_init__(self) -> None:
        if self.close_kernel < 1 or self.close_kernel % 2 == 0:
            raise ValueError("close kernel must be a positive odd size")
        if self.min_hole_area_px <= 0.0 or self.border_margin_px < 0.0:
            raise ValueError("area and border margin must be non-negative, area positive")
        if not 0.0 < self.ring_ratio_min < self.ring_ratio_max:
            raise ValueError("ring-ratio window must be positive and ordered")
        if self.frontal_ring_ratio <= 0.0:
            raise ValueError("frontal ring ratio must be positive")
        if self.max_reproj_px <= 0.0 or self.min_range_m <= 0.0:
            raise ValueError("reprojection and minimum range must be positive")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("maximum range must exceed minimum range")
        if self.approx_epsilon_ratio <= 0.0:
            raise ValueError("polygon epsilon ratio must be positive")


class GateDetector:
    """Find gate openings in one RGBA frame, largest apparent hole first."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        intrinsics: CameraIntrinsics = FPV_INTRINSICS,
        mount: CameraMount = FPV_MOUNT,
    ) -> None:
        self.config = DetectorConfig() if config is None else config
        self.intrinsics = intrinsics
        self.mount = mount

    def detect(
        self,
        frame_rgba: np.ndarray,
        frame_time: float,
        inner_size: float,
    ) -> tuple[GateCandidate, ...]:
        """Return accepted candidates. The input frame is not modified."""

        frame = np.asarray(frame_rgba)
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 4
            or not math.isfinite(frame_time)
            or not math.isfinite(inner_size)
            or inner_size <= 0.0
        ):
            return ()
        height, width = frame.shape[:2]
        mask = _orange_mask(frame, self.config)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None or len(contours) == 0:
            return ()
        hierarchy = hierarchy[0]
        accepted: list[tuple[float, GateCandidate]] = []
        for index, contour in enumerate(contours):
            parent = int(hierarchy[index][3])
            if parent < 0:
                continue
            candidate = self._candidate_from_hole(
                frame,
                contour,
                contours[parent],
                width,
                height,
                float(frame_time),
                float(inner_size),
            )
            if candidate is not None:
                accepted.append(candidate)
        accepted.sort(key=lambda item: item[0], reverse=True)
        return tuple(item[1] for item in accepted)

    def _candidate_from_hole(
        self,
        frame: np.ndarray,
        hole: np.ndarray,
        parent: np.ndarray,
        width: int,
        height: int,
        frame_time: float,
        inner_size: float,
    ) -> tuple[float, GateCandidate] | None:
        config = self.config
        area = float(cv2.contourArea(hole))
        if area < config.min_hole_area_px:
            return None
        epsilon = config.approx_epsilon_ratio * float(cv2.arcLength(hole, True))
        approx = cv2.approxPolyDP(hole, epsilon, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            return None
        vertices = approx.reshape(4, 2).astype(np.float64)
        if _touches_border(vertices, width, height, config.border_margin_px):
            return None
        ring_ratio = _ring_ratio(hole, parent)
        parent_clipped = _touches_border(
            parent.reshape(-1, 2).astype(np.float64),
            width,
            height,
            config.border_margin_px,
        )
        if ring_ratio is None:
            return None
        if not parent_clipped and not (
            config.ring_ratio_min <= ring_ratio <= config.ring_ratio_max
        ):
            return None
        corners = _refined_corners(frame, hole, vertices, config)
        if corners is None or _touches_border(corners, width, height, config.border_margin_px):
            return None
        preliminary = candidate_from_corners(
            frame_time,
            corners,
            1.0,
            inner_size,
            self.intrinsics,
            self.mount,
        )
        pose = preliminary.pose
        rms = preliminary.reprojection_error_px
        if pose is None or rms is None or rms > config.max_reproj_px:
            return None
        if not config.min_range_m <= pose.range_m <= config.max_range_m:
            return None
        confidence = _confidence(
            preliminary.corners_px,
            None if parent_clipped else ring_ratio,
            rms,
            config.frontal_ring_ratio,
            config.max_reproj_px,
        )
        candidate = GateCandidate(
            preliminary.sample_time,
            preliminary.corners_px,
            confidence,
            pose,
            rms,
            preliminary.corner_order_ambiguous,
        )
        return area, candidate


def detect_fresh(detector: GateDetector, update: GuidanceUpdate) -> tuple[GateCandidate, ...]:
    """Run the detector only when this tick carries a new frame."""

    if not update.frame_fresh or update.frame is None or update.frame_time is None:
        return ()
    if update.gate_inner_size is None:
        return ()
    return detector.detect(update.frame, float(update.frame_time), float(update.gate_inner_size))


def _orange_mask(frame: np.ndarray, config: DetectorConfig) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([config.hue_min, config.sat_min, config.val_min], dtype=np.uint8),
        np.array([config.hue_max, 255, 255], dtype=np.uint8),
    )
    kernel = np.ones((config.close_kernel, config.close_kernel), dtype=np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # findContours is allowed to rewrite the image it traces.
    return np.ascontiguousarray(closed)


def _touches_border(points: np.ndarray, width: int, height: int, margin: float) -> bool:
    xs = points[:, 0]
    ys = points[:, 1]
    return bool(
        np.any(xs <= margin)
        or np.any(ys <= margin)
        or np.any(xs >= (width - 1) - margin)
        or np.any(ys >= (height - 1) - margin)
    )


def _ring_ratio(hole: np.ndarray, parent: np.ndarray) -> float | None:
    hole_area = float(cv2.contourArea(hole))
    parent_area = float(cv2.contourArea(parent))
    if hole_area <= 0.0 or parent_area <= hole_area:
        return None
    return (parent_area - hole_area) / hole_area


def _refined_corners(
    frame: np.ndarray, hole: np.ndarray, vertices: np.ndarray, config: DetectorConfig
) -> np.ndarray | None:
    contour = hole.reshape(-1, 2).astype(np.float64)
    indices = [_nearest_index(contour, vertex) for vertex in vertices]
    if any(index is None for index in indices):
        return None
    centroid = vertices.mean(axis=0) + config.pixel_center_offset
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for edge in range(4):
        points = _edge_points(contour, indices[edge], indices[(edge + 1) % 4])
        if points.shape[0] < 2:
            return None
        shifted = _crossing_points(frame, points, centroid, config.pixel_center_offset)
        direction, origin = _fit_line(shifted)
        if direction is None or origin is None:
            return None
        origin = _shift_toward(direction, origin, centroid, config.corner_inset_px)
        lines.append((direction, origin))
    corners = []
    for edge in range(4):
        direction, origin = lines[edge]
        next_direction, next_origin = lines[(edge + 1) % 4]
        point = _intersect(direction, origin, next_direction, next_origin)
        if point is None or not np.all(np.isfinite(point)):
            return None
        corners.append(point)
    return np.asarray(corners, dtype=np.float64)


def _nearest_index(contour: np.ndarray, point: np.ndarray) -> int | None:
    distance = np.max(np.abs(contour - point), axis=1)
    matches = np.flatnonzero(distance <= 0.5 + 1e-6)
    if matches.size == 0:
        nearest = int(np.argmin(distance))
        if distance[nearest] > 1.5:
            return None
        return nearest
    return int(matches[0])


def _edge_points(contour: np.ndarray, start: int, end: int) -> np.ndarray:
    if end >= start:
        return contour[start : end + 1]
    return np.vstack((contour[start:], contour[: end + 1]))


def _crossing_points(
    frame: np.ndarray, points: np.ndarray, centroid: np.ndarray, center_offset: float
) -> np.ndarray:
    """Move contour pixels to the orange-to-hole crossing along the edge normal."""

    inward = centroid - (points.mean(axis=0) + center_offset)
    norm = float(np.linalg.norm(inward))
    if norm < 1e-6:
        return points + center_offset
    inward = inward / norm
    step = np.array([int(np.round(inward[0])), int(np.round(inward[1]))], dtype=int)
    if step[0] == 0 and step[1] == 0:
        step = np.array([int(np.sign(inward[0])) or 0, int(np.sign(inward[1])) or 0])
    height, width = frame.shape[:2]
    shifted = np.empty_like(points, dtype=np.float64)
    for index, point in enumerate(points):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        coverage = _orange_fraction(frame, x, y, step, width, height)
        shifted[index] = point + center_offset + (coverage - 0.5) * step.astype(np.float64)
    return shifted


def _orange_fraction(
    frame: np.ndarray, x: int, y: int, step: np.ndarray, width: int, height: int
) -> float:
    """Fraction of this pixel that is orange, 0 at the outer side and 1 at the hole side."""

    outer = (x - int(step[0]), y - int(step[1]))
    inner = (x + int(step[0]), y + int(step[1]))
    if not _inside_image(x, y, width, height):
        return 0.5
    here = float(frame[y, x, 0])
    if not _inside_image(*outer, width, height) or not _inside_image(*inner, width, height):
        return 0.5
    outward = float(frame[outer[1], outer[0], 0])
    inward = float(frame[inner[1], inner[0], 0])
    span = outward - inward
    if abs(span) < 5.0:
        return 0.5
    return float(np.clip((here - inward) / span, 0.0, 1.0))


def _inside_image(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    fitted = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = np.asarray(fitted, dtype=np.float64).reshape(-1)
    direction = np.array([vx, vy], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None, None
    return direction / norm, np.array([x0, y0], dtype=np.float64)


def _shift_toward(
    direction: np.ndarray, origin: np.ndarray, centroid: np.ndarray, inset: float
) -> np.ndarray:
    if inset == 0.0:
        return origin
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    if float(np.dot(centroid - origin, normal)) < 0.0:
        normal = -normal
    return origin + inset * normal


def _intersect(
    direction_a: np.ndarray,
    origin_a: np.ndarray,
    direction_b: np.ndarray,
    origin_b: np.ndarray,
) -> np.ndarray | None:
    basis = np.column_stack((direction_a, -direction_b))
    if abs(float(np.linalg.det(basis))) < 1e-10:
        return None
    step = np.linalg.solve(basis, origin_b - origin_a)
    return origin_a + float(step[0]) * direction_a


def _edge_agreement(corners: np.ndarray) -> float:
    lengths = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index])) for index in range(4)
    ]

    def pair(left: float, right: float) -> float:
        return 1.0 - abs(left - right) / max(left + right, 1e-6)

    return 0.5 * (pair(lengths[0], lengths[2]) + pair(lengths[1], lengths[3]))


def _confidence(
    corners: np.ndarray,
    ring_ratio: float | None,
    rms: float,
    frontal_ring_ratio: float,
    reproj_scale: float,
) -> float:
    edges = min(1.0, max(0.0, _edge_agreement(corners)))
    if ring_ratio is None:
        ring = 1.0
    else:
        ring = math.exp(-abs(ring_ratio - frontal_ring_ratio) / frontal_ring_ratio)
    reprojection = math.exp(-rms / reproj_scale)
    return float(min(1.0, max(0.0, edges * ring * reprojection)))
