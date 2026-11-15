"""Detector acceptance on recorded renderer frames.

The fixture is a small range-binned extract of a Package E truth run. Bearing
uses the body-origin direction to the gate at the stored nominal ``frame_time``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from gate_detector import GateDetector
from synthetic_vision import _world_rotation

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rendered_gate_frames.npz"


def test_recorded_frames_meet_detection_and_bearing() -> None:
    data = np.load(_FIXTURE)
    center = np.asarray(data["gate_center"], dtype=np.float64)
    inner_size = float(data["inner_size"])
    detector = GateDetector()
    errors = []
    detected = 0
    for frame, pose in zip(data["frames"], data["world_pos"], strict=True):
        found = detector.detect(frame, 0.0, inner_size)
        if not found or found[0].pose is None:
            continue
        detected += 1
        truth = _body_bearing(center, pose)
        errors.append(_angle_deg(found[0].pose.bearing_body, truth))

    assert data["frames"].shape[0] > 0
    assert detected / data["frames"].shape[0] >= 0.90
    assert float(np.median(errors)) <= 2.0


def _body_bearing(center: np.ndarray, pose: np.ndarray) -> np.ndarray:
    body = _world_rotation(pose[:4]).T @ (center - pose[4:7])
    return body / np.linalg.norm(body)


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(
        np.clip(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)), -1.0, 1.0)
    )
    return math.degrees(math.acos(cosine))
