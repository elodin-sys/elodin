"""Score the offline navigator on a Package H replay bundle.

Sensors and detector candidates drive ``navigation.py``. Truth is loaded only
here, for the printed errors. Missing bundles exit 0; this is not a CI gate.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from gate_detector import GateDetector
from navigation import GateNavigator, TrackCode
from replay import load_sensor_bundle, load_truth
from vision import CameraIntrinsics, CameraMount, FPV_INTRINSICS, FPV_MOUNT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the offline gate navigator")
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--on-pad-s", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.bundle is None or not (args.bundle / "sensors.npz").is_file():
        print("no replay bundle; navigator acceptance is the synthetic suite")
        return 0
    _report(args.bundle, args.on_pad_s)
    return 0


def _report(bundle: Path, on_pad_s: float) -> None:
    sensors = load_sensor_bundle(bundle)
    truth = load_truth(bundle)
    intrinsics, mount = _camera(sensors.meta)
    detector = GateDetector(intrinsics=intrinsics, mount=mount)
    inner = sensors.meta.get("inner_size_m")
    if inner is None:
        raise SystemExit("bundle has no inner_size_m")
    candidates = _detect(detector, sensors, float(inner))
    navigator = GateNavigator()
    origin: np.ndarray | None = None
    position_errors: list[float] = []
    bearing_errors: list[float] = []
    track_time = 0.0
    coast_time = 0.0
    lost = False
    baro_index = 0
    mag_index = 0
    frame_index = 0
    gate_center = np.asarray(truth.gate_center[0], dtype=np.float64)
    for index, stamp in enumerate(sensors.imu_time):
        baro_index = _advance(sensors.baro_time, baro_index, float(stamp))
        mag_index = _advance(sensors.mag_time, mag_index, float(stamp))
        batch = []
        while frame_index < len(sensors.frame_time) and sensors.frame_time[frame_index] <= stamp:
            batch.extend(candidates[frame_index])
            frame_index += 1
        estimate = navigator.step(
            float(stamp),
            sensors.gyro_mean[index],
            sensors.accel_mean[index],
            barometer=float(sensors.baro[baro_index]) if len(sensors.baro) else None,
            magnetometer=sensors.mag[mag_index] if len(sensors.mag) else None,
            candidates=batch,
            on_pad=float(stamp) <= on_pad_s,
        )
        if not estimate.ready:
            continue
        truth_pose = _truth_pose(truth, index)
        if truth_pose is None:
            continue
        if origin is None:
            origin = truth_pose[1].copy()
        position_errors.append(float(np.linalg.norm(estimate.position - (truth_pose[1] - origin))))
        if estimate.gate is None:
            continue
        if estimate.gate.track is TrackCode.TRACKING:
            track_time += _dt(sensors.imu_time, index)
        elif estimate.gate.track is TrackCode.COASTING:
            coast_time += _dt(sensors.imu_time, index)
        elif estimate.gate.track is TrackCode.LOST:
            lost = True
        relative = estimate.relative_body()[:3]
        truth_relative = truth_pose[0].T @ (gate_center - truth_pose[1])
        if float(np.linalg.norm(relative)) > 1e-6 and float(np.linalg.norm(truth_relative)) > 1e-6:
            bearing_errors.append(_angle_deg(relative, truth_relative))
    position = float(np.median(position_errors)) if position_errors else float("nan")
    bearing = float(np.median(bearing_errors)) if bearing_errors else float("nan")
    print(f"position_error_m {position:.3f}")
    print(f"bearing_error_deg {bearing:.3f}")
    print(f"track_time_s {track_time:.2f}")
    print(f"coast_time_s {coast_time:.2f}")
    print(f"lost {str(lost).lower()}")


def _detect(detector: GateDetector, sensors, inner: float) -> list[tuple]:
    found = []
    for frame, stamp in zip(sensors.frames, sensors.frame_time, strict=True):
        found.append(detector.detect(frame, float(stamp), inner))
    return found


def _camera(meta: dict) -> tuple[CameraIntrinsics, CameraMount]:
    raw = meta.get("intrinsics")
    if not isinstance(raw, dict):
        return FPV_INTRINSICS, FPV_MOUNT
    intrinsics = CameraIntrinsics(
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
    )
    mount = CameraMount(
        position_body=np.asarray(meta["mount_position_body"], dtype=np.float64),
        tilt_deg=float(meta["tilt_deg"]),
    )
    return intrinsics, mount


def _advance(times: np.ndarray, index: int, stamp: float) -> int:
    while index + 1 < len(times) and float(times[index + 1]) <= stamp:
        index += 1
    return index


def _dt(times: np.ndarray, index: int) -> float:
    if index + 1 >= len(times):
        return 0.0
    return max(0.0, float(times[index + 1] - times[index]))


def _truth_pose(truth, index: int) -> tuple[np.ndarray, np.ndarray] | None:
    if index >= len(truth.world_pos):
        return None
    pose = np.asarray(truth.world_pos[index], dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        return None
    return _rotation(pose[:4]), pose[4:7].copy()


def _rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


if __name__ == "__main__":
    raise SystemExit(main())
