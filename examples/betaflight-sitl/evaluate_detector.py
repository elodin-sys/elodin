"""Offline detection metrics for synthetic images, fixtures, and replay bundles.

Timing is reported against a 5 ms target and never fails the process. Accuracy
checks are opt-in via ``--check``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from course import Gate
from gate_detector import GateDetector
from replay import load_sensor_bundle, load_truth
from synthetic_vision import _world_rotation, project_gate_corners

_EXAMPLE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = _EXAMPLE / "tests" / "fixtures" / "rendered_gate_frames.npz"
_INNER = 2.5
_TIME_TARGET_MS = 5.0


@dataclass(frozen=True, slots=True)
class SampleResult:
    range_m: float
    detected: bool
    bearing_error_deg: float | None
    reprojection_px: float | None
    elapsed_ms: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the classical gate detector")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--fixture", nargs="?", const=DEFAULT_FIXTURE, type=Path)
    source.add_argument("--bundle", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--inner-size", type=float, default=None)
    args = parser.parse_args(argv)

    if args.synthetic:
        results = _synthetic_results()
    elif args.fixture is not None:
        results = _fixture_results(args.fixture, args.inner_size)
    else:
        results = _bundle_results(args.bundle, args.inner_size)
    _report(results)
    if args.check and not _passes(results):
        return 1
    return 0


def _synthetic_results() -> list[SampleResult]:
    tests = str(_EXAMPLE / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    from gate_images import CORE_BEARINGS_DEG, CORE_RANGES_M, camera_pose_for_range, render_gates

    gate = Gate(0, (10.0, 0.0, 1.8), 0.0, _INNER)
    detector = GateDetector()
    results = []
    for range_m in CORE_RANGES_M:
        for bearing in CORE_BEARINGS_DEG:
            pose = camera_pose_for_range(gate, range_m, bearing_deg=bearing)
            image, corners = render_gates([gate], pose)
            if not _fully_visible(corners[0]):
                continue
            truth = _body_bearing(gate.center, pose)
            results.append(_measure(detector, image, range_m, _INNER, truth))
    return results


def _fixture_results(path: Path, inner_size: float | None) -> list[SampleResult]:
    data = np.load(path)
    size = float(data["inner_size"] if inner_size is None else inner_size)
    center = np.asarray(data["gate_center"], dtype=np.float64)
    detector = GateDetector()
    results = []
    for frame, world_pos in zip(data["frames"], data["world_pos"], strict=True):
        truth_range = float(np.linalg.norm(world_pos[4:7] - center))
        results.append(
            _measure(detector, frame, truth_range, size, _body_bearing(center, world_pos))
        )
    return results


def _bundle_results(path: Path, inner_size: float | None) -> list[SampleResult]:
    """Score frames whose nearest fully visible gate is the truth target."""

    sensors = load_sensor_bundle(path)
    truth = load_truth(path)
    size = inner_size
    if size is None:
        size = sensors.meta.get("inner_size_m")
    if size is None:
        raise SystemExit("bundle has no inner_size_m; pass --inner-size")
    if truth.gate_center.shape[0] < 1:
        raise SystemExit("bundle truth has no gate")
    gates = [
        Gate(int(index), tuple(float(v) for v in center), float(yaw), float(gate_size))
        for index, center, yaw, gate_size in zip(
            truth.gate_index,
            truth.gate_center,
            truth.gate_yaw,
            truth.gate_inner_size,
            strict=True,
        )
    ]
    detector = GateDetector()
    results = []
    skipped = 0
    for frame, world_pos in zip(sensors.frames, truth.frame_world_pos, strict=True):
        target = _nearest_visible_gate(gates, world_pos)
        if target is None:
            skipped += 1
            continue
        truth_range = float(np.linalg.norm(world_pos[4:7] - np.asarray(target.center)))
        results.append(
            _measure(
                detector, frame, truth_range, float(size), _body_bearing(target.center, world_pos)
            )
        )
    print(f"skipped {skipped} frames without a fully visible gate or finite truth")
    return results


def _nearest_visible_gate(gates: list[Gate], world_pos: np.ndarray) -> Gate | None:
    if not np.all(np.isfinite(world_pos)):
        return None
    visible = [gate for gate in gates if project_gate_corners(gate, world_pos) is not None]
    if not visible:
        return None
    return min(visible, key=lambda gate: np.linalg.norm(world_pos[4:7] - np.asarray(gate.center)))


def _measure(detector, frame, range_m, inner_size, truth_bearing) -> SampleResult:
    started = time.perf_counter()
    found = detector.detect(frame, 0.0, inner_size)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not found or found[0].pose is None:
        return SampleResult(range_m, False, None, None, elapsed_ms)
    candidate = found[0]
    return SampleResult(
        range_m,
        True,
        _angle_deg(candidate.pose.bearing_body, truth_bearing),
        candidate.reprojection_error_px,
        elapsed_ms,
    )


def _report(results: list[SampleResult]) -> None:
    print("range_m  frames  detected  rate")
    for low in range(4, 20, 2):
        group = [item for item in results if _range_bin(item.range_m) == low]
        detected = sum(item.detected for item in group)
        rate_text = "n/a" if not group else f"{detected / len(group):.3f}"
        print(f"{low:2d}-{low + 2:<2d}  {len(group):6d}  {detected:8d}  {rate_text}")
    detected = [item for item in results if item.detected]
    print(f"frames {len(results)} detected {len(detected)}")
    _print_stat("bearing_deg", [item.bearing_error_deg for item in detected])
    _print_stat("reproj_px", [item.reprojection_px for item in detected])
    times = [item.elapsed_ms for item in results]
    _print_stat("time_ms", times)
    if times:
        print(f"time_target_ms {_TIME_TARGET_MS:.1f} (reported only, not a failure)")


def _print_stat(name: str, values: list[float | None]) -> None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        print(f"{name} median n/a p95 n/a")
        return
    ordered = np.sort(np.asarray(finite))
    median = float(np.median(ordered))
    p95 = float(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)])
    print(f"{name} median {median:.4f} p95 {p95:.4f}")


def _passes(results: list[SampleResult]) -> bool:
    results = [item for item in results if _range_bin(item.range_m) is not None]
    if not results:
        print("check failed: no frames in 4-20 m")
        return False
    detected = [item for item in results if item.detected and item.bearing_error_deg is not None]
    rate = len(detected) / len(results)
    median = (
        float(np.median([item.bearing_error_deg for item in detected])) if detected else math.inf
    )
    print(f"check detection_rate {rate:.3f} bearing_median_deg {median:.4f}")
    return rate >= 0.90 and median <= 2.0


def _range_bin(range_m: float) -> int | None:
    """Lower edge of the 2 m bin. The 20 m endpoint stays in 18–20."""

    if range_m < 4.0 or range_m > 20.0:
        return None
    if math.isclose(range_m, 20.0):
        return 18
    return int(math.floor(range_m / 2.0) * 2)


def _fully_visible(corners: np.ndarray) -> bool:
    return bool(
        np.all(corners[:, 0] >= 0.0)
        and np.all(corners[:, 0] <= 640.0)
        and np.all(corners[:, 1] >= 0.0)
        and np.all(corners[:, 1] <= 360.0)
    )


def _body_bearing(center, pose: np.ndarray) -> np.ndarray:
    body = _world_rotation(pose[:4]).T @ (np.asarray(center, dtype=np.float64) - pose[4:7])
    return body / np.linalg.norm(body)


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(
        np.clip(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)), -1.0, 1.0)
    )
    return math.degrees(math.acos(cosine))


if __name__ == "__main__":
    raise SystemExit(main())
