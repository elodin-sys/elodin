"""Bundle evaluation uses the recorded camera and matches each detection."""

import math

import numpy as np
from course import Gate
from gate_detector import GateDetector
from gate_images import camera_pose_for_range, render_gates
from replay import ReplayRecorder
from vision import FPV_INTRINSICS, FPV_MOUNT, CameraIntrinsics, CameraMount

import evaluate_detector


def _record(directory, frames, poses, gates, intrinsics, mount) -> None:
    recorder = ReplayRecorder(
        directory,
        simulation_rate_hz=8.0,
        block_ticks=1,
        frame_shape=(intrinsics.height, intrinsics.width, 4),
        intrinsics={
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "width": intrinsics.width,
            "height": intrinsics.height,
        },
        mount_position_body=tuple(float(value) for value in mount.position_body),
        tilt_deg=float(mount.tilt_deg),
        inner_size_m=gates[0].inner_size,
        gates=tuple((gate.index, gate.center, gate.yaw, gate.inner_size) for gate in gates),
    )
    for tick, (frame, pose) in enumerate(zip(frames, poses, strict=True)):
        recorder.observe(
            tick=tick,
            sim_time=float(tick),
            gyro=np.zeros(3),
            accel=np.array([0.0, 0.0, 9.81]),
            world_pos=np.asarray(pose, dtype=np.float64),
            world_vel=np.zeros(6),
            barometer=None,
            barometer_fresh=False,
            magnetometer=None,
            magnetometer_fresh=False,
            frame=frame,
            frame_time=float(tick),
            frame_fresh=True,
        )
    recorder.finish()


def _bearing_median(output: str) -> float:
    line = next(item for item in output.splitlines() if item.startswith("bearing_deg "))
    return float(line.split()[2])


def test_bundle_evaluation_uses_the_recorded_calibration(tmp_path, capsys):
    intrinsics = CameraIntrinsics(fx=500.0, fy=420.0, cx=300.0, cy=200.0, width=640, height=360)
    mount = CameraMount(position_body=np.array([0.12, 0.03, -0.05]), tilt_deg=10.0)
    gate = Gate(0, (10.0, 0.0, 1.8), 0.0, 2.5)
    frames = []
    poses = []
    for range_m in (6.0, 8.0, 10.0):
        pose = camera_pose_for_range(gate, range_m, mount=mount)
        image, _corners = render_gates([gate], pose, intrinsics=intrinsics, mount=mount)
        frames.append(image)
        poses.append(pose)
    _record(tmp_path, frames, poses, [gate], intrinsics, mount)

    assert evaluate_detector.main(["--bundle", str(tmp_path), "--check"]) == 0
    assert _bearing_median(capsys.readouterr().out) < 0.1

    default = GateDetector()
    errors = []
    for frame, pose in zip(frames, poses, strict=True):
        found = default.detect(frame, 0.0, gate.inner_size)
        if not found or found[0].pose is None:
            errors.append(180.0)
            continue
        errors.append(
            evaluate_detector._angle_deg(
                found[0].pose.bearing_body,
                evaluate_detector._body_bearing(gate.center, pose),
            )
        )
    assert float(np.median(errors)) > 2.0


def _polygon_area(corners: np.ndarray) -> float:
    x, y = corners[:, 0], corners[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def test_scoring_matches_the_nearest_gate_not_the_largest_opening(tmp_path, capsys):
    far = Gate(0, (14.0, 0.0, 1.8), 0.0, 2.5)
    near = Gate(1, (8.0, 1.6, 1.8), math.radians(-62.0), 2.5)
    pose = camera_pose_for_range(far, 10.0)
    image, corners = render_gates([far, near], pose)
    assert len(corners) == 2
    far_area, near_area = (_polygon_area(item) for item in corners)
    assert far_area > near_area

    _record(tmp_path, [image], [pose], [far, near], FPV_INTRINSICS, FPV_MOUNT)
    assert evaluate_detector.main(["--bundle", str(tmp_path), "--check"]) == 0
    output = capsys.readouterr().out
    assert _bearing_median(output) < 0.1
    assert "first_candidate_is_target 0.000" in output
