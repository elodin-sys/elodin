"""Compact characterization of representative original E1 controller behavior.

Normal pytest replays these eight deterministic scenarios and compares a SHA-256
of each current output trace against digests derived from the original expected
rows. Full expected rows remain available in git history; tests never read history
or generate expected values.

Capture only in an isolated tree containing the ORIGINAL 806b0772 guidance.py
and controls.py:

    python test_guidance_golden.py --capture

Capture prints current trace digests and never updates this file. See
fixtures/README.md for the provenance and safe reproduction procedure.
"""

import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controls import GatePose, TruthGuidanceUpdate, TruthState
from guidance import TruthGuidance, TruthGuidanceConfig

BASELINE = "806b07721797a15e92b1614c0a206516c036f2e3"
SOURCE_SHA256 = "6783e94e03cd715c5cfdc66a3ca2d09c17fe79be6c1f41c41696af4a729d3b69"
CONTROLS_SHA256 = "1bf2504d66ed8666755c648f8c936d81931ff23d4f8ffcbfdefa054cc2b9948f"
EXPECTED_DIGESTS = {
    "full_mission_and_hold_timeout": "cb8cb98cb02327d5f5e2bfb89b6c3f370dc9b85a7c74a0631bfd2ab29ff478bf",
    "braking_yaw_saturation": "59733de5c3652a2c35914bda22af5662a5396b9b43be371f6f62691aa4500377",
    "yawed_active_latching_braking_crossing": "a9e4dd97bf2480468351df2011e740f126e8ab673430c6b0e0105e8a5a40bdb5",
    "full_attitude_and_missing_truth": "ac82d25725f572de71a4154bcc7e9ed678416df2827a7e076197e5e9f2d38f44",
    "missing_truth_armed": "0654f6e4590760cc8f34e218678e2dc8f571fd6690114bd861a4e6947911e9c1",
    "gyro_nan_x_armed": "41ea01c5f8cc9dce82e72b0cd860bb99870156088753e977f29cda6648923c8e",
    "nonmonotonic_True": "10aac0b115d29ee8212a8ff297468f4a7c321f0a0c26de736444644389a35a93",
    "velocity_and_launch_xy_dwell_interruptions": "3edfb4cecce35da0b1cd24fa80fb61fd2508e03712ec64219d45c819cb1f9378",
}


def scenarios():
    start = [{"time": t} for t in (0.0, 4.999, 5.0, 5.599, 5.6)]
    acquire = start + [{"time": t, "position": [0.0, 0.0, 1.8]} for t in (6.0, 6.5, 7.0, 7.75)]
    hold = acquire + [{"time": t, "position": [6.0, 0.0, 1.8]} for t in (8.0, 8.99, 9.0)]
    yield (
        "full_mission_and_hold_timeout",
        {},
        hold
        + [
            {
                "time": 9.1,
                "position": [5.8, 0.1, 1.7],
                "velocity": [0.2, -0.1, 0.03],
                "yaw": 0.1,
                "yaw_rate": -0.03,
            },
            {"time": 69.0, "position": [6.2, -0.1, 1.9]},
            {"time": 69.1, "position": [6.5, -0.4, 2.0], "velocity": [0.1, 0.2, -0.3]},
        ],
    )
    yield (
        "braking_yaw_saturation",
        {},
        start
        + [
            {"time": 5.7 + i * 0.001, "position": p, "velocity": v, "yaw": yaw, "yaw_rate": rate}
            for i, (p, v, yaw, rate) in enumerate(
                [
                    ([-100, -100, -100], [-30, 30, -20], 0.4, 2.0),
                    ([1, -1, 100], [0.2, -0.2, 20], -0.4, -2.0),
                    ([7.9, 0.3, 1.7], [3, 0.2, 0.1], 1.0, 0.1),
                    ([8, -0.2, 1.8], [0.05, -0.1, 0], -1.0, -0.1),
                    ([8.5, 0, 1.8], [0.5, 0, 0], math.pi / 2, 0.0),
                    ([8.7, 0, 1.8], [-0.1, 0, 0], -math.pi + 0.01, 0.03),
                ]
            )
        ],
    )
    yawed_gate = [0, [4.0, -3.0, 2.5], math.pi / 2, 2.5]
    yield (
        "yawed_active_latching_braking_crossing",
        {},
        [
            {"time": 0.0, "position": [0.0, -8.0, 0.1], "gates": [yawed_gate]},
            {
                "time": 5.0,
                "position": [0.0, -8.0, 0.1],
                "gates": [[1, [20.0, 0.0, 1.8], 0.0, 2.5]],
                "next_gate_index": 1,
            },
            {"time": 5.6, "position": [0.0, -8.0, 0.1], "gates": []},
            {"time": 5.8, "position": [3.0, -3.1, 2.3], "velocity": [0.1, 0.5, 0.0], "yaw": 1.3},
            {"time": 5.9, "position": [4.0, -2.9, 2.5]},
        ],
    )
    yield (
        "full_attitude_and_missing_truth",
        {},
        [{"time": t} for t in (0.0, 5.0, 5.6)]
        + [
            {
                "time": 5.7,
                "position": [0.2, -0.3, 0.5],
                "velocity": [0.1, -0.2, 0.3],
                "quaternion": [0.3, -0.4, 0.7, 1.2],
                "yaw_rate": -0.23,
            },
            {"time": 5.8, "truth": False, "yaw_rate": 0.31},
            {"time": 5.9, "quaternion": [-0.1, 0.25, -0.4, 0.8], "yaw_rate": -0.17},
            {"time": 6.0, "truth": False, "yaw_rate": 0.13},
        ],
    )
    yield (
        "missing_truth_armed",
        {},
        start
        + [
            {
                "time": 5.8,
                "position": [0.3, -0.4, 0.5],
                "velocity": [0.1, -0.2, 0.3],
                "yaw": 0.2,
                "yaw_rate": 0.6,
                "truth": False,
            },
            {"time": 5.9, "truth": False, "gyro": [0.0, 0.0, "nan"]},
            {"time": 6.0, "position": [-0.2, 0.3, 0.7], "yaw": -0.2, "yaw_rate": -0.4},
        ],
    )
    yield (
        "gyro_nan_x_armed",
        {},
        start
        + [
            {
                "time": 5.8,
                "position": [0.3, -0.4, 0.5],
                "velocity": [0.1, -0.2, 0.3],
                "yaw": 0.2,
                "yaw_rate": 0.6,
                "gyro": ["nan", 0.0, 0.7],
            },
            {"time": 5.9, "truth": False, "gyro": [0.0, 0.0, "nan"]},
            {"time": 6.0, "position": [-0.2, 0.3, 0.7], "yaw": -0.2, "yaw_rate": -0.4},
        ],
    )
    yield (
        "nonmonotonic_True",
        {},
        start
        + [
            {"time": 0.5, "position": [0.4, 0.2, 0.6], "yaw_rate": 0.3},
            {"time": 6.0, "truth": False},
        ],
    )
    yield (
        "velocity_and_launch_xy_dwell_interruptions",
        {},
        [{"time": t} for t in (0.0, 5.0, 5.6)]
        + [
            {"time": t, "position": p, "velocity": v}
            for t, p, v in (
                (6.0, [0, 0, 1.8], [0, 0, 0]),
                (6.4, [0, 0, 1.8], [0, 0, 0.21]),
                (6.5, [0, 0, 1.8], [0, 0, 0]),
                (6.9, [0.21, 0, 1.8], [0, 0, 0]),
                (7.0, [0, 0, 1.8], [0, 0, 0]),
                (7.5, [0, 0, 1.8], [0, 0, 0]),
                (8.0, [0, 0, 1.8], [0.19, 0, 0]),
                (8.1, [0, 0, 1.8], [0, 0, 0]),
                (8.25, [0, 0, 1.8], [0, 0, 0]),
                (8.85, [0, 0, 1.8], [0, 0, 0]),
                (9.0, [6, 0, 1.8], [0, 0, 0]),
                (9.9, [6, 0, 1.8], [0.21, 0, 0]),
                (10.0, [6, 0, 1.8], [0, 0, 0]),
                (11.0, [6, 0, 1.8], [0, 0, 0]),
            )
        ],
    )


def make_update(spec, tick):
    yaw = spec.get("yaw", 0.0)
    return TruthGuidanceUpdate(
        sim_time=float(spec["time"]),
        tick=tick,
        gyro=np.array([float(v) for v in spec.get("gyro", [0, 0, spec.get("yaw_rate", 0)])]),
        accel=np.zeros(3),
        truth_state=TruthState(
            spec.get("position", [0.0, 0.0, 0.1]),
            spec.get("velocity", [0.0, 0.0, 0.0]),
            spec.get("quaternion", [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]),
        )
        if spec.get("truth", True)
        else None,
        gate_poses=tuple(
            GatePose(*gate) for gate in spec.get("gates", [[0, [10.0, 0.0, 1.8], 0.0, 2.5]])
        ),
        next_gate_index=spec.get("next_gate_index", 0),
        last_gate_passed=spec.get("last_gate_passed", -1),
        gate_count=len(spec.get("gates", [0])),
    )


def snapshot(controller, command):
    return {
        "phase": controller.phase.name,
        "phase_code": controller.phase_code,
        "failure_reason": controller.failure_reason,
        "braking_active": controller.braking_active,
        "target": None
        if controller.target_enu is None
        else [value.hex() for value in controller.target_enu],
        "command": {
            key: value.hex() if isinstance(value, float) else value
            for key, value in asdict(command).items()
        },
    }


def canonical_digest(rows):
    canonical = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def run_scenario(config, inputs):
    controller = TruthGuidance(TruthGuidanceConfig(**config))
    return [
        snapshot(controller, controller.update(make_update(spec, tick)))
        for tick, spec in enumerate(inputs)
    ]


def assert_original_source():
    root = Path(__file__).resolve().parents[1]
    for name, expected in (("guidance.py", SOURCE_SHA256), ("controls.py", CONTROLS_SHA256)):
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(
                f"--capture requires ORIGINAL {name} from {BASELINE} "
                f"(SHA256 {expected}); found {actual}. No baseline was captured."
            )


def capture_digests():
    assert_original_source()
    digests = {
        name: canonical_digest(run_scenario(config, inputs)) for name, config, inputs in scenarios()
    }
    print(json.dumps(digests, sort_keys=True, separators=(",", ":")))


SCENARIOS = tuple(scenarios())


def test_characterization_scope():
    names = [name for name, _, _ in SCENARIOS]
    assert len(names) == 8
    assert sum(len(inputs) for _, _, inputs in SCENARIOS) == 78
    assert set(names) == set(EXPECTED_DIGESTS)


@pytest.mark.parametrize("name,config,inputs", SCENARIOS, ids=[name for name, _, _ in SCENARIOS])
def test_original_controller_characterization(name, config, inputs):
    actual_rows = run_scenario(config, inputs)
    actual_digest = canonical_digest(actual_rows)
    expected_digest = EXPECTED_DIGESTS[name]
    assert actual_digest == expected_digest, (
        f"{name}: expected SHA256 {expected_digest}, actual SHA256 {actual_digest}\n"
        f"current output rows:\n{json.dumps(actual_rows, sort_keys=True, indent=2, ensure_ascii=True)}"
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--capture"]:
        raise SystemExit(f"usage: {Path(__file__).name} --capture")
    capture_digests()
