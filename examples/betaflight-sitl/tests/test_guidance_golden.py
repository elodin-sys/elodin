"""Frozen Package E traces; generate ONLY from baseline 806b0772 guidance.py.

From the repository root:
  python examples/betaflight-sitl/tests/test_guidance_golden.py --capture

Capture refuses a different controller source. Normal pytest never regenerates
expected outputs. All floating-point outputs use hex strings (including signed
zero), so replay checks exact values, not a numeric tolerance.

After extraction, use a NEW isolated baseline tree under $HOME; see
fixtures/README.md. Never restore/overwrite the working controller to capture.
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
FIXTURE = Path(__file__).with_name("fixtures") / "guidance_package_e.json"
FIXTURE_SHA256 = "95f9257340cd5f0fa3a7a436403018c9e55fa340297d97c5a7de3f630af92bb3"


def test_golden_provenance_and_coverage():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["baseline"] == BASELINE
    assert fixture["source_sha256"] == SOURCE_SHA256
    rows = [row for case in fixture["cases"] for row in case["expected"]]
    assert len(fixture["cases"]) == 34
    assert len(rows) == 229
    assert {row["phase"] for row in rows} == {"ARM", "TAKEOFF", "HOVER", "ACQUIRE", "HOLD", "FAULT"}
    assert {row["failure_reason"] for row in rows} == {
        None,
        "invalid_sim_time",
        "nonmonotonic_sim_time",
        "invalid_truth",
        "missing_gate",
        "gate_crossing",
        "arm_timeout",
        "takeoff_timeout",
        "hover_timeout",
        "acquire_timeout",
        "hold_timeout",
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
    yield "arm_timeout", {"arm_timeout_s": 5.1}, [{"time": 0.0}, {"time": 5.1}]
    yield "takeoff_timeout", {"takeoff_timeout_s": 0.1}, start + [{"time": 5.71}]
    yield (
        "hover_timeout",
        {"hover_timeout_s": 0.1},
        acquire[:-2] + [{"time": 6.61, "position": [0.0, 0.0, 1.8]}],
    )
    yield (
        "acquire_timeout",
        {"acquire_timeout_s": 0.1},
        acquire + [{"time": 7.86, "position": [2.0, 0.0, 1.8]}],
    )
    invalids = {
        "negative_time": {"time": -1.0},
        "nan_time": {"time": "nan"},
        "inf_time": {"time": "inf"},
        "missing_truth": {"truth": False},
        "gyro_shape": {"gyro": [0.0, 0.0]},
        "gyro_nan_x": {"gyro": ["nan", 0.0, 0.7]},
        "gyro_inf_y": {"gyro": [0.0, "inf", 0.7]},
        "gyro_nan_z": {"gyro": [0.0, 0.0, "nan"]},
    }
    for name, bad in invalids.items():
        for armed in (False, True):
            prefix = start if armed else []
            yield (
                f"{name}_{'armed' if armed else 'disarmed'}",
                {},
                prefix
                + [
                    {
                        "time": 5.8 if armed else 0.0,
                        "position": [0.3, -0.4, 0.5],
                        "velocity": [0.1, -0.2, 0.3],
                        "yaw": 0.2,
                        "yaw_rate": 0.6,
                        **bad,
                    },
                    {"time": 5.9, "truth": False, "gyro": [0.0, 0.0, "nan"]},
                    {"time": 6.0, "position": [-0.2, 0.3, 0.7], "yaw": -0.2, "yaw_rate": -0.4},
                ],
            )
    for armed in (False, True):
        prefix = start if armed else [{"time": 1.0}]
        yield (
            f"nonmonotonic_{armed}",
            {},
            prefix
            + [
                {"time": 0.5, "position": [0.4, 0.2, 0.6], "yaw_rate": 0.3},
                {"time": 6.0, "truth": False},
            ],
        )
    yield "missing_gate", {}, [{"time": 0.0, "gates": []}, {"time": 5.0}]
    for phase, prefix in (
        ("arm", [{"time": 0.0}]),
        ("takeoff", start),
        ("hover", acquire[:-2]),
        ("acquire", acquire),
        ("hold", hold),
    ):
        yield (
            f"crossing_{phase}",
            {},
            prefix
            + [
                {"time": 10.0, "position": [9.9, 0.0, 1.8]},
                {"time": 10.1, "position": [10.0, 0.0, 1.8], "velocity": [0.5, 0.1, 0.0]},
            ],
        )
    yield "referee_event", {}, start + [{"time": 5.8, "last_gate_passed": 0}]
    yield (
        "braking_yaw_saturation",
        {},
        start
        + [
            {"time": 5.7 + i * 0.001, "position": p, "velocity": v, "yaw": yaw, "yaw_rate": rate}
            for i, (p, v, yaw, rate) in enumerate(
                [
                    ((-100, -100, -100), (-30, 30, -20), 0.4, 2.0),
                    ((1, -1, 100), (0.2, -0.2, 20), -0.4, -2.0),
                    ((7.9, 0.3, 1.7), (3, 0.2, 0.1), 1.0, 0.1),
                    ((8, -0.2, 1.8), (0.05, -0.1, 0), -1.0, -0.1),
                    ((8.5, 0, 1.8), (0.5, 0, 0), math.pi / 2, 0.0),
                    ((8.7, 0, 1.8), (-0.1, 0, 0), -math.pi + 0.01, 0.03),
                ]
            )
        ],
    )
    yawed_gate = [0, [4.0, -3.0, 2.5], math.pi / 2, 2.5]
    yield (
        "yawed_gate_latching_and_crossing",
        {"standoff_distance_m": 3.0},
        [
            {"time": 0.0, "position": [0.0, -8.0, 0.1], "gates": [yawed_gate]},
            {"time": 5.0, "gates": [[1, [20.0, 0.0, 1.8], 0.0, 2.5]], "next_gate_index": 1},
            {"time": 5.6, "gates": [], "position": [0.0, -8.0, 0.1]},
            {"time": 5.8, "position": [3.0, -3.1, 2.3], "velocity": [0.1, 0.5, 0.0], "yaw": 1.3},
            {"time": 5.9, "position": [4.0, -2.9, 2.5]},
        ],
    )
    yield (
        "yaw_wrap",
        {},
        [
            {**item, "gates": [[0, [10.0, 0.0, 1.8], -math.pi + 0.01, 2.5]], "yaw": math.pi - 0.01}
            for item in start + [{"time": 5.8, "yaw_rate": 0.03}]
        ],
    )
    yield (
        "dwell_resets",
        {},
        start
        + [
            {"time": t, "position": [0.0, 0.0, z]}
            for t, z in (
                (6.0, 1.8),
                (6.4, 1.0),
                (6.5, 1.8),
                (7.0, 1.8),
                (7.5, 1.0),
                (7.6, 1.8),
                (8.35, 1.8),
            )
        ]
        + [
            {"time": t, "position": [x, 0.0, 1.8]}
            for t, x in ((9.0, 6.0), (9.9, 5.0), (10.0, 6.0), (11.0, 6.0))
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
            GatePose(*g) for g in spec.get("gates", [[0, [10.0, 0.0, 1.8], 0.0, 2.5]])
        ),
        next_gate_index=spec.get("next_gate_index", 0),
        last_gate_passed=spec.get("last_gate_passed", -1),
        gate_count=len(spec.get("gates", [0])),
    )


def snapshot(source, command):
    return {
        "phase": source.phase.name,
        "phase_code": source.phase_code,
        "failure_reason": source.failure_reason,
        "braking_active": source.braking_active,
        "target": None if source.target_enu is None else [v.hex() for v in source.target_enu],
        "command": {k: v.hex() if isinstance(v, float) else v for k, v in asdict(command).items()},
    }


def assert_original_source():
    root = Path(__file__).resolve().parents[1]
    for name, digest in (("guidance.py", SOURCE_SHA256), ("controls.py", CONTROLS_SHA256)):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, (
            f"Golden capture requires ORIGINAL {name}; do not regenerate from extracted code"
        )


def capture():
    assert_original_source()
    cases = []
    for name, config, inputs in scenarios():
        controller = TruthGuidance(TruthGuidanceConfig(**config))
        cases.append(
            {
                "name": name,
                "config": config,
                "inputs": inputs,
                "expected": [
                    snapshot(controller, controller.update(make_update(s, i)))
                    for i, s in enumerate(inputs)
                ],
            }
        )
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(
        json.dumps({"baseline": BASELINE, "source_sha256": SOURCE_SHA256, "cases": cases}, indent=2)
        + "\n"
    )
    print(f"{len(cases)} cases, {sum(len(c['inputs']) for c in cases)} samples")
    print(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE)


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURE.read_text())["cases"] if FIXTURE.exists() else [],
    ids=lambda case: case["name"],
)
def test_original_package_e_golden(case):
    controller = TruthGuidance(TruthGuidanceConfig(**case["config"]))
    actual = [
        snapshot(controller, controller.update(make_update(s, i)))
        for i, s in enumerate(case["inputs"])
    ]
    assert actual == case["expected"]


if __name__ == "__main__":
    assert sys.argv[1:] == ["--capture"]
    capture()
