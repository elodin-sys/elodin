"""Supplemental ORIGINAL-controller traces from the pre-extraction review.

Capture only in an isolated baseline tree, as documented in fixtures/README.md.
The initial 34-case fixture remains unchanged, including its early yawed fault.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
from test_guidance_golden import (
    BASELINE,
    SOURCE_SHA256,
    TruthGuidance,
    TruthGuidanceConfig,
    assert_original_source,
    make_update,
    snapshot,
)

FIXTURE = Path(__file__).with_name("fixtures") / "guidance_package_e_supplement.json"
FIXTURE_SHA256 = "53fe85c570bb477d1889536fc08256179092acb3fe6ea7e0241e90420c8f348f"


def scenarios():
    gate = [0, [4.0, -3.0, 2.5], math.pi / 2, 2.5]
    yield (
        "yawed_active_latching_braking_crossing",
        {},
        [
            {"time": 0.0, "position": [0.0, -8.0, 0.1], "gates": [gate]},
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
    start = [{"time": t} for t in (0.0, 5.0, 5.6)]
    # Nonunit quaternion with roll, pitch and yaw tests truth normalization,
    # full heading conversion, then missing-truth feedback with CURRENT gyro.
    yield (
        "full_attitude_and_missing_truth",
        {},
        start
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
    for name, config in (
        ("tilt_cap", {"max_tilt_rad": 0.03}),
        ("stick_cap", {"max_horizontal_stick": 0.04}),
    ):
        yield (
            name,
            config,
            start
            + [
                {"time": 5.7, "position": [-10.0, -10.0, 0.1], "velocity": [-1.0, -1.0, 0.0]},
                {"time": 5.8, "position": [1.0, 1.0, 0.1], "velocity": [1.0, 1.0, 0.0]},
            ],
        )
    yield (
        "velocity_and_launch_xy_dwell_interruptions",
        {},
        start
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
    print(f"{len(cases)} cases / {sum(len(c['inputs']) for c in cases)} samples")
    print(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), FIXTURE)


def test_supplement_provenance_and_active_coverage():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["baseline"] == BASELINE
    assert fixture["source_sha256"] == SOURCE_SHA256
    cases = {case["name"]: case for case in fixture["cases"]}
    rows = cases["yawed_active_latching_braking_crossing"]["expected"]
    assert [row["phase"] for row in rows] == ["ARM", "ARM", "TAKEOFF", "TAKEOFF", "FAULT"]
    assert [row["failure_reason"] for row in rows] == [None, None, None, None, "gate_crossing"]
    assert rows[3]["braking_active"] is True
    assert rows[3]["command"]["armed"] is True
    assert [
        row["phase"] for row in cases["velocity_and_launch_xy_dwell_interruptions"]["expected"]
    ] == [
        "ARM",
        "ARM",
        "TAKEOFF",
        "TAKEOFF",
        "TAKEOFF",
        "TAKEOFF",
        "TAKEOFF",
        "TAKEOFF",
        "HOVER",
        "HOVER",
        "HOVER",
        "HOVER",
        "ACQUIRE",
        "ACQUIRE",
        "ACQUIRE",
        "ACQUIRE",
        "HOLD",
    ]
    # Both independent horizontal caps actually bind, not merely bound output.
    stick = cases["stick_cap"]["expected"][3]["command"]
    assert abs(float.fromhex(stick["roll"])) == 0.04
    assert abs(float.fromhex(stick["pitch"])) == 0.04
    tilt = cases["tilt_cap"]["expected"][3]["command"]
    cap = 0.03 / math.radians(20)
    assert abs(float.fromhex(tilt["roll"])) == cap
    assert abs(float.fromhex(tilt["pitch"])) == cap


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURE.read_text())["cases"] if FIXTURE.exists() else [],
    ids=lambda case: case["name"],
)
def test_supplement_package_e_golden(case):
    controller = TruthGuidance(TruthGuidanceConfig(**case["config"]))
    actual = [
        snapshot(controller, controller.update(make_update(s, i)))
        for i, s in enumerate(case["inputs"])
    ]
    assert actual == case["expected"]


if __name__ == "__main__":
    assert sys.argv[1:] == ["--capture"]
    capture()
