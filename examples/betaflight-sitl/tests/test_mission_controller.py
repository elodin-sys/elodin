"""Direct generic-controller contracts and thin truth-adapter delegation."""

import ast
import inspect
import math
from dataclasses import FrozenInstanceError, replace

import pytest
from controls import SemanticControl
from guidance import (
    GateTarget,
    GuidancePhase,
    MissionController,
    NavState,
    TrackState,
    TruthGuidance,
    TruthGuidanceConfig,
)
from test_guidance_golden import make_update


def nav(position=(0.0, 0.0, 0.1), velocity=(0.0, 0.0, 0.0), yaw=0.0, yaw_rate=0.0):
    return NavState(position, velocity, yaw, yaw_rate)


def gate(**changes):
    return replace(GateTarget(0, (10.0, 0.0, 1.8), 0.0, 2.5, TrackState.TRACKING, 0.0), **changes)


def armed_controller():
    controller = MissionController()
    for time in (0.0, 5.0, 5.6):
        controller.update(time, nav(), gate())
    assert controller.phase is GuidancePhase.TAKEOFF
    return controller


def test_generic_controller_runs_all_hold_phases_without_truth_updates():
    controller = armed_controller()
    for time in (6.0, 6.5):
        controller.update(time, nav((0.0, 0.0, 1.8)), None)
    assert controller.phase is GuidancePhase.HOVER
    for time in (7.0, 7.75):
        controller.update(time, nav((0.0, 0.0, 1.8)), None)
    assert controller.phase is GuidancePhase.ACQUIRE
    for time in (8.0, 9.0):
        command = controller.update(time, nav((6.0, 0.0, 1.8)), None)
    assert controller.phase is GuidancePhase.HOLD
    assert controller.phase_code == 5
    assert controller.failure_reason is None
    assert controller.target == (6.0, 0.0, 1.8)
    assert command == SemanticControl(throttle=0.13, armed=True, angle_mode=True)


def test_generic_controller_latches_first_gate_despite_progress_and_metadata():
    controller = armed_controller()
    first = controller.gate_target
    controller.update(
        5.7,
        nav(),
        gate(index=1, center=(20.0, 0.0, 2.0), track=TrackState.LOST, age=99.0),
        next_gate_index=1,
        gate_count=2,
    )
    assert controller.gate_target is first
    assert controller.target == (6.0, 0.0, 1.8)
    assert controller.phase is GuidancePhase.TAKEOFF
    controller.update(5.8, nav(), None, last_gate_passed=0)
    assert controller.failure_reason == "gate_crossing"


@pytest.mark.parametrize("track", list(TrackState))
def test_track_metadata_does_not_introduce_vision_policy(track):
    controller = MissionController()
    controller.update(0.0, nav(), gate(track=track, age=2.0))
    command = controller.update(5.0, nav(), None)
    assert command.armed
    assert controller.phase is GuidancePhase.ARM


def test_controller_uses_same_geometry_in_translated_rotated_frame():
    original, transformed = MissionController(), MissionController()
    angle = 0.7
    offset = (23.0, -17.0, 4.0)

    def rotate(v):
        c, s = math.cos(angle), math.sin(angle)
        return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])

    def translate(p):
        return tuple(v + o for v, o in zip(rotate(p), offset, strict=True))

    target = gate()
    rotated_gate = replace(target, center=translate(target.center), yaw=target.yaw + angle)
    for time, state in (
        (0.0, nav()),
        (5.0, nav()),
        (5.6, nav()),
        (5.8, nav((-0.4, 0.3, 0.2), (0.1, -0.2, 0.03), 0.2, 0.04)),
    ):
        a = original.update(time, state, target)
        b = transformed.update(
            time,
            NavState(
                translate(state.position), rotate(state.velocity), state.yaw + angle, state.yaw_rate
            ),
            rotated_gate,
        )
        assert (b.roll, b.pitch, b.throttle, b.yaw) == pytest.approx(
            (a.roll, a.pitch, a.throttle, a.yaw), abs=1e-12, rel=0
        )
        assert (b.armed, b.angle_mode) == (a.armed, a.angle_mode)
        assert transformed.phase is original.phase


def test_error_ingress_keeps_feedback_separate_from_accepted_snapshot():
    controller = armed_controller()
    accepted = controller.last_valid_nav
    feedback = nav((0.4, -0.2, 0.7), (0.1, 0.2, -0.1), 0.2, 0.3)
    command = controller.reject_update(5.8, "invalid_truth", feedback)
    assert controller.last_valid_nav is accepted
    assert controller._last_time == 5.8
    assert controller._fault_target == accepted.position
    assert controller.phase is GuidancePhase.FAULT
    assert controller.failure_reason == "invalid_truth"
    assert command == controller._position_control(
        feedback, accepted.position, accepted.position[2], 0.0
    )
    # A valid tick in FAULT updates the accepted cache, not the latched target.
    later = nav((0.7, 0.3, 0.8), yaw_rate=-0.2)
    controller.update(5.9, later, None)
    assert controller.last_valid_nav is later
    assert controller._fault_target == accepted.position
    controller.reject_update(6.0, "different_error", replace(later, yaw_rate=0.4))
    assert controller.failure_reason == "invalid_truth"
    assert controller.last_valid_nav is later
    assert controller._fault_target == accepted.position


@pytest.mark.parametrize(
    "time,reason",
    [
        (float("nan"), "invalid_sim_time"),
        (-1.0, "invalid_sim_time"),
        (4.0, "nonmonotonic_sim_time"),
    ],
)
def test_error_ingress_time_priority_and_no_time_or_snapshot_advance(time, reason):
    controller = armed_controller()
    accepted = controller.last_valid_nav
    feedback = nav((0.4, 0.2, 0.7), yaw_rate=0.3)
    controller.reject_update(time, "invalid_truth", feedback)
    assert controller.failure_reason == reason
    assert controller.phase_started_at == 5.6
    assert controller._last_time == 5.6
    assert controller.last_valid_nav is accepted
    assert controller._fault_target == accepted.position


def test_generic_missing_input_fails_safe_and_uses_legacy_reason():
    controller = MissionController()
    assert controller.update(0.0, None, None) == SemanticControl.safe()
    assert controller.failure_reason == "invalid_truth"
    no_gate = MissionController()
    assert no_gate.update(0.0, nav(), None) == SemanticControl.safe()
    assert no_gate.failure_reason == "missing_gate"


def test_controller_logic_has_no_truth_update_or_io_dependencies():
    tree = ast.parse(inspect.getsource(MissionController))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert names.isdisjoint(
        {
            "TruthState",
            "TruthGuidanceUpdate",
            "GatePose",
            "GuidanceUpdate",
            "TruthGuidance",
            "_yaw_from_xyzw",
            "StepContext",
        }
    )
    assert attributes.isdisjoint(
        {
            "truth_state",
            "gate_poses",
            "position_enu",
            "velocity_enu",
            "orientation_xyzw",
            "gyro",
            "accel",
            "frame",
            "read_component",
            "write_component",
        }
    )
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    assert list(inspect.signature(MissionController.update).parameters) == [
        "self",
        "sim_time",
        "nav",
        "gate",
        "last_gate_passed",
        "next_gate_index",
        "gate_count",
    ]


def test_adapter_delegates_each_tick_and_uses_first_gate_tracking_zero_age(monkeypatch):
    config = TruthGuidanceConfig()
    source = TruthGuidance(config)
    real_update = source.controller.update
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_update(*args, **kwargs)

    monkeypatch.setattr(source.controller, "update", spy)
    for time in (0.0, 0.000125, 5.0, 5.6):
        source.update(
            make_update(
                {
                    "time": time,
                    "gates": [[4, [10, 0, 1.8], 0, 2.5], [0, [20, 0, 1.8], 0, 2.5]],
                    "next_gate_index": 0,
                },
                0,
            )
        )
    assert len(calls) == 4
    assert [args[0] for args, _ in calls] == [0.0, 0.000125, 5.0, 5.6]
    assert all(isinstance(args[1], NavState) for args, _ in calls)
    assert all(args[2] is calls[0][0][2] for args, _ in calls)
    target = calls[0][0][2]
    assert target.index == 4  # Array order, not next_gate_index or minimum index.
    assert target.track is TrackState.TRACKING
    assert target.age == 0.0
    assert calls[0][1] == {"last_gate_passed": -1, "next_gate_index": 0, "gate_count": 2}
    assert source.config is config
    assert source.phase is source.controller.phase
    assert source.phase_code == source.controller.phase_code
    assert source.phase_started_at == source.controller.phase_started_at
    assert source.failure_reason == source.controller.failure_reason
    assert source.braking_active == source.controller.braking_active
    assert source.target_enu == source.controller.target


def test_adapter_rejection_passes_only_generic_current_feedback(monkeypatch):
    source = TruthGuidance()
    for time in (0.0, 5.0, 5.6):
        source.update(make_update({"time": time}, 0))
    real_reject = source.controller.reject_update
    calls = []

    def spy(*args):
        calls.append(args)
        return real_reject(*args)

    monkeypatch.setattr(source.controller, "reject_update", spy)
    previous = source.controller.last_valid_nav
    source.update(make_update({"time": 5.7, "truth": False, "yaw_rate": 0.6}, 0))
    assert calls[-1] == (5.7, "invalid_truth", replace(previous, yaw_rate=0.6))
    source.update(
        make_update({"time": 5.8, "position": [0.3, 0.2, 0.4], "gyro": ["nan", 0.0, 0.7]}, 0)
    )
    feedback = calls[-1][2]
    assert isinstance(feedback, NavState)
    assert feedback.position == (0.3, 0.2, 0.4)
    assert feedback.yaw_rate == 0.0  # Any bad axis invalidates even a finite z.
    assert source.controller.last_valid_nav is previous


def test_generic_values_copy_mutable_vectors_and_are_frozen():
    position, velocity, center = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    state = NavState(position, velocity, 0, 0)
    target = GateTarget(0, center, 0, 2.5, TrackState.TRACKING, 0)
    position[0] = velocity[0] = center[0] = 100
    assert state.position == (1.0, 2.0, 3.0)
    assert state.velocity == (4.0, 5.0, 6.0)
    assert target.center == (7.0, 8.0, 9.0)
    with pytest.raises(FrozenInstanceError):
        state.yaw = 1.0
    with pytest.raises(FrozenInstanceError):
        target.age = 1.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("position", [1, 2]),
        ("position", [0, 0, float("nan")]),
        ("position", None),
        ("velocity", [0, float("inf"), 0]),
        ("velocity", [[0], 0, 0]),
        ("yaw", float("nan")),
        ("yaw", None),
        ("yaw_rate", float("inf")),
    ],
)
def test_nav_input_contract_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(nav(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("index", True),
        ("index", -1),
        ("index", 0.5),
        ("center", [0, 0]),
        ("center", [0, float("nan"), 0]),
        ("yaw", float("inf")),
        ("inner_size", 0),
        ("inner_size", -1),
        ("inner_size", float("nan")),
        ("track", "TRACKING"),
        ("track", 2),
        ("age", -0.1),
        ("age", float("inf")),
    ],
)
def test_gate_input_contract_rejects_invalid_values(field, value):
    with pytest.raises(TypeError if field == "track" else ValueError):
        replace(gate(), **{field: value})
