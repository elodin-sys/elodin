"""Frame-neutral F profile: deterministic sequencing, no SDK or self-grading."""

import math
from dataclasses import replace

import pytest
from controls import SemanticControl
from guidance import (
    GUIDANCE_PHASE_NAMES,
    GateTarget,
    MissionController,
    NavState,
    TrackState,
    TruthGuidance,
    TruthGuidanceConfig,
)
from guidance import (
    GuidancePhase as Phase,
)
from test_guidance import _update

GATES = tuple(
    GateTarget(index, (10.0 + 10 * index, 0.0, 1.8), 0.0, 2.5, TrackState.TRACKING, 0.0)
    for index in range(3)
)


def tick(
    controller,
    time,
    *,
    passed=-1,
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    yaw=0.0,
    track=TrackState.TRACKING,
):
    target = replace(GATES[passed + 1], track=track) if passed < 2 else None
    return controller.update(
        time,
        NavState(position, velocity, yaw, 0.0),
        target,
        last_gate_passed=passed,
        next_gate_index=passed + 1 if passed < 2 else None,
        gate_count=3,
    )


def launch():
    controller = MissionController(TruthGuidanceConfig.for_course("c1_straight"))
    for time in (0.0, 5.0, 5.6):
        tick(controller, time)
    assert controller.phase is Phase.TAKEOFF
    for time in (6.0, 6.5, 7.0, 7.75):
        tick(controller, time, position=(0.0, 0.0, 1.8))
    assert controller.phase is Phase.ACQUIRE
    return controller


def commit(controller, start=8.0, index=0):
    position = (6.0 + 10 * index, 0.0, 1.8)
    tick(controller, start, passed=index - 1, position=position)
    assert controller.phase is Phase.ACQUIRE
    tick(controller, start + 1.0, passed=index - 1, position=position)
    assert controller.phase is Phase.HOLD
    tick(controller, start + 1.999, passed=index - 1, position=position)
    assert controller.phase is Phase.HOLD
    command = tick(controller, start + 2.0, passed=index - 1, position=position)
    assert controller.phase is Phase.COMMIT
    assert command.armed and command.angle_mode and command.pitch > 0


def finish():
    controller = launch()
    for index, time in enumerate((8.0, 12.0, 16.0)):
        commit(controller, time, index)
        tick(controller, time + 2.5, passed=index, position=(10.01 + 10 * index, 0.0, 1.8))
        assert controller.phase is (Phase.FINISH if index == 2 else Phase.ACQUIRE)
    return controller


def test_profile_factory_preserves_exact_hold_configuration_and_clear_phase_codes():
    base = TruthGuidanceConfig()
    assert base.profile == "hold"
    assert TruthGuidanceConfig.for_course("single") == base
    assert TruthGuidanceConfig.for_course("none") == base
    race = TruthGuidanceConfig.for_course("c1_straight")
    assert race == replace(
        base,
        profile="race",
        simulation_duration_s=60,
        takeoff_climb_rate_mps=0.4,
        kp_xy=1.2,
        kv_xy=2.5,
        max_horizontal_speed_mps=1.5,
        max_horizontal_accel_mps2=2.0,
    )
    assert TruthGuidance(race).config is race
    assert dict(GUIDANCE_PHASE_NAMES) == {
        0: "DISABLED",
        1: "ARM",
        2: "TAKEOFF",
        3: "HOVER",
        4: "ACQUIRE",
        5: "HOLD",
        6: "FAULT",
        7: "COMMIT",
        8: "SEARCH",
        9: "FINISH",
    }
    with pytest.raises(ValueError, match="unknown guidance course"):
        TruthGuidanceConfig.for_course("not-a-course")
    with pytest.raises(ValueError, match="profile"):
        replace(base, profile="not-a-profile")


def test_three_ordered_passes_retarget_then_finish_and_held_events_do_not_reset():
    controller = launch()
    for index, time in enumerate((8.0, 12.0, 16.0)):
        commit(controller, time, index)
        tick(controller, time + 2.5, passed=index, position=(10.01 + 10 * index, 0.0, 1.8))
        assert controller.failure_reason is None
        if index < 2:
            assert controller.phase is Phase.ACQUIRE
            assert controller.gate_target.index == index + 1
            assert controller.target == (16.0 + 10 * index, 0.0, 1.8)
        else:
            assert controller.phase is Phase.FINISH
            assert controller.target == (32.0, 0.0, 1.8)
        began = controller.phase_started_at
        tick(controller, time + 2.6, passed=index, position=(10.1 + 10 * index, 0.0, 1.8))
        assert controller.phase_started_at == began


def test_geometry_never_awards_pass_and_miss_limit_is_strictly_more_than_1_5m():
    controller = launch()
    commit(controller)
    for time, x in ((10.1, 9.9), (10.2, 10.1), (10.3, 11.5)):
        tick(controller, time, position=(x, 0.0, 1.8))
        assert controller.phase is Phase.COMMIT
        assert controller.gate_target.index == 0
    command = tick(controller, 10.4, position=(11.50001, 0.0, 1.8))
    assert controller.phase is Phase.FAULT
    assert controller.failure_reason == "missed_gate"
    assert command.armed and command.angle_mode  # Truth FAULT holds; never self-passes.


def test_referee_event_takes_priority_over_old_gate_miss_distance():
    controller = launch()
    commit(controller)
    tick(controller, 10.5, passed=0, position=(11.6, 0.0, 1.8))
    assert controller.phase is Phase.ACQUIRE
    assert controller.gate_target.index == 1
    assert controller.failure_reason is None


def test_alignment_requires_continuous_one_second_position_speed_tracking_and_yaw():
    controller = launch()
    for time in (8.0, 9.0):
        tick(controller, time, position=(6.0, 0.0, 1.8), yaw=math.radians(5.0))
    assert controller.phase is Phase.HOLD
    # Exactly 5 degrees is not "under 5 degrees" and cannot start the dwell.
    tick(controller, 10.0, position=(6.0, 0.0, 1.8), yaw=math.radians(5.0))
    tick(controller, 10.1, position=(6.0, 0.0, 1.8))
    tick(controller, 10.9, position=(6.0, 0.0, 1.8), velocity=(0.3, 0.0, 0.0))
    tick(controller, 11.0, position=(6.0, 0.0, 1.8))
    tick(controller, 11.9, position=(6.5, 0.0, 1.8))
    tick(controller, 12.0, position=(6.0, 0.0, 1.8))
    tick(controller, 12.9, position=(6.0, 0.0, 1.8), track=TrackState.COASTING)
    tick(controller, 13.0, position=(6.0, 0.0, 1.8))
    tick(controller, 13.999, position=(6.0, 0.0, 1.8))
    assert controller.phase is Phase.HOLD
    tick(controller, 14.0, position=(6.0, 0.0, 1.8))
    assert controller.phase is Phase.COMMIT


def test_commit_continues_through_coasting_but_lost_enters_truth_fault_hold():
    controller = launch()
    commit(controller)
    command = tick(
        controller,
        10.1,
        position=(9.0, 0.0, 1.8),
        velocity=(0.5, 0.0, 0.0),
        track=TrackState.COASTING,
    )
    assert controller.phase is Phase.COMMIT
    assert not controller.braking_active
    assert command.pitch > 0
    command = tick(controller, 10.2, position=(9.1, 0.0, 1.8), track=TrackState.LOST)
    assert controller.phase is Phase.FAULT
    assert controller.failure_reason == "lost_track"
    assert command.armed and command.angle_mode


def test_commit_times_out_even_without_crossing():
    controller = launch()
    commit(controller)
    tick(controller, 17.999, position=(9.0, 0.0, 1.8))
    assert controller.phase is Phase.COMMIT
    tick(controller, 18.0, position=(9.0, 0.0, 1.8))
    assert controller.phase is Phase.FAULT
    assert controller.failure_reason == "commit_timeout"


def test_commit_has_its_own_speed_bound():
    slow = launch()
    slow.config = replace(slow.config, commit_speed_mps=0.2)
    fast = launch()
    for controller in (slow, fast):
        commit(controller)
    a = tick(slow, 10.1, position=(9.0, 0.0, 1.8), velocity=(0.3, 0.0, 0.0))
    b = tick(fast, 10.1, position=(9.0, 0.0, 1.8), velocity=(0.3, 0.0, 0.0))
    assert a.pitch < 0 < b.pitch


@pytest.mark.parametrize("passed", (0, 1, 2))
def test_premature_or_skipped_events_fault(passed):
    controller = launch()
    command = tick(controller, 8.0, passed=passed, position=(6.0, 0.0, 1.8))
    assert controller.phase is Phase.FAULT
    assert controller.failure_reason == "unexpected_pass_event"
    assert command.armed and command.angle_mode


def test_progress_reversal_and_uncommitted_crossing_fail_closed():
    controller = launch()
    commit(controller)
    tick(controller, 10.5, passed=0, position=(10.1, 0.0, 1.8))
    tick(controller, 10.6, passed=-1, position=(10.2, 0.0, 1.8))
    assert controller.failure_reason == "unexpected_pass_event"
    controller = launch()
    tick(controller, 8.0, position=(10.1, 0.0, 1.8))
    assert controller.failure_reason == "uncommitted_crossing"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"gate_count": 2}, "course_changed"),
        ({"next_gate_index": 2}, "invalid_progress"),
        ({"next_gate_index": 0.0}, "invalid_progress"),
        ({"last_gate_passed": -2}, "invalid_progress"),
        ({"last_gate_passed": 3, "next_gate_index": None}, "invalid_progress"),
        ({"last_gate_passed": True}, "invalid_progress"),
        ({"gate": GATES[1]}, "missing_gate"),
        ({"gate": None}, "missing_gate"),
    ],
)
def test_malformed_progress_and_missing_current_target_fault(changes, reason):
    controller = launch()
    arguments = {
        "sim_time": 8.0,
        "nav": NavState((6.0, 0.0, 1.8), (0.0, 0.0, 0.0), 0.0, 0.0),
        "gate": GATES[0],
        "last_gate_passed": -1,
        "next_gate_index": 0,
        "gate_count": 3,
    }
    arguments.update(changes)
    controller.update(**arguments)
    assert controller.phase is Phase.FAULT
    assert controller.failure_reason == reason


def test_missing_next_target_on_a_valid_pass_fails_closed():
    controller = launch()
    commit(controller)
    command = controller.update(
        10.5,
        NavState((10.1, 0.0, 1.8), (0.0, 0.0, 0.0), 0.0, 0.0),
        GATES[0],  # Old geometry must not be relabelled into the next target.
        last_gate_passed=0,
        next_gate_index=1,
        gate_count=3,
    )
    assert controller.phase is Phase.FAULT
    assert controller.failure_reason == "missing_gate"
    assert command.armed and command.angle_mode


def test_race_is_invariant_to_constant_navigation_origin_and_heading():
    original, transformed = (
        MissionController(TruthGuidanceConfig.for_course("c1_straight")) for _ in range(2)
    )
    angle, offset = 0.7, (23.0, -17.0, 4.0)

    def rotate(vector):
        cosine, sine = math.cos(angle), math.sin(angle)
        x, y, z = vector
        return (cosine * x - sine * y, sine * x + cosine * y, z)

    def translate(position):
        return tuple(a + b for a, b in zip(rotate(position), offset, strict=True))

    sequence = [(time, -1, (0.0, 0.0, 0.0)) for time in (0.0, 5.0, 5.6)]
    sequence += [(time, -1, (0.0, 0.0, 1.8)) for time in (6.0, 6.5, 7.0, 7.75)]
    for index, start in enumerate((8.0, 12.0, 16.0)):
        sequence += [
            (time, index - 1, (6.0 + 10 * index, 0.0, 1.8))
            for time in (start, start + 1.0, start + 1.999, start + 2.0)
        ]
        sequence.append((start + 2.5, index, (10.01 + 10 * index, 0.0, 1.8)))
    sequence += [
        (19.0, 2, (32.0, 0.0, 1.8)),
        (19.5, 2, (32.0, 0.0, 1.8)),
        (20.0, 2, (32.0, 0.0, 1.8)),
        (24.0, 2, (32.0, 0.0, 0.15)),
        (25.0, 2, (32.0, 0.0, 0.0)),
        (25.5, 2, (32.0, 0.0, 0.0)),
    ]
    for time, passed, position in sequence:
        target = GATES[passed + 1] if passed < 2 else None
        commands = []
        for controller, shifted in ((original, False), (transformed, True)):
            state = NavState(
                translate(position) if shifted else position,
                (0.0, 0.0, 0.0),
                angle if shifted else 0.0,
                0.0,
            )
            selected = (
                replace(target, center=translate(target.center), yaw=target.yaw + angle)
                if shifted and target is not None
                else target
            )
            commands.append(
                controller.update(
                    time,
                    state,
                    selected,
                    last_gate_passed=passed,
                    next_gate_index=passed + 1 if passed < 2 else None,
                    gate_count=3,
                )
            )
        first, second = commands
        assert (second.roll, second.pitch, second.throttle, second.yaw) == pytest.approx(
            (first.roll, first.pitch, first.throttle, first.yaw), abs=1e-12, rel=0
        )
        assert (second.armed, second.angle_mode) == (first.armed, first.angle_mode)
        assert transformed.phase is original.phase
        assert transformed.failure_reason == original.failure_reason
        assert transformed.braking_active == original.braking_active
    assert original.phase is transformed.phase is Phase.FINISH
    assert not first.armed and not second.armed


def test_pad_reference_uses_settled_prearm_nav_not_spawn_and_freezes_before_power():
    controller = MissionController(TruthGuidanceConfig.for_course("c1_straight"))
    tick(controller, 0.0, position=(0.0, 0.0, 0.1))
    tick(controller, 0.1, position=(0.0, 0.0, 0.05), velocity=(0.0, 0.0, -0.5))
    tick(controller, 0.2)
    assert controller._pad_altitude is None
    tick(controller, 0.69)
    assert controller._pad_altitude is None
    tick(controller, 0.7)
    assert controller._pad_altitude == 0.0
    arm = tick(controller, 5.0)
    assert arm.armed
    assert controller._pad_altitude == 0.0
    # Even while still in ARM settle, powered motion must not redefine the pad.
    tick(controller, 5.1, position=(0.0, 0.0, 0.2), velocity=(0.0, 0.0, 0.4))
    tick(controller, 5.6, position=(0.0, 0.0, 0.3))
    assert controller.phase is Phase.TAKEOFF
    assert controller._pad_altitude == 0.0
    assert controller._launch_position[2] == 0.1  # E's launch latch remains separate.


def test_unsettled_pad_delays_arming_and_times_out_disarmed():
    controller = MissionController(TruthGuidanceConfig.for_course("c1_straight"))
    for time in (0.0, 4.9, 5.0, 11.9):
        command = tick(controller, time, velocity=(0.0, 0.0, -0.1))
        assert command == SemanticControl.safe()
        assert controller.phase is Phase.ARM
    command = tick(controller, 12.0, velocity=(0.0, 0.0, -0.1))
    assert command == SemanticControl.safe()
    assert controller.phase is Phase.FAULT
    assert controller.failure_reason == "unsettled_pad"


def test_pad_settling_can_complete_after_boot_without_arming_on_a_moving_sample():
    controller = MissionController(TruthGuidanceConfig.for_course("c1_straight"))
    tick(controller, 0.0, position=(0.0, 0.0, 0.1))
    tick(controller, 4.7, velocity=(0.0, 0.0, -0.1))
    tick(controller, 4.8)
    assert tick(controller, 5.0) == SemanticControl.safe()
    assert controller._pad_altitude is None
    assert tick(controller, 5.3).armed
    assert controller._pad_altitude == 0.0


def test_finish_brakes_and_requires_continuous_settling_before_powered_descent():
    controller = finish()
    tick(controller, 19.0, passed=2, position=(32.0, 0.0, 1.8), velocity=(0.3, 0.0, 0.0))
    assert controller._descent_started_at is None
    tick(controller, 19.5, passed=2, position=(32.0, 0.0, 1.8))
    tick(controller, 19.9, passed=2, position=(32.0, 0.0, 1.8), velocity=(0.3, 0.0, 0.0))
    tick(controller, 20.0, passed=2, position=(32.0, 0.0, 1.8))
    tick(controller, 20.499, passed=2, position=(32.0, 0.0, 1.8))
    assert controller._descent_started_at is None
    tick(controller, 20.5, passed=2, position=(32.0, 0.0, 1.8))
    assert controller._descent_started_at == 20.5
    assert controller._descent_start_altitude == 1.8
    assert not controller.braking_active  # Old gate slab cannot interfere behind it.


def test_finish_ramp_is_0_4mps_and_bounded_below_the_captured_pad(monkeypatch):
    controller = finish()
    references = []
    real_control = controller._position_control

    def spy(state, target, altitude, yaw):
        references.append(altitude)
        return real_control(state, target, altitude, yaw)

    monkeypatch.setattr(controller, "_position_control", spy)
    for time in (19.0, 19.5, 20.0, 22.0, 24.75, 30.0):
        command = tick(controller, time, passed=2, position=(32.0, 0.0, 1.8))
        assert command.armed and command.angle_mode
    assert references == pytest.approx([1.8, 1.8, 1.6, 0.8, -0.3, -0.3])
    assert controller._pad_altitude == 0.0


def test_suspended_0_15_never_disarms_and_near_pad_speed_height_interruptions_reset_dwell():
    controller = finish()
    for time in (19.0, 19.5):
        tick(controller, time, passed=2, position=(32.0, 0.0, 1.8))
    for time in (24.0, 25.0):
        assert tick(controller, time, passed=2, position=(32.0, 0.0, 0.15)).armed
        assert controller._dwell_since is None
    assert tick(
        controller, 25.1, passed=2, position=(32.0, 0.0, 0.01), velocity=(0.0, 0.0, -0.19)
    ).armed
    assert controller._dwell_since is None
    tick(controller, 25.2, passed=2, position=(32.0, 0.0, 0.01))
    tick(controller, 25.6, passed=2, position=(32.0, 0.0, 0.02))
    assert controller._dwell_since is None
    tick(controller, 25.7, passed=2, position=(32.0, 0.0, 0.01))
    tick(controller, 26.1, passed=2, position=(32.0, 0.0, 0.01), velocity=(0.19, 0.0, 0.0))
    assert controller._dwell_since is None
    tick(controller, 26.2, passed=2, position=(32.0, 0.0, 0.01))
    assert tick(controller, 26.699, passed=2, position=(32.0, 0.0, 0.01)).armed
    assert tick(controller, 26.7, passed=2, position=(32.0, 0.0, 0.01)) == SemanticControl.safe()
    # No later height, malformed feedback or time error may restore arming.
    assert tick(controller, 27.0, passed=2, position=(32.0, 0.0, 0.15)) == SemanticControl.safe()
    assert controller.reject_update(27.1, "invalid_truth", None) == SemanticControl.safe()
    assert controller.phase is Phase.FAULT
    assert tick(controller, 27.2, passed=2) == SemanticControl.safe()
    assert controller.reject_update(26.0, "invalid_truth", None) == SemanticControl.safe()


def test_race_adapter_selects_next_index_not_pose_array_order():
    from controls import GatePose

    poses = tuple(GatePose(g.index, g.center, g.yaw, g.inner_size) for g in reversed(GATES))
    source = TruthGuidance(TruthGuidanceConfig.for_course("c1_straight"))
    for time in (0.0, 5.0, 5.6, 6.0, 6.5, 7.0, 7.75, 8.0, 9.0, 10.0):
        position = (0.0, 0.0, 0.0) if time <= 5.6 else (0.0, 0.0, 1.8)
        if time >= 8.0:
            position = (6.0, 0.0, 1.8)
        source.update(_update(time, gates=poses, position=position))
    assert source.phase is Phase.COMMIT
    source.update(
        _update(
            10.5,
            gates=poses,
            position=(10.01, 0.0, 1.8),
            last_gate_passed=0,
            next_gate_index=1,
        )
    )
    assert source.phase is Phase.ACQUIRE
    assert source.target_enu == (16.0, 0.0, 1.8)
    assert source.controller.gate_target.track is TrackState.TRACKING
    assert source.controller.gate_target.age == 0.0


@pytest.mark.parametrize(
    "field",
    (
        "race_hold_dwell_s",
        "race_alignment_yaw_rad",
        "commit_timeout_s",
        "commit_exit_distance_m",
        "commit_speed_mps",
        "missed_gate_distance_m",
        "pad_settle_s",
        "pad_velocity_tolerance_mps",
        "finish_settle_s",
        "landing_rate_mps",
        "landing_reference_depth_m",
        "landing_altitude_tolerance_m",
        "landing_speed_mps",
        "landing_dwell_s",
    ),
)
@pytest.mark.parametrize("invalid", (0, -1, float("nan"), float("inf"), True))
def test_race_configuration_rejects_invalid_positive_limits(field, invalid):
    with pytest.raises(ValueError, match=field):
        replace(TruthGuidanceConfig.for_course("c1_straight"), **{field: invalid})


def test_missing_nav_before_launch_is_safe():
    controller = MissionController(TruthGuidanceConfig.for_course("c1_straight"))
    assert controller.update(0.0, None, GATES[0], gate_count=3) == SemanticControl.safe()
    assert controller.phase is Phase.FAULT
