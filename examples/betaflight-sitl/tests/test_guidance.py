"""Deterministic tests for the pure truth guidance state machine/controller."""

import math

import numpy as np
import pytest
from controls import GatePose, SemanticControl, TruthGuidanceUpdate, TruthState, semantic_to_rc
from course import course_from_name
from guidance import (
    GUIDANCE_PHASE_NAMES,
    GuidancePhase,
    TruthGuidance,
    TruthGuidanceConfig,
    wrap_to_pi,
)


def _gate(center=(10.0, 0.0, 1.8), yaw=0.0, index=0):
    return GatePose(index, center, yaw, 2.5)


def _update(
    sim_time,
    *,
    position=(0.0, 0.0, 0.1),
    velocity=(0.0, 0.0, 0.0),
    yaw=0.0,
    yaw_rate=0.0,
    gate=None,
    gates=None,
    next_gate_index=0,
    last_gate_passed=-1,
    truth=True,
    gyro_shape=(3,),
):
    q = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    poses = tuple(gates) if gates is not None else ((gate or _gate()),)
    return TruthGuidanceUpdate(
        sim_time=sim_time,
        tick=round(sim_time * 8_000),
        gyro=np.array([0.0, 0.0, yaw_rate]) if gyro_shape == (3,) else np.zeros(2),
        accel=np.zeros(3),
        truth_state=TruthState(position, velocity, q) if truth else None,
        gate_poses=poses,
        next_gate_index=next_gate_index,
        last_gate_passed=last_gate_passed,
        gate_count=len(poses),
    )


def _start_takeoff(source, *, position=(0.0, 0.0, 0.1), gate=None):
    source.update(_update(0.0, position=position, gate=gate))
    assert source.phase is GuidancePhase.ARM
    source.update(_update(5.0, position=position, gate=gate))
    assert source.phase is GuidancePhase.ARM
    source.update(_update(5.61, position=position, gate=gate))
    assert source.phase is GuidancePhase.TAKEOFF


def _reach_acquire(source):
    # Hover above the launch XY, at gate altitude and stationary.
    for t in (6.0, 6.5):
        source.update(_update(t, position=(0.0, 0.0, 1.8)))
    assert source.phase is GuidancePhase.HOVER
    for t in (7.0, 7.75):
        source.update(_update(t, position=(0.0, 0.0, 1.8)))
    assert source.phase is GuidancePhase.ACQUIRE


def test_numeric_phase_codes_and_single_gate_standoff_target() -> None:
    source = TruthGuidance()
    source.update(_update(0.0))

    assert source.phase_code == 1
    assert source.target_enu == pytest.approx((6.0, 0.0, 1.8))
    assert int(GuidancePhase.ARM) == 1
    assert int(GuidancePhase.TAKEOFF) == 2
    assert int(GuidancePhase.HOVER) == 3
    assert int(GuidancePhase.ACQUIRE) == 4
    assert int(GuidancePhase.HOLD) == 5
    assert dict(GUIDANCE_PHASE_NAMES) == {
        0: "DISABLED",
        1: "ARM",
        2: "TAKEOFF",
        3: "HOVER",
        4: "ACQUIRE",
        5: "HOLD",
        6: "FAULT",
    }


def test_standoff_target_uses_arbitrary_gate_yaw_and_config_distance() -> None:
    source = TruthGuidance(TruthGuidanceConfig(standoff_distance_m=3.0))
    gate = _gate(center=(4.0, -3.0, 2.5), yaw=math.pi / 2)
    source.update(_update(0.0, position=(0.0, -8.0, 0.1), gate=gate))

    assert source.target_enu == pytest.approx((4.0, -6.0, 2.5))


def test_arm_has_five_simulated_seconds_disarmed_then_angle_arm_settle() -> None:
    source = TruthGuidance()
    before = source.update(_update(4.999))
    assert before == SemanticControl.safe()
    assert source.phase is GuidancePhase.ARM

    arm = source.update(_update(5.0))
    assert source.phase is GuidancePhase.ARM
    assert arm.armed is True
    assert arm.angle_mode is True
    assert arm.throttle == 0.0

    settling = source.update(_update(5.599))
    assert source.phase is GuidancePhase.ARM
    assert settling.throttle == 0.0

    takeoff = source.update(_update(5.6))
    assert source.phase is GuidancePhase.TAKEOFF
    assert takeoff.armed and takeoff.angle_mode


def test_full_phase_sequence_uses_capture_dwell_and_never_commits() -> None:
    source = TruthGuidance()
    _start_takeoff(source)
    _reach_acquire(source)

    source.update(_update(8.0, position=(6.0, 0.0, 1.8)))
    assert source.phase is GuidancePhase.ACQUIRE
    source.update(_update(8.99, position=(6.0, 0.0, 1.8)))
    assert source.phase is GuidancePhase.ACQUIRE
    source.update(_update(9.0, position=(6.0, 0.0, 1.8)))
    assert source.phase is GuidancePhase.HOLD
    command = source.update(_update(9.1, position=(6.0, 0.0, 1.8)))

    assert source.phase is GuidancePhase.HOLD
    assert command.armed and command.angle_mode
    assert command.roll == pytest.approx(0.0)
    assert command.pitch == pytest.approx(0.0)
    assert not hasattr(GuidancePhase, "COMMIT")


def test_positive_forward_and_left_acceleration_use_measured_stick_signs() -> None:
    pitch_source = TruthGuidance()
    _start_takeoff(pitch_source)
    pitch = pitch_source.update(_update(5.8, position=(-1.0, 0.0, 0.1)))
    assert pitch.pitch > 0.0  # Semantic positive pitch moves forward/nose-down.

    roll_source = TruthGuidance()
    _start_takeoff(roll_source)
    roll = roll_source.update(_update(5.8, position=(0.0, -1.0, 0.1)))
    assert roll.roll < 0.0  # World/body-left acceleration requires negative roll.

    reverse_pitch_source = TruthGuidance()
    _start_takeoff(reverse_pitch_source)
    reverse_pitch = reverse_pitch_source.update(_update(5.8, position=(1.0, 0.0, 0.1)))
    assert reverse_pitch.pitch < 0.0

    reverse_roll_source = TruthGuidance()
    _start_takeoff(reverse_roll_source)
    reverse_roll = reverse_roll_source.update(_update(5.8, position=(0.0, 1.0, 0.1)))
    assert reverse_roll.roll > 0.0


def test_heading_frame_rotation_preserves_world_acceleration_direction() -> None:
    source = TruthGuidance()
    gate = _gate(center=(10.0, 10.0, 1.8), yaw=math.pi / 2)
    _start_takeoff(source, position=(0.0, 0.0, 0.1), gate=gate)
    # A vehicle facing north must pitch forward to move north toward launch Y=0
    # from Y=-1; world +Y acceleration is body-forward at this heading.
    command = source.update(_update(5.8, position=(0.0, -1.0, 0.1), yaw=math.pi / 2, gate=gate))
    assert command.pitch > 0.0
    assert command.roll == pytest.approx(0.0, abs=1e-12)


def test_attitude_throttle_and_rc_outputs_are_saturated() -> None:
    source = TruthGuidance(TruthGuidanceConfig(max_horizontal_stick=0.4))
    _start_takeoff(source)
    command = source.update(
        _update(5.8, position=(-100.0, -100.0, -100.0), velocity=(-30.0, 30.0, -20.0))
    )

    assert abs(command.pitch) <= 0.4
    assert abs(command.roll) <= 0.4
    assert 0.10 <= command.throttle <= 0.32
    assert abs(command.yaw) <= 0.15
    assert semantic_to_rc(command).as_array().tolist()[0:4] == pytest.approx(
        [1500 + round(command.roll * 500), 1500 + round(command.pitch * 500),
         1000 + round(command.throttle * 1000), 1500 - round(command.yaw * 500)]
    )


def test_yaw_error_wraps_at_pi_and_semantic_sign_is_clockwise() -> None:
    assert wrap_to_pi(-math.pi + 0.02 - (math.pi - 0.02)) == pytest.approx(0.04)
    source = TruthGuidance()
    gate = _gate(yaw=-math.pi + 0.01)
    _start_takeoff(source, gate=gate)
    command = source.update(_update(5.8, yaw=math.pi - 0.01, gate=gate))

    assert command.yaw < 0.0
    assert abs(command.yaw) < 0.02


def test_altitude_pd_increases_throttle_for_low_altitude_and_downward_velocity() -> None:
    source = TruthGuidance()
    _start_takeoff(source)
    rising = source.update(_update(5.8, position=(0.0, 0.0, 0.1), velocity=(0.0, 0.0, -0.2)))
    descending = source.update(_update(5.9, position=(0.0, 0.0, 0.3), velocity=(0.0, 0.0, 0.2)))

    assert rising.throttle > source.config.hover_rc_trim
    assert descending.throttle < source.config.hover_rc_trim


def test_gate_braking_barrier_never_requests_forward_acceleration_near_plane() -> None:
    source = TruthGuidance()
    _start_takeoff(source, position=(8.5, 0.0, 0.1))
    command = source.update(_update(5.8, position=(8.5, 0.0, 0.1), velocity=(0.5, 0.0, 0.0)))

    assert source.braking_active is True
    assert command.pitch < 0.0


def test_capture_timeout_fault_keeps_angle_armed_safe_hold() -> None:
    config = TruthGuidanceConfig(acquire_timeout_s=0.1)
    source = TruthGuidance(config)
    _start_takeoff(source)
    _reach_acquire(source)
    source.update(_update(8.0, position=(2.0, 0.0, 1.8)))
    control = source.update(_update(8.1, position=(2.0, 0.0, 1.8)))

    assert source.phase is GuidancePhase.FAULT
    assert source.failure_reason == "acquire_timeout"
    assert control.armed and control.angle_mode
    assert 0.10 <= control.throttle <= 0.32


def test_invalid_truth_and_nonmonotonic_time_fail_closed() -> None:
    source = TruthGuidance()
    safe = source.update(_update(0.0, truth=False))
    assert source.phase is GuidancePhase.FAULT
    assert safe == SemanticControl.safe()

    source = TruthGuidance()
    _start_takeoff(source)
    invalid = source.update(_update(5.7, truth=False))
    assert source.phase is GuidancePhase.FAULT
    assert invalid.armed and invalid.angle_mode
    assert 0.10 <= invalid.throttle <= 0.32

    nonmonotonic = TruthGuidance()
    nonmonotonic.update(_update(1.0))
    nonmonotonic.update(_update(0.5))
    assert nonmonotonic.phase is GuidancePhase.FAULT
    assert nonmonotonic.failure_reason == "nonmonotonic_sim_time"


def test_invalid_sensor_shape_faults_and_target_never_retargets_on_progress() -> None:
    bad = TruthGuidance()
    bad.update(_update(0.0, gyro_shape=(2,)))
    assert bad.phase is GuidancePhase.FAULT
    assert bad.failure_reason == "invalid_truth"

    source = TruthGuidance()
    first = _gate()
    later = _gate(center=(20.0, 0.0, 1.8), index=1)
    source.update(_update(0.0, gates=(first, later), next_gate_index=0))
    source.update(_update(1.0, gates=(first, later), next_gate_index=1))
    assert source.target_enu == pytest.approx((6.0, 0.0, 1.8))
    assert source.phase is GuidancePhase.ARM


def test_phase_timeouts_fault_safely_in_takeoff_hover_and_hold() -> None:
    takeoff = TruthGuidance(TruthGuidanceConfig(takeoff_timeout_s=0.1))
    _start_takeoff(takeoff)
    takeoff_command = takeoff.update(_update(5.711))
    assert takeoff.phase is GuidancePhase.FAULT
    assert takeoff.failure_reason == "takeoff_timeout"
    assert takeoff_command.armed and takeoff_command.angle_mode

    hover = TruthGuidance(TruthGuidanceConfig(hover_timeout_s=0.1))
    _start_takeoff(hover)
    hover.update(_update(6.0, position=(0.0, 0.0, 1.8)))
    hover.update(_update(6.5, position=(0.0, 0.0, 1.8)))
    assert hover.phase is GuidancePhase.HOVER
    hover_command = hover.update(_update(6.61, position=(0.0, 0.0, 1.8)))
    assert hover.phase is GuidancePhase.FAULT
    assert hover.failure_reason == "hover_timeout"
    assert hover_command.armed and hover_command.angle_mode

    hold = TruthGuidance(TruthGuidanceConfig(hold_timeout_s=0.1))
    _start_takeoff(hold)
    _reach_acquire(hold)
    for t in (8.0, 8.99, 9.0):
        hold.update(_update(t, position=(6.0, 0.0, 1.8)))
    assert hold.phase is GuidancePhase.HOLD
    hold_command = hold.update(_update(9.11, position=(6.0, 0.0, 1.8)))
    assert hold.phase is GuidancePhase.FAULT
    assert hold.failure_reason == "hold_timeout"
    assert hold_command.armed and hold_command.angle_mode


def test_arm_timeout_is_a_safe_fault() -> None:
    source = TruthGuidance(TruthGuidanceConfig(arm_timeout_s=5.1))
    source.update(_update(0.0))
    command = source.update(_update(5.1))

    assert source.phase is GuidancePhase.FAULT
    assert source.failure_reason == "arm_timeout"
    assert command == SemanticControl.safe()


def test_course_single_gate_matches_immutable_guidance_gate_value() -> None:
    gate = course_from_name("single").gates[0]
    pose = GatePose(gate.index, gate.center, gate.yaw, gate.inner_size)
    source = TruthGuidance()
    source.update(_update(0.0, gate=pose))

    assert source.target_enu == pytest.approx((6.0, 0.0, 1.8))
