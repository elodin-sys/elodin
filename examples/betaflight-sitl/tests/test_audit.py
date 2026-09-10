"""Tests for deterministic Package D audit guidance and physical assessment."""

import numpy as np

from audit import AuditGuidance, AxisAudit
from controls import GuidanceUpdate, RcCommand, SemanticControl


def _update(sim_time: float, tick: int = 0) -> GuidanceUpdate:
    return GuidanceUpdate(
        sim_time=sim_time,
        tick=tick,
        gyro=np.zeros(3),
        accel=np.zeros(3),
    )


def test_audit_guidance_phase_boundaries_follow_simulation_time() -> None:
    source = AuditGuidance()
    safe = SemanticControl.safe()
    armed = SemanticControl(armed=True)
    flying = SemanticControl(throttle=0.15, armed=True)
    expected = [
        (0.0, safe, "boot"),
        (4.999, safe, "boot"),
        (5.0, armed, "arm"),
        (6.0, flying, "throttle"),
        (8.0, SemanticControl(roll=0.25, throttle=0.15, armed=True), "roll-right"),
        (9.0, SemanticControl(roll=-0.25, throttle=0.15, armed=True), "roll-left"),
        (10.0, flying, "center"),
        (
            11.0,
            SemanticControl(pitch=0.25, throttle=0.15, armed=True),
            "pitch-forward",
        ),
        (
            12.0,
            SemanticControl(pitch=-0.25, throttle=0.15, armed=True),
            "pitch-back",
        ),
        (13.0, flying, "center"),
        (14.0, SemanticControl(yaw=0.25, throttle=0.15, armed=True), "yaw-right"),
        (15.0, SemanticControl(yaw=-0.25, throttle=0.15, armed=True), "yaw-left"),
        (16.0, flying, "center"),
        (17.0, SemanticControl(throttle=0.30, armed=True), "high-throttle"),
        (18.0, armed, "settle"),
        (20.0, safe, "safe"),
        (24.0, safe, "safe"),
    ]

    # Every update deliberately has the same tick: only sim_time may select a phase.
    for sim_time, control, phase in expected:
        assert source.update(_update(sim_time, tick=123_456)) == control
        assert source.phase == phase


def test_audit_guidance_exercises_every_axis_while_armed_in_angle_mode() -> None:
    source = AuditGuidance()
    samples = [source.update(_update(t)) for t in (8.5, 9.5, 11.5, 12.5, 14.5, 15.5, 17.5)]

    assert all(sample.armed and sample.angle_mode for sample in samples)
    assert [sample.roll for sample in samples[:2]] == [0.25, -0.25]
    assert [sample.pitch for sample in samples[2:4]] == [0.25, -0.25]
    assert [sample.yaw for sample in samples[4:6]] == [0.25, -0.25]
    assert samples[6].throttle == 0.30


def test_axis_audit_requires_all_axes_angle_and_throttle_response() -> None:
    audit = AxisAudit()
    motors = np.array([0.2, 0.2, 0.2, 0.2])

    audit.observe(RcCommand(roll=1650, throttle=1300), np.array([0.3, 0.0, 0.0]), motors)
    audit.observe(RcCommand(pitch=1650, throttle=1300), np.array([0.0, 0.4, 0.0]), motors)
    audit.observe(RcCommand(yaw=1350, throttle=1300), np.array([0.0, 0.0, -0.5]), motors)

    assert audit.passed is True
    assert audit.format() == (
        "[D-AUDIT] angle=true roll_right_rad_s=0.300 pitch_forward_rad_s=0.400 "
        "yaw_right_rad_s=0.500 max_motor=0.200 status=PASS"
    )


def test_axis_audit_rejects_opposite_physical_signs() -> None:
    audit = AxisAudit()
    motors = np.full(4, 0.2)
    audit.observe(RcCommand(roll=1650, throttle=1300), np.array([-0.3, 0.0, 0.0]), motors)
    audit.observe(RcCommand(pitch=1650, throttle=1300), np.array([0.0, -0.4, 0.0]), motors)
    audit.observe(RcCommand(yaw=1350, throttle=1300), np.array([0.0, 0.0, 0.5]), motors)

    assert audit.passed is False
