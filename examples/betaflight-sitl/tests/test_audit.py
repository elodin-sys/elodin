"""Tests for deterministic Package D physical-audit assessment."""

import numpy as np

from audit import AxisAudit
from controls import RcCommand


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
