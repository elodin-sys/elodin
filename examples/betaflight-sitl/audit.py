"""Opt-in physical response audit for Package D.

The manual controller's ``--audit`` sequence injects bounded commands. This
collector observes the command actually exchanged with Betaflight and the
resulting physical body rates/motor output; it never participates in control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from controls import RC_ANGLE, RC_CENTER, RcCommand


@dataclass(slots=True)
class AxisAudit:
    max_roll_right_rate: float = float("-inf")
    max_pitch_forward_rate: float = float("-inf")
    max_yaw_right_rate: float = float("-inf")
    max_throttle_motor: float = 0.0
    angle_requested: bool = False

    def observe(
        self,
        command: RcCommand,
        gyro_flu: NDArray[np.float64],
        motors: NDArray[np.float64],
    ) -> None:
        """Observe one exchange using measured FLU physical sign conventions."""

        gyro = np.asarray(gyro_flu)
        motors = np.asarray(motors)
        self.angle_requested |= command.mode >= RC_ANGLE
        if command.roll > RC_CENTER + 50:
            # +X FLU rotation is right-wing-down roll.
            self.max_roll_right_rate = max(self.max_roll_right_rate, float(gyro[0]))
        if command.pitch > RC_CENTER + 50:
            # +Y FLU rotation pitches the body +X axis down.
            self.max_pitch_forward_rate = max(self.max_pitch_forward_rate, float(gyro[1]))
        if command.yaw < RC_CENTER - 50:
            # A right/clockwise yaw is rotation about body -Z in FLU. The
            # measured Betaflight SITL RC mapping requires PWM below center.
            self.max_yaw_right_rate = max(self.max_yaw_right_rate, float(-gyro[2]))
        if command.throttle > 1200:
            self.max_throttle_motor = max(self.max_throttle_motor, float(np.max(motors)))

    @property
    def passed(self) -> bool:
        return (
            self.angle_requested
            and self.max_roll_right_rate > 0.05
            and self.max_pitch_forward_rate > 0.05
            and self.max_yaw_right_rate > 0.05
            and self.max_throttle_motor > 0.06
        )

    def format(self) -> str:
        def finite(value: float) -> float:
            return value if np.isfinite(value) else 0.0

        return (
            f"[D-AUDIT] angle={str(self.angle_requested).lower()} "
            f"roll_right_rad_s={finite(self.max_roll_right_rate):.3f} "
            f"pitch_forward_rad_s={finite(self.max_pitch_forward_rate):.3f} "
            f"yaw_right_rad_s={finite(self.max_yaw_right_rate):.3f} "
            f"max_motor={self.max_throttle_motor:.3f} "
            f"status={'PASS' if self.passed else 'FAIL'}"
        )
