"""Pass/fail assessment for the default scripted SITL scenario."""

from dataclasses import dataclass


MIN_MOTOR_RESPONSE = 0.06
MIN_TAKEOFF_DELTA_M = 0.1


@dataclass(frozen=True)
class C0Result:
    """Measured outcome of the default scripted SITL integration run."""

    lockstep_steps: int
    max_motor: float
    takeoff_delta_m: float

    @property
    def motor_response(self) -> bool:
        return self.max_motor > MIN_MOTOR_RESPONSE

    @property
    def passed(self) -> bool:
        return (
            self.lockstep_steps > 0
            and self.motor_response
            and self.takeoff_delta_m >= MIN_TAKEOFF_DELTA_M
        )

    def format(self) -> str:
        """Return the stable, machine-readable C0 result line."""
        motor_response = str(self.motor_response).lower()
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[C0] lockstep_steps={self.lockstep_steps} "
            f"motor_response={motor_response} "
            f"max_motor={self.max_motor:.3f} "
            f"takeoff_delta_m={self.takeoff_delta_m:.3f} "
            f"status={status}"
        )


def evaluate_c0(
    *, lockstep_steps: int, max_motor: float, initial_altitude: float, max_altitude: float
) -> C0Result:
    """Evaluate the Package A criteria from observed integration metrics."""
    return C0Result(
        lockstep_steps=lockstep_steps,
        max_motor=max_motor,
        takeoff_delta_m=max_altitude - initial_altitude,
    )
