"""RocketPy-inspired motor model wrapping the OpenRocket motor implementation."""

from __future__ import annotations

from collections.abc import Sequence

from openrocket_motor import Motor as OpenRocketMotor


class Motor:
    """Thin wrapper around :class:`openrocket_motor.Motor` with RocketPy naming."""

    def __init__(self, motor: OpenRocketMotor) -> None:
        self._motor = motor

    @classmethod
    def from_openrocket(cls, motor: OpenRocketMotor) -> "Motor":
        return cls(motor)

    # ------------------------------------------------------------------
    # RocketPy-style accessors
    # ------------------------------------------------------------------
    def thrust(self, time: float) -> float:
        return float(self._motor.get_thrust(time))

    def mass(self, time: float) -> float:
        return float(self._motor.get_mass(time))

    def cg(self, time: float) -> float:
        return float(self._motor.get_cg(time))

    def inertia(self, time: float) -> tuple[float, float, float]:
        return tuple(float(v) for v in self._motor.get_inertia(time))

    @property
    def burn_time(self) -> float:
        return float(self._motor.burn_time)

    @property
    def propellant_mass(self) -> float:
        return float(self._motor.propellant_mass)

    @property
    def dry_mass(self) -> float:
        return float(self._motor.case_mass)

    @property
    def case_mass(self) -> float:
        return float(self._motor.case_mass)

    @property
    def total_mass(self) -> float:
        return float(self._motor.total_mass_initial)

    @property
    def total_mass_initial(self) -> float:
        return self.total_mass

    @property
    def length(self) -> float:
        return float(self._motor.length)

    @property
    def diameter(self) -> float:
        return float(self._motor.diameter)

    @property
    def nozzle_radius(self) -> float:
        # The OpenRocket motor model does not expose a dedicated nozzle radius.
        # RocketPy expects this property, so default to half the motor diameter.
        return float(getattr(self._motor, "nozzle_radius", self._motor.diameter / 2.0))

    # ------------------------------------------------------------------
    # Convenience properties for existing OpenRocket data tables
    # ------------------------------------------------------------------
    @property
    def times(self) -> Sequence[float]:
        return tuple(float(t) for t in self._motor.times)

    @property
    def thrusts(self) -> Sequence[float]:
        return tuple(float(v) for v in self._motor.thrusts)

    def get_thrust(self, time: float) -> float:
        return self.thrust(time)

    def get_mass(self, time: float) -> float:
        return self.mass(time)

    def get_cg(self, time: float) -> float:
        return self.cg(time)

    def get_inertia(self, time: float) -> tuple[float, float, float]:
        return self.inertia(time)

    @property
    def openrocket(self) -> OpenRocketMotor:
        return self._motor
