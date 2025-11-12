"""Dynamic wind model inspired by RocketPy's implementation.

This module provides a deterministic wind field that captures:
- Altitude dependent mean wind profile (power law or custom profile points)
- Directional shear with altitude
- Deterministic pseudo-gusts along, cross and vertical components using
  sinusoidal harmonics (similar in spirit to RocketPy's Dryden-based gust model)

The model intentionally avoids random numbers inside the simulation loop so that
it can be integrated with JAX/Elodin without breaking purity assumptions.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ProfilePoint:
    """Single profile point describing wind state at a given altitude."""

    altitude: float              # metres
    speed: float                 # m/s
    direction_rad: float         # radians, 0 = north, pi/2 = east
    vertical: float = 0.0        # optional up/downdraft (m/s)


def _normalise_profile_point(point: Tuple[float, float, float] | dict | ProfilePoint) -> ProfilePoint:
    """Convert incoming tuple/dict/ProfilePoint into ProfilePoint."""

    if isinstance(point, ProfilePoint):
        return point
    if isinstance(point, dict):
        altitude = float(point.get("altitude", 0.0))
        speed = float(point.get("speed", 0.0))
        direction = point.get("direction", point.get("direction_deg", 0.0))
        if "direction_deg" in point and "direction" not in point:
            direction = math.radians(float(point["direction_deg"]))
        else:
            direction = float(direction)
        vertical = float(point.get("vertical", 0.0))
        return ProfilePoint(altitude=altitude, speed=speed, direction_rad=direction, vertical=vertical)
    if isinstance(point, (list, tuple)):
        if len(point) == 3:
            altitude, speed, direction_deg = point
            return ProfilePoint(float(altitude), float(speed), math.radians(float(direction_deg)))
        if len(point) == 4:
            altitude, speed, direction_deg, vertical = point
            return ProfilePoint(float(altitude), float(speed), math.radians(float(direction_deg)), float(vertical))
    raise TypeError(f"Unsupported profile point format: {point!r}")


class DynamicWindModel:
    """Deterministic dynamic wind model with altitude shear and gusts."""

    def __init__(
        self,
        profile_points: Iterable[Tuple[float, float, float]] | Iterable[dict] | None = None,
        surface_speed: float = 3.0,
        surface_direction_deg: float = 90.0,
        shear_exponent: float = 0.143,
        reference_height: float = 10.0,
        gust_along: Sequence[Tuple[float, float]] | None = None,
        gust_cross: Sequence[Tuple[float, float]] | None = None,
        gust_vertical: Sequence[Tuple[float, float]] | None = None,
        gust_decay_scale: float = 4500.0,
        turbulence_intensity: float = 0.15,
        seed: int = 1,
    ) -> None:
        """
        Args:
            profile_points: Optional iterable of points (altitude [m], speed [m/s],
                direction [deg]) sorted by altitude. If omitted, a power-law wind
                profile is used.
            surface_speed: Reference wind speed at the reference_height.
            surface_direction_deg: Direction (deg from north, clockwise).
            shear_exponent: Power-law exponent (1/7 by default).
            reference_height: Reference height for power law (m).
            gust_along / gust_cross / gust_vertical: Sequences of (amplitude [m/s],
                frequency [Hz]) tuples describing deterministic gust harmonics for
                along-wind, cross-wind and vertical components.
            gust_decay_scale: Exponential decay scale (m) applied to gust amplitude
                with altitude (mimics decreasing turbulence aloft).
            turbulence_intensity: Scalar multiplier applied to gust amplitudes.
            seed: RNG seed used to pick deterministic phases for each harmonic.
        """

        self.reference_height = max(reference_height, 1.0)
        self.surface_speed = float(surface_speed)
        self.surface_direction = math.radians(surface_direction_deg)
        self.shear_exponent = shear_exponent
        self.gust_decay_scale = max(gust_decay_scale, 1.0)
        self.turbulence_intensity = max(turbulence_intensity, 0.0)

        # Process profile points (if provided)
        self._profile_alts: List[float] = []
        self._profile_north: List[float] = []
        self._profile_east: List[float] = []
        self._profile_up: List[float] = []
        if profile_points:
            points = sorted((_normalise_profile_point(p) for p in profile_points), key=lambda p: p.altitude)
            for point in points:
                self._profile_alts.append(point.altitude)
                self._profile_north.append(point.speed * math.cos(point.direction_rad))
                self._profile_east.append(point.speed * math.sin(point.direction_rad))
                self._profile_up.append(point.vertical)

        rng = random.Random(seed)

        def _prepare(series: Sequence[Tuple[float, float]] | None):
            prepared: List[Tuple[float, float, float]] = []
            if not series:
                return prepared
            for amplitude, frequency in series:
                if amplitude == 0.0 or frequency == 0.0:
                    continue
                phase = rng.uniform(0.0, 2.0 * math.pi)
                prepared.append((float(amplitude), float(frequency), phase))
            return prepared

        self._gust_along = _prepare(gust_along)
        self._gust_cross = _prepare(gust_cross)
        self._gust_vertical = _prepare(gust_vertical)

    # ------------------------------------------------------------------
    # Mean wind profile helpers
    # ------------------------------------------------------------------

    def _mean_components(self, altitude: float) -> Tuple[float, float, float]:
        altitude = max(0.0, altitude)
        if self._profile_alts:
            if altitude <= self._profile_alts[0]:
                return (
                    self._profile_north[0],
                    self._profile_east[0],
                    self._profile_up[0],
                )
            if altitude >= self._profile_alts[-1]:
                return (
                    self._profile_north[-1],
                    self._profile_east[-1],
                    self._profile_up[-1],
                )
            idx = bisect.bisect_right(self._profile_alts, altitude) - 1
            next_idx = idx + 1
            span = self._profile_alts[next_idx] - self._profile_alts[idx]
            frac = (altitude - self._profile_alts[idx]) / span if span > 1e-6 else 0.0
            north = self._profile_north[idx] + frac * (self._profile_north[next_idx] - self._profile_north[idx])
            east = self._profile_east[idx] + frac * (self._profile_east[next_idx] - self._profile_east[idx])
            up = self._profile_up[idx] + frac * (self._profile_up[next_idx] - self._profile_up[idx])
            return north, east, up

        # Power law profile (1/7 rule by default)
        ref = self.reference_height
        scaled_alt = (altitude + ref) / ref
        speed = self.surface_speed * scaled_alt ** self.shear_exponent
        north = speed * math.cos(self.surface_direction)
        east = speed * math.sin(self.surface_direction)
        return north, east, 0.0

    # ------------------------------------------------------------------
    # Gust helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _harmonic_sum(series: Sequence[Tuple[float, float, float]], time: float) -> float:
        if not series:
            return 0.0
        total = 0.0
        two_pi = 2.0 * math.pi
        for amplitude, frequency, phase in series:
            total += amplitude * math.sin(two_pi * frequency * time + phase)
        return total

    def _gust_scale(self, altitude: float) -> float:
        return math.exp(-max(altitude, 0.0) / self.gust_decay_scale) * self.turbulence_intensity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_wind(self, altitude: float, time: float) -> Tuple[float, float, float]:
        """Compute wind vector (north, east, up) at a given altitude and time."""

        north_mean, east_mean, up_mean = self._mean_components(altitude)
        base_speed = math.hypot(north_mean, east_mean)

        if base_speed > 1e-6:
            along_unit_n = north_mean / base_speed
            along_unit_e = east_mean / base_speed
        else:
            along_unit_n = math.cos(self.surface_direction)
            along_unit_e = math.sin(self.surface_direction)
            base_speed = max(base_speed, 1e-6)

        cross_unit_n = -along_unit_e
        cross_unit_e = along_unit_n

        scale = self._gust_scale(altitude)

        along_gust = scale * self._harmonic_sum(self._gust_along, time)
        cross_gust = scale * self._harmonic_sum(self._gust_cross, time)
        vertical_gust = scale * self._harmonic_sum(self._gust_vertical, time)

        north = north_mean + along_gust * along_unit_n + cross_gust * cross_unit_n
        east = east_mean + along_gust * along_unit_e + cross_gust * cross_unit_e
        up = up_mean + vertical_gust
        return north, east, up

