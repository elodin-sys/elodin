"""Scenario and numerics configuration (no aircraft data lives here).

Aircraft constants come exclusively from the validated package via
`bdx_model.load()`; class-D fallbacks are a separate, logged opt-in. This
module owns what the handoff calls scenario settings (site, altitude, wind,
initial state) and numerical settings (rates, durations) — guide §23.

Scenario selection (env, keeping the CI invocation contract stable):

    ELODIN_RC_JET_SCENARIO      "demo" (default) | "validation"
    ELODIN_RC_JET_ALTITUDE_M    demo-only override; triggers the trim solver
    ELODIN_RC_JET_SPEED_MPS     demo-only override; triggers the trim solver
    ELODIN_RC_JET_HEADING_DEG   initial heading (default 350)

`demo` flies from the Mojave RC field (35.350664 N, 117.809027 W, field
elevation 589.274 m) inside the `mojave_rc_field` terrain region, defaulting
to 300 m AGL. `validation` stays on the Death Valley floor at the package
cruise row (300 m MSL) so the CI trim-hold does not spawn underground at
the higher Mojave pad (guide §9.5). `demo` solves its own equilibrium and
is the interactive default.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

import bdx_model
from bdx_model import BdxModel
from class_d_fallbacks import FALLBACKS, ClassDFallbacks
from frames import enu_basis, geodetic_to_ecef, level_attitude_ecef, quaternion_xyzw_from_matrix
from trim import TrimSolution, solve_level_trim


@dataclass(frozen=True)
class Site:
    name: str
    lat_deg: float
    lon_deg: float
    field_elevation_m: float

    def format_latlon(self) -> str:
        lat_h = "N" if self.lat_deg >= 0.0 else "S"
        lon_h = "E" if self.lon_deg >= 0.0 else "W"
        return f"{abs(self.lat_deg):.4f} {lat_h}, {abs(self.lon_deg):.4f} {lon_h}"


DEATH_VALLEY_FLOOR = Site(
    name="Death Valley floor (CA)",
    lat_deg=36.2300,
    lon_deg=-116.9700,
    field_elevation_m=-60.0,
)

# Pilot's ground position; region center of the mojave_rc_field world_mesh.
MOJAVE_RC_FIELD = Site(
    name="Mojave RC field (CA)",
    lat_deg=35.350664,
    lon_deg=-117.809027,
    field_elevation_m=589.274,
)


@dataclass(frozen=True)
class Numerics:
    dt: float = 1.0 / 300.0
    simulation_time_s: float = 180.0

    @property
    def total_ticks(self) -> int:
        return int(self.simulation_time_s / self.dt)


@dataclass(frozen=True)
class InitialState:
    pos_ecef: np.ndarray
    quat_xyzw: np.ndarray
    vel_ecef: np.ndarray
    alpha_rad: float
    elevator_rad: float
    throttle: float  # effective throttle == stick command (floored at idle)
    fuel_kg: float
    wind_ecef: np.ndarray


@dataclass(frozen=True)
class Scenario:
    name: str
    site: Site
    altitude_m: float
    tas_mps: float
    heading_deg: float
    wind_enu: tuple[float, float, float]
    initial: InitialState
    trim: TrimSolution


def _pitch_up(alpha_rad: float) -> np.ndarray:
    """Body-frame rotation lifting the nose by alpha (about +Y-left by -alpha)."""
    cos_a, sin_a = math.cos(alpha_rad), math.sin(alpha_rad)
    return np.array([[cos_a, 0.0, -sin_a], [0.0, 1.0, 0.0], [sin_a, 0.0, cos_a]])


def _initial_state(
    model: BdxModel,
    site: Site,
    altitude_m: float,
    tas_mps: float,
    heading_deg: float,
    alpha_rad: float,
    elevator_rad: float,
    throttle: float,
    wind_enu: tuple[float, float, float],
) -> InitialState:
    lat = math.radians(site.lat_deg)
    lon = math.radians(site.lon_deg)
    pos = np.asarray(geodetic_to_ecef(lat, lon, altitude_m))
    level = np.asarray(level_attitude_ecef(lat, lon, heading_deg))
    attitude = level @ _pitch_up(alpha_rad)
    basis = np.asarray(enu_basis(lat, lon))
    wind_ecef = basis.T @ np.asarray(wind_enu, dtype=np.float64)
    # Level flight path: air-relative velocity along the local horizontal
    # forward direction; ground velocity adds the wind.
    vel = tas_mps * level[:, 0] + wind_ecef
    return InitialState(
        pos_ecef=pos,
        quat_xyzw=quaternion_xyzw_from_matrix(attitude),
        vel_ecef=vel,
        alpha_rad=alpha_rad,
        elevator_rad=elevator_rad,
        throttle=throttle,
        fuel_kg=model.mass.fuel_mass_kg,
        wind_ecef=wind_ecef,
    )


def load_scenario(
    model: BdxModel,
    fallbacks: ClassDFallbacks = FALLBACKS,
    name: str | None = None,
    wind_enu: tuple[float, float, float] | None = None,
) -> Scenario:
    name = name or os.environ.get("ELODIN_RC_JET_SCENARIO", "demo")
    if name not in ("demo", "validation"):
        raise ValueError(f"unknown scenario {name!r} (expected 'demo' or 'validation')")

    # Both scenarios fly full 6-DOF, which the package alone cannot support:
    # explicit class-D opt-in, logged at startup (guide §9.2).
    model.require_mode(bdx_model.MODE_CLASS_D_6DOF, allow_class_d=True)
    model.require_credibility("analysis-correlated")
    fallbacks.log_selection(name)

    # validation keeps the Death Valley pad so the package cruise row
    # (300 m MSL) is above ground; demo flies the Mojave RC field.
    if name == "validation":
        site = DEATH_VALLEY_FLOOR
    else:
        site = MOJAVE_RC_FIELD
    heading_deg = float(os.environ.get("ELODIN_RC_JET_HEADING_DEG", "350.0"))
    wind_enu = wind_enu if wind_enu is not None else (0.0, 0.0, 0.0)
    cruise = model.trim_rows["cruise"]

    if name == "validation":
        altitude_m, tas_mps = cruise.altitude_m, cruise.tas_mps
    else:
        default_alt = site.field_elevation_m + 300.0
        altitude_m = float(os.environ.get("ELODIN_RC_JET_ALTITUDE_M", default_alt))
        tas_mps = float(os.environ.get("ELODIN_RC_JET_SPEED_MPS", cruise.tas_mps))

    solution = solve_level_trim(model, fallbacks, site.lat_deg, site.lon_deg, altitude_m, tas_mps)
    if not solution.valid:
        raise RuntimeError(
            f"scenario {name!r} has no valid equilibrium at {altitude_m} m / "
            f"{tas_mps} m/s (alpha {math.degrees(solution.alpha_rad):.2f} deg, "
            f"throttle {solution.effective_throttle:.3f}); refusing to spawn "
            "off-equilibrium (guide §9.5)"
        )

    if name == "validation":
        # Package trim row verbatim (guide §9.5); the solver supplies only the
        # elevator that balances the thrust-line moment our plant adds.
        alpha_rad = math.radians(cruise.alpha_deg)
        throttle = cruise.throttle
    else:
        alpha_rad = solution.alpha_rad
        throttle = solution.effective_throttle

    initial = _initial_state(
        model,
        site,
        altitude_m,
        tas_mps,
        heading_deg,
        alpha_rad,
        solution.elevator_rad,
        throttle,
        wind_enu,
    )
    return Scenario(
        name=name,
        site=site,
        altitude_m=altitude_m,
        tas_mps=tas_mps,
        heading_deg=heading_deg,
        wind_enu=wind_enu,
        initial=initial,
        trim=solution,
    )
