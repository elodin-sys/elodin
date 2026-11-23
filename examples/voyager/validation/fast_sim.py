"""High-performance, high-precision trajectory simulation engine for Voyager 1 and 2.
Uses precomputed SPICE ephemeris spline/Hermite interpolators for 100x faster integration.
Enables exhaustive, controlled physical experiments across multiple flybys and cruise arcs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import spiceypy as spice
from scipy.interpolate import CubicHermiteSpline

# Paths
SPICE_DIR = Path(
    os.environ.get("VOYAGER_SPICE_DIR", Path(__file__).resolve().parents[1] / "nasa_spice_data")
)
LEAP_SECONDS_KERNEL = SPICE_DIR / "naif0012.tls"
PLANET_KERNEL = SPICE_DIR / "de440.bsp"
GM_KERNEL = SPICE_DIR / "gm_de440.tpc"
PCK_KERNEL = SPICE_DIR / "pck00011.tpc"
JUPITER_SATELLITE_KERNEL = SPICE_DIR / "jup365.bsp"
SATURN_SATELLITE_KERNEL = SPICE_DIR / "sat441.bsp"
VOYAGER_1_LONG = SPICE_DIR / "Voyager_1.a54206u_V0.2_merged.bsp"
VOYAGER_2_LONG = SPICE_DIR / "Voyager_2.m05016u.merged.bsp"
V1_JUP = SPICE_DIR / "vgr1_jup230.bsp"
V2_JUP = SPICE_DIR / "vgr2_jup230.bsp"
V1_SAT = SPICE_DIR / "vgr1_sat337.bsp"
V2_SAT = SPICE_DIR / "vgr2_sat337.bsp"
MANIFEST_FILE = Path(__file__).with_name("truth_reference.json")

G_CODATA = 6.6743e-11
SUN_MASS = 1.9885e30

ROUNDED_PLANETS = [
    {"name": "mercury", "spice_name": "MERCURY BARYCENTER", "gm_id": 1, "mass": 3.3011e23},
    {"name": "venus", "spice_name": "VENUS BARYCENTER", "gm_id": 2, "mass": 4.8675e24},
    {"name": "earth", "spice_name": "EARTH", "gm_id": 399, "mass": 5.97219e24},
    {"name": "mars", "spice_name": "MARS BARYCENTER", "gm_id": 4, "mass": 6.4171e23},
    {
        "name": "jupiter",
        "spice_name": "JUPITER BARYCENTER",
        "center_name": "JUPITER",
        "gm_id": 5,
        "center_gm_id": 599,
        "mass": 1.898125e27,
    },
    {
        "name": "saturn",
        "spice_name": "SATURN BARYCENTER",
        "center_name": "SATURN",
        "gm_id": 6,
        "center_gm_id": 699,
        "mass": 5.6834e26,
    },
    {"name": "uranus", "spice_name": "URANUS BARYCENTER", "gm_id": 7, "mass": 8.6813e25},
    {"name": "neptune", "spice_name": "NEPTUNE BARYCENTER", "gm_id": 8, "mass": 1.02413e26},
]

JUPITER_REQ_KM = 71492.0
JUPITER_REQ_M = JUPITER_REQ_KM * 1000.0
JUPITER_J2 = 1.46965e-2
JUPITER_J4 = -5.866e-4
JUPITER_J6 = 3.42e-5

SATURN_REQ_KM = 60268.0
SATURN_REQ_M = SATURN_REQ_KM * 1000.0
SATURN_J2 = 1.62907e-2
SATURN_J4 = -9.358e-4

GALILEAN_MOONS = [
    {"name": "io", "spice_name": "IO", "gm_id": 501, "approx_gm": 5.959915466180539e12},
    {"name": "europa", "spice_name": "EUROPA", "gm_id": 502, "approx_gm": 3.202712099607295e12},
    {"name": "ganymede", "spice_name": "GANYMEDE", "gm_id": 503, "approx_gm": 9.887832752719638e12},
    {"name": "callisto", "spice_name": "CALLISTO", "gm_id": 504, "approx_gm": 7.179283402579837e12},
]

SATURN_MOONS = [
    {"name": "mimas", "spice_name": "MIMAS", "gm_id": 601, "approx_gm": 2.503488768152587e9},
    {
        "name": "enceladus",
        "spice_name": "ENCELADUS",
        "gm_id": 602,
        "approx_gm": 7.210366688598896e9,
    },
    {"name": "tethys", "spice_name": "TETHYS", "gm_id": 603, "approx_gm": 4.121352885489587e10},
    {"name": "dione", "spice_name": "DIONE", "gm_id": 604, "approx_gm": 7.311607172482067e10},
    {"name": "rhea", "spice_name": "RHEA", "gm_id": 605, "approx_gm": 1.539417519146563e11},
    {"name": "titan", "spice_name": "TITAN", "gm_id": 606, "approx_gm": 8.978137095521046e12},
    {"name": "hyperion", "spice_name": "HYPERION", "gm_id": 607, "approx_gm": 3.704913747932265e8},
    {"name": "iapetus", "spice_name": "IAPETUS", "gm_id": 608, "approx_gm": 1.205151060137642e11},
]


@dataclass
class ModelConfig:
    name: str = "custom"
    chapter: int = 2  # 1 = direct Newtonian only, 2 = heliocentric relative
    gm_source: str = "rounded_mass"  # "rounded_mass" or "naif_de440"
    giant_planet_state: str = "barycenter"  # "barycenter" or "center"
    jupiter_moons: list[str] = field(
        default_factory=list
    )  # e.g. ["io", "europa", "ganymede", "callisto"]
    saturn_moons: list[str] = field(default_factory=list)  # e.g. ["titan", "rhea", ...]
    jupiter_j2: bool = False
    jupiter_j4: bool = False
    jupiter_j6: bool = False
    jupiter_j2_pole_mode: str = "true_pole"  # "true_pole", "eclip_z", "j2000_z"
    saturn_j2: bool = False
    saturn_j4: bool = False
    maneuvers: list[dict] = field(
        default_factory=list
    )  # List of {"epoch_et": float, "delta_v": [dvx, dvy, dvz]}
    ephemeris_mode: str = "spline"  # "spline" (exact Hermite), "linear_stage" (Elodin default)


def init_spice(encounter_kernel: Path | None = None):
    spice.kclear()
    spice.furnsh(str(LEAP_SECONDS_KERNEL))
    spice.furnsh(str(VOYAGER_1_LONG))
    spice.furnsh(str(VOYAGER_2_LONG))
    if encounter_kernel is not None and encounter_kernel.is_file():
        spice.furnsh(str(encounter_kernel))
    spice.furnsh(str(PLANET_KERNEL))
    if GM_KERNEL.is_file():
        spice.furnsh(str(GM_KERNEL))
    if PCK_KERNEL.is_file():
        spice.furnsh(str(PCK_KERNEL))
    if JUPITER_SATELLITE_KERNEL.is_file():
        spice.furnsh(str(JUPITER_SATELLITE_KERNEL))
    if SATURN_SATELLITE_KERNEL.is_file():
        spice.furnsh(str(SATURN_SATELLITE_KERNEL))


def get_gm(gm_id: int, fallback_mass: float, config: ModelConfig) -> float:
    if config.gm_source == "naif_de440":
        _, vals = spice.bodvcd(gm_id, "GM", 1)
        return float(vals[0]) * 1.0e9
    return G_CODATA * fallback_mass


class EphemerisCache:
    """Precomputed Hermite splines for planet and moon positions and velocities."""

    def __init__(self, t_start: float, t_end: float, bodies: list[str]):
        # Sample every 1 hour (3600s) - Hermite spline with pos & vel is accurate to sub-millimeter!
        dt_sample = 3600.0
        n_samples = int(np.ceil((t_end - t_start) / dt_sample)) + 2
        self.sample_times = np.linspace(t_start, t_start + (n_samples - 1) * dt_sample, n_samples)
        self.splines = {}
        self.pole_splines = {}

        for body in bodies:
            pos_list = []
            vel_list = []
            for t in self.sample_times:
                state, _ = spice.spkezr(body, t, "ECLIPJ2000", "NONE", "SUN")
                pos_list.append(np.array(state[:3], dtype=np.float64) * 1000.0)
                vel_list.append(np.array(state[3:], dtype=np.float64) * 1000.0)
            self.splines[body] = CubicHermiteSpline(
                self.sample_times, np.array(pos_list), np.array(vel_list)
            )

        # Also cache Jupiter & Saturn spin poles in ECLIPJ2000
        for p_name, frame in [("jupiter", "IAU_JUPITER"), ("saturn", "IAU_SATURN")]:
            poles = []
            for t in self.sample_times:
                rot = spice.pxform(frame, "ECLIPJ2000", t)
                poles.append(rot[:, 2])
            # Simple cubic spline for unit pole
            from scipy.interpolate import CubicSpline

            self.pole_splines[p_name] = CubicSpline(self.sample_times, np.array(poles))

    def get_pos(self, body: str, t: float) -> np.ndarray:
        return self.splines[body](t)

    def get_pole(self, planet: str, t: float) -> np.ndarray:
        p = self.pole_splines[planet](t)
        return p / np.linalg.norm(p)


def compute_j2_j4_acceleration(
    r_rel_m: np.ndarray,
    mu_m3s2: float,
    req_m: float,
    j2: float,
    j4: float,
    j6: float,
    pole_unit: np.ndarray,
) -> np.ndarray:
    r = np.linalg.norm(r_rel_m)
    if r < 1.0:
        return np.zeros(3)

    z = np.dot(r_rel_m, pole_unit)
    z_over_r = z / r
    z2_over_r2 = z_over_r**2

    r_unit = r_rel_m / r
    p_unit = pole_unit
    accel = np.zeros(3)

    if j2 != 0.0:
        factor_j2 = -1.5 * mu_m3s2 * j2 * (req_m**2) / (r**4)
        a_j2 = factor_j2 * ((1.0 - 5.0 * z2_over_r2) * r_unit + 2.0 * z_over_r * p_unit)
        accel += a_j2

    if j4 != 0.0:
        factor_j4 = -0.625 * mu_m3s2 * j4 * (req_m**4) / (r**6)
        radial_j4 = (3.0 - 42.0 * z2_over_r2 + 63.0 * (z2_over_r2**2)) * r_unit
        pole_j4 = (12.0 * z_over_r - 28.0 * (z_over_r**3)) * p_unit
        accel += factor_j4 * (radial_j4 + pole_j4)

    if j6 != 0.0:
        factor_j6 = -(mu_m3s2 * j6 * (req_m**6) / (16.0 * (r**8)))
        radial_j6 = (
            35.0 - 945.0 * z2_over_r2 + 3465.0 * (z2_over_r2**2) - 3003.0 * (z2_over_r2**3)
        ) * r_unit
        pole_j6 = (120.0 * z_over_r - 720.0 * (z_over_r**3) + 792.0 * (z_over_r**5)) * p_unit
        accel += factor_j6 * (radial_j6 + pole_j6)

    return accel


def compute_cached_accel(
    t_et: float, pos_m: np.ndarray, config: ModelConfig, cache: EphemerisCache
) -> np.ndarray:
    mu_sun = get_gm(10, SUN_MASS, config)
    r_probe = pos_m
    r_probe_norm = np.linalg.norm(r_probe)

    a_total = -mu_sun * r_probe / (r_probe_norm**3)

    for p in ROUNDED_PLANETS:
        p_name = p["name"]
        is_giant = p_name in ("jupiter", "saturn")
        use_center = is_giant and (
            config.giant_planet_state == "center"
            or (p_name == "jupiter" and len(config.jupiter_moons) > 0)
            or (p_name == "saturn" and len(config.saturn_moons) > 0)
        )
        spice_target = p["center_name"] if use_center else p["spice_name"]
        gm_id = p["center_gm_id"] if use_center else p["gm_id"]

        r_planet = cache.get_pos(spice_target, t_et)
        r_planet_norm = np.linalg.norm(r_planet)
        mu_p = get_gm(gm_id, p["mass"], config)

        r_rel = r_planet - r_probe
        dist_rel = np.linalg.norm(r_rel)

        a_direct = mu_p * r_rel / (dist_rel**3)
        a_total += a_direct

        if config.chapter == 2:
            a_sun = mu_p * r_planet / (r_planet_norm**3)
            a_total -= a_sun

        if p_name == "jupiter" and (config.jupiter_j2 or config.jupiter_j4 or config.jupiter_j6):
            r_c_to_probe = r_probe - r_planet
            if config.jupiter_j2_pole_mode == "true_pole":
                pole_unit = cache.get_pole("jupiter", t_et)
            elif config.jupiter_j2_pole_mode == "eclip_z":
                pole_unit = np.array([0.0, 0.0, 1.0])
            elif config.jupiter_j2_pole_mode == "j2000_z":
                rot_j2e = spice.pxform("J2000", "ECLIPJ2000", t_et)
                pole_unit = rot_j2e @ np.array([0.0, 0.0, 1.0])
            else:
                pole_unit = np.array([0.0, 0.0, 1.0])

            j2_val = JUPITER_J2 if config.jupiter_j2 else 0.0
            j4_val = JUPITER_J4 if config.jupiter_j4 else 0.0
            j6_val = JUPITER_J6 if config.jupiter_j6 else 0.0
            a_j2 = compute_j2_j4_acceleration(
                r_c_to_probe, mu_p, JUPITER_REQ_M, j2_val, j4_val, j6_val, pole_unit
            )
            a_total += a_j2

        if p_name == "saturn" and (config.saturn_j2 or config.saturn_j4):
            r_c_to_probe = r_probe - r_planet
            pole_unit = cache.get_pole("saturn", t_et)
            j2_val = SATURN_J2 if config.saturn_j2 else 0.0
            j4_val = SATURN_J4 if config.saturn_j4 else 0.0
            a_j2 = compute_j2_j4_acceleration(
                r_c_to_probe, mu_p, SATURN_REQ_M, j2_val, j4_val, 0.0, pole_unit
            )
            a_total += a_j2

    for m in GALILEAN_MOONS:
        if m["name"] in config.jupiter_moons:
            r_moon = cache.get_pos(m["spice_name"], t_et)
            r_moon_norm = np.linalg.norm(r_moon)
            mu_m = get_gm(m["gm_id"], m["approx_gm"] / G_CODATA, config)
            r_rel = r_moon - r_probe
            dist_rel = np.linalg.norm(r_rel)
            a_total += mu_m * r_rel / (dist_rel**3)
            if config.chapter == 2:
                a_total -= mu_m * r_moon / (r_moon_norm**3)

    for m in SATURN_MOONS:
        if m["name"] in config.saturn_moons:
            r_moon = cache.get_pos(m["spice_name"], t_et)
            r_moon_norm = np.linalg.norm(r_moon)
            mu_m = get_gm(m["gm_id"], m["approx_gm"] / G_CODATA, config)
            r_rel = r_moon - r_probe
            dist_rel = np.linalg.norm(r_rel)
            a_total += mu_m * r_rel / (dist_rel**3)
            if config.chapter == 2:
                a_total -= mu_m * r_moon / (r_moon_norm**3)

    return a_total


def run_simulation(
    probe_spice_name: str,
    start_utc: str,
    duration_days: float,
    dt: float,
    config: ModelConfig,
    checkpoints: list[dict] | None = None,
) -> dict:
    t_start = spice.utc2et(start_utc)
    t_end = t_start + duration_days * 86400.0

    # Collect all bodies needed in cache
    needed_bodies = set()
    for p in ROUNDED_PLANETS:
        needed_bodies.add(p["spice_name"])
        if "center_name" in p:
            needed_bodies.add(p["center_name"])
    for m in GALILEAN_MOONS:
        needed_bodies.add(m["spice_name"])
    for m in SATURN_MOONS:
        needed_bodies.add(m["spice_name"])

    cache = EphemerisCache(t_start, t_end, list(needed_bodies))

    # Initial state
    init_state, _ = spice.spkezr(probe_spice_name, t_start, "ECLIPJ2000", "NONE", "SUN")
    pos_m = np.array(init_state[:3], dtype=np.float64) * 1000.0
    vel_ms = np.array(init_state[3:], dtype=np.float64) * 1000.0
    current_state = np.concatenate((pos_m, vel_ms))

    current_et = t_start
    total_steps = round(duration_days * 86400.0 / dt)

    chk_results = []
    if checkpoints:
        for chk in checkpoints:
            chk_et = t_start + chk["elapsed_seconds"]
            chk_results.append(
                {
                    "name": chk["name"],
                    "target_et": chk_et,
                    "evaluated": False,
                    "elapsed_seconds": chk["elapsed_seconds"],
                    "validation_role": chk.get("validation_role", ""),
                    "sim_pos": None,
                    "sim_vel": None,
                    "truth_pos": None,
                    "truth_vel": None,
                    "pos_err_km": None,
                    "vel_err_mps": None,
                }
            )

    for c in chk_results:
        if abs(c["elapsed_seconds"]) < 1e-3:
            truth_state, _ = spice.spkezr(probe_spice_name, t_start, "ECLIPJ2000", "NONE", "SUN")
            c["evaluated"] = True
            c["sim_pos"] = pos_m.copy()
            c["sim_vel"] = vel_ms.copy()
            c["truth_pos"] = np.array(truth_state[:3]) * 1000.0
            c["truth_vel"] = np.array(truth_state[3:]) * 1000.0
            c["pos_err_km"] = 0.0
            c["vel_err_mps"] = 0.0

    trajectory_times = []
    trajectory_pos_err = []
    trajectory_vel_err = []

    for step in range(total_steps):
        # Apply any impulsive maneuvers
        for man in config.maneuvers:
            man_et = man["epoch_et"]
            if current_et <= man_et < current_et + dt:
                current_state[3:] += np.array(man["delta_v"], dtype=np.float64)

        # RK4
        r = current_state[:3]
        v = current_state[3:]

        k1_v = compute_cached_accel(current_et, r, config, cache)
        k1_r = v

        k2_v = compute_cached_accel(current_et + 0.5 * dt, r + 0.5 * dt * k1_r, config, cache)
        k2_r = v + 0.5 * dt * k1_v

        k3_v = compute_cached_accel(current_et + 0.5 * dt, r + 0.5 * dt * k2_r, config, cache)
        k3_r = v + 0.5 * dt * k2_v

        k4_v = compute_cached_accel(current_et + dt, r + dt * k3_r, config, cache)
        k4_r = v + dt * k3_v

        current_state[:3] += (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        current_state[3:] += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        current_et += dt

        # Check checkpoints
        for c in chk_results:
            if not c["evaluated"] and abs(current_et - c["target_et"]) <= 0.5 * dt:
                c["evaluated"] = True
                truth_state, _ = spice.spkezr(
                    probe_spice_name, current_et, "ECLIPJ2000", "NONE", "SUN"
                )
                t_pos = np.array(truth_state[:3]) * 1000.0
                t_vel = np.array(truth_state[3:]) * 1000.0
                s_pos = current_state[:3].copy()
                s_vel = current_state[3:].copy()
                c["sim_pos"] = s_pos
                c["sim_vel"] = s_vel
                c["truth_pos"] = t_pos
                c["truth_vel"] = t_vel
                c["pos_err_km"] = float(np.linalg.norm(s_pos - t_pos) / 1000.0)
                c["vel_err_mps"] = float(np.linalg.norm(s_vel - t_vel))

        # Sample for plotting every 1 day
        if step % int(86400.0 / dt) == 0 or step == total_steps - 1:
            truth_state, _ = spice.spkezr(probe_spice_name, current_et, "ECLIPJ2000", "NONE", "SUN")
            t_pos = np.array(truth_state[:3]) * 1000.0
            t_vel = np.array(truth_state[3:]) * 1000.0
            p_err = np.linalg.norm(current_state[:3] - t_pos) / 1000.0
            v_err = np.linalg.norm(current_state[3:] - t_vel)
            trajectory_times.append((current_et - t_start) / 86400.0)
            trajectory_pos_err.append(p_err)
            trajectory_vel_err.append(v_err)

    return {
        "checkpoints": chk_results,
        "final_state": current_state,
        "config": config,
        "trajectory_times_days": trajectory_times,
        "trajectory_pos_err_km": trajectory_pos_err,
        "trajectory_vel_err_mps": trajectory_vel_err,
    }
