"""Automated regression and physics verification tests for Voyager trajectory simulation.
Run with pytest.
"""

import json

import numpy as np
from fast_sim import (
    JUPITER_J2,
    JUPITER_REQ_M,
    MANIFEST_FILE,
    V1_JUP,
    V1_SAT,
    ModelConfig,
    compute_cached_accel,
    compute_j2_j4_acceleration,
    init_spice,
    run_simulation,
)


def test_j2_pure_equatorial_radial_outward():
    """On the equator (z=0), J2 disturbance points radially inward (increases attraction)."""
    r_rel = np.array([1e8, 0.0, 0.0])  # on equatorial plane
    pole = np.array([0.0, 0.0, 1.0])
    mu = 1.267e17
    req = JUPITER_REQ_M
    j2 = JUPITER_J2

    a_j2 = compute_j2_j4_acceleration(r_rel, mu, req, j2, 0.0, 0.0, pole)
    # At z=0, factor_j2 = -1.5 * mu * j2 * R^2 / r^4 < 0.
    # a_j2 = factor_j2 * r_unit -> points in -r_rel direction!
    assert a_j2[0] < 0.0
    assert abs(a_j2[1]) < 1e-15
    assert abs(a_j2[2]) < 1e-15


def test_j2_pure_polar_axial():
    """Over the pole (x=y=0, z=r), J2 disturbance opposes central gravity."""
    r_rel = np.array([0.0, 0.0, 1e8])
    pole = np.array([0.0, 0.0, 1.0])
    mu = 1.267e17
    req = JUPITER_REQ_M
    j2 = JUPITER_J2

    a_j2 = compute_j2_j4_acceleration(r_rel, mu, req, j2, 0.0, 0.0, pole)
    # At z=r, (1 - 5*(z/r)^2)*r_unit + 2*(z/r)*p_unit = -4*z_hat + 2*z_hat = -2*z_hat
    # a_j2 = -1.5 * mu * j2 * R^2 / r^4 * (-2) * z_hat > 0 (points in +z direction, outward!)
    assert a_j2[2] > 0.0
    assert abs(a_j2[0]) < 1e-15
    assert abs(a_j2[1]) < 1e-15


def test_baseline_clean_approach_reproduction():
    """Test that Chapter 2 baseline reproduces clean_approach error within 0.01 km."""
    init_spice(V1_JUP)
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    case = next(c for c in manifest["cases"] if c["id"] == "v1_jupiter")

    cfg = ModelConfig(chapter=2, gm_source="rounded_mass")
    res = run_simulation(
        "VOYAGER 1",
        case["initialization_utc"],
        duration_days=20.0,
        dt=100.0,
        config=cfg,
        checkpoints=case["checkpoints"],
    )

    clean_chk = next(c for c in res["checkpoints"] if c["name"] == "clean_approach")
    assert abs(clean_chk["pos_err_km"] - 33.699) < 0.01


def test_moons_improve_voyager2_jupiter_approach():
    """Verify that Galilean moons reduce Voyager 2 Day 20 clean approach error by >70%."""
    from fast_sim import V2_JUP

    init_spice(V2_JUP)
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    case = next(c for c in manifest["cases"] if c["id"] == "v2_jupiter")

    cfg_base = ModelConfig(chapter=2)
    res_base = run_simulation(
        "VOYAGER 2",
        case["initialization_utc"],
        duration_days=20.0,
        dt=100.0,
        config=cfg_base,
        checkpoints=case["checkpoints"],
    )
    err_base = next(c for c in res_base["checkpoints"] if c["name"] == "clean_approach")[
        "pos_err_km"
    ]

    cfg_moons = ModelConfig(
        chapter=2,
        giant_planet_state="center",
        jupiter_moons=["io", "europa", "ganymede", "callisto"],
    )
    res_moons = run_simulation(
        "VOYAGER 2",
        case["initialization_utc"],
        duration_days=20.0,
        dt=100.0,
        config=cfg_moons,
        checkpoints=case["checkpoints"],
    )
    err_moons = next(c for c in res_moons["checkpoints"] if c["name"] == "clean_approach")[
        "pos_err_km"
    ]

    reduction = (err_base - err_moons) / err_base
    assert reduction > 0.70


def test_saturn_j2_reduces_encounter_error():
    """Verify that incorporating Saturn J2/J4 reduces V1 Saturn closest approach error below 25 km."""
    import spiceypy as spice

    enc_kernel = V1_SAT
    init_spice(enc_kernel)
    t_peri = spice.utc2et("1980-11-12T23:46:30")
    t_start = t_peri - 3.0 * 86400.0

    needed = set()
    from fast_sim import ROUNDED_PLANETS, SATURN_MOONS, EphemerisCache

    for p in ROUNDED_PLANETS:
        needed.add(p["spice_name"])
        if "center_name" in p:
            needed.add(p["center_name"])
    for m in SATURN_MOONS:
        needed.add(m["spice_name"])
    cache = EphemerisCache(t_start - 86400, t_peri + 86400, list(needed))

    st0, _ = spice.spkezr("-31", t_start, "ECLIPJ2000", "NONE", "SUN")
    st_target, _ = spice.spkezr("-31", t_peri, "ECLIPJ2000", "NONE", "SUN")

    cfg = ModelConfig(
        giant_planet_state="center",
        saturn_moons=["titan", "rhea", "iapetus", "dione", "tethys", "enceladus", "mimas"],
        saturn_j2=True,
        saturn_j4=True,
    )

    curr = np.concatenate((np.array(st0[:3]) * 1000.0, np.array(st0[3:]) * 1000.0))
    t = t_start
    dt = 120.0
    while t < t_peri:
        h = min(dt, t_peri - t)
        k1_v = compute_cached_accel(t, curr[:3], cfg, cache)
        k1_r = curr[3:]
        k2_v = compute_cached_accel(t + 0.5 * h, curr[:3] + 0.5 * h * k1_r, cfg, cache)
        k2_r = curr[3:] + 0.5 * h * k1_v
        k3_v = compute_cached_accel(t + 0.5 * h, curr[:3] + 0.5 * h * k2_r, cfg, cache)
        k3_r = curr[3:] + 0.5 * h * k2_v
        k4_v = compute_cached_accel(t + h, curr[:3] + h * k3_r, cfg, cache)
        k4_r = curr[3:] + h * k3_v
        curr[:3] += (h / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        curr[3:] += (h / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        t += h

    pos_err = np.linalg.norm(curr[:3] - np.array(st_target[:3]) * 1000.0) / 1000.0
    assert pos_err < 20.0  # Proves sub-20 km accuracy on Saturn encounter
