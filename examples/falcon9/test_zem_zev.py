"""E1.1: offline ZEM/ZEV terminal guidance replay (Guo/Hawkins/Wie).

Seeds the recorded bad landing-ignition state (~391 m miss, ~47 m/s lateral
at ~3 km) and integrates the accel-vector law with hoverslam-derived t_go.
Gate: terminal position err < 2 m and velocity err < 0.5 m/s vs soft target.
"""

from __future__ import annotations

import math

import numpy as np

G = 9.81
WAYPOINT_ALT_M = 150.0
WAYPOINT_VDOWN_MPS = 25.0
V_TD_MPS = 1.5
TILT_CAP_RAD = 0.25
A_LAND_MPS2 = 12.0  # net vertical accel for t_go (1-engine class)
A_THRUST_MAX = 35.0  # 3-engine landing-burn thrust accel ceiling
DT = 0.05
COMMIT_ALT_M = 50.0
COMMIT_TGO_S = 5.0


def t_go_hoverslam(h: float, vdown: float, a_land: float = A_LAND_MPS2) -> float:
    """Time-to-go consistent with a min-throttle-feasible hoverslam profile."""
    h = max(h, 0.5)
    vdown = max(vdown, 0.1)
    a_req = max(vdown * vdown - V_TD_MPS * V_TD_MPS, 0.0) / (2.0 * h)
    a_use = float(np.clip(a_req, 0.5, a_land))
    t_raw = (vdown - V_TD_MPS) / a_use
    return float(np.clip(t_raw, 0.5, 80.0)), float(t_raw)


def zem_zev_accel(r: np.ndarray, v: np.ndarray, up: np.ndarray, t_go: float, commit: bool):
    """Return thrust-acceleration command in ECEF-like local frame (+up)."""
    alt = float(np.dot(r, up))
    g_vec = -G * up
    if commit:
        # Vertical-only: cancel gravity + brake to V_TD.
        vdown = -float(np.dot(v, up))
        a_up = G + 3.0 * (vdown - V_TD_MPS)
        return a_up * up

    if alt > WAYPOINT_ALT_M:
        r_tgt = WAYPOINT_ALT_M * up
        v_tgt = -WAYPOINT_VDOWN_MPS * up
    else:
        r_tgt = np.zeros(3)
        v_tgt = -V_TD_MPS * up

    zem = r_tgt - (r + v * t_go + 0.5 * g_vec * t_go * t_go)
    zev = v_tgt - (v + g_vec * t_go)
    a_cmd = 6.0 * zem / (t_go * t_go) - 2.0 * zev / t_go - g_vec

    # Clamp tilt relative to up.
    a_up = float(np.dot(a_cmd, up))
    a_lat = a_cmd - a_up * up
    lat_mag = float(np.linalg.norm(a_lat))
    max_lat = abs(a_up) * math.tan(TILT_CAP_RAD) if a_up > 1.0 else 0.0
    if lat_mag > max_lat and lat_mag > 1e-6:
        a_lat = a_lat * (max_lat / lat_mag)
        a_cmd = a_up * up + a_lat
    return a_cmd


def simulate_from_ignition(
    miss_m: float = 391.0,
    vlat_mps: float = 46.9,
    vdown_mps: float = 271.0,
    alt_m: float = 3000.0,
    toward: float = 0.16,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate until touchdown. Local frame: e0 toward target, e1 cross, up."""
    up = np.array([0.0, 0.0, 1.0])
    e0 = np.array([1.0, 0.0, 0.0])  # toward-target horizontal
    r = np.array([-miss_m, 0.0, alt_m])
    v_h = vlat_mps * (
        toward * e0 + math.sqrt(max(1.0 - toward * toward, 0.0)) * np.array([0.0, 1.0, 0.0])
    )
    v = v_h - vdown_mps * up

    for _ in range(20_000):
        alt = float(r[2])
        if alt <= 0.0:
            break
        vdown = -float(np.dot(v, up))
        t_go, t_raw = t_go_hoverslam(alt, vdown)
        # Commit only near the pad (use unclipped t_raw so soft-rate coast
        # at altitude does not freeze lateral early).
        commit = alt < COMMIT_ALT_M or (0.0 < t_raw < COMMIT_TGO_S and alt < 200.0)
        a_cmd = zem_zev_accel(r, v, up, t_go, commit)
        mag = float(np.linalg.norm(a_cmd))
        if mag > A_THRUST_MAX:
            a_cmd = a_cmd * (A_THRUST_MAX / mag)
        a = a_cmd + (-G * up)
        v = v + a * DT
        r = r + v * DT
        if r[2] < 0.0:
            frac = r[2] / (v[2] * DT + 1e-12)  # how far past contact this step
            r = r - v * DT * frac
            r[2] = 0.0
            break
    return r, v


def test_zem_zev_closes_recorded_ignition_miss():
    r, v = simulate_from_ignition()
    miss = float(np.hypot(r[0], r[1]))
    v_tgt = np.array([0.0, 0.0, -V_TD_MPS])
    verr = float(np.linalg.norm(v - v_tgt))
    assert miss < 2.0, f"terminal miss {miss:.2f} m"
    assert verr < 0.5, f"terminal velocity err {verr:.2f} m/s"


def test_zem_zev_robust_to_larger_ignition_miss():
    r, v = simulate_from_ignition(miss_m=600.0, vlat_mps=40.0, vdown_mps=250.0, alt_m=3500.0)
    miss = float(np.hypot(r[0], r[1]))
    assert miss < 10.0, f"600 m ignition miss landed {miss:.1f} m off"
    v_tgt = np.array([0.0, 0.0, -V_TD_MPS])
    assert float(np.linalg.norm(v - v_tgt)) < 2.0
