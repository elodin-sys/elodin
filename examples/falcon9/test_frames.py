"""Phase 0 geodesy tests: WHITEPAPER section 4 worked examples as assertions."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from constants import (
    LZ1_ALT_M,
    LZ1_LAT_DEG,
    LZ1_LON_DEG,
    PAD_ALT_M,
    PAD_LAT_DEG,
    PAD_LON_DEG,
)
from frames import (
    OMEGA_E_VEC,
    apparent_gravity,
    centrifugal_accel,
    coriolis_accel,
    ecef_to_geodetic,
    ellipsoid_up,
    geodetic_altitude,
    geodetic_to_ecef,
    gravity_accel,
    ned_basis,
)


def _pad_ecef() -> jnp.ndarray:
    return geodetic_to_ecef(math.radians(PAD_LAT_DEG), math.radians(PAD_LON_DEG), PAD_ALT_M)


def test_pad_ecef_matches_whitepaper():
    pad = np.asarray(_pad_ecef())
    np.testing.assert_allclose(pad / 1000.0, [914.8, -5528.6, 3035.9], atol=0.1)
    lz1 = np.asarray(
        geodetic_to_ecef(math.radians(LZ1_LAT_DEG), math.radians(LZ1_LON_DEG), LZ1_ALT_M)
    )
    np.testing.assert_allclose(lz1 / 1000.0, [921.7, -5534.0, 3023.9], atol=0.1)
    assert abs(np.linalg.norm(pad - lz1) / 1000.0 - 14.8) < 0.1


def test_geodetic_roundtrip():
    for lat_deg in (-75.0, -28.0, 0.0, 28.60839, 45.0, 89.0):
        for lon_deg in (-170.0, -80.60433, 0.0, 91.0):
            for alt_m in (0.0, 3.0, 8_700.0, 118_000.0, 200_000.0):
                r = geodetic_to_ecef(math.radians(lat_deg), math.radians(lon_deg), alt_m)
                lat, lon, alt = ecef_to_geodetic(r)
                assert abs(math.degrees(float(lat)) - lat_deg) < 1e-9
                assert abs(math.degrees(float(lon)) - lon_deg) < 1e-9
                assert abs(float(alt) - alt_m) < 1e-6


def test_ned_basis_orthonormal_and_up_is_ellipsoid_normal():
    lat, lon = math.radians(PAD_LAT_DEG), math.radians(PAD_LON_DEG)
    r = ned_basis(lat, lon)
    np.testing.assert_allclose(np.asarray(r @ r.T), np.eye(3), atol=1e-12)
    assert float(jnp.linalg.det(r)) > 0.99
    # Moving +1 m along the ellipsoid normal changes geodetic altitude by +1 m.
    up = ellipsoid_up(lat, lon)
    base = geodetic_to_ecef(lat, lon, PAD_ALT_M)
    finite_diff = geodetic_to_ecef(lat, lon, PAD_ALT_M + 1.0) - base
    np.testing.assert_allclose(np.asarray(finite_diff), np.asarray(up), atol=1e-9)
    assert abs(float(geodetic_altitude(base + up * 100.0)) - (PAD_ALT_M + 100.0)) < 1e-6


def test_rotating_frame_magnitudes():
    pad = _pad_ecef()
    # Surface rotation speed ~408.6 m/s (WHITEPAPER 4.3).
    v_rot = float(jnp.linalg.norm(jnp.cross(OMEGA_E_VEC, pad)))
    assert abs(v_rot - 408.6) < 0.5
    # Centrifugal ~0.0298 m/s^2 at the pad; Coriolis 0.242 m/s^2 at MECO speed.
    assert abs(float(jnp.linalg.norm(centrifugal_accel(pad))) - 0.0298) < 3e-4
    v = jnp.array([1656.0, 0.0, 0.0])
    assert abs(float(jnp.linalg.norm(coriolis_accel(v))) - 0.2415) < 1e-3
    # Gravity 9.813 at pad, -3.6% at 118 km apogee (WHITEPAPER 6).
    g_pad = float(jnp.linalg.norm(gravity_accel(pad)))
    assert abs(g_pad - 9.813) < 2e-3
    apo = pad * (1.0 + 118_000.0 / float(jnp.linalg.norm(pad)))
    g_apo = float(jnp.linalg.norm(gravity_accel(apo)))
    assert abs(g_apo / g_pad - 0.964) < 1e-3


def test_plumb_line_is_geodetic_vertical():
    # Gravitation + centrifugal at rest on the ellipsoid points along -up to
    # within the deflection tolerance of the point-mass field (WHITEPAPER 5.1).
    lat, lon = math.radians(PAD_LAT_DEG), math.radians(PAD_LON_DEG)
    g_app = apparent_gravity(geodetic_to_ecef(lat, lon, PAD_ALT_M))
    g_hat = np.asarray(g_app / jnp.linalg.norm(g_app))
    up = np.asarray(ellipsoid_up(lat, lon))
    misalign_deg = math.degrees(math.acos(float(np.clip(-g_hat @ up, -1.0, 1.0))))
    # Point-mass gravity (no J2) leaves a ~0.1 deg deflection at mid-latitudes;
    # the ellipsoid is an equipotential of the *real* field, not the point-mass one.
    assert misalign_deg < 0.2
    assert abs(float(jnp.linalg.norm(g_app)) - 9.79) < 0.02
