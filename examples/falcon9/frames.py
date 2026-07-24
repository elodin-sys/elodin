"""WGS84 geodesy and rotating-frame helpers (WHITEPAPER sections 4-5).

Pure JAX-compatible functions shared by the plant systems, the reference
tooling, and the tests. All positions/velocities are ECEF meters unless
stated; angles are radians. The Python SDK has no geodesy helpers, so these
follow examples/geo-frames and the WHITEPAPER derivations directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from constants import (
    MU_EARTH_M3S2,
    OMEGA_EARTH_RADPS,
    WGS84_A_M,
    WGS84_B_M,
    WGS84_E2,
    WGS84_EP2,
    WGS84_F,
)

jax.config.update("jax_enable_x64", True)

OMEGA_E_VEC = jnp.array([0.0, 0.0, OMEGA_EARTH_RADPS])


def geodetic_to_ecef(lat_rad, lon_rad, alt_m) -> jnp.ndarray:
    """Geodetic (lat, lon, ellipsoid height) to ECEF position."""
    sin_lat = jnp.sin(lat_rad)
    cos_lat = jnp.cos(lat_rad)
    n = WGS84_A_M / jnp.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    return jnp.array(
        [
            (n + alt_m) * cos_lat * jnp.cos(lon_rad),
            (n + alt_m) * cos_lat * jnp.sin(lon_rad),
            (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
        ]
    )


def ecef_to_geodetic(r_ecef: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ECEF position to (lat_rad, lon_rad, ellipsoid height).

    Bowring's method with a fixed iteration count (JIT-friendly; converges to
    sub-millimeter for all altitudes this mission sees).
    """
    x, y, z = r_ecef[0], r_ecef[1], r_ecef[2]
    lon = jnp.arctan2(y, x)
    p = jnp.hypot(x, y)
    # Reduced latitude seed, then Bowring fixed-point iterations.
    beta = jnp.arctan2(z, (1.0 - WGS84_F) * p)
    lat = beta
    for _ in range(4):
        lat = jnp.arctan2(
            z + WGS84_EP2 * WGS84_B_M * jnp.sin(beta) ** 3,
            p - WGS84_E2 * WGS84_A_M * jnp.cos(beta) ** 3,
        )
        beta = jnp.arctan((1.0 - WGS84_F) * jnp.tan(lat))
    sin_lat = jnp.sin(lat)
    # h = p cos(lat) + z sin(lat) - a*W is robust at all latitudes (no cos division).
    w = jnp.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    alt = p * jnp.cos(lat) + z * sin_lat - WGS84_A_M * w
    return lat, lon, alt


def geodetic_altitude(r_ecef: jnp.ndarray) -> jnp.ndarray:
    """Ellipsoid height only (the altitude observable, WHITEPAPER 12.3)."""
    return ecef_to_geodetic(r_ecef)[2]


def ned_basis(lat_rad, lon_rad) -> jnp.ndarray:
    """Rows are the local NED unit vectors (north, east, down) in ECEF.

    v_ned = R @ v_ecef; v_ecef = R.T @ v_ned.
    """
    sin_lat, cos_lat = jnp.sin(lat_rad), jnp.cos(lat_rad)
    sin_lon, cos_lon = jnp.sin(lon_rad), jnp.cos(lon_rad)
    north = jnp.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    east = jnp.array([-sin_lon, cos_lon, 0.0])
    down = jnp.array([-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat])
    return jnp.stack([north, east, down])


def ellipsoid_up(lat_rad, lon_rad) -> jnp.ndarray:
    """The geodetic vertical (ellipsoid normal, pointing away from Earth)."""
    return -ned_basis(lat_rad, lon_rad)[2]


def gravity_accel(r_ecef: jnp.ndarray) -> jnp.ndarray:
    """Point-mass gravitation g(r) = -mu r / |r|^3 (WHITEPAPER 6)."""
    r_norm = jnp.linalg.norm(r_ecef)
    return -MU_EARTH_M3S2 * r_ecef / r_norm**3


def coriolis_accel(v_ecef: jnp.ndarray) -> jnp.ndarray:
    """-2 omega x v, the Coriolis term of the rotating frame (WHITEPAPER 5.1)."""
    return -2.0 * jnp.cross(OMEGA_E_VEC, v_ecef)


def centrifugal_accel(r_ecef: jnp.ndarray) -> jnp.ndarray:
    """-omega x (omega x r), the centrifugal term (WHITEPAPER 5.1)."""
    return -jnp.cross(OMEGA_E_VEC, jnp.cross(OMEGA_E_VEC, r_ecef))


def frame_accel(r_ecef: jnp.ndarray, v_ecef: jnp.ndarray) -> jnp.ndarray:
    """Total fictitious acceleration of the rotating ECEF frame."""
    return coriolis_accel(v_ecef) + centrifugal_accel(r_ecef)


def apparent_gravity(r_ecef: jnp.ndarray) -> jnp.ndarray:
    """Gravitation + centrifugal: what a plumb line at rest experiences."""
    return gravity_accel(r_ecef) + centrifugal_accel(r_ecef)
