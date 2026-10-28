"""Shared WGS84 conversions used by terrain sampling and simulation.

Positions and ellipsoid heights are in meters; angles are in radians.
The existing formulas use JAX operations and support jit/vmap.
"""

import jax.numpy as jnp


WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)


def ecef_to_geodetic(r_ecef: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ECEF position to (lat_rad, lon_rad, ellipsoid height).

    Bowring's method with a fixed iteration count (JIT-friendly).
    """
    x, y, z = r_ecef[0], r_ecef[1], r_ecef[2]
    lon = jnp.arctan2(y, x)
    p = jnp.hypot(x, y)
    beta = jnp.arctan2(z, (1.0 - WGS84_F) * p)
    lat = beta
    for _ in range(4):
        lat = jnp.arctan2(
            z + WGS84_EP2 * WGS84_B_M * jnp.sin(beta) ** 3,
            p - WGS84_E2 * WGS84_A_M * jnp.cos(beta) ** 3,
        )
        beta = jnp.arctan((1.0 - WGS84_F) * jnp.tan(lat))
    sin_lat = jnp.sin(lat)
    w = jnp.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    alt = p * jnp.cos(lat) + z * sin_lat - WGS84_A_M * w
    return lat, lon, alt


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


def enu_basis(lat_rad, lon_rad) -> jnp.ndarray:
    """Rows are the local ENU unit vectors (east, north, up) in ECEF."""
    ned = ned_basis(lat_rad, lon_rad)
    return jnp.stack([ned[1], ned[0], -ned[2]])
