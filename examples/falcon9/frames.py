"""WGS84 geodesy and rotating-frame helpers (WHITEPAPER sections 4-5).

Pure JAX-compatible functions shared by the plant systems, the reference
tooling, and the tests. All positions/velocities are ECEF meters unless
stated; angles are radians. Geodesy primitives come from elodin.geodesy.
"""

from __future__ import annotations

from elodin.geodesy import (
    ecef_to_geodetic as ecef_to_geodetic,
    geodetic_to_ecef as geodetic_to_ecef,
    ned_basis as ned_basis,
)

import jax
import jax.numpy as jnp

from constants import (
    MU_EARTH_M3S2,
    OMEGA_EARTH_RADPS,
)

jax.config.update("jax_enable_x64", True)

OMEGA_E_VEC = jnp.array([0.0, 0.0, OMEGA_EARTH_RADPS])


def geodetic_altitude(r_ecef: jnp.ndarray) -> jnp.ndarray:
    """Ellipsoid height only (the altitude observable, WHITEPAPER 12.3)."""
    return ecef_to_geodetic(r_ecef)[2]


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
