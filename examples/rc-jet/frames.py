"""WGS84 geodesy and rotating-frame helpers for the ECEF world.

Geodesy primitives come from elodin.geodesy. All positions/velocities are
ECEF meters; angles are radians. The world frame is rotating WGS84 ECEF, so
plant forces include Coriolis and centrifugal terms.
"""

from __future__ import annotations

from elodin.geodesy import (
    ecef_to_geodetic as ecef_to_geodetic,
    geodetic_to_ecef as geodetic_to_ecef,
    ned_basis as ned_basis,
    enu_basis as enu_basis,
)

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

MU_EARTH_M3S2 = 3.986004418e14
OMEGA_EARTH_RADPS = 7.292115e-5

OMEGA_E_VEC = jnp.array([0.0, 0.0, OMEGA_EARTH_RADPS])


def geodetic_altitude(r_ecef: jnp.ndarray) -> jnp.ndarray:
    """Ellipsoid height only (drives atmosphere and ground contact)."""
    return ecef_to_geodetic(r_ecef)[2]


def ellipsoid_up(lat_rad, lon_rad) -> jnp.ndarray:
    """The geodetic vertical (ellipsoid normal, pointing away from Earth)."""
    return -ned_basis(lat_rad, lon_rad)[2]


def gravity_accel(r_ecef: jnp.ndarray) -> jnp.ndarray:
    """Point-mass gravitation g(r) = -mu r / |r|^3."""
    r_norm = jnp.linalg.norm(r_ecef)
    return -MU_EARTH_M3S2 * r_ecef / r_norm**3


def coriolis_accel(v_ecef: jnp.ndarray) -> jnp.ndarray:
    """-2 omega x v, the Coriolis term of the rotating frame."""
    return -2.0 * jnp.cross(OMEGA_E_VEC, v_ecef)


def centrifugal_accel(r_ecef: jnp.ndarray) -> jnp.ndarray:
    """-omega x (omega x r), the centrifugal term."""
    return -jnp.cross(OMEGA_E_VEC, jnp.cross(OMEGA_E_VEC, r_ecef))


def frame_accel(r_ecef: jnp.ndarray, v_ecef: jnp.ndarray) -> jnp.ndarray:
    """Total fictitious acceleration of the rotating ECEF frame."""
    return coriolis_accel(v_ecef) + centrifugal_accel(r_ecef)


def apparent_gravity(r_ecef: jnp.ndarray) -> jnp.ndarray:
    """Gravitation + centrifugal: what a plumb line at rest experiences."""
    return gravity_accel(r_ecef) + centrifugal_accel(r_ecef)


def level_attitude_ecef(lat_rad: float, lon_rad: float, heading_deg: float) -> np.ndarray:
    """Rotation matrix (columns = body axes in ECEF) for a level attitude.

    Body X points along the aviation heading (0 = north, 90 = east) in the
    local horizontal plane, body Z along the geodetic up, body Y left.
    """
    basis = np.asarray(enu_basis(lat_rad, lon_rad))
    east, north, up = basis[0], basis[1], basis[2]
    heading = np.deg2rad(heading_deg)
    forward = np.sin(heading) * east + np.cos(heading) * north
    left = np.cross(up, forward)
    return np.column_stack([forward, left, up])


def quaternion_xyzw_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion [x, y, z, w] (Shepperd's method)."""
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)
