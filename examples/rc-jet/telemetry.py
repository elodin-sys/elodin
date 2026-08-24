"""Derived geodetic and local-ENU telemetry components.

The ECEF world position is unreadable on graphs, so publish geodetic
lat/lon/altitude and the position in the site's local ENU frame (meters east/
north/up of the field origin). The ground-track plot reads `pos_enu`.
"""

import typing as ty

import elodin as el
import jax
import jax.numpy as jnp

from frames import ecef_to_geodetic, enu_basis, geodetic_to_ecef

Geodetic = ty.Annotated[
    jax.Array,
    el.Component(
        "geodetic",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": "73", "element_names": "lat_deg,lon_deg,alt_m"},
    ),
]
PosENU = ty.Annotated[
    jax.Array,
    el.Component(
        "pos_enu",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": "72", "element_names": "east_m,north_m,up_m"},
    ),
]


def build_geodetic_telemetry(lat_deg: float, lon_deg: float, elevation_m: float):
    lat = jnp.deg2rad(lat_deg)
    lon = jnp.deg2rad(lon_deg)
    origin = geodetic_to_ecef(lat, lon, elevation_m)
    basis = enu_basis(lat, lon)

    @el.map
    def derive_geodetic(pos: el.WorldPos, _prev: Geodetic) -> tuple[Geodetic, PosENU]:
        # `_prev` scopes the query to entities that carry the telemetry
        # components: a world_pos-only input would also match static scene
        # markers, which lack the outputs.
        lat_v, lon_v, alt_v = ecef_to_geodetic(pos.linear())
        geodetic = jnp.array([jnp.rad2deg(lat_v), jnp.rad2deg(lon_v), alt_v])
        pos_enu = basis @ (pos.linear() - origin)
        return geodetic, pos_enu

    return derive_geodetic
