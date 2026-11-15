"""Terrain declared before flight: coarse CRS-12 corridor and nested launch/landing areas."""

import math
import os

import elodin as el
from constants import LZ1_LAT_DEG, LZ1_LON_DEG, PAD_LAT_DEG, PAD_LON_DEG
from reference import build_reference

EARTH_RADIUS_M = 6371008.8
# (half-width m, max_error m): reserve fine detail for the ground views;
# the wider rings provide progressively coarser context during flight.
# Leave headroom above provider levels (~4, 16, 128, 514 and 1027 m), rather than
# accidentally requesting the next finer level with power-of-two thresholds.
SITE_LEVELS = ((500.0, 5.0), (2000.0, 20.0), (8000.0, 150.0), (32000.0, 600.0))
CORRIDOR_MARGIN_M = 20000.0
CORRIDOR_MAX_ERROR_M = 1200.0


def _margins(half_m, lat_deg):
    d_lat = math.degrees(half_m / EARTH_RADIUS_M)
    return d_lat / math.cos(math.radians(lat_deg)), d_lat


def regions():
    reference = build_reference(os.environ.get("ELODIN_FALCON9_MISSION", "crs12"))
    bearing = math.radians(47.67)
    lat = math.radians(PAD_LAT_DEG)
    lon = math.radians(PAD_LON_DEG)
    points = [(PAD_LON_DEG, PAD_LAT_DEG), (LZ1_LON_DEG, LZ1_LAT_DEG)]
    for distance in [min(reference.downrange_m), max(reference.downrange_m)]:
        arc = distance / EARTH_RADIUS_M
        end_lat = math.asin(
            math.sin(lat) * math.cos(arc) + math.cos(lat) * math.sin(arc) * math.cos(bearing)
        )
        end_lon = lon + math.atan2(
            math.sin(bearing) * math.sin(arc) * math.cos(lat),
            math.cos(arc) - math.sin(lat) * math.sin(end_lat),
        )
        points.append((math.degrees(end_lon), math.degrees(end_lat)))
    # A declared operating region, not a guarantee for arbitrary Monte Carlo
    # trajectories or free-camera views.
    margin_lon, margin_lat = _margins(CORRIDOR_MARGIN_M, PAD_LAT_DEG)
    bounds = (
        min(p[0] for p in points) - margin_lon,
        min(p[1] for p in points) - margin_lat,
        max(p[0] for p in points) + margin_lon,
        max(p[1] for p in points) + margin_lat,
    )
    result = [el.TerrainRegion(bounds, max_error=CORRIDOR_MAX_ERROR_M)]
    for site_lon, site_lat in [(PAD_LON_DEG, PAD_LAT_DEG), (LZ1_LON_DEG, LZ1_LAT_DEG)]:
        for half_m, max_error in SITE_LEVELS:
            d_lon, d_lat = _margins(half_m, site_lat)
            result.append(
                el.TerrainRegion(
                    (site_lon - d_lon, site_lat - d_lat, site_lon + d_lon, site_lat + d_lat),
                    max_error=max_error,
                )
            )
    return tuple(result)
