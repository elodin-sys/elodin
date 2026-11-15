"""Terrain declared before flight: nominal CRS-12 corridor and finer launch/landing areas."""

import math
import os

import elodin as el
from constants import LZ1_LAT_DEG, LZ1_LON_DEG, PAD_LAT_DEG, PAD_LON_DEG
from reference import build_reference


def regions():
    reference = build_reference(os.environ.get("ELODIN_FALCON9_MISSION", "crs12"))
    bearing = math.radians(47.67)
    lat = math.radians(PAD_LAT_DEG)
    lon = math.radians(PAD_LON_DEG)
    points = [(PAD_LON_DEG, PAD_LAT_DEG), (LZ1_LON_DEG, LZ1_LAT_DEG)]
    for distance in [min(reference.downrange_m), max(reference.downrange_m)]:
        arc = distance / 6371008.8
        end_lat = math.asin(
            math.sin(lat) * math.cos(arc) + math.cos(lat) * math.sin(arc) * math.cos(bearing)
        )
        end_lon = lon + math.atan2(
            math.sin(bearing) * math.sin(arc) * math.cos(lat),
            math.cos(arc) - math.sin(lat) * math.sin(end_lat),
        )
        points.append((math.degrees(end_lon), math.degrees(end_lat)))
    # 40 km around the nominal ground track. This is a declared operating region,
    # not a guarantee for arbitrary Monte Carlo trajectories or free-camera views.
    margin_lat = math.degrees(40000 / 6371008.8)
    margin_lon = margin_lat / math.cos(lat)
    bounds = (
        min(p[0] for p in points) - margin_lon,
        min(p[1] for p in points) - margin_lat,
        max(p[0] for p in points) + margin_lon,
        max(p[1] for p in points) + margin_lat,
    )
    result = [el.TerrainRegion(bounds, max_error=256.0)]
    for site_lon, site_lat in [(PAD_LON_DEG, PAD_LAT_DEG), (LZ1_LON_DEG, LZ1_LAT_DEG)]:
        result.extend(
            [
                el.TerrainRegion(
                    (site_lon - 0.06, site_lat - 0.06, site_lon + 0.06, site_lat + 0.06),
                    max_error=16.0,
                ),
                el.TerrainRegion(
                    (site_lon - 0.01, site_lat - 0.01, site_lon + 0.01, site_lat + 0.01),
                    max_error=4.0,
                ),
            ]
        )
    return tuple(result)
