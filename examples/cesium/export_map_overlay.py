"""Prepare the Mojave map demo from Overpass and an existing Elodin height grid.

Run from the repository root in nix develop with uv. Runtime rendering never
contacts Overpass. Elevation labels use OSM's mean-sea-level `ele` tag; placement
uses ellipsoid heights from the same grid as RC-jet physics.
"""

import argparse
import json
import math
import re
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

import elodin as el

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rc-jet"))
from frames import ellipsoid_up, geodetic_to_ecef

jax.config.update("jax_enable_x64", True)


def clip_segment(a, b, bounds):
    west, south, east, north = bounds
    dx, dy = b[0] - a[0], b[1] - a[1]
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, a[0] - west), (dx, east - a[0]), (-dy, a[1] - south), (dy, north - a[1])):
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            lo = max(lo, q / p)
        else:
            hi = min(hi, q / p)
    if lo > hi:
        return None
    return ((a[0] + lo * dx, a[1] + lo * dy), (a[0] + hi * dx, a[1] + hi * dy))


def building_outline_height(tags):
    """Preview extrusion only; preserve absent/unknown OSM heights as absent."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(m|ft)?\s*", tags.get("height", ""))
    if match:
        height = float(match[1]) * (0.3048 if match[2] == "ft" else 1.0)
        if 0.5 <= height <= 200:
            return height, "osm_height"
    try:
        height = float(tags.get("building:levels", "")) * 3.0
        if 0.5 <= height <= 200:
            return height, "osm_levels_3m"
    except ValueError:
        pass
    return 4.0, "illustrative_default"


def is_water(tags):
    return (
        "waterway" in tags or tags.get("natural") == "water" or tags.get("landuse") == "reservoir"
    )


def geometries(element):
    if element["type"] == "way":
        return [element.get("geometry", [])]
    return [
        member["geometry"]
        for member in element.get("members", [])
        if member["type"] == "way"
        and member.get("role", "") in ("outer", "inner", "")
        and len(member.get("geometry", [])) >= 2
    ]


def clipped_parts(geometries, bounds):
    parts = []
    for geometry in geometries:
        # Keep separate relation members separate, including inner shorelines.
        member_parts = []
        for a, b in zip(geometry, geometry[1:]):
            clipped = clip_segment((a["lon"], a["lat"]), (b["lon"], b["lat"]), bounds)
            if clipped is None:
                continue
            a, b = clipped
            # Approximate length is for tessellation only, never geodesy.
            length = 111320 * math.hypot((b[0] - a[0]) * math.cos(math.radians(a[1])), b[1] - a[1])
            count = max(1, math.ceil(length / 60))
            part = [
                (a[0] + (b[0] - a[0]) * i / count, a[1] + (b[1] - a[1]) * i / count)
                for i in range(count + 1)
            ]
            if member_parts and math.dist(member_parts[-1][-1], part[0]) < 1e-9:
                member_parts[-1].extend(part[1:])
            else:
                member_parts.append(part)
        parts.extend(member_parts)
    return parts


def prepare(source, terrain, detail_terrain=None):
    features = []
    positions = []
    tags_to_keep = (
        "name",
        "ref",
        "ele",
        "place",
        "natural",
        "highway",
        "surface",
        "maxspeed",
        "lanes",
        "access",
        "bridge",
        "tunnel",
        "building",
        "building:levels",
        "height",
        "waterway",
        "water",
        "landuse",
        "intermittent",
        "seasonal",
        "tidal",
        "width",
    )
    water_members = {
        member["ref"]
        for element in source["elements"]
        if element["type"] == "relation" and is_water(element.get("tags", {}))
        for member in element.get("members", [])
        if member["type"] == "way"
        and member.get("role", "") in ("outer", "inner", "")
        and len(member.get("geometry", [])) >= 2
    }
    for element in source["elements"]:
        tags = element.get("tags", {})
        if element["type"] == "node":
            kind = "peak" if tags.get("natural") == "peak" else "place"
            parts = [[(element["lon"], element["lat"])]]
        elif element["type"] in ("way", "relation"):
            if "building" in tags:
                kind = "building"
            elif "highway" in tags:
                kind = "road"
            elif is_water(tags):
                kind = "water"
            else:
                continue
            if element["type"] == "relation" and kind != "water":
                continue
            if element["type"] == "way" and kind == "water" and element["id"] in water_members:
                continue
            parts = clipped_parts(geometries(element), terrain.bounds)
        else:
            continue
        if not parts:
            continue
        fallback = (
            tags.get("waterway", tags.get("water", "Water"))
            if kind == "water"
            else ("Building" if kind == "building" else tags.get("highway", "Place"))
        )
        feature = {
            "osm_id": element["id"],
            "osm_type": element["type"],
            "kind": kind,
            "name": tags.get("name") or tags.get("ref") or fallback.replace("_", " ").title(),
            "tags": {key: tags[key] for key in tags_to_keep if key in tags},
            "parts": [],
        }
        if kind == "building":
            feature["outline_height_m"], feature["outline_height_source"] = building_outline_height(
                tags
            )
        for part in parts:
            feature["parts"].append([len(positions), len(part)])
            positions.extend(part)
        features.append(feature)
    result = {
        "version": 1,
        "attribution": "© OpenStreetMap contributors",
        "copyright_url": "https://www.openstreetmap.org/copyright",
        "osm_timestamp": source["osm3s"]["timestamp_osm_base"],
        "bounds_degrees": terrain.bounds.tolist(),
        "heightmap_sha256": terrain.sha256,
        "detail_heightmap_sha256": detail_terrain.sha256 if detail_terrain is not None else None,
        "features": features,
    }
    if not positions:
        return result
    points = np.asarray(positions)
    lon, lat = np.radians(points[:, 0]), np.radians(points[:, 1])
    heights, _, covered = jax.vmap(terrain.sample_geodetic)(jnp.asarray(lat), jnp.asarray(lon))
    assert np.asarray(covered).all(), "Clipped geometry must stay inside the height grid"
    if detail_terrain is not None:
        detail_heights, _, valid = jax.vmap(detail_terrain.sample_geodetic)(
            jnp.asarray(lat), jnp.asarray(lon)
        )
        heights = jnp.where(valid, detail_heights, heights)
    lifts = jnp.asarray(
        [
            lift
            for feature in features
            for _, count in feature["parts"]
            for lift in [3.0 if feature["kind"] in ("road", "building", "water") else 15.0] * count
        ]
    )
    # Use RC-jet's existing forward conversion, not a new formula.
    ecef = np.asarray(geodetic_to_ecef(jnp.asarray(lat), jnp.asarray(lon), heights + lifts)).T
    up = np.asarray(jax.vmap(ellipsoid_up)(jnp.asarray(lat), jnp.asarray(lon)))
    for feature in features:
        parts = []
        for start, count in feature["parts"]:
            base = ecef[start : start + count]
            parts.append(base.round(3).tolist())
            if feature["kind"] == "building":
                roof = base + up[start : start + count] * feature["outline_height_m"]
                parts.append(roof.round(3).tolist())
                parts.extend(
                    np.stack((a, b)).round(3).tolist() for a, b in zip(base[:-1], roof[:-1])
                )
        feature["parts"] = parts
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Raw Overpass JSON export")
    parser.add_argument(
        "--heightmap", type=Path, required=True, help="Cached Elodin heightmap JSON"
    )
    parser.add_argument(
        "--detail-heightmap", type=Path, help="Optional finer grid near the flight site"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        json.loads(args.source.read_text()),
        el.TerrainHeightmap.load(args.heightmap),
        el.TerrainHeightmap.load(args.detail_heightmap) if args.detail_heightmap else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Prepared {len(result['features'])} features in {args.output}")


if __name__ == "__main__":
    main()
