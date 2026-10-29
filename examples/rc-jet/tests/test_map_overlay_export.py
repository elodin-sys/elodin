"""Map preparation preserves coverage boundaries and distinct height references."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from elodin.geodesy import ecef_to_geodetic

spec = importlib.util.spec_from_file_location(
    "export_map_overlay", Path(__file__).resolve().parents[2] / "cesium/export_map_overlay.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_clip_crossing_segment_and_reject_outside_segment():
    bounds = [0.0, 0.0, 1.0, 1.0]
    assert module.clip_segment((-1.0, 0.5), (2.0, 0.5), bounds) == ((0.0, 0.5), (1.0, 0.5))
    assert module.clip_segment((-1.0, 2.0), (2.0, 2.0), bounds) is None


def test_osm_sea_level_label_is_not_used_as_ellipsoid_placement():
    class Terrain:
        bounds = np.array([-118.0, 35.0, -117.0, 36.0])
        sha256 = "test-heightmap"

        def sample_geodetic(self, latitude, longitude):
            return jnp.asarray(1000.0), jnp.zeros(3), jnp.asarray(True)

    source = {
        "osm3s": {"timestamp_osm_base": "2026-10-28T00:00:00Z"},
        "elements": [
            {
                "type": "node",
                "id": 1,
                "lat": 35.3,
                "lon": -117.8,
                "tags": {"name": "Peak", "natural": "peak", "ele": "975"},
            }
        ],
    }
    result = module.prepare(source, Terrain())
    feature = result["features"][0]
    assert feature["tags"]["ele"] == "975"
    assert feature["osm_id"] == 1
    _, _, height = ecef_to_geodetic(jnp.asarray(feature["parts"][0][0]))
    assert float(height) == pytest.approx(1015.0, abs=0.002)


def test_building_outline_is_explicit_about_assumed_height():
    assert module.building_outline_height({"height": "20 ft"}) == (6.096, "osm_height")
    assert module.building_outline_height({"building:levels": "2"}) == (6.0, "osm_levels_3m")
    tags = {"building": "yes"}
    assert module.building_outline_height(tags) == (4.0, "illustrative_default")
    assert "height" not in tags


class FlatTerrain:
    bounds = np.array([-118.0, 35.0, -117.0, 36.0])
    sha256 = "water-test-grid"

    def sample_geodetic(self, latitude, longitude):
        return jnp.asarray(1000.0), jnp.zeros(3), jnp.asarray(True)


def test_stream_is_clipped_and_preserves_intermittent_tag():
    source = {
        "osm3s": {"timestamp_osm_base": "2026-10-28T00:00:00Z"},
        "elements": [
            {
                "type": "way",
                "id": 10,
                "tags": {"waterway": "stream", "intermittent": "yes"},
                "geometry": [{"lon": -118.01, "lat": 35.3}, {"lon": -117.99, "lat": 35.3}],
            }
        ],
    }
    (feature,) = module.prepare(source, FlatTerrain())["features"]
    assert feature["kind"] == "water"
    assert feature["tags"]["intermittent"] == "yes"
    assert feature["name"] == "Stream"
    for point in feature["parts"][0]:
        _, lon, height = ecef_to_geodetic(jnp.asarray(point))
        assert -118.0000001 <= float(np.degrees(lon)) <= -117.9899999
        assert float(height) == pytest.approx(1003.0, abs=0.002)


def test_lake_relation_keeps_inner_shoreline_without_duplicate_member_way():
    def ring(west, south, size):
        return [
            {"lon": lon, "lat": lat}
            for lon, lat in [
                (west, south),
                (west + size, south),
                (west + size, south + size),
                (west, south + size),
                (west, south),
            ]
        ]

    outer, inner = ring(-117.9, 35.3, 0.003), ring(-117.899, 35.301, 0.0005)
    source = {
        "osm3s": {"timestamp_osm_base": "2026-10-28T00:00:00Z"},
        "elements": [
            {"type": "way", "id": 100, "tags": {"natural": "water"}, "geometry": outer},
            {
                "type": "relation",
                "id": 200,
                "tags": {
                    "type": "multipolygon",
                    "natural": "water",
                    "water": "lake",
                    "name": "Test Lake",
                },
                "members": [
                    {"type": "way", "ref": 100, "role": "outer", "geometry": outer},
                    {"type": "way", "ref": 101, "role": "inner", "geometry": inner},
                ],
            },
        ],
    }
    (feature,) = module.prepare(source, FlatTerrain())["features"]
    assert (feature["osm_type"], feature["osm_id"], feature["kind"]) == ("relation", 200, "water")
    assert feature["name"] == "Test Lake"
    assert feature["tags"]["water"] == "lake"
    assert len(feature["parts"]) == 2
    for shoreline in feature["parts"]:
        np.testing.assert_array_equal(shoreline[0], shoreline[-1])


def test_water_outside_heightmap_is_omitted_without_empty_geometry_failure():
    source = {
        "osm3s": {"timestamp_osm_base": "2026-10-28T00:00:00Z"},
        "elements": [
            {
                "type": "way",
                "id": 11,
                "tags": {"natural": "water"},
                "geometry": [{"lon": -119, "lat": 35.3}, {"lon": -118.5, "lat": 35.3}],
            }
        ],
    }
    assert module.prepare(source, FlatTerrain())["features"] == []
