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
