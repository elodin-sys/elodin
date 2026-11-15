import json
import math
import struct
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from elodin import Terrain, TerrainRegion
from elodin.terrain_prepare import (
    GoogleSource,
    ROOT,
    canonical,
    intersects_geographic,
    prepare,
    region_aabb,
    tile_aabb,
    validate_package,
)


def glb():
    document = b'{"asset":{"version":"2.0"}}'
    document += b" " * (-len(document) % 4)
    return (
        struct.pack("<4sIIII", b"glTF", 2, 20 + len(document), len(document), 0x4E4F534A) + document
    )


def tile(bounds, uri=None, error=1):
    result = {
        "boundingVolume": {"region": [*np.radians(bounds), -100, 100]},
        "geometricError": error,
    }
    if uri:
        result["content"] = {"uri": uri}
    return result


class Source:
    attributions = [{"html": "Test attribution"}]

    def __init__(self):
        self.calls = []
        root = tile((-180, -89, 180, 89), "parent.glb?key=private", 100)
        root["children"] = [
            tile((-0.01, -0.01, 0.01, 0.01), "child.glb?session=private"),
            tile((1, 1, 1.01, 1.01), "outside.glb"),
        ]
        self.data = {
            ROOT: json.dumps(
                {"asset": {"version": "1.0"}, "geometricError": 100, "root": root}
            ).encode()
        }
        for name in ["parent", "child", "outside"]:
            self.data[ROOT.replace("root.json", f"{name}.glb")] = glb()

    def get(self, url):
        key = canonical(url)
        self.calls.append(key)
        return self.data[key]


def request(tmp_path, bounds=(-0.02, -0.02, 0.02, 0.02)):
    return {
        "version": 1,
        "asset_id": 2275207,
        "regions": [{"bounds": bounds, "max_error": 8}],
        "output": str(tmp_path / "terrain"),
    }


def test_declared_region_downloads_without_camera_or_database(tmp_path):
    source = Source()
    spec = request(tmp_path)
    report = prepare(spec, source)
    assert report["tiles"] == 2
    assert not any("outside.glb" in call for call in source.calls)
    directory = Path(spec["output"])
    validate_package(directory, spec)
    for path in directory.rglob("*.json"):
        assert "private" not in path.read_text()
        if path.parent.name == "source" or path.name in {"package.json", "prepared-region.json"}:
            continue

        def references(value):
            if isinstance(value, dict):
                for name, child in value.items():
                    if name == "uri":
                        assert (path.parent / child).is_file()
                        assert "?" not in child
                    references(child)
            elif isinstance(value, list):
                for child in value:
                    references(child)

        references(json.loads(path.read_text()))


def test_prepared_package_reuses_then_extends_coverage(tmp_path):
    source = Source()
    spec = request(tmp_path)
    prepare(spec, source)
    source.calls.clear()
    prepare(spec, source)
    assert source.calls == []
    expanded = request(tmp_path, (-0.02, -0.02, 1.02, 1.02))
    with pytest.raises(ValueError, match="cover"):
        validate_package(Path(spec["output"]), expanded)
    assert prepare(expanded, source)["tiles"] == 3
    validate_package(Path(spec["output"]), spec)


def test_missing_or_changed_tile_fails_package_validation(tmp_path):
    spec = request(tmp_path)
    prepare(spec, Source())
    directory = Path(spec["output"])
    next(directory.glob("tiles/*.glb")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt"):
        validate_package(directory, spec)


def test_failed_extension_preserves_the_previous_package(tmp_path):
    source = Source()
    spec = request(tmp_path)
    prepare(spec, source)
    before = (Path(spec["output"]) / "tileset.json").read_bytes()
    source.data[ROOT.replace("root.json", "outside.glb")] = b"invalid"
    with pytest.raises(ValueError):
        prepare(request(tmp_path, (-0.02, -0.02, 1.02, 1.02)), source)
    assert (Path(spec["output"]) / "tileset.json").read_bytes() == before
    validate_package(Path(spec["output"]), spec)
    assert not (tmp_path / ".terrain.prepare.lock").exists()


def test_existing_geometry_is_reused_when_the_download_cache_is_empty(tmp_path, monkeypatch):
    spec = request(tmp_path)
    prepare(spec, Source())
    downloads = []

    def download(self, request):
        downloads.append(request.full_url)
        assert request.full_url == "https://api.cesium.com/v1/assets/2275207/endpoint"
        return json.dumps({"externalType": "3DTILES", "options": {"url": ROOT}}).encode()

    monkeypatch.setattr(GoogleSource, "download", download)
    source = GoogleSource("test", tmp_path / "empty-cache", Path(spec["output"]))
    assert source.get(ROOT.replace("root.json", "child.glb")) == glb()
    assert len(downloads) == 1
    assert source.hits == 1


def test_valid_package_preparation_needs_no_provider_connection_or_token(tmp_path, monkeypatch):
    from elodin.terrain_prepare import main

    spec = request(tmp_path)
    prepare(spec, Source())
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["terrain_prepare", str(request_path)])
    monkeypatch.delenv("ELODIN_CESIUM_TOKEN", raising=False)

    def no_connection(*args, **kwargs):
        pytest.fail("A validated package must not contact the provider")

    monkeypatch.setattr(GoogleSource, "__init__", no_connection)
    main()


@pytest.mark.parametrize("manifest", [[], {"regions": None}, {"regions": [{}]}])
def test_malformed_package_metadata_has_a_clear_error(tmp_path, manifest):
    spec = request(tmp_path)
    prepare(spec, Source())
    (Path(spec["output"]) / "prepared-region.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest|object|regions"):
        validate_package(Path(spec["output"]), spec)


def test_source_selection_uses_prepared_assets_and_offline_rejects_a_miss(tmp_path, monkeypatch):
    terrain = Terrain(regions=(TerrainRegion((-0.02, -0.02, 0.02, 0.02), 8),))
    source_file = tmp_path / "simulation.py"
    spec = terrain._preparation_request(source_file)
    assert terrain._render_source(source_file) == (2275207, None)
    monkeypatch.setenv("ELODIN_TERRAIN_OFFLINE", "1")
    with pytest.raises(ValueError, match="run elodin terrain prepare"):
        terrain._render_source(source_file)
    prepare(spec, Source())
    assert terrain._render_source(source_file) == (None, "terrain/tileset.json")


def test_preparation_honors_the_db_asset_root_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ELODIN_ASSETS", str(tmp_path / "shared-assets"))
    terrain = Terrain(regions=(TerrainRegion((-0.02, -0.02, 0.02, 0.02), 8),))
    source_file = tmp_path / "models" / "simulation.py"
    spec = terrain._preparation_request(source_file)
    assert spec["output"] == str(tmp_path / "shared-assets" / "terrain")
    prepare(spec, Source())
    monkeypatch.setenv("ELODIN_TERRAIN_OFFLINE", "1")
    assert terrain._render_source(source_file) == (None, "terrain/tileset.json")


def test_geographic_aabb_contains_surface_and_curvature():
    bounds = (-82, 27, -78, 31)
    box = region_aabb(bounds)
    for latitude in np.linspace(27, 31, 21):
        for longitude in np.linspace(-82, -78, 21):
            lat, lon = np.radians([latitude, longitude])
            radius = 6378137 / math.sqrt(1 - 0.0066943799901413165 * math.sin(lat) ** 2)
            xyz = np.array(
                [
                    radius * math.cos(lat) * math.cos(lon),
                    radius * math.cos(lat) * math.sin(lon),
                    radius * (1 - 0.0066943799901413165) * math.sin(lat),
                ]
            )
            assert np.all(box[:, 0] <= xyz) and np.all(xyz <= box[:, 1])


def test_box_transform_and_region_ignores_transform():
    matrix = np.eye(4)
    matrix[:3, 3] = [100, 200, 300]
    volume = {"box": [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]}
    np.testing.assert_allclose(tile_aabb(volume, matrix), [[99, 101], [198, 202], [297, 303]])
    volume = tile((-1, -1, 1, 1))["boundingVolume"]
    np.testing.assert_equal(tile_aabb(volume, matrix), tile_aabb(volume, np.eye(4)))


def test_angular_filter_keeps_region_edges_and_rejects_remote_tiles():
    region = (-81, 28, -79, 30)
    for latitude in [28, 29, 30]:
        for longitude in [-81, -80, -79]:
            small = tile(
                (longitude - 0.001, latitude - 0.001, longitude + 0.001, latitude + 0.001)
            )["boundingVolume"]
            assert intersects_geographic(tile_aabb(small, np.eye(4)), region)
    outside = tile((-80, 25, -79.99, 25.01))["boundingVolume"]
    assert not intersects_geographic(tile_aabb(outside, np.eye(4)), region)


def test_external_tileset_keeps_relative_references(tmp_path):
    source = Source()
    root = tile((-1, -1, 1, 1), "nested.json?key=private", 100)
    source.data[ROOT] = json.dumps(
        {"asset": {"version": "1.0"}, "geometricError": 100, "root": root}
    ).encode()
    source.data[ROOT.replace("root.json", "nested.json")] = json.dumps(
        {
            "asset": {"version": "1.0"},
            "geometricError": 1,
            "root": tile((-0.01, -0.01, 0.01, 0.01), "child.glb"),
        }
    ).encode()
    spec = request(tmp_path)
    prepare(spec, source)
    directory = Path(spec["output"])
    root_doc = json.loads((directory / "tileset.json").read_text())
    nested = directory / root_doc["root"]["content"]["uri"]
    nested_doc = json.loads(nested.read_text())
    assert (nested.parent / nested_doc["root"]["content"]["uri"]).is_file()


def test_python_preparation_reads_configuration_without_starting_recipes(tmp_path):
    script = tmp_path / "sim.py"
    marker = tmp_path / "sitl-started"
    output = tmp_path / "request.json"
    command = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    script.write_text(f"""import sys
import elodin as el
from elodin import ui
world = el.World()
world.spawn(el.Body(), name="body")
world.schematic(ui.schematic(ui.viewport(), terrain=el.Terrain(regions=(el.TerrainRegion((-1, -1, 1, 1)),))))
world.recipe(el.s10.PyRecipe.process(name="sentinel", cmd=sys.executable, args=["-c", {command!r}]))
@el.map
def noop(pos: el.WorldPos) -> el.WorldPos:
    return pos
world.run(noop)
""")
    completed = subprocess.run(
        [sys.executable, str(script), "terrain-prepare", str(output)],
        env={**os.environ, "ELODIN_TERRAIN_PREPARE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert json.loads(output.read_text())["output"] == str(tmp_path / "assets/terrain")


def test_provider_detail_limit_is_reported_without_fabricating_finer_tiles(tmp_path):
    source = Source()
    root = tile((-0.01, -0.01, 0.01, 0.01), "child.glb", 20)
    source.data[ROOT] = json.dumps(
        {"asset": {"version": "1.0"}, "geometricError": 20, "root": root}
    ).encode()
    report = prepare(request(tmp_path), source)
    assert report["coarser_leaf_tiles"] == 1
    assert report["largest_leaf_error"] == 20


@pytest.mark.parametrize("bounds", [(0, 0, 0, 1), (179, 0, -179, 1), (0, -91, 1, 1)])
def test_invalid_region_rejected_before_download(bounds):
    with pytest.raises(ValueError):
        TerrainRegion(bounds)
