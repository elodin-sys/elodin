import json
import subprocess
from typing import Annotated

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def grid():
    return dict(
        version=1,
        ellipsoid="WGS84",
        asset_id=1,
        bounds_degrees=[-0.001, -0.001, 0.001, 0.001],
        rows=2,
        columns=2,
        heights_m=[100, 120, 100, 120],
    )


def test_ecef_lookup_jit_vmap_and_invalid_positions():
    terrain = el.TerrainHeightmap(grid())
    position = jnp.array([6378137.0 + 200, 0, 0])
    sample = jax.jit(terrain.sample)(position)
    assert sample.height_m == pytest.approx(110)
    assert sample.valid
    assert sample.normal_ecef[1] < 0
    assert np.linalg.norm(sample.normal_ecef) == pytest.approx(1)
    batch = jax.jit(jax.vmap(terrain.sample))(
        jnp.stack([position, position + jnp.array([0, 1000, 0])])
    )
    np.testing.assert_array_equal(batch.valid, [True, False])
    for invalid in [jnp.zeros(3), jnp.array([jnp.nan, 0, 0]), jnp.array([jnp.inf, 0, 0])]:
        assert not jax.jit(terrain.sample)(invalid).valid


def test_ecef_lookup_in_compiled_elodin_system():
    terrain = el.TerrainHeightmap(grid())
    Height = Annotated[jax.Array, el.Component("terrain_test_height", el.ComponentType.F64)]

    @el.dataclass
    class Telemetry(el.Archetype):
        height: Height

    @el.map
    def sample(pos: el.WorldPos) -> Height:
        return terrain.sample(pos.linear()).height_m

    world = el.World()
    world.spawn(
        [
            el.Body(world_pos=el.SpatialTransform(linear=jnp.array([6378337.0, 0, 0]))),
            Telemetry(jnp.asarray(0.0)),
        ],
        name="body",
    )
    simulation = world.build(sample)
    simulation.run(2, show_progress=False)
    history = simulation.history(["body.terrain_test_height"])
    assert np.asarray(history["body.terrain_test_height"].to_numpy())[-1] == pytest.approx(110)


def test_ecef_query_away_from_equator():
    lat, lon = np.radians([35.35, -117.81])
    radius = 6378137 / np.sqrt(1 - 0.0066943799901413165 * np.sin(lat) ** 2)
    position = np.array(
        [
            (radius + 900) * np.cos(lat) * np.cos(lon),
            (radius + 900) * np.cos(lat) * np.sin(lon),
            (radius * (1 - 0.0066943799901413165) + 900) * np.sin(lat),
        ]
    )
    terrain = el.TerrainHeightmap(
        {
            **grid(),
            "bounds_degrees": [-117.82, 35.34, -117.80, 35.36],
            "heights_m": [800, 800, 800, 800],
        }
    )
    result = jax.jit(terrain.sample)(position)
    assert result.valid
    assert result.height_m == pytest.approx(800)
    np.testing.assert_allclose(
        result.normal_ecef,
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        atol=1e-12,
    )


def fake_backend(monkeypatch):
    calls = []

    def run(args, **kwargs):
        from pathlib import Path

        request = json.loads(Path(args[2]).read_text())
        calls.append(request)
        # Match the last-bit rounding possible in native JSON float parsing.
        request = {**request, "bounds_degrees": np.nextafter(request["bounds_degrees"], 0).tolist()}
        assert (
            kwargs["timeout"]
            == 600 * ((request["rows"] * request["columns"] + 999_999) // 1_000_000) + 60
        )
        # Height grows with latitude across the full grid.
        south, north = request["bounds_degrees"][1::2]
        heights = np.repeat(np.linspace(south, north, request["rows"]), request["columns"])
        Path(args[3]).write_text(
            json.dumps(
                {
                    "version": 1,
                    "ellipsoid": "WGS84",
                    **request,
                    "heights_m": heights.tolist(),
                }
            )
        )

    monkeypatch.setenv("CESIUM_ION_TOKEN", "test-only")
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_cache_hit_offline_and_portable_grid(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    args = dict(center_ecef=[6378137, 0, 0], dimensions=(100, 200), spacing=10, cache_dir=tmp_path)
    first = el.terrain_heightmap(**args)
    assert first.heights.shape == (21, 11)
    assert len(calls) == 1
    monkeypatch.delenv("CESIUM_ION_TOKEN")
    second = el.terrain_heightmap(**args, offline=True)
    assert len(calls) == 1
    assert first.sha256 == second.sha256
    assert el.TerrainHeightmap.load(first.path).sha256 == first.sha256
    with pytest.raises(FileNotFoundError, match="offline"):
        el.terrain_heightmap(**{**args, "spacing": 20}, offline=True)


def test_large_grid_uses_one_cli_request(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    terrain = el.terrain_heightmap([6378137, 0, 0], (10000, 10000), 10, cache_dir=tmp_path)
    assert len(calls) == 1
    assert calls[0]["rows"] == calls[0]["columns"] == 1001
    expected = np.linspace(terrain.bounds[1], terrain.bounds[3], 1001)
    np.testing.assert_allclose(terrain.heights[:, 0], expected, atol=1e-12)


@pytest.mark.parametrize(
    "change",
    [
        {"dimensions": (0, 100)},
        {"dimensions": (100,)},
        {"spacing": 0},
        {"spacing": float("nan")},
        {"center_ecef": [0, 0, 0]},
        {"center_ecef": [float("nan"), 0, 0]},
        {"center_ecef": [0, 0, 6356752]},
        {"center_ecef": [-6378137, 0, 0]},
        {"asset_id": -1},
        {"asset_id": True},
        {"timeout": 0},
        {"dimensions": (1000000, 1000000)},
    ],
)
def test_bad_geometry_fails_before_network(tmp_path, change):
    args = dict(center_ecef=[6378137, 0, 0], dimensions=(100, 100), spacing=10, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        el.terrain_heightmap(**{**args, **change})
    assert list(tmp_path.iterdir()) == []


def test_failed_download_does_not_publish_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CESIUM_ION_TOKEN", "test-only")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "elodin")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        el.terrain_heightmap([6378137, 0, 0], (100, 100), 10, cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_wrong_cached_asset_is_rejected(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    args = dict(center_ecef=[6378137, 0, 0], dimensions=(100, 100), spacing=10, cache_dir=tmp_path)
    terrain = el.terrain_heightmap(**args)
    data = json.loads(terrain.path.read_text())
    data["asset_id"] = 2
    terrain.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="match"):
        el.terrain_heightmap(**args, offline=True)


def test_transient_backend_failure_retries_preparation(tmp_path, monkeypatch):
    calls = fake_backend(monkeypatch)
    backend = subprocess.run
    attempts = []

    def intermittent(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(1, "elodin")
        return backend(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", intermittent)
    monkeypatch.setattr("elodin.terrain.time.sleep", lambda _: None)
    terrain = el.terrain_heightmap([6378137, 0, 0], (100, 100), 10, cache_dir=tmp_path)
    assert len(attempts) == 2 and len(calls) == 1
    assert terrain.path.exists()


@pytest.mark.parametrize(
    "change", [{"asset_id": 2}, {"heights_m": []}, {"heights_m": [float("nan")] * 121}]
)
def test_invalid_output_is_not_published(tmp_path, monkeypatch, change):
    from pathlib import Path

    fake_backend(monkeypatch)
    backend = subprocess.run

    def invalid(args, **kwargs):
        backend(args, **kwargs)
        path = Path(args[3])
        path.write_text(json.dumps({**json.loads(path.read_text()), **change}))

    monkeypatch.setattr(subprocess, "run", invalid)
    with pytest.raises(ValueError):
        el.terrain_heightmap([6378137, 0, 0], (100, 100), 10, cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_concurrent_cache_publication_returns_the_winning_file(tmp_path, monkeypatch):
    from pathlib import Path

    fake_backend(monkeypatch)

    def publish_other(source, destination):
        data = json.loads(Path(source).read_text())
        data["heights_m"] = [42] * (data["rows"] * data["columns"])
        Path(destination).write_text(json.dumps(data))
        raise FileExistsError

    monkeypatch.setattr("elodin.terrain.os.link", publish_other)
    terrain = el.terrain_heightmap([6378137, 0, 0], (100, 100), 10, cache_dir=tmp_path)
    assert np.all(terrain.heights == 42)
    assert terrain.sha256 == el.TerrainHeightmap.load(terrain.path).sha256
    assert list(tmp_path.iterdir()) == [terrain.path]
