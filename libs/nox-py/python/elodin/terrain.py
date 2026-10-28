"""Fixed WGS84 terrain for simulation, prepared once and queried with JAX."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
import numpy as np

from .geodesy import WGS84_A_M, WGS84_E2, ecef_to_geodetic, enu_basis


class TerrainSample(NamedTuple):
    """Ellipsoid height (meters), outward ECEF unit normal, and coverage validity."""

    height_m: Array
    normal_ecef: Array
    valid: Array


class TerrainHeightmap:
    """Bilinear WGS84 ellipsoid heights and surface normals on a bounded grid."""

    def __init__(self, data: dict):
        if data.get("version") != 1 or data.get("ellipsoid") != "WGS84":
            raise ValueError("Terrain heights require version 1 and the WGS84 ellipsoid")
        self.asset_id = data["asset_id"]
        self.bounds = np.asarray(data["bounds_degrees"], dtype=np.float64)
        rows, columns = data["rows"], data["columns"]
        if (
            type(self.asset_id) is not int
            or self.asset_id <= 0
            or self.bounds.shape != (4,)
            or not np.all(np.isfinite(self.bounds))
            or type(rows) is not int
            or type(columns) is not int
            or min(rows, columns) < 2
            or rows * columns > 25_000_000
        ):
            raise ValueError("Invalid terrain grid geometry")
        west, south, east, north = self.bounds
        if not (-180 <= west < east <= 180 and -90 < south < north < 90):
            raise ValueError("Invalid terrain coverage")
        heights = np.asarray(data["heights_m"], dtype=np.float64)
        if heights.size != rows * columns or not np.all(np.isfinite(heights)):
            raise ValueError("Terrain grid has missing or invalid heights")
        self.heights = jnp.asarray(heights.reshape(rows, columns))
        self.west, self.south, self.east, self.north = np.radians(self.bounds)
        self.dx = (self.east - self.west) / (columns - 1)
        self.dy = (self.north - self.south) / (rows - 1)

    @classmethod
    def load(cls, path: Path) -> TerrainHeightmap:
        """Load a saved grid without network access; expose its path and SHA-256."""
        path = Path(path)
        content = path.read_bytes()
        result = cls(json.loads(content))
        result.path = path.resolve()
        result.sha256 = hashlib.sha256(content).hexdigest()
        return result

    def sample_geodetic(self, latitude, longitude):
        """Query radians; outside coverage, values are clamped and valid is false."""
        finite = jnp.isfinite(latitude) & jnp.isfinite(longitude)
        latitude = jnp.where(finite, latitude, self.south)
        longitude = jnp.where(finite, longitude, self.west)
        rows, columns = self.heights.shape
        x = jnp.clip((longitude - self.west) / self.dx, 0.0, columns - 1.0)
        y = jnp.clip((latitude - self.south) / self.dy, 0.0, rows - 1.0)
        ix = jnp.minimum(jnp.floor(x).astype(jnp.int32), columns - 2)
        iy = jnp.minimum(jnp.floor(y).astype(jnp.int32), rows - 2)
        u, v = x - ix, y - iy
        h00, h10 = self.heights[iy, ix], self.heights[iy, ix + 1]
        h01, h11 = self.heights[iy + 1, ix], self.heights[iy + 1, ix + 1]
        height = (1 - v) * ((1 - u) * h00 + u * h10) + v * ((1 - u) * h01 + u * h11)
        dh_lon = ((1 - v) * (h10 - h00) + v * (h11 - h01)) / self.dx
        dh_lat = ((1 - u) * (h01 - h00) + u * (h11 - h10)) / self.dy
        w = jnp.sqrt(1 - WGS84_E2 * jnp.sin(latitude) ** 2)
        east_scale = (WGS84_A_M / w + height) * jnp.cos(latitude)
        north_scale = WGS84_A_M * (1 - WGS84_E2) / w**3 + height
        east, north, up = enu_basis(latitude, longitude)
        normal = up - east * dh_lon / east_scale - north * dh_lat / north_scale
        normal = normal / jnp.linalg.norm(normal)
        covered = (
            finite
            & (longitude >= self.west)
            & (longitude <= self.east)
            & (latitude >= self.south)
            & (latitude <= self.north)
        )
        return TerrainSample(height, normal, covered)

    def sample(self, position_ecef):
        """Query one ECEF position in meters; compatible with jax.jit and jax.vmap.

        Always check ``valid`` before using the result. No I/O occurs here.
        Height is above the WGS84 ellipsoid; normal is an ECEF unit vector.
        """
        position = jnp.asarray(position_ecef)
        if position.shape != (3,):
            raise ValueError("Expected an ECEF position with shape (3,)")
        latitude, longitude, _ = ecef_to_geodetic(position)
        result = self.sample_geodetic(latitude, longitude)
        valid = result.valid & jnp.all(jnp.isfinite(position)) & (jnp.linalg.norm(position) > 1)
        return TerrainSample(result.height_m, result.normal_ecef, valid)


def terrain_heightmap(
    center_ecef,
    dimensions: tuple[float, float],
    spacing: float,
    *,
    asset_id: int = 1,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    timeout: int = 600,
) -> TerrainHeightmap:
    """Prepare a fixed Cesium heightmap before building a simulation.

    ``center_ecef`` is a WGS84 ECEF position in meters. ``dimensions`` gives
    east-west and north-south coverage in meters at the center; ``spacing``
    is the maximum sample interval there. The grid follows latitude/longitude
    lines. Areas crossing a pole or the antimeridian are not supported.

    Cache misses use the installed Elodin CLI and its Bevy-Cesium backend.
    Set CESIUM_ION_TOKEN for downloads. Existing cached data is never refreshed
    automatically. ``offline=True`` requires a cache hit. For portable replay,
    retain ``result.path`` and load it with ``TerrainHeightmap.load(path)``.
    Network and disk access happen only here, never in ``sample``.
    ``timeout`` applies to each native batch of up to one million positions.
    A failed preparation is retried up to twice with a fresh native session.
    """
    center = np.asarray(center_ecef, dtype=np.float64)
    size = np.asarray(dimensions, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)) or np.linalg.norm(center) < 1:
        raise ValueError("center_ecef must be a finite nonzero ECEF position (3,)")
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError("dimensions must contain two positive finite lengths in meters")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive finite meters")
    if type(asset_id) is not int or asset_id <= 0:
        raise ValueError("asset_id must be a positive integer")
    if type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ValueError("timeout must be an integer between 1 and 3600 seconds per query")
    intervals = [float(length) / spacing for length in size]
    if any(not math.isfinite(value) or value >= 25_000_000 for value in intervals):
        raise ValueError("Terrain grid is too large; reduce dimensions or increase spacing")
    columns, rows = (math.ceil(value) + 1 for value in intervals)
    if rows * columns > 25_000_000:
        raise ValueError("Terrain grid is too large; reduce dimensions or increase spacing")
    latitude, longitude, _ = ecef_to_geodetic(jnp.asarray(center))
    lat, lon = float(latitude), float(longitude)
    w = math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    north_scale = WGS84_A_M * (1 - WGS84_E2) / w**3
    east_scale = WGS84_A_M / w * math.cos(lat)
    if east_scale < 1:
        raise ValueError("Terrain grids at a pole are not supported")
    half_lat, half_lon = size[1] / (2 * north_scale), size[0] / (2 * east_scale)
    bounds = np.degrees([lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat]).tolist()
    west, south, east, north = bounds
    if not (-180 <= west < east <= 180 and -90 < south < north < 90):
        raise ValueError("Terrain coverage must not cross a pole or the antimeridian")
    request = dict(asset_id=asset_id, bounds_degrees=bounds, columns=columns, rows=rows)
    key = hashlib.sha256(json.dumps({"version": 1, **request}, sort_keys=True).encode()).hexdigest()
    cache = (
        Path(cache_dir)
        if cache_dir is not None
        else (
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "elodin/terrain-heights"
        )
    )
    path = cache / f"{key}.json"
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Terrain cache missing in offline mode: {path}")
        if not os.environ.get("CESIUM_ION_TOKEN"):
            raise RuntimeError("Set CESIUM_ION_TOKEN to prepare terrain heights")
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix="prepare-") as temporary:
            input_path = Path(temporary) / "request.json"
            output_path = Path(temporary) / "heights.json"
            input_path.write_text(json.dumps(request))
            batches = math.ceil(rows * columns / 1_000_000)
            for attempt in range(3):
                try:
                    subprocess.run(
                        [
                            os.environ.get("ELODIN_EDITOR_BIN", "elodin"),
                            "terrain-heights",
                            str(input_path),
                            str(output_path),
                            "--timeout-seconds",
                            str(timeout),
                        ],
                        check=True,
                        timeout=timeout * batches + 60,
                        stdout=subprocess.DEVNULL,
                    )
                    break
                except FileNotFoundError as error:
                    raise RuntimeError(
                        "Terrain preparation needs the Elodin CLI with Cesium support"
                    ) from error
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    if attempt == 2:
                        raise
                    output_path.unlink(missing_ok=True)
                    time.sleep(attempt + 1)
            result = _load_requested(output_path, request)
            # Publish a complete file atomically, without replacing another worker's cache.
            try:
                os.link(output_path, path)
            except FileExistsError:
                pass
            else:
                result.path = path.resolve()
                return result
    return _load_requested(path, request)


def _load_requested(path, request):
    result = TerrainHeightmap.load(path)
    # JSON float parsing in the native CLI may round the last binary digit.
    if (
        not np.allclose(result.bounds, request["bounds_degrees"], rtol=0, atol=1e-12)
        or result.asset_id != request["asset_id"]
        or result.heights.shape != (request["rows"], request["columns"])
    ):
        raise ValueError("Terrain heights do not match the requested asset or coverage")
    return result
