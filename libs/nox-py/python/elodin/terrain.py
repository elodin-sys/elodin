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


class TerrainSample(NamedTuple):
    """Ellipsoid height (meters), outward ECEF unit normal, and coverage validity."""

    height_m: Array
    normal_ecef: Array
    valid: Array


WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)


def _ecef_to_geodetic(r_ecef: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ECEF position to (lat_rad, lon_rad, ellipsoid height).

    Bowring's method with a fixed iteration count (JIT-friendly).
    """
    x, y, z = r_ecef[0], r_ecef[1], r_ecef[2]
    lon = jnp.arctan2(y, x)
    p = jnp.hypot(x, y)
    beta = jnp.arctan2(z, (1.0 - WGS84_F) * p)
    lat = beta
    for _ in range(4):
        lat = jnp.arctan2(
            z + WGS84_EP2 * WGS84_B_M * jnp.sin(beta) ** 3,
            p - WGS84_E2 * WGS84_A_M * jnp.cos(beta) ** 3,
        )
        beta = jnp.arctan((1.0 - WGS84_F) * jnp.tan(lat))
    sin_lat = jnp.sin(lat)
    w = jnp.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    alt = p * jnp.cos(lat) + z * sin_lat - WGS84_A_M * w
    return lat, lon, alt


def _ned_basis(lat_rad, lon_rad) -> jnp.ndarray:
    """Rows are the local NED unit vectors (north, east, down) in ECEF.

    v_ned = R @ v_ecef; v_ecef = R.T @ v_ned.
    """
    sin_lat, cos_lat = jnp.sin(lat_rad), jnp.cos(lat_rad)
    sin_lon, cos_lon = jnp.sin(lon_rad), jnp.cos(lon_rad)
    north = jnp.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    east = jnp.array([-sin_lon, cos_lon, 0.0])
    down = jnp.array([-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat])
    return jnp.stack([north, east, down])


def _enu_basis(lat_rad, lon_rad) -> jnp.ndarray:
    """Rows are the local ENU unit vectors (east, north, up) in ECEF."""
    ned = _ned_basis(lat_rad, lon_rad)
    return jnp.stack([ned[1], ned[0], -ned[2]])


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
        east, north, up = _enu_basis(latitude, longitude)
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
        latitude, longitude, _ = _ecef_to_geodetic(position)
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
    if rows * columns > 25_000_000 or columns > 500_000:
        raise ValueError("Terrain grid is too large; reduce dimensions or increase spacing")
    latitude, longitude, _ = _ecef_to_geodetic(jnp.asarray(center))
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
            heights = []
            strip_rows = 1_000_000 // columns
            for start in range(0, rows - 1, strip_rows - 1):
                end = min(start + strip_rows - 1, rows - 1)
                strip = {
                    **request,
                    "rows": end - start + 1,
                    "bounds_degrees": [
                        west,
                        south + (north - south) * start / (rows - 1),
                        east,
                        south + (north - south) * end / (rows - 1),
                    ],
                }
                input_path.write_text(json.dumps(strip))
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
                            timeout=timeout + 60,
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
                data = json.loads(output_path.read_text())
                _validate_request(data, strip)
                TerrainHeightmap(data)
                chunk = data["heights_m"]
                heights.extend(chunk if start == 0 else chunk[columns:])
                output_path.unlink()
            output_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "ellipsoid": "WGS84",
                        **request,
                        "heights_m": heights,
                    }
                )
            )
            TerrainHeightmap.load(output_path)
            # Publish a complete file atomically, without replacing another worker's cache.
            try:
                os.link(output_path, path)
            except FileExistsError:
                pass
    result = TerrainHeightmap.load(path)
    _validate_request(
        {
            "asset_id": result.asset_id,
            "bounds_degrees": result.bounds.tolist(),
            "rows": result.heights.shape[0],
            "columns": result.heights.shape[1],
        },
        request,
    )
    return result


def _validate_request(data, request):
    # JSON float parsing in the native CLI may round the last binary digit.
    bounds = np.asarray(data.get("bounds_degrees", []), dtype=np.float64)
    if (
        bounds.shape != (4,)
        or not np.allclose(bounds, request["bounds_degrees"], rtol=0, atol=1e-12)
        or any(data.get(key) != request[key] for key in ("asset_id", "rows", "columns"))
    ):
        raise ValueError("Terrain heights do not match the requested asset or coverage")
