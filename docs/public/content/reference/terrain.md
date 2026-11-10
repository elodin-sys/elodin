+++
title = "Terrain"
description = "Place your simulation, display scenery and query ground heights."
weight = 104
aliases = ["/reference/cesium/", "/reference/terrain-heights/"]

[extra]
toc = true
order = 4
+++

Choose the scenery and location in your Python schematic. Prepare ground heights
from the same terrain when your simulation needs them.

## Scenery

| Mode | `terrain` | Display |
| --- | --- | --- |
| Legacy | `None` | The existing GLB or world mesh defined in your schematic. |
| Aerial | `el.Terrain.aerial()` | Streamed terrain with aerial images. |
| Photo 3D | `el.Terrain()` | Streamed photogrammetry, including buildings where covered. |
| Prepared assets | `el.Terrain.from_asset("terrain/Falcon9/tileset.json")` | Scenery served with the DB, in native and WASM editors. |

The editor streams aerial terrain and Photo 3D through Bevy-Cesium.
Local models are loaded by the schematic, as in
[RC-jet's Mojave terrain](https://github.com/elodin-sys/elodin/blob/main/examples/rc-jet/terrain_schematic.py).

### Scenery stored with the DB

Place the complete prepared tileset under your simulation's `assets/` directory.
Use relative links inside its JSON/glTF files so child tiles, buffers and textures
move with the folder. Select its entry point with `Terrain.from_asset(path)`;
the recorded schematic stores a `db:` key and the editor resolves it against the
connected DB's asset server. No machine-local path or ion credential is stored.

```python
terrain = el.Terrain.from_asset("terrain/Falcon9/tileset.json")
```

This loads existing data; it does not download or prepare a geographic region.
Google Photorealistic 3D Tiles, used by `Terrain()` by default, are streamed
online by default. This integration records the dataset selection in the DB schematic,
not an archive of the streamed tiles.
The DB ingests its source asset tree on first creation. Use a fresh DB when adding
a prepared folder, or explicitly upload the new assets to an existing DB.
See [DB Asset Server](/reference/db-asset-server/) for storage and replication.

The browser supports explicit 3D Tiles and bounded EPSG:4326/TMS quantized-mesh
layers with static availability. Photo 3D ion sources also work; `Terrain.aerial()`
with ion imagery is currently native-only. Browser detail retention is disabled.
Heightmaps remain independent: load a prepared grid with `TerrainHeightmap.load()`;
`Terrain.from_asset(...).heightmap()` does not prepare one from scenery.

### Exporting a cached Photo 3D snapshot

For a local offline prototype, first visit the required views in the native editor
with Photo 3D enabled. Then export the tiles already present in Cesium Native's
SQLite cache, from the repository root inside `nix develop`:

```bash
uv run python scripts/export_photo3d_cache.py examples/falcon9/assets/terrain/Falcon9
FALCON9_TERRAIN_ASSET=terrain/Falcon9/tileset.json elodin editor examples/falcon9/main.py
```

Use a fresh recording so the DB ingests this folder. The exporter makes no network
requests and leaves the source cache unchanged. It copies self-contained GLBs,
rewrites tileset links to relative paths, preserves attribution, and excludes
access keys and session parameters from resource URLs. `--cache` selects a
different `requests.sqlite`; the output directory must not already exist.

`cache-manifest.json` reports tile counts, bytes and missing references. This is
only the reachable cached subset: missing branches are pruned, so new camera
positions may reveal gaps. It does not guarantee coverage of a whole region or
flight. Implicit tilesets and GLBs with external dependencies are rejected.
This prototype is separate from geographic preparation and heightmap generation.

### Startup and outages

When `elodin editor` launches a simulation with streamed terrain, the simulation
stays at t=0 until the initial terrain is ready. Loading has no interactive
deadline. If Cesium reports an error, the editor disables streamed terrain,
restores the schematic's fallback scenery, and starts the simulation. Reload the
schematic to retry.

Live connections are never paused. Automated fixed-view captures remain strict:
`ELODIN_CESIUM_CAPTURE_TIMEOUT` bounds terrain loading, and an error or timeout
exits without writing a terrain screenshot.

```python
import elodin as el
import elodin.ui as ui

location = ui.coordinate(
    frame="ECEF",
    lat=28.60839, lon=-80.60433,
    alt=-7.063,
)
terrain = el.Terrain()

scene = ui.schematic(
    ui.viewport(name="Flight"),
    coordinate=location,
    terrain=terrain,
)
world = el.World()
world.schematic(scene)
```

Latitude and longitude are in degrees; altitude is in meters above the WGS84
ellipsoid.

## Terrain Heights

Prepare a grid from the selected aerial terrain or Photo 3D before running the simulation:

```python
import jax.numpy as jnp
from elodin.geodesy import geodetic_to_ecef

location = geodetic_to_ecef(
    jnp.deg2rad(28.485738),
    jnp.deg2rad(-80.542948),
    0.0,
)
heights = terrain.heightmap(
    location, dimensions=(200.0, 200.0), spacing=10.0,
)
```

Dimensions and spacing are in meters. The grid is cached and reused on later
runs. No editor is needed.

During the simulation:

```python
sample = heights.sample(position_ecef)
# sample.height_m, sample.normal_ecef, sample.valid
```

Use `sample.height_m` and `sample.normal_ecef` in your ground-contact model,
after checking `sample.valid`. Heights are above the WGS84 ellipsoid; normals
are in ECEF. Queries work inside JAX systems.

See [API Reference](#api-reference) for parameters and return values.

## Caching

The native editor and height preparation use a local terrain cache automatically.
Available tiles are reused; missing or expired data is fetched and cached as you
explore new areas. Cache misses produce a terminal warning and loading continues.
Provider cache rules apply to terrain and imagery responses.

Prepared height grids are also saved and reused on subsequent runs. Use
`offline=True` when preparing heights only if the grid must already exist and
network access is not allowed.

Tiles are stored in `$XDG_CACHE_HOME/bevy-cesium/terrain`, or
`~/.cache/bevy-cesium/terrain`. Height grids are stored in
`$XDG_CACHE_HOME/elodin/terrain-heights`, or `~/.cache/elodin/terrain-heights`.

## Run Falcon 9

```sh
elodin editor examples/falcon9/main.py
```

Falcon 9 flies from LC-39A to Landing Zone 1 at Cape Canaveral. See its
[README](https://github.com/elodin-sys/elodin/tree/main/examples/falcon9)
for the simulation and reference flights.

In `examples/falcon9/constants.py`, set `TERRAIN_MODE` to `"legacy"`, `"aerial"`
or `"photo3d"`. This selects both the scenery and the matching launch and landing
ground heights. `"legacy"` uses Falcon 9's original globe. Photo 3D is the default.

## API Reference

### _class_ `elodin.Terrain`

`Terrain(asset_id=2275207)`

| Parameter | Default | Description |
| --- | --- | --- |
| `asset_id` | `2275207` | Positive signed 64-bit ion asset ID. The default is Google Photorealistic 3D Tiles; `1` selects Cesium World Terrain with Bing aerial imagery in the editor. |

Constructing a `Terrain` performs no network or disk access.

- `Terrain.aerial()` → `Terrain`

    Select Cesium World Terrain with aerial imagery. Heights come from Cesium
    World Terrain; imagery does not change the surface used for height queries.

- `heightmap(center_ecef, dimensions, spacing, *, cache_dir=None, offline=False, timeout=600)` → `TerrainHeightmap`

    Prepare a grid from this terrain before running the simulation.

    | Parameter | Default | Description |
    | --- | --- | --- |
    | `center_ecef` | Required | Three-element WGS84 ECEF position, in meters. |
    | `dimensions` | Required | `(east_west, north_south)` coverage in meters; both lengths must be positive. |
    | `spacing` | Required | Maximum sample spacing in meters at the grid center; must be positive. |
    | `cache_dir` | `None` | Directory for saved height grids. Defaults to `$XDG_CACHE_HOME/elodin/terrain-heights`, or `~/.cache/elodin/terrain-heights`. Accepts a string or `Path`. |
    | `offline` | `False` | Require an existing cached grid; raise `FileNotFoundError` on a cache miss. |
    | `timeout` | `600` | Integer seconds per native batch, from `1` to `3600`. Failed preparation is retried up to twice. |

    The grid includes both edges and contains at least two samples along each
    axis. Coverage must not cross a pole or the antimeridian. Grids are limited
    to 25 million samples. An existing cached grid is reused; preparation needs
    the Elodin CLI on a cache miss. The editor does not need to be open.

    `elodin.terrain_heightmap(center_ecef, dimensions, spacing, *, asset_id=2275207, cache_dir=None, offline=False, timeout=600)`
    provides the same operation with an explicit asset ID.

### _class_ `elodin.TerrainHeightmap`

- `TerrainHeightmap.load(path)` → `TerrainHeightmap`

    Load a saved JSON grid without network access. `path` accepts a string or `Path`.

- `TerrainHeightmap(data)` → `TerrainHeightmap`

    Construct from a decoded grid dictionary. Required fields are `version=1`,
    `ellipsoid="WGS84"`, `asset_id`, `bounds_degrees=[west, south, east, north]`,
    `rows`, `columns`, and `heights_m` in row-major order from south to north.
    Heights must be finite, with `rows * columns` values.

- `sample(position_ecef)` → `TerrainSample`

    Query a three-element ECEF position in meters. Compatible with `jax.jit`
    and `jax.vmap`; performs no I/O.

- `sample_geodetic(latitude, longitude)` → `TerrainSample`

    Query geodetic latitude and longitude in radians.

Queries interpolate heights bilinearly. Outside coverage, values are clamped
to the grid boundary and `valid` is false. Invalid input positions also return
`valid=False`; check it before using the height or normal.

Prepared and loaded grids expose `path` (absolute file path) and `sha256` (file
checksum). `asset_id` identifies the dataset; `bounds` contains west, south,
east and north limits in degrees; `heights` is the two-dimensional JAX height array.

### _class_ `elodin.TerrainSample`

| Field | Description |
| --- | --- |
| `height_m` | Surface altitude in meters above the WGS84 ellipsoid. |
| `normal_ecef` | Outward unit surface normal in ECEF coordinates. |
| `valid` | Whether the input position is valid and inside the grid coverage. |
