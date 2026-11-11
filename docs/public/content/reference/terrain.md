+++
title = "Terrain"
description = "Run with cached terrain and streaming; prepare assets for offline simulations and replays."
weight = 104
aliases = ["/reference/cesium/", "/reference/terrain-heights/"]

[extra]
toc = true
order = 4
+++

Choose terrain in your Python schematic, then use the usual commands.
Available terrain assets and cached data are reused; missing tiles stream as needed.
No terrain flag or preparation step is required for normal online use.

## Run and view

| Command | Result |
| --- | --- |
| `elodin run sim.py` | Run the simulation and its sensor renderer, if configured. |
| `elodin editor sim.py` | Run the simulation with the desktop editor. |
| `elodin editor dbs/flight` | Open a recorded run without rerunning the simulation. |

All three use cached terrain first and stream missing data by default.
Add `--offline` to disable terrain-provider requests. DB and SITL connections
remain available.

## Scenery

| Mode | `terrain` | Display |
| --- | --- | --- |
| Legacy | `None` | The existing GLB or world mesh defined in your schematic. |
| Aerial | `el.Terrain.aerial()` | Streamed terrain with aerial images. |
| Photo 3D | `el.Terrain()` | Streamed photogrammetry, including buildings where covered. |
| Prepared assets | `el.Terrain.from_asset("terrain/Falcon9/tileset.json")` | Scenery served with the DB, in native and WASM editors. |

The editor streams aerial terrain and Photo 3D through Bevy-Cesium.
Add [geographic overlays](@/reference/map-overlays.md) for place names, roads and
surfaces stored with the DB, independently of the terrain provider.
Local models are loaded by the schematic, as in
[RC-jet's Mojave terrain](https://github.com/elodin-sys/elodin/blob/main/examples/rc-jet/terrain_schematic.py).

## Prepare for offline runs and SITL

Prepare terrain once when a run needs reproducible, versioned scenery or must
work without a terrain-provider connection. Declare the operating regions in
the Python schematic. Bounds are west, south,
east and north in degrees; `max_error` is the target geometric error in meters.

```python
terrain = el.Terrain(
    regions=(el.TerrainRegion((-80.7, 28.4, -80.5, 28.7), max_error=16.0),),
    directory="terrain/Flight",
)
world.schematic(ui.schematic(ui.viewport(), terrain=terrain))
```

```sh
elodin terrain prepare sim.py
```

- Preparation reads the script configuration without running simulation steps or starting its recipes.
- It writes `assets/terrain/Flight/` beside the Python schematic, or uses the existing `ELODIN_ASSETS` override. Keep the schematic and simulation in the same asset root.
- The package is selected automatically when it covers the declared regions and passes its file checks.
- Repeating preparation reuses a valid package. Extending the regions reuses downloaded geometry and fetches missing resources.
- Overlapping regions use the finer target. Where the provider has no finer tile, its best available geometry is retained and the limitation is reported.
- The declared regions must include the operating area and terrain visible to the sensors. Camera resolution and FOV stay in the existing camera definitions.
- Offline startup rejects a missing, corrupt or insufficiently prepared package.

Test the prepared simulation, with or without the editor:

```sh
elodin run --offline sim.py
elodin editor --offline sim.py
```

The first command runs without the editor; the second opens it. Recorded sensor
cameras wait for initial terrain loading before emitting images; their frames remain
asynchronous. Preparation does not add simulation lockstep.

After validation, version the package with the simulation:

```sh
git lfs track "assets/terrain/**"
git add .gitattributes assets/terrain
git commit -m "Prepare terrain for simulation"
```

Use the asset path relative to your repository. Colleagues and CI can reuse the
checked-out package without preparing it again.

Region preparation currently supports Google Photo 3D explicit tilesets with
self-contained GLB content. It does not prepare Cesium aerial imagery or heightmaps.
The regions describe the intended operating area; they do not cover arbitrary
camera movement beyond it.

## Record and replay

```sh
ELODIN_DB_PATH=dbs/flight elodin run sim.py
elodin editor dbs/flight
```

- The first command creates a recording and includes the simulation's asset tree.
- The second opens that recording, including its prepared terrain and overlays.
- Both retain the normal cache-and-streaming behavior. Add `--offline` to either command when required.
- Runtime streaming does not add tiles to the DB's assets. Rerun preparation to extend the versioned package.
- Assets are ingested when a DB is first created. Use a fresh DB after changing the source assets.

To select an existing package explicitly, put the complete tileset under `assets/`
and use relative child references:

```python
terrain = el.Terrain.from_asset("terrain/Flight/tileset.json")
```

The editor resolves this path through the connected DB's asset server. The package
travels with the recording; provider credentials are not included.
See [DB Asset Server](/reference/db-asset-server/) for storage and replication.

The browser supports explicit 3D Tiles and bounded EPSG:4326/TMS quantized-mesh
layers with static availability. Photo 3D ion sources also work; `Terrain.aerial()`
with ion imagery is currently native-only. Browser detail retention is disabled.
Heightmaps remain independent: load a prepared grid with `TerrainHeightmap.load()`;
`Terrain.from_asset(...).heightmap()` does not prepare one from scenery.

## Browser replay

Connect the browser editor with `?db=HOST:PORT`. Terrain assets come from the DB
automatically; missing Photo 3D tiles stream as needed. Add `&offline=1` to disable
terrain-provider requests while retaining access to the DB and its assets.

Particles and atmosphere require WebGPU. See the
[browser editor README](https://github.com/elodin-sys/elodin/tree/main/apps/elodin-web)
for deployment and a separate asset-server address.

## Startup and outages

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

Cache locations, relative to `$XDG_CACHE_HOME` (default `~/.cache`):

| Data | Directory |
| --- | --- |
| Native streamed terrain | `bevy-cesium/terrain` |
| Geometry reused by `terrain prepare` | `elodin/terrain-photo3d` |
| Prepared height grids | `elodin/terrain-heights` |

The reusable download cache and the versioned package under `assets/` are separate.

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

`Terrain(asset_id=2275207, *, regions=(), directory="terrain")`

| Parameter | Default | Description |
| --- | --- | --- |
| `asset_id` | `2275207` | Positive signed 64-bit ion asset ID. The default is Google Photorealistic 3D Tiles; `1` selects Cesium World Terrain with Bing aerial imagery in the editor. |
| `regions` | `()` | `TerrainRegion(bounds, max_error=16.0)` declarations for preparation. Empty means streamed scenery. |
| `directory` | `"terrain"` | Prepared package directory relative to the source asset root. |

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
