# Cesium terrain and RC-jet

These examples connect Elodin to [Bevy-Cesium](https://github.com/elodin-sys/bevy-cesium)
for streamed terrain and imagery. RC-jet also uses the shared nox-py height API
for ground contact and produces color and infrared camera previews.

## Architecture

There are two paths through the integration:

| Path | Preparation | During simulation |
| --- | --- | --- |
| Rendering | Configure a tileset or Cesium ion asset | Cesium Native selects and downloads tiles; Bevy-Cesium renders them for the Editor and sensor cameras. |
| Physics heights | `el.terrain_heightmap()` asks the Elodin CLI to sample and cache a fixed grid through Bevy-Cesium's `TerrainSampler`. No rendering window is needed. | JAX reads the prepared grid to return height, surface normal and coverage, without network or disk access. |

Elodin supplies camera poses and adapts simulation coordinates to the renderer.
nox-py owns the public height API and cache orchestration. RC-jet supplies the
requested area and implements its own contact response; it has no custom terrain
download or cache script. The CLI is needed for online preparation, while a saved
grid can be loaded offline. See the
[height API reference](../../docs/public/content/reference/python-api.md#terrain-heights).

The physics grid and the visible terrain mesh have different resolutions and
can disagree locally. A cached height grid does not make rendering offline:
streamed terrain and imagery are not yet packaged persistently between runs.

## Build

Run from the Elodin repository root:

```sh
export ELODIN_SHELL_ID=cesium
nix develop
just install py
just install editor
```

`just install editor` prepares the Cesium Native SDK. For direct Cargo commands,
first run `just prepare-cesium` and set `CESIUM_NATIVE_DIR` to its installation
directory, normally `~/.cache/elodin/cesium-0.65.0/install`.

## RC-jet

Inside the Nix shell, run:

```sh
export CESIUM_ION_TOKEN_FILE="$HOME/.config/elodin/cesium-ion-token.txt"
bash examples/cesium/run_rc_jet_terrain.sh
```

Keep the token file outside the repository. The launcher runs
`examples/rc-jet/main.py` with its existing controls, instruments and views. It
sets `ELODIN_RC_JET_TERRAIN=cesium`, defaults to Google Maps Photorealistic 3D
Tiles at the existing Mojave site, and uses World Terrain (`1`) for physics
heights. The original launch without the Cesium script retains planar terrain.

### Launch options

All three options use the existing Mojave location and initial 350-degree route:

| Option | Command (inside the Nix shell) | Rendering and contact |
| --- | --- | --- |
| Historical terrain | `ELODIN_RC_JET_TERRAIN=planar elodin editor examples/rc-jet/main.py` | Existing local Mojave terrain; fixed-altitude ground contact. Unset `CESIUM_ION_ASSET_ID` and `CESIUM_ION_TOKEN` if inherited from another run. |
| Cesium + heightmap | `bash examples/cesium/run_rc_jet_terrain.sh --photo3d off` | Streamed terrain and aerial imagery; physical contact from the prepared height grid. |
| Cesium + heightmap + Photo 3D (default) | `bash examples/cesium/run_rc_jet_terrain.sh` | Streamed Google photogrammetry, including buildings where covered; the same height-grid physics. |

Bevy-Cesium streams the visual tiles through Cesium Native and provides the
sampler used by the Elodin CLI to prepare heights. The nox-py API caches that
grid; JAX reads it during flight. Photo 3D changes the visual source only.

### Photo 3D on / off

```sh
bash examples/cesium/run_rc_jet_terrain.sh --photo3d on
bash examples/cesium/run_rc_jet_terrain.sh --photo3d off
```

On is the default: Google Maps Photorealistic 3D Tiles (ion asset `2275207`),
without a separate imagery overlay. Off selects Cesium terrain (`1` by default)
with aerial imagery (`2` by default). Enable that asset in your ion account and give the token access to it
and the terrain asset used for heights. Coverage and detail vary by location.
The switch is resolved once by the launcher; it is not a live UI toggle.

Physics uses `RuntimeConfig.height_asset_id`, independently of the visual
tileset. The launcher defaults to World Terrain (`1`) for Photo 3D and preserves
the selected terrain asset in standard mode. Set
`ELODIN_RC_JET_HEIGHT_ASSET_ID` explicitly to use another height source. Direct
Python launches also default heights to World Terrain (`1`). Photogrammetry is
visual geometry, not a collision mesh: buildings can intersect the aircraft.

On a 16 GB M2, use one Editor instance. The launcher defaults to 48 px screen
error and no detail-retention margin, requesting coarser tiles and releasing
detail sooner outside the current views. For sharper terrain, override
`CESIUM_MAXIMUM_SCREEN_ERROR=16` (at a higher rendering and streaming cost).
Neither setting is a cap on total process or GPU memory. Validate the
color and infrared views together before claiming a frame rate. Do not run a
standalone viewer alongside RC-jet when measuring.



`runtime_config.load_config()` reads and validates this mode at the entry point:
unset or `planar` selects the planar preset, and `cesium` selects the Cesium preset.
Other values fail explicitly. The resulting immutable `RuntimeConfig` is passed
to the world and schematic builders. The standalone schematic/watch entry point
uses the same loader when called without a configuration.

Camera recording settings are independent of the terrain selection. The presets
retain today's raw versus H.264 defaults, but Python callers can replace them
without changing the terrain, for example:

```python
from dataclasses import replace
from runtime_config import PREVIEW_CAMERAS, load_config

config = replace(load_config(), cameras=PREVIEW_CAMERAS)
```

This example-local configuration does not change Cesium's renderer environment
variables or introduce a general terrain interface.

### Height preparation and contact

RC-jet requests a 24 × 24 km grid with 10 m spacing around its starting ECEF
position. Heights are prepared or loaded from cache before flight; startup prints
the grid path and SHA-256. The aircraft follows the original 350-degree route.

A spring-and-damper response pushes the aircraft out of the ground, and a
post-integration correction prevents its reference point from passing through
the estimated surface. The aircraft continues after impact; there is no damage
model. At the coverage boundary it holds its last valid pose (ground status 2).

Contact protects the aircraft's reference point, not its entire airframe or
forward-mounted camera. The camera can still cross the visible mesh because of
that offset and differences between the grid and rendered terrain. The chase
camera stays above the sampled ground. The heightmap is not a collision mesh.

### Color and infrared previews

| Preview | Resolution | Configured cadence | Recording |
| --- | --- | --- | --- |
| Color | 480 × 360 | 24 fps | H.264 |
| Infrared | 480 × 384, 18° horizontal field of view | 24 fps | H.264 |

Both panels are enabled and share the same configured cadence. Reduced preview
resolution cuts requested sensor pixels per second by about 39% compared with
640×480 at 30 fps plus 640×512 at 15 fps. Color shadows are disabled, and the
Cesium layout omits the redundant target and FPV scene viewports. Infrared
retains the Boson-style field of view and LWIR effect; planar mode retains its
original full-resolution cameras and 60 fps raw infrared configuration.
H.264 previews are lossy; use full-resolution raw captures for pixel comparisons.

Co-located, aligned sensors share the wider view's terrain selection throughout
the flight. This deliberately sacrifices independent zoom refinement: repeated
requests for finer IR tiles could keep the shared renderer on an old published
selection until terrain disappeared from view. Sensors at different positions,
orientations or clipping ranges still contribute their own terrain requirements.


Cesium imagery contains no measured temperatures. Terrain infrared uses Elodin's
lighting-based estimate; tagged objects use their configured temperatures.
Thermal fidelity and longer flights still need validation. Achieved frame rates
and loading times depend on the machine and connection.

### Loading and controls

The Editor displays a loading message until its initial terrain is ready. The
simulation continues during that load. Sensor cameras load their first pose
before emitting images, then follow the live simulation. On macOS, the Editor
is limited to about 30 fps with up to two reduced previews (at most 480×384
and 24 fps each), or about 24 fps for heavier sensor configurations. This leaves
GPU capacity for the sensor renderer; it does not guarantee those frame rates.

The launcher uses no detail-retention margin and 48 px screen error. Override
`CESIUM_LOD_RETENTION_METERS` or `CESIUM_MAXIMUM_SCREEN_ERROR` to compare settings;
a zero margin disables retention.

For an unattended five-minute flight, set `ELODIN_NON_INTERACTIVE=1` and
`ELODIN_MAX_TICKS=90000`. This keeps the initial trim commands and disables the
keyboard controller. Interactive keyboard input is global: typing in another
application can steer the aircraft or change its throttle.

## Drone sample without an ion account

The AGI headquarters photogrammetry sample needs no account or API key:

```sh
export CESIUM_DEMO=agi
export CESIUM_TILESET_URL=https://pelican-public.s3.amazonaws.com/3dtiles/agi-hq/tileset.json
export CESIUM_CREDIT="AGI headquarters sample"
elodin editor examples/cesium/main.py
```

Unset `CESIUM_ION_ASSET_ID` and `CESIUM_ION_TOKEN` first if they are still set from
another example: ion takes precedence over a tileset URL.

Chase, Onboard / Ground and Follow views share one tileset. Telemetry shows a
north-up map, WGS84 position and ground track. The drone uses the existing
`talon-quad-v2.glb` model; keep the attribution overlay visible.

The simulation runs at 60 Hz for five minutes, integrating commanded forces along
a two-minute target loop. It has no terrain collision model. Plotted altitude is
relative to the geographic origin, not ground clearance.

For public Swiss building geometry without a textured ground surface:

```sh
export CESIUM_DEMO=bern
export CESIUM_TILESET_URL=https://3d.geo.admin.ch/ch.swisstopo.swissbuildings3d.3d/v1/tileset.json
export CESIUM_CREDIT="Data: swisstopo"
elodin editor examples/cesium/main.py
```

For another site, adapt the origin and flight path. Data sources:
[AGI / Cesium samples](https://github.com/CesiumGS/3d-tiles/blob/main/RESOURCES.md)
and [swisstopo](https://docs.geo.admin.ch/visualize-data/3d-tiles.html).

## Rendering scope and controlled captures

Bevy-Cesium is pinned to a Git revision and owns the Rust/C++ bridge, native SDK
build and reusable renderer. The Editor supplies camera views, floating-origin
placement and the credits overlay. Streaming starts only when a source is configured.

Supported features include multi-camera streaming, tile eviction, triangle meshes,
base-color textures, metallic/roughness and emissive factors, alpha modes and
double-sided surfaces. Ion terrain supports one process-wide imagery overlay via
`CESIUM_ION_IMAGERY_ASSET_ID`; meshes are replaced as imagery arrives. Full glTF
materials, general overlay composition, GPU instancing, point clouds, collision
meshes and persistent offline packages are not implemented.

`ELODIN_CESIUM_CAPTURE_VIEWS` accepts a JSON file of reference cameras.
`rc-jet-reference-views.json` covers a sample 2 km corridor; use references that
cover the actual flight and sensor requirements. See the
[Bevy-Cesium README](https://github.com/elodin-sys/bevy-cesium) for renderer details.

## Architecture review

Rendering, infrared and the height API are reviewed together in
[#914](https://github.com/elodin-sys/elodin/pull/914), which incorporates #915.
Rendering source and physical height source are selected independently. Camera
recording settings are explicit; credentials stay outside the repository. The
height SDK and RC-jet reuse `elodin.geodesy` for shared ECEF conversions.

Remaining discussion: allowing other providers to prepare height grids without
changing the simulation lookup, and validating performance and visual/physics
agreement on longer flights.


### Offline map overlay demonstration

The RC-jet launcher enables an offline OSM overlay: named summits with their
published elevations, named places, and beige roads/tracks with metadata on
hover. The Map menu toggles each layer. Sensor images remain independent.

`data/mojave-overpass.json` is the original Overpass response and
`data/mojave.overpassql` is the exact query. The active prepared overlay uses the original
24 × 24 km flight area. `data/mojave-context-overpass.json` and its query
retain a wider, approximately 100 × 100 km context for later preparation.
Preparing the enlarged terrain grid currently times out; it is not enabled
in the rendered demo. The simulation's physics grid is unchanged.

`data/mojave-overlay.json` contains prepared ECEF geometry. The active overlay uses the existing 10 m flight-site grid; an optional
context grid can be combined with it when available.
Runtime rendering performs no Overpass requests or height sampling.
All data is © OpenStreetMap contributors, ODbL 1.0; source IDs, tags, data
timestamp and heightmap SHA-256 hashes are preserved. See
https://www.openstreetmap.org/copyright.

Rebuild inside `nix develop`:

```sh
uv run examples/cesium/export_map_overlay.py \
  --source examples/cesium/data/mojave-overpass.json \
  --heightmap /path/to/context-heightmap.json \
  --detail-heightmap /path/to/flight-heightmap.json \
  --output examples/cesium/data/mojave-overlay.json
```

OSM `ele` is metres above mean sea level. Placement uses WGS84 ellipsoid
heights; missing OSM elevations are not invented. Roads are tessellated at
approximately 60 m intervals and clipped to the context grid. Existing
RC-jet geodesy performs the ECEF conversion.

Beige roads use one depth-tested 3D mesh, visible in Editor viewports only.
Labels and hover cards are terrain-occlusion tested; up to 12 labels are
tested per viewport per frame to bound CPU cost. Labels are clipped and
decluttered. This is a visualization prototype, not a navigation map; road
placement inherits the sampled grid's accuracy.

Right-drag detaches the demonstration viewport from the moving aircraft,
so orbiting uses a stable terrain reference. **Follow aircraft** or the cube's reset button restores
the live chase view. The demo hides the yellow flight trail and body
axis labels. Set `ELODIN_CESIUM_OVERLAY_FILE=''` to disable the map overlay.
