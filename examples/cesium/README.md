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

On a 16 GB M2, start with one Editor instance and the default 16 px screen error.
Increasing `CESIUM_MAXIMUM_SCREEN_ERROR` to `24` or `32` reduces requested detail;
setting `CESIUM_LOD_RETENTION_METERS=0` releases detail sooner outside the current
views. Neither setting is a cap on total process or GPU memory. Validate the
color and infrared views together before claiming a frame rate. Do not run a
standalone viewer alongside RC-jet when measuring.

Photo 3D keeps a compact Google Maps attribution next to the Sources menu;
provider details are available in that menu. Preserve the required attributions
in shared screenshots and video. The current integration does not provide a persistent offline terrain
package or a frame-by-frame video export. For a promotional Google-tile video,
keep the clip at most 30 seconds and include the required promotional notice;
see the [provider's capture and attribution guidance](https://developers.google.com/maps/documentation/tile/policies).

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
| Color | 640 × 480 | 30 fps | H.264 |
| Infrared | 640 × 512, 18° horizontal field of view | 15 fps | H.264 |

Both panels are enabled. Infrared retains the Boson-style optics and LWIR effect;
the planar mode retains its original 60 fps raw infrared configuration. H.264
previews are lossy; use raw captures for pixel comparisons.

Co-located sensors with matching orientations load terrain through the wider
view first. Once it is ready, the zoomed view requests additional detail without
blocking either video stream. Initial IR images can therefore use coarser terrain
while refinement continues. Fixed-reference captures still wait for their
requested terrain detail.

Cesium imagery contains no measured temperatures. Terrain infrared uses Elodin's
lighting-based estimate; tagged objects use their configured temperatures.
Thermal fidelity and longer flights still need validation. Achieved frame rates
and loading times depend on the machine and connection.

### Loading and controls

The Editor displays a loading message until its initial terrain is ready. The
simulation continues during that load. Sensor cameras load their first pose
before emitting images, then follow the live simulation. On macOS, the Editor
is limited to about 24 fps when sensor panels are present to leave GPU capacity
for the sensor renderer.

The launcher uses a 500 m detail-retention margin and 16 px screen error. Override
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
