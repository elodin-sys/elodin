# Cesium examples

A simulated drone flies over the AGI headquarters photogrammetry sample.
The Editor uses [bevy-cesium](https://github.com/elodin-sys/bevy-cesium):
Cesium Native streams and decodes the tiles; Bevy renders them in Elodin's viewports.
No account or API key is needed for this dataset.

## Run

From the Elodin repository root:

```sh
export ELODIN_SHELL_ID=cesium
nix develop
just install py
just install editor

export CESIUM_DEMO=agi
export CESIUM_TILESET_URL=https://pelican-public.s3.amazonaws.com/3dtiles/agi-hq/tileset.json
export CESIUM_CREDIT="AGI headquarters sample"
elodin editor examples/cesium/main.py
```

## What to look for

- **Chase:** a close rear view that keeps the drone clearly visible, with an interactive ENU viewcube.
- **Onboard / Ground:** a downward-facing camera showing the roofs, roads and parking areas below.
- **Follow:** a close three-quarter view of the drone against the streamed buildings.
- **Telemetry:** a north-up site map, WGS84 position, altitude and East/North ground track. The orange trail uses simulated positions.

The cameras share one tileset. The drone uses the existing `talon-quad-v2.glb` model.
Allow the site to load before recording; keep the attribution overlay visible.

The simulation runs at 60 Hz for five minutes. Its target follows a two-minute
loop, with altitude varying by 10 m around 70 m above the geographic origin.
Elodin integrates the commanded forces: this is not a prerecorded trajectory.
The display heading follows velocity; there is no attitude controller or terrain
collision model. The plotted altitude is relative to the origin, **not ground clearance**.

## RC-jet

For RC-jet over Cesium World Terrain near Ridgecrest, run inside the Nix shell:

```sh
export CESIUM_ION_TOKEN_FILE="$HOME/.config/elodin/cesium-ion-token.txt"
bash examples/cesium/run_rc_jet_terrain.sh
```

The token file contains only the ion token and stays outside the repository.
The launcher runs the original
`examples/rc-jet/main.py`, preserving its controls, instruments and 3D views.
It selects ion asset `1` for relief and asset `2` for aerial imagery.

RC-jet calls `el.terrain_heightmap()` with its starting ECEF position, a
24 × 24 km area and 10 m spacing. The SDK prepares and caches heights before
flight. The example contains no download or cache script. The original
350-degree route uses these heights for ground contact.

The lookup returns height, slope normal and a coverage flag using JAX operations.
There are no network calls during physics steps. The grid file and its SHA-256
are printed at startup. See the [terrain API](../../docs/public/content/reference/python-api.md#terrain-heights)
for cache, offline loading and coverage options.

**Why the aircraft bounces:** a spring-and-damper response pushes it out of the
ground. A correction after integration prevents its reference point from passing
through the estimated surface. The aircraft keeps moving after an impact;
there is no damage model. At the grid boundary it holds its last valid pose
(ground status 2), rather than inventing terrain beyond the prepared area.

**Why the onboard camera can cross the ground:** heights are interpolated between
grid points and can differ from the visible mesh. Contact protects the aircraft's
reference point, not the whole airframe or its forward-mounted camera. The chase
camera stays above the sampled terrain; full airframe and camera collisions
remain future work. This heightmap is not a collision mesh.

Both sensor cameras are enabled. The color preview runs at 30 fps; the infrared
preview retains the Boson-style 640 × 512 resolution, 18° horizontal lens and
LWIR effect, with a 15 fps cadence. Both use H.264 to limit recording size.
These are lossy video previews; use raw captures for pixel comparisons in CI.
The planar-terrain mode retains its original 60 fps raw infrared configuration.

At startup, co-located sensors with matching orientations load terrain through
their widest view first. Once that view is ready, the zoomed view requests its
extra detail without blocking either video stream. The first infrared images
can therefore use coarser terrain while refinement continues. Fixed-reference
captures still wait for their requested terrain detail.

Cesium imagery supplies no measured temperature: infrared terrain uses Elodin's
lighting-based estimate, while tagged objects use their configured temperature.
Rendering cadence and online loading time depend on the machine and connection;
this preview configuration does not claim the hardware preset's full 60 Hz.
For an unattended flight, set `ELODIN_NON_INTERACTIVE=1` and, for example,
`ELODIN_MAX_TICKS=90000` for five minutes. This keeps the initial trim commands
and does not start the keyboard controller. Interactive keyboard input is global:
typing in another application can steer the jet or change its throttle.

The launcher keeps acquired terrain detail with a 500 m margin and uses 16 px
screen error. Override `CESIUM_LOD_RETENTION_METERS` or `CESIUM_MAXIMUM_SCREEN_ERROR`
to compare settings. A zero margin disables detail retention.
At startup, Elodin shows a loading message until the first terrain and images
are ready. The simulation keeps running during loading. Online loading time
depends on the connection; rendering data is not yet cached between runs.
The sensors load their first pose before emitting images, then follow the
live simulation. On macOS, the Editor is limited to about 24 fps when sensor
panels are present, leaving GPU capacity for the sensor renderer.

## Other data sources

For public Swiss building geometry (without a textured ground surface):

```sh
export CESIUM_DEMO=bern
export CESIUM_TILESET_URL=https://3d.geo.admin.ch/ch.swisstopo.swissbuildings3d.3d/v1/tileset.json
export CESIUM_CREDIT="Data: swisstopo"
elodin editor examples/cesium/main.py
```

`CESIUM_DEMO` selects the example origin and route. For another site, adapt the
origin and flight path to the dataset.

For a Cesium ion **3D Tiles** asset, set `CESIUM_ION_ASSET_ID` and
`CESIUM_ION_TOKEN` in your environment. Ion takes precedence over the URL;
unset those variables to return to a public URL. Never commit tokens.
`CESIUM_CREDIT` supplies an additional plain-text acknowledgement; native credits
are also displayed.

Data sources: [AGI sample / Cesium resources](https://github.com/CesiumGS/3d-tiles/blob/main/RESOURCES.md)
and [swisstopo](https://docs.geo.admin.ch/visualize-data/3d-tiles.html).

## Integration and limits

`bevy-cesium` is pinned to a Git revision. It owns the Rust/C++ bridge, native SDK
build and reusable renderer. The Editor provides ECEF camera views, floating-origin
placement and the credits overlay. Cesium is part of the standard Editor build;
streaming starts only when a source is configured.

`just install editor` prepares the native SDK. For direct Cargo commands, first
run `just prepare-cesium` and set `CESIUM_NATIVE_DIR` to its installation directory
(normally `~/.cache/elodin/cesium-0.65.0/install`).

Supported: multi-camera streaming, tile eviction, triangle meshes, base-color
textures, metallic/roughness and emissive factors, alpha modes and double-sided
surfaces. Ion terrain supports one ion imagery overlay, selected process-wide
with `CESIUM_ION_IMAGERY_ASSET_ID`. Terrain meshes are replaced as imagery
arrives. Full glTF materials, general overlay composition, GPU instancing, point
clouds, collision meshes and persistent offline packages are not implemented.
The RC-jet workflow keeps the terrain views free of the sample credits panel.

For terrain stability, controlled captures and the remaining work, see the
[Bevy-Cesium README](https://github.com/elodin-sys/bevy-cesium).
`ELODIN_CESIUM_CAPTURE_VIEWS` accepts a JSON file of reference cameras;
`rc-jet-reference-views.json` is a sample for a 2 km corridor near the RC-jet site.
Supply references that cover the actual flight and sensor requirements.
