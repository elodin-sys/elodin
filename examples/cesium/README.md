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
It selects ion asset `1` for relief and asset `2` for aerial imagery. The
flight model still uses its reference ground elevation.

The Cesium launcher starts toward the valley at a 45-degree heading. The original
350-degree route enters mountains higher than the aircraft after a few minutes;
the cameras then see the undersides and edges of terrain tiles. Override
`ELODIN_RC_JET_HEADING_DEG` to choose another route. Terrain-aware ground contact
is still separate work: the simulation does not stop an aircraft flying into a hill.

The color sensor camera stays enabled. The zoomed infrared camera is temporarily
disabled in Cesium mode because it greatly increases initial loading time.
Both video panels remain visible; the infrared panel shows "No video at this time".
The next step is a shared `el.terrain_heightmap()` API for loading fixed heights
before simulation and querying them with JAX. Ground contact will use that API
in a separate change, keeping custom download and cache logic out of RC-jet.

After the terrain API, make infrared usable: measure its costs, choose
a sustainable frame rate and recording format, prevent it from delaying the color
stream, and check heat contrast on terrain and a tagged target. Enable it only
after both cameras work together during a long flight.
The 18° infrared view requests finer detail and shares the initial loading gate
with the color camera. Its 640 × 512, 60 fps preset also adds thermal rendering,
GPU-to-CPU copies and about 20 MB/s of raw grayscale recording. These costs need
separate measurements; the terrain retention CPU benchmark does not cover them.
Cesium imagery supplies no measured temperature: infrared terrain currently uses
Elodin's lighting-based estimate. See the
[investigation notes](https://github.com/elodin-sys/bevy-cesium#infrared-remains-disabled).

The color preview uses H.264 compression to keep recording size small. This is
a lossy video preview; use raw captures for future pixel comparisons in CI.
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
The color sensor loads its first pose before emitting images, then follows the
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
