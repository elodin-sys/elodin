# Cesium drone example

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

The cameras share one tileset. The orange/navy drone is a material-only variant of `talon-quad-v2.glb`, stored as
`assets/talon-quad-cesium.glb`. Geometry and scale are unchanged; visibility
comes from closer camera positions and more screen space for the three flight views.
Allow the site to load before recording; keep the attribution overlay visible.

The simulation runs at 60 Hz for five minutes. Its target follows a two-minute
loop, with altitude varying by 10 m around 70 m above the geographic origin.
Elodin integrates the commanded forces: this is not a prerecorded trajectory.
The display heading follows velocity; there is no attitude controller or terrain
collision model. The plotted altitude is relative to the origin, **not ground clearance**.

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
(normally `~/.cache/elodin/cesium-0.65.0/install`). Nix packaging and cross-platform
build validation remain outstanding.

Supported: multi-camera streaming, tile eviction, triangle meshes and base-color
textures. Raster imagery overlays, full glTF materials, GPU instancing, point
clouds and collision meshes are not implemented. Cesium World Terrain with a
separate imagery layer is not a supported configuration. The credits UI still
needs complete HTML/logo handling for providers requiring it.

The standalone plugin has seven passing tests, including a
[headless ECEF integration test](https://github.com/elodin-sys/bevy-cesium/blob/main/tests/ecef_end_to_end.rs)
covering two cameras, origin shifts and mesh placement. These are not GPU image
regression tests. Editor visual checks have been performed on Apple Silicon macOS;
authenticated ion access and Linux/Windows builds remain unvalidated.
