# Cesium Native in the Elodin Editor (experimental)

Cesium Native selects, downloads and decodes the tiles. Bevy renders the
resulting meshes in the Editor's existing viewports. This is not a replacement
3D Tiles implementation and does not depend on `bevy_3d_tiles`.

## Build and run

From the repository root:

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

The default example is a simple position-controlled vehicle over the **AGI
headquarters photogrammetry sample**, listed in
[Cesium's open-data resources](https://github.com/CesiumGS/3d-tiles/blob/main/RESOURCES.md).
Elodin integrates its motion, while both cameras drive a shared Cesium tileset.
No account or API key is needed for this sample.
The simulation runs at 60 Hz in real time for five minutes. The route repeats
every two minutes; it is a position-control example without terrain collisions.

The dashboard shows a close chase camera aligned with motion, a downward-looking
onboard view, and a wider following camera. Below them are a north-up Cesium site
map, WGS84 coordinates, height above the local origin, and the measured East/North
ground track. The orange trail comes from the simulated positions. Height varies
by 10 m around the nominal altitude; it is not terrain clearance. Display heading
follows velocity and is not produced by an attitude controller.
The site map uses `effects=False` to omit the skybox. The simulation repeats its
flight profile continuously during the five-minute run, without resetting the
physics at each lap.

The alternative Bern scene uses public Swiss building geometry, without a
textured ground surface:

```sh
export CESIUM_DEMO=bern
export CESIUM_TILESET_URL=https://3d.geo.admin.ch/ch.swisstopo.swissbuildings3d.3d/v1/tileset.json
export CESIUM_CREDIT="Data: swisstopo"
elodin editor examples/cesium/main.py
```

`CESIUM_DEMO` selects the example's geographic origin and flight route; it does
not change the streaming engine. For another dataset,
adapt the `coordinate` origin and cameras in the example to that dataset.
Swiss data: [swisstopo](https://docs.geo.admin.ch/visualize-data/3d-tiles.html).

To use your own Cesium ion account, set `CESIUM_ION_ASSET_ID` to a **3D Tiles**
asset and `CESIUM_ION_TOKEN` to a token with read access. The asset ID takes
precedence over `CESIUM_TILESET_URL`. Free and paid accounts use the same
integration; availability and quotas depend on your account and dataset.
Keep tokens in your local environment, never in Python, KDL or committed files.
`CESIUM_CREDIT` adds a plain-text source acknowledgement for URL datasets whose
files do not contain their own attribution. Cesium's supplied credits are also
forwarded to the overlay.

## Standard build, runtime activation

The `cesium` Cargo feature only isolated the first prototype. Cesium is now
included in the standard Editor build and starts streaming only when a source
is configured. `just install editor` prepares the native SDK if missing.
Before direct Cargo commands (including CI), run `just prepare-cesium`, or set
`CESIUM_NATIVE_DIR` to an existing installation. Cross-platform builds remain
unvalidated; the SDK is not yet packaged as a Nix derivation.

## Scope of this first integration

- Native streaming, multi-camera tile selection, CPU decoding and tile eviction.
- ECEF placement through Elodin's geographic frames and floating origin.
- Triangle meshes, base-color textures and material colors.
- Standard Editor dependency, activated by runtime source configuration.

Raster imagery overlays, the material features beyond base color, GPU instancing,
point clouds and collision meshes are not implemented yet. Cesium World Terrain
with a separate imagery layer is consequently not a supported demo configuration.
The first version targets textured 3D Tiles, such as photogrammetry.
The existing credit display is a prototype and needs complete HTML/logo handling
before distributing an integration for providers requiring logo attribution.

## Validation

Validated locally on macOS ARM64: release Editor build, Clippy with warnings
denied, local textured/georeferenced glTF test, public Swiss streaming test,
and the AGI sample in multiple Editor viewports. Cesium ion with a user token and
Linux/Windows builds have not been validated.

```sh
just prepare-cesium
cargo test -p cesium-native-bridge
cargo run -p cesium-native-bridge --example stream
```

This bounded streaming test requires at least one visible tile and checks mesh
array lengths and index bounds. It does not replace visual validation in the Editor.

The bridge uses Cesium Native **0.65.0**, pinned to commit
`1bca8021711294cc83a5a032599ac72d9b3166c9`, and the maintained CXX binding generator.
The sole `unsafe extern` boundary is in `libs/cesium-native-bridge`; no native
pointers escape to the Editor and sessions are kept on the creating thread.

Upstream integration guide:
https://cesium.com/learn/cesium-native/ref-doc/rendering-3d-tiles.html
