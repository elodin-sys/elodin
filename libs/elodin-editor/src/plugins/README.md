# Plugins Documentation Overview

This folder gathers the Bevy utilities and plugins used by the Elodin editor.

| Plugin | First appearance | Status | Description | Docs |
| --- | --- | --- | --- | --- |
| `navigation_gizmo` | 2024-02-13 | Legacy | Original navigation gizmo systems (cube/camera sync/helpers). The old gizmo UX is obsolete and replaced by Cube-Viewer (`view_cube`). | [README.md](navigation_gizmo/README.md) |
| `gizmos` | 2024-03-19 | Active | Vector arrows, body axes, label rendering, and cleanup systems. | [README.md](gizmos/README.md) |
| `web_asset` | 2024-03-28 | Active | Adds `http`/`https` asset sources, with ETag-aware caching support. | [README.md](web_asset/README.md) |
| `editor_cam_touch` | 2024-04-07 | Active | Touch gesture input for `EditorCam` (orbit/pan/zoom). | [README.md](editor_cam_touch/README.md) |
| `asset_cache` | 2024-08-22 | Active | Traits and FS/`NoCache` implementations for caching downloaded assets. | [README.md](asset_cache/README.md) |
| `logical_key` | 2024-09-26 | Active | Tracks logical keys pressed/just-pressed/just-released for keyboard handling. | [README.md](logical_key/README.md) |
| `env_asset_source` | 2025-09-26 | Active | Configures the default `AssetServer` source from `ELODIN_ASSETS_DIR` and warns on invalid paths. | [README.md](env_asset_source/README.md) |
| `camera_anchor` | 2025-11-24 | Active | Computes a safe camera anchor (view ↔ origin) to avoid depth `NaN`s. | [README.md](camera_anchor/README.md) |
| `view_cube` (Cube-Viewer) | 2026-02-12 | Active | Current interactive Cube-Viewer (snap/zoom buttons, synchronized overlay rendering). | [README.md](view_cube/README.md) |
| `osm_world` | 2026-02-14 | Prototype | Adds streamed OpenStreetMap context around the tracked vehicle using a local tile cache and Overpass-backed fetches. | [README.md](#osm_world-plugin) |
| `frustum` | 2026-02-19 | Active | Draws created viewport frustums (`create_frustum`) onto viewports that opt in (`show_frustums`), with per-source color/thickness. | [README.md](frustum/README.md) |
| `frustum_intersection` | 2026-02-24 | Active | Volume coverage (frustum∩ellipsoid) and 2D projection on far plane; Inspector controls are gated by `show_frustums` and ellipsoid detection. | [README.md](frustum_intersection/README.md) |
| `kdl_asset_source` | 2026-03-18 | Active | Registers a custom Bevy `AssetSource` for `.kdl` files, using Bevy's built-in `FileWatcher` for hot-reload. | [README.md](kdl_asset_source/README.md) |
| `kdl_document` | 2026-03-18 | Active | KDL schematic document lifecycle: loading, saving, hot-reload, and message-based command API. | [README.md](kdl_document/README.md) |

## `osm_world` plugin

`osm_world` adds lightweight geospatial context around the current simulated vehicle:

- building meshes (extruded from OSM polygons),
- roads and land/water areas,
- hover metadata on buildings,
- tile streaming with local cache and incremental frame-by-frame rendering.

### Prototype scope

This data flow is primarily a **geodata integration pipeline** for simulation work:

- ingest polygon geometry,
- keep polygon key/value properties available,
- let systems use those attributes to enrich simulation behavior and semantics.

The use of **OpenStreetMap + Overpass** in this plugin is intentionally a
**prototype** choice.
It is meant to quickly validate editor UX, rendering, and polygon-attribute-driven
simulation ideas. It is not the final production geodata pipeline.

### Future direction

A pipeline based on **GeoParquet** is a strong candidate to explore for production:

- better control of schemas and data contracts,
- offline/replicable data packaging,
- scalable querying and processing compared to ad-hoc API fetches.

### How it works

1. Determine the tracked center (`bdx.world_pos`, `drone.world_pos`, `target.world_pos`, or a configured component).
2. Convert the center to local tiles.
3. Load tile data from cache when available.
4. Fetch missing tiles from Overpass, then persist JSON locally.
5. Build renderables and apply them progressively over frames.

### Default behavior

- Enabled by default in editor builds (non-wasm).
- Default origin is fixed near Century City:
  - lat: `34.054085661510506`
  - lon: `-118.42558289792434`
- Local cache defaults to: `~/.cache/elodin/osm_tiles`

### Main tuning environment variables

- `ELODIN_OSM_ENABLED`
- `ELODIN_OSM_ORIGIN_LAT`, `ELODIN_OSM_ORIGIN_LON`
- `ELODIN_OSM_TILE_SIZE_M`
- `ELODIN_OSM_TILE_RADIUS`
- `ELODIN_OSM_PREFETCH_TILE_RADIUS`
- `ELODIN_OSM_MAX_INFLIGHT_FETCHES`
- `ELODIN_OSM_SPAWN_PER_FRAME`
- `ELODIN_OSM_CACHE_DIR`
- `ELODIN_OSM_OVERPASS_URLS`
- `ELODIN_OSM_TRACK_COMPONENT`

### Quick run

From repository root:

```bash
cargo run -p elodin -- editor examples/drone/main.py
```

or:

```bash
cargo run -p elodin -- editor examples/rc-jet/main.py
```
