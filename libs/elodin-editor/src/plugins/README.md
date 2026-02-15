# Plugins Documentation Overview

Inventory of Bevy plugins and support modules used by `elodin-editor`.

| Plugin / Module | First appearance | Status | Description | Docs |
| --- | --- | --- | --- | --- |
| `navigation_gizmo` | 2024-02-13 | Legacy | Original navigation gizmo systems (cube/camera sync/helpers). The old gizmo UX is obsolete and replaced by Cube-Viewer (`view_cube`). | [README.md](navigation_gizmo/README.md) |
| `gizmos` | 2024-03-19 | Active | Vector arrows, body axes, label rendering, and cleanup systems. | [README.md](gizmos/README.md) |
| `web_asset` | 2024-03-28 | Active | Adds `http`/`https` asset sources, with ETag-aware caching support. | [README.md](web_asset/README.md) |
| `editor_cam_touch` | 2024-04-07 | Active | Touch gesture input for `EditorCam` (orbit/pan/zoom). | [README.md](editor_cam_touch/README.md) |
| `asset_cache` | 2024-08-22 | Active | Traits and FS/`NoCache` implementations for caching downloaded assets used by plugin asset fetchers. | [README.md](asset_cache/README.md) |
| `logical_key` | 2024-09-26 | Active | Tracks logical keys pressed/just-pressed/just-released for keyboard handling. | [README.md](logical_key/README.md) |
| `env_asset_source` | 2025-09-26 | Active | Configures the default `AssetServer` source from `ELODIN_ASSETS_DIR` and warns on invalid paths. | [README.md](env_asset_source/README.md) |
| `camera_anchor` | 2025-11-24 | Active | Computes a safe camera anchor (view ↔ origin) to avoid depth `NaN`s. | [README.md](camera_anchor/README.md) |
| `view_cube` (Cube-Viewer) | 2026-02-12 | Active | Current interactive Cube-Viewer (snap/zoom buttons, synchronized overlay rendering). | [README.md](view_cube/README.md) |
| `osm_world` | 2026-02-14 | Exploratory (POC) | Geodata tile streaming plus polygon metadata for simulation enrichment and editor context. | [osm_world.md](./osm_world.md) |
| `frustum` | 2026-02-19 | Active | Draws created viewport frustums (`create_frustum`) onto viewports that opt in (`show_frustums`), with per-source color and thickness. | [README.md](frustum/README.md) |
| `frustum_intersection` | 2026-02-24 | Active | Volume coverage (frustum∩ellipsoid) and 2D projection on the far plane; Inspector controls are gated by `show_frustums` and ellipsoid detection. | [README.md](frustum_intersection/README.md) |
| `kdl_asset_source` | 2026-03-18 | Active | Registers a custom Bevy `AssetSource` for `.kdl` files, using Bevy's built-in `FileWatcher` for hot-reload. | [README.md](kdl_asset_source/README.md) |
| `kdl_document` | 2026-03-18 | Active | KDL schematic document lifecycle: loading, saving, hot-reload, and message-based command API. | [README.md](kdl_document/README.md) |

`asset_cache.rs`, `camera_anchor.rs`, and `env_asset_source.rs` are support modules
used by the plugins above.
