# Editor Plugins

Inventory of Bevy plugins used by `elodin-editor`.

| Plugin | Main role | Status | Documentation |
| --- | --- | --- | --- |
| `GizmoPlugin` | Render vector arrows, body axes, and in-viewport labels/gizmos. | Stable | `gizmos.rs` |
| `NavigationGizmoPlugin` | Manage viewport navigation gizmo camera, layer, and interactions. | Stable | `navigation_gizmo.rs` |
| `ViewCubePlugin` | CAD-style orientation cube and camera snap controls. | Stable | [`view_cube/README.md`](./view_cube/README.md) |
| `EditorCamTouchPlugin` | Touch gestures (pan/pinch/rotate) for editor camera controls. | Stable | `editor_cam_touch.rs` |
| `LogicalKeyPlugin` | Frame-accurate logical keyboard state (`pressed/just_pressed/...`). | Stable | `logical_key.rs` |
| `WebAssetPlugin` | Register `http://` and `https://` Bevy asset sources with cache support. | Stable | `web_asset.rs` |
| `OsmWorldPlugin` | Geodata tile streaming + polygon metadata for simulation enrichment. Activated only through `examples/drone/main_osm_world.py` and `examples/rc-jet/main_osm_world.py`; default `examples/*/main.py` runs keep standard non-geodata behavior. | Exploratory (POC) | [`osm_world/README.md`](./osm_world/README.md) |

`asset_cache.rs`, `camera_anchor.rs`, and `env_asset_source.rs` are support modules used by plugins above.
