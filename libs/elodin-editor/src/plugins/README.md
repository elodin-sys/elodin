# Plugins
This folder gathers the Bevy utilities and plugins used by the Elodin editor.

| Plugin | Description | Docs |
| --- | --- | --- |
| `asset_cache` | Traits and FS/`NoCache` implementations for caching downloaded assets. | — |
| `camera_anchor` | Computes a Bevy camera anchor (view ↔ origin) to avoid depth `NaN`s. | — |
| `editor_cam_touch` | Plugin that provides a touch input system for `EditorCam`, covering one- and two-finger gestures (orbit, pan, zoom). | — |
| `env_asset_source` | Plugin that configures the `AssetServer` from the `ELODIN_ASSETS_DIR` environment variable and warns when the path is invalid. | — |
| `gizmos` | Plugin rendering vector arrows, body axes, UI labels, and cleanup systems related to gizmos. | — |
| `logical_key` | Plugin that tracks logical keys pressed, just pressed, and just released to simplify keyboard input handling. | — |
| `navigation_gizmo` | Plugin + helpers for 3D navigation (cube, render layers, dedicated camera, drag/drop, animation). | — |
| `view_cube` | ViewCube plugin (interactive cube + snap/zoom buttons) with synchronized rendering logic. | [README.md](view_cube/README.md) |
| `web_asset` | Plugin that adds `http`/`https` asset sources to the pipeline, including caching and ETag support. | — |

Docs links are relative to this directory and point to more detailed documentation where available.
