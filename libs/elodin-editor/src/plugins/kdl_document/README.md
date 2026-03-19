# kdl_document

Manages the lifecycle of KDL schematic documents: loading, saving, reloading, and asset integration.

## What it does
- Registers `SchematicDocumentAsset` as a Bevy asset with a dedicated `AssetLoader`.
- Loads root and secondary (multi-window) KDL schematics as a single document.
- Provides a message-based command API (`OpenDocumentRequest`, `SaveCurrentDocumentRequest`, `OpenDocumentFromContentRequest`).
- Emits lifecycle events: `DocumentLoaded`, `DocumentSaved`, `DocumentReloaded`, `DocumentLoadFailed`, `DocumentCommandFailed`, `DocumentCleared`.
- Tracks the current document via `CurrentDocument`, with change detection based on applied KDL snapshots.
- Bridges `DbConfig` metadata into document loading via `apply_initial_kdl_path` / `sync_document_from_config`.
- Integrates with the `kdl_asset_source` plugin for hot-reload via the Bevy asset pipeline.

## Main API
- `plugin(app)` — registers the plugin.
- `CurrentDocument` — resource tracking the active document.
- `InitialKdlPath` — resource for the CLI `--kdl` path.
- `SchematicDocumentAsset` / `SchematicDocumentLoader` — Bevy asset types.
- `OpenDocumentRequest(PathBuf)` — message to load a KDL file.
- `OpenDocumentFromContentRequest` — message to load from inline KDL content.
- `SaveCurrentDocumentRequest` — message to save the current schematic to disk.
- `DocumentLoaded`, `DocumentReloaded`, `DocumentSaved`, `DocumentCleared`, `DocumentLoadFailed`, `DocumentCommandFailed` — lifecycle events.

## Allowed KDL schema

A schematic document is a `.kdl` file. The following top-level nodes are recognized:

### Panels (layout & visualization)

| Node | Description |
|---|---|
| `viewport` | 3D viewport with camera controls |
| `graph` | Time-series telemetry graph |
| `component_monitor` | Live component value table |
| `query_table` | SQL query result table |
| `query_plot` | SQL query-driven plot |
| `action_pane` | Button executing a Lua command |
| `video_stream` | H.264 video stream viewer |
| `sensor_view` | Raw RGBA sensor camera view |
| `log_stream` | Scrolling log message viewer |
| `data_overview` | Auto-generated component overview |
| `dashboard` | Composite dashboard layout |
| `schematic_tree` | Schematic tree navigator |
| `inspector` | Entity inspector sidebar |
| `hierarchy` | Entity hierarchy sidebar |

### Layout containers

| Node | Description |
|---|---|
| `tabs` | Tabbed container — children are panels |
| `hsplit` | Horizontal split — children are panels, optional `name`, `share` |
| `vsplit` | Vertical split — children are panels, optional `name`, `share` |

### 3D scene elements

| Node | Description |
|---|---|
| `object_3d` | 3D object (GLTF, primitives) bound to an EQL expression |
| `line_3d` | 3D line visualization |
| `vector_arrow` | 3D vector arrow visualization |

### Document-level configuration

| Node | Description |
|---|---|
| `window` | Secondary window descriptor (`path`, `title`, `screen`, `screen_rect`) |
| `theme` | Color scheme / mode (`scheme`, `mode`) |
| `timeline` | Timeline appearance (`played_color`, `future_color`, `follow_latest`) |

### Example

```kdl
theme mode="dark" scheme="elodin"

window path="motor-panel.kdl" title="Motors" screen=1
window path="rate-control-panel.kdl" title="Rate Control"

tabs {
    viewport name="Main" fov=60 show_grid=true
    graph "drone.gyro" name="Gyroscope"
}

object_3d "drone.world_pos" mesh="assets/drone.glb"

timeline follow_latest=true
```

## Status
Active.
