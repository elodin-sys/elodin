---
name: elodin-editor-dev
description: Contribute to the Elodin Editor, the 3D viewer and graphing tool. Use when editing files in libs/elodin-editor/ or apps/elodin/, working on the Bevy/Egui UI, modifying viewport rendering, telemetry graphs, video streaming, KDL schematics, or the command palette.
---

# Elodin Editor Development

The Elodin Editor is a 3D visualization and telemetry graphing tool built with Bevy (ECS game engine) and Egui (immediate-mode UI). It connects to Elodin-DB via Impeller2 for real-time data.

## Running

```bash
# Standard
cargo run --bin elodin editor examples/three-body/main.py

# With hot-reload on editor code changes (requires cargo-watch)
cargo watch --watch libs/elodin-editor \
    -x 'run --bin elodin editor examples/three-body/main.py'

# Connect to a running database instead of a simulation
cargo run --bin elodin editor 127.0.0.1:2240
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ELODIN_ASSETS_DIR` | `./assets` | Directory for meshes, images, GLB files |
| `ELODIN_KDL_DIR` | `.` (cwd) | Directory for `.kdl` schematic files |

## Cargo features

Optional features declared in `libs/elodin-editor/Cargo.toml` (re-exported by `apps/elodin/Cargo.toml`):

- `big_space` (default): upstream `big_space` 0.12 floating-origin layer.
- `no_prop_rot`: restore the pre-upgrade fork behavior where `MainCamera.GlobalTransform.rotation` ignores its parent viewport's rotation. See `restore_main_camera_world_rotation` in `lib.rs`.
- `inspector`: adds the `bevy-inspector-egui` runtime entity inspector.
- `debug`: enables `big_space`'s debug diagnostics.
- `tracy`: enables Tracy profiling (see `.cursor/skills/elodin-tracy/SKILL.md`).

Enable with `cargo run -p elodin --features "<list>" -- editor ...`.

## Source Layout

The editor is split across two crates:

### `libs/elodin-editor/` — Core library

```
src/
├── lib.rs              # Plugin registration, app setup
├── run.rs              # Main run loop and Bevy app builder
├── object_3d.rs        # 3D object spawning from KDL schematics
├── vector_arrow.rs     # Vector visualization (force/velocity arrows)
├── offset_parse.rs     # EQL viewport formula parsing
├── icon_rasterizer.rs  # Icon rendering
├── iter.rs             # Iterator utilities
├── ui/                 # Egui UI layer
│   ├── mod.rs          # Top-level UI orchestration
│   ├── tiles.rs        # Tiled panel layout system
│   ├── tiles/sidebar.rs
│   ├── inspector/      # Component inspector panels
│   │   ├── mod.rs
│   │   ├── entity.rs   # Entity property inspector
│   │   ├── viewport.rs # Viewport panel
│   │   ├── graph.rs    # Graph panel configuration
│   │   ├── dashboard.rs
│   │   ├── monitor.rs
│   │   ├── object3d.rs # 3D object inspector
│   │   └── ...
│   ├── plot/           # Telemetry graph rendering
│   │   ├── mod.rs
│   │   ├── data.rs     # Data fetching and buffering
│   │   ├── gpu.rs      # GPU-accelerated plot rendering
│   │   ├── widget.rs   # Egui plot widget
│   │   └── state.rs    # Plot state management
│   ├── plot_3d/        # 3D plot visualization
│   ├── schematic/      # KDL schematic loading and rendering
│   │   ├── mod.rs
│   │   ├── load.rs     # KDL file parsing
│   │   └── tree.rs     # Schematic tree view
│   ├── timeline/       # Playback timeline
│   │   ├── mod.rs
│   │   ├── timeline_controls.rs
│   │   └── timeline_slider.rs
│   ├── command_palette/ # Command palette (Ctrl+P)
│   ├── video_stream.rs  # Live video rendering
│   ├── theme.rs         # Color theme
│   ├── colors/          # Color system and presets
│   └── ...
└── plugins/            # Bevy plugins
    ├── mod.rs
    ├── view_cube/      # 3D orientation cube
    ├── camera_anchor/  # Camera tracking and anchoring
    ├── gizmos/         # Transform gizmos
    ├── navigation_gizmo/
    ├── editor_cam_touch/ # Touch input for camera
    ├── asset_cache/    # Asset caching
    ├── env_asset_source/ # ELODIN_ASSETS_DIR integration
    ├── web_asset/      # Web asset loading
    └── logical_key/    # Keyboard input handling
```

### `apps/elodin/` — CLI binary

The main entry point that ties together the editor with nox-py, s10, and Impeller2. Handles CLI argument parsing (`elodin editor`, `elodin run`, etc.).

## Key Subsystems

### Bevy Plugin Architecture

The editor registers as a set of Bevy plugins. Each feature area (view cube, camera, gizmos) is a self-contained plugin with its own components, systems, and resources. New features should follow this pattern.

### Egui UI Layer

The `ui/` module contains all immediate-mode UI rendering. Egui runs inside Bevy via `bevy_egui`. The tile-based layout system (`ui/tiles.rs`) manages panel arrangement (viewports, graphs, inspectors).

### Telemetry Data Flow

1. Impeller2 client subscribes to component streams from Elodin-DB
2. Data arrives as time-series samples
3. Plot system buffers and renders via GPU-accelerated rendering (`ui/plot/gpu.rs`)
4. Inspector panels show latest values

### KDL Schematics

KDL files define 3D objects and viewport configurations. The loading pipeline:
1. `ui/schematic/load.rs` — Parses `.kdl` files
2. `object_3d.rs` — Spawns Bevy entities (meshes, GLB models, shapes)
3. `offset_parse.rs` — Evaluates EQL viewport formulas (rotate, translate)

### Video Streaming

`ui/video_stream.rs` handles live H.264/AV1 video decoding and rendering as textures in the 3D viewport. Uses the `video-toolbox` crate on macOS for hardware acceleration.

## Dependencies

Key crates used in the editor:

| Crate | Purpose |
|-------|---------|
| `bevy` | ECS game engine, 3D rendering, windowing |
| `bevy_egui` | Egui integration for Bevy |
| `egui` | Immediate-mode UI framework |
| `impeller2-bevy` | Bevy plugin for Impeller2 telemetry |
| `arrow` | Arrow data format for time-series |
| `eql` | Elodin Query Language parser |
| `nox` | Spatial math types |

## Development Tips

- Use `cargo watch` for fast iteration on UI changes
- The editor hot-reloads KDL schematics on file change
- Test with `examples/three-body/main.py` for a lightweight simulation
- GPU plot rendering is in `ui/plot/gpu.rs` — changes here affect all telemetry graphs
- The command palette (`ui/command_palette/`) is the entry point for user actions

## Key References

- Bevy Tips (ECS): [../bevy/SKILL.md](../bevy/SKILL.md)
- Editor README: [apps/elodin/README.md](../../../apps/elodin/README.md)
- KDL schematic syntax: [docs/public/content/reference/schematic.md](../../../docs/public/content/reference/schematic.md)
- Command palette reference: [docs/public/content/reference/command-palette.md](../../../docs/public/content/reference/command-palette.md)
