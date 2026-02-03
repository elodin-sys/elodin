# ViewCube Plugin

A CAD-style 3D orientation widget for Bevy applications.

https://github.com/user-attachments/assets/view-cube-demo.mp4

![ViewCube Demo](ViewCubeDemo.mp4)

## Features

- **Interactive cube**: Click on faces, edges, or corners to snap the camera to that view
- **Rotation arrows**: 6 buttons for incremental camera rotation (up, down, left, right, roll)
- **Hover highlighting**: Visual feedback when hovering over interactive elements
- **Overlay mode**: Renders as a fixed overlay in the corner of the screen
- **Camera sync**: Cube rotation mirrors the main camera orientation
- **Coordinate systems**: Supports ENU (East-North-Up) and NED (North-East-Down)

## Quick Start

```rust
use bevy::prelude::*;
use elodin_editor::plugins::view_cube::{
    ViewCubeConfig, ViewCubePlugin, ViewCubeTargetCamera,
    spawn::spawn_view_cube,
};

fn main() {
    let config = ViewCubeConfig {
        use_overlay: true,       // Render as overlay in corner
        sync_with_camera: true,  // Cube shows world orientation
        auto_rotate: true,       // Plugin handles camera rotation on click
        ..default()
    };

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ViewCubePlugin { config })
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<ViewCubeConfig>,
) {
    // Spawn main camera with ViewCubeTargetCamera marker
    let camera_entity = commands
        .spawn((
            Transform::from_xyz(5.0, 4.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            Camera3d::default(),
            ViewCubeTargetCamera,  // Required: marks this as the camera to control
            Name::new("main_camera"),
        ))
        .id();

    // Spawn the ViewCube
    spawn_view_cube(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut materials,
        &config,
        camera_entity,
    );
}
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_overlay` | `bool` | `false` | Render as fixed overlay with dedicated camera |
| `sync_with_camera` | `bool` | `false` | Cube rotation mirrors main camera |
| `auto_rotate` | `bool` | `true` | Plugin handles camera rotation on click |
| `overlay_size` | `u32` | `160` | Viewport size in pixels |
| `overlay_margin` | `f32` | `8.0` | Margin from window edge |
| `camera_distance` | `f32` | `3.5` | Distance from cube (affects apparent size) |
| `scale` | `f32` | `0.95` | Cube scale factor |
| `system` | `CoordinateSystem` | `ENU` | Coordinate system (ENU or NED) |

## Run the Demo

```bash
cargo run --example view_cube_demo -p elodin-editor
```

## Required Assets

The plugin expects these assets in your assets folder:
- `axes-cube.glb` - The 3D cube model
- `fonts/Roboto-Bold.ttf` - Font for face labels
