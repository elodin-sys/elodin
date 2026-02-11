# ViewCube Plugin

A CAD-style 3D orientation widget for Bevy editor viewports (overlay + `LookToTrigger` snaps).

## Features

- **Interactive cube**: Click on faces, edges, or corners to snap the camera to that view
- **Rotation arrows**: top-left/top-right roll around the camera axis; left/right rotate around camera up; up/down rotate around camera right
- **Hover highlighting**: Visual feedback with CAD-style grouped edge hover (4 edges for active front frame, 2-3 edges for hidden-face groups)
- **Overlay rendering**: Dedicated ViewCube camera rendered in viewport corner
- **Camera sync**: Cube mirrors the main camera orientation
- **Coordinate systems**: Supports ENU (E, N, U labels) and NED (N, E, D labels)
- **Colored axes**: Corner-mounted X/Y/Z axes, colored according to the active coordinate system

## Interaction Rules

- **Front face click**: no-op (clicking a face already in front does nothing)
- **Oblique face click**: snaps to the clicked face
- **Front-frame group**: only when a single face is truly visible (face-on), hovering a border highlights the full frame group (4 frame edges + 4 frame corners); clicking that border snaps to the opposite face
- **Hidden-face groups**: in oblique views, hovering a border tied to a truly hidden face highlights a coherent mixed group (border edges plus adjacent corners, with at least 2 edges), clicking that border snaps to the hidden face
- **Inactive edges**: no hover highlight, no click action
- **Corners**: hover highlights only the targeted corner and click snaps to that corner view
- **Front corner click**: no-op when the clicked corner is already aligned with the camera axis

## Quick Start

```rust
use bevy::prelude::*;
use bevy_editor_cam::prelude::EditorCam;
use elodin_editor::plugins::{
    navigation_gizmo::{NavGizmoCamera, NavGizmoParent, NavigationGizmoPlugin},
    view_cube::{
        NeedsInitialSnap, ViewCubeConfig, ViewCubePlugin, ViewCubeTargetCamera,
        spawn::spawn_view_cube,
    },
};

fn main() {
    let config = ViewCubeConfig::editor_mode();

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(NavigationGizmoPlugin)
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
    let camera_entity = commands
        .spawn((
            Transform::from_xyz(5.0, 4.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            Camera3d::default(),
            EditorCam::default(),
            ViewCubeTargetCamera,
            NeedsInitialSnap,
            Name::new("main_camera"),
        ))
        .id();

    let spawned = spawn_view_cube(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut materials,
        &config,
        camera_entity,
    );

    if let Some(view_cube_camera) = spawned.camera {
        commands.entity(view_cube_camera).insert((
            NavGizmoParent {
                main_camera: camera_entity,
            },
            NavGizmoCamera,
        ));
    }
}
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sync_with_camera` | `bool` | `true` | Cube rotation mirrors main camera |
| `camera_distance` | `f32` | `2.5` | Overlay camera distance from cube |
| `scale` | `f32` | `0.6` | Cube scale factor |
| `rotation_increment` | `f32` | `15°` | Arrow click angular step |
| `axis_correction` | `Quat` | `IDENTITY` | Extra rotation applied after the system correction |
| `render_layer` | `u8` | `31` | Render layer used by ViewCube camera and entities |
| `system` | `CoordinateSystem` | `ENU` | Coordinate system (ENU or NED) |

`ViewCubeConfig::editor_mode()` is the canonical preset and currently matches `Default`.

## Required Assets

The plugin expects:
- `axes-cube.glb` in your runtime assets folder
- `fonts/Roboto-Bold.ttf` for face labels (embedded in `elodin_editor`)
- `icons/chevron_right.png` for left/right/up/down arrow buttons (embedded in `elodin_editor`)
- `icons/loop.png` for roll-left/roll-right buttons (embedded in `elodin_editor`)

When used inside `elodin_editor`, these font/icon assets are embedded by `EmbeddedAssetPlugin`.
