use bevy::prelude::*;
use bevy::window::{Window, WindowRef, WindowResolution, WindowTheme, WindowClosed};
use bevy::render::camera::RenderTarget;
use bevy::render::view::RenderLayers;
use std::collections::HashMap;

use crate::ui::tiles::Pane;

/// Resource to track pending secondary window creation requests
#[derive(Resource, Default)]
pub struct SecondaryWindowRequests {
    pub requests: Vec<SecondaryWindowRequest>,
}

/// A request to create a secondary window with specific content
#[derive(Clone)]
pub struct SecondaryWindowRequest {
    pub pane: Pane,
    pub pane_title: String,
    pub tile_id: Option<egui_tiles::TileId>,
}

/// Component to identify and track secondary windows
#[derive(Component)]
pub struct SecondaryWindowHandle {
    pub content: WindowContent,
    pub camera: Option<Entity>,
    pub original_tile_id: Option<egui_tiles::TileId>,
}

/// Specifies what type of content should be rendered in the secondary window
#[derive(Clone)]
pub enum WindowContent {
    Pane(Pane),
}

/// Component to mark cameras that belong to secondary windows
#[derive(Component)]
pub struct SecondaryWindowCamera;

/// Component to mark UI cameras for secondary windows
#[derive(Component)]
pub struct SecondaryWindowUiCamera;

/// Resource to track all active secondary windows
#[derive(Resource, Default)]
pub struct SecondaryWindows {
    pub windows: HashMap<Entity, SecondaryWindowInfo>,
}

pub struct SecondaryWindowInfo {
    pub window_entity: Entity,
    pub camera_entity: Option<Entity>,
    pub ui_camera_entity: Option<Entity>,
    pub content: WindowContent,
    pub original_tile_id: Option<egui_tiles::TileId>,
}

/// Plugin to handle multi-window functionality
pub struct MultiWindowPlugin;

impl Plugin for MultiWindowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SecondaryWindowRequests>()
            .init_resource::<SecondaryWindows>()
            .add_systems(Update, (
                spawn_secondary_windows,
                handle_window_closed,
                update_secondary_window_viewports,
            ).chain());
    }
}

/// System to spawn secondary windows from requests
fn spawn_secondary_windows(
    mut commands: Commands,
    mut requests: ResMut<SecondaryWindowRequests>,
    mut secondary_windows: ResMut<SecondaryWindows>,
    asset_server: Res<AssetServer>,
) {
    for request in requests.requests.drain(..) {
        // Create the window entity
        let window_entity = commands.spawn(Window {
            title: format!("Elodin - {}", request.pane_title),
            resolution: WindowResolution::new(1024.0, 768.0),
            window_theme: Some(WindowTheme::Dark),
            decorations: true,
            ..default()
        }).id();

        // For viewport panes, we need a 3D camera
        let camera_entity = match &request.pane {
            Pane::Viewport(_) => {
                Some(spawn_viewport_camera(&mut commands, window_entity, &asset_server))
            },
            _ => None,
        };

        // Spawn UI camera for all secondary windows
        let ui_camera_entity = commands.spawn((
            Camera2d,
            Camera {
                target: RenderTarget::Window(WindowRef::Entity(window_entity)),
                order: 100, // Render UI after 3D content
                ..default()
            },
            SecondaryWindowUiCamera,
        )).id();

        // Add the secondary window handle
        commands.entity(window_entity).insert(SecondaryWindowHandle {
            content: WindowContent::Pane(request.pane.clone()),
            camera: camera_entity,
            original_tile_id: request.tile_id,
        });

        // Track in our resource
        secondary_windows.windows.insert(
            window_entity,
            SecondaryWindowInfo {
                window_entity,
                camera_entity,
                ui_camera_entity: Some(ui_camera_entity),
                content: WindowContent::Pane(request.pane),
                original_tile_id: request.tile_id,
            }
        );
    }
}

/// Spawn a 3D camera for viewport rendering in secondary window
fn spawn_viewport_camera(
    commands: &mut Commands,
    window_entity: Entity,
    asset_server: &AssetServer,
) -> Entity {
    use bevy::core_pipeline::bloom::Bloom;
    use bevy::core_pipeline::tonemapping::Tonemapping;
    use bevy::render::camera::{Exposure, PhysicalCameraParameters};
    use bevy_editor_cam::prelude::{EditorCam, OrbitConstraint};
    use crate::EnvironmentMapLight;

    let camera = commands.spawn((
        Transform::from_translation(Vec3::new(5.0, 5.0, 5.0))
            .looking_at(Vec3::ZERO, Vec3::Y),
        Camera3d::default(),
        Camera {
            target: RenderTarget::Window(WindowRef::Entity(window_entity)),
            hdr: true,
            clear_color: bevy::render::camera::ClearColorConfig::Default,
            order: 1,
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            fov: 45.0_f32.to_radians(),
            ..default()
        }),
        Tonemapping::TonyMcMapface,
        Exposure::from_physical_camera(PhysicalCameraParameters {
            aperture_f_stops: 2.8,
            shutter_speed_s: 1.0 / 200.0,
            sensitivity_iso: 400.0,
            sensor_height: 24.0 / 1000.0,
        }),
        RenderLayers::layer(1), // Use layer 1 for secondary windows
        SecondaryWindowCamera,
        EditorCam {
            orbit_constraint: OrbitConstraint::Fixed {
                up: Vec3::Y,
                can_pass_tdc: false,
            },
            last_anchor_depth: 2.0,
            ..default()
        },
        Bloom::default(),
        EnvironmentMapLight {
            diffuse_map: asset_server.load("embedded://elodin_editor/assets/diffuse.ktx2"),
            specular_map: asset_server.load("embedded://elodin_editor/assets/specular.ktx2"),
            intensity: 2000.0,
            ..default()
        },
    )).id();

    camera
}

/// Handle window close events and clean up resources
fn handle_window_closed(
    mut commands: Commands,
    mut closed_events: EventReader<WindowClosed>,
    mut secondary_windows: ResMut<SecondaryWindows>,
    _query: Query<&SecondaryWindowHandle>,
) {
    for event in closed_events.read() {
        if let Some(info) = secondary_windows.windows.remove(&event.window) {
            // Clean up camera entities
            if let Some(camera) = info.camera_entity {
                commands.entity(camera).despawn();
            }
            if let Some(ui_camera) = info.ui_camera_entity {
                commands.entity(ui_camera).despawn();
            }

            // TODO: Return pane to main window if needed
            // This would require communicating with TileState
        }
    }
}

/// Update viewport settings for secondary window cameras
fn update_secondary_window_viewports(
    _windows: Query<(&Window, Entity), With<SecondaryWindowHandle>>,
    _cameras: Query<&mut Camera, With<SecondaryWindowCamera>>,
) {
    // This system can be used to update camera viewports if needed
    // For now, the default full-window viewport should work
}

/// Helper function to request a new secondary window
pub fn request_secondary_window(
    requests: &mut SecondaryWindowRequests,
    pane: Pane,
    tile_id: Option<egui_tiles::TileId>,
) {
    let pane_title = match &pane {
        Pane::Viewport(vp) => vp.label.clone(),
        Pane::Graph(g) => g.label.clone(),
        Pane::Monitor(m) => m.label.clone(),
        Pane::QueryTable(_) => "Query Table".to_string(),
        Pane::QueryPlot(_) => "Query Plot".to_string(),
        Pane::Dashboard(d) => d.label.clone(),
        Pane::Hierarchy => "Hierarchy".to_string(),
        Pane::Inspector => "Inspector".to_string(),
        Pane::SchematicTree(_) => "Schematic Tree".to_string(),
        Pane::ActionTile(a) => a.label.clone(),
        Pane::VideoStream(_) => "Video Stream".to_string(),
    };

    requests.requests.push(SecondaryWindowRequest {
        pane,
        pane_title,
        tile_id,
    });
}
