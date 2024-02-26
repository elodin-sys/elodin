use std::f32::consts;

use bevy::{
    app::{App, Plugin, PostUpdate, Startup},
    asset::{AssetServer, Assets, Handle},
    core_pipeline::{
        clear_color::ClearColorConfig,
        core_3d::{Camera3d, Camera3dBundle},
    },
    ecs::{
        component::Component,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut, Resource},
    },
    math::{Quat, UVec2, Vec2, Vec3},
    pbr::{PbrBundle, StandardMaterial},
    render::{
        camera::{Camera, Viewport},
        color::Color,
        mesh::{shape, Mesh},
        texture::Image,
    },
    transform::components::Transform,
    ui::camera_config::UiCameraConfig,
    utils::default,
    window::Window,
};

use bevy_egui::EguiContexts;
use bevy_mod_picking::prelude::*;
use bevy_panorbit_camera::PanOrbitCamera;

use crate::NAVIGATION_GIZMO_LAYER;

#[derive(Resource)]
pub struct SharedCameraState {
    target_alpha: f32,
    target_beta: f32,
    // orthographic: bool,
}

impl Default for SharedCameraState {
    fn default() -> Self {
        Self {
            target_alpha: 0.0,
            target_beta: 0.0,
            // orthographic: false,
        }
    }
}

pub struct NavigationGizmoPlugin;

impl Plugin for NavigationGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SharedCameraState>()
            .add_systems(Startup, nav_gizmo)
            .add_systems(Startup, nav_gizmo_camera)
            .add_systems(PostUpdate, set_camera_viewport)
            .add_systems(PostUpdate, sync_nav_camera);
    }
}

//------------------------------------------------------------------------------

#[derive(Component)]
pub struct NavGizmoCamera;

fn cube_color_highlight(
    event: Listener<Pointer<Over>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    target_query: Query<&Handle<StandardMaterial>>,
) {
    if let Ok(material_asset) = target_query.get(event.target) {
        if let Some(material) = materials.get_mut(material_asset) {
            material.base_color = Color::TURQUOISE;
        }
    }
}

fn cube_color_reset(
    event: Listener<Pointer<Out>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    target_query: Query<&Handle<StandardMaterial>>,
) {
    if let Ok(material_asset) = target_query.get(event.target) {
        if let Some(material) = materials.get_mut(material_asset) {
            material.base_color = Color::WHITE;
        }
    }
}

// NOTE: Reduces jumping
fn get_closest_angle(current: f32, target: f32) -> f32 {
    if current > (target - consts::PI) {
        target
    } else {
        target - (consts::PI * 2.0)
    }
}

struct NavGizmoMaterials {
    cube_side_top: StandardMaterial,
    cube_side_bottom: StandardMaterial,
    cube_side_right: StandardMaterial,
    cube_side_left: StandardMaterial,
    cube_side_front: StandardMaterial,
    cube_side_back: StandardMaterial,
    cube_side_corner: StandardMaterial,
    cube_side_edge: StandardMaterial,
}

impl NavGizmoMaterials {
    pub fn new(asset_server: Res<AssetServer>) -> Self {
        Self {
            cube_side_top: Self::material(
                asset_server.load("embedded://elodin_editor/assets/textures/cube_side_top.png"),
            ),
            cube_side_bottom: Self::material(
                asset_server.load("embedded://elodin_editor/assets/textures/cube_side_bottom.png"),
            ),
            cube_side_right: Self::material(
                asset_server.load("embedded://elodin_editor/assets/textures/cube_side_right.png"),
            ),
            cube_side_left: Self::material(
                asset_server.load("embedded://elodin_editor/assets/textures/cube_side_left.png"),
            ),
            cube_side_front: Self::material(
                asset_server.load("embedded://elodin_editor/assets/textures/cube_side_front.png"),
            ),
            cube_side_back: Self::material(
                asset_server.load("embedded://elodin_editor/assets/textures/cube_side_back.png"),
            ),
            cube_side_corner: Self::material(
                asset_server.load("embedded://elodin_editor/assets/textures/cube_side_corner.png"),
            ),
            cube_side_edge: Self::material(
                asset_server.load("embedded://elodin_editor/assets/textures/cube_side_edge.png"),
            ),
        }
    }

    fn material(texture: Handle<Image>) -> StandardMaterial {
        StandardMaterial {
            base_color_texture: Some(texture),
            base_color: Color::WHITE,
            unlit: true,
            ..default()
        }
    }
}

pub fn nav_gizmo(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let nav_materials = NavGizmoMaterials::new(asset_server);

    let quad_mesh = meshes.add(Mesh::from(shape::Quad::new(Vec2::new(0.6, 0.6))));
    let edge_mesh = meshes.add(Mesh::from(shape::Box::new(0.2, 0.2, 0.6)));
    let corner_mesh = meshes.add(Mesh::from(shape::Box::new(0.2, 0.2, 0.2)));

    //--------------------------------------------------------------------------

    // top-right edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(0.4, 0.4, 0.0),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI / 2.0);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI / 4.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // bottom-right edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(0.4, -0.4, 0.0),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI / 2.0);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI * 1.75);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // top-left edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(-0.4, 0.4, 0.0),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 1.5);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI / 4.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // bottom-left edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(-0.4, -0.4, 0.0),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 1.5);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI * 1.75);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    //--------------------------------------------------------------------------

    // front-right edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(0.4, 0.0, 0.4)
                .with_rotation(Quat::from_rotation_x(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI / 4.0);
                    nav_orbit.target_beta = get_closest_angle(nav_orbit.target_beta, 0.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // left-front edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(-0.4, 0.0, 0.4)
                .with_rotation(Quat::from_rotation_x(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 1.75);
                    nav_orbit.target_beta = get_closest_angle(nav_orbit.target_beta, 0.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // right-back edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(0.4, 0.0, -0.4)
                .with_rotation(Quat::from_rotation_x(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 0.75);
                    nav_orbit.target_beta = get_closest_angle(nav_orbit.target_beta, 0.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // back-left edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(-0.4, 0.0, -0.4)
                .with_rotation(Quat::from_rotation_x(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 1.25);
                    nav_orbit.target_beta = get_closest_angle(nav_orbit.target_beta, 0.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    //--------------------------------------------------------------------------

    // top-front edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(0.0, 0.4, 0.4)
                .with_rotation(Quat::from_rotation_y(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha = get_closest_angle(nav_orbit.target_alpha, 0.0);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI / 4.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // bottom-front edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(0.0, -0.4, 0.4)
                .with_rotation(Quat::from_rotation_y(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha = get_closest_angle(nav_orbit.target_alpha, 0.0);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI * 1.75);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // top-back edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(0.0, 0.4, -0.4)
                .with_rotation(Quat::from_rotation_y(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha = get_closest_angle(nav_orbit.target_alpha, consts::PI);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI / 4.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // bottom-back edge
    commands.spawn((
        PbrBundle {
            mesh: edge_mesh.clone(),
            material: materials.add(nav_materials.cube_side_edge.clone()),
            transform: Transform::from_xyz(0.0, -0.4, -0.4)
                .with_rotation(Quat::from_rotation_y(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha = get_closest_angle(nav_orbit.target_alpha, consts::PI);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI * 1.75);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    //--------------------------------------------------------------------------

    // top-front-right corner
    commands.spawn((
        PbrBundle {
            mesh: corner_mesh.clone(),
            material: materials.add(nav_materials.cube_side_corner.clone()),
            transform: Transform::from_xyz(0.4, 0.4, 0.4),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI / 4.0);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI / 4.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // top-left-right corner
    commands.spawn((
        PbrBundle {
            mesh: corner_mesh.clone(),
            material: materials.add(nav_materials.cube_side_corner.clone()),
            transform: Transform::from_xyz(-0.4, 0.4, 0.4),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 1.75);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI / 4.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // top-right-back corner
    commands.spawn((
        PbrBundle {
            mesh: corner_mesh.clone(),
            material: materials.add(nav_materials.cube_side_corner.clone()),
            transform: Transform::from_xyz(0.4, 0.4, -0.4),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 0.75);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI / 4.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // top-back-left corner
    commands.spawn((
        PbrBundle {
            mesh: corner_mesh.clone(),
            material: materials.add(nav_materials.cube_side_corner.clone()),
            transform: Transform::from_xyz(-0.4, 0.4, -0.4),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 1.25);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI / 4.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // bottom-front-right corner
    commands.spawn((
        PbrBundle {
            mesh: corner_mesh.clone(),
            material: materials.add(nav_materials.cube_side_corner.clone()),
            transform: Transform::from_xyz(0.4, -0.4, 0.4),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI / 4.0);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI * 1.75);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // bottom-left-right corner
    commands.spawn((
        PbrBundle {
            mesh: corner_mesh.clone(),
            material: materials.add(nav_materials.cube_side_corner.clone()),
            transform: Transform::from_xyz(-0.4, -0.4, 0.4),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 1.75);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI * 1.75);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // bottom-right-back corner
    commands.spawn((
        PbrBundle {
            mesh: corner_mesh.clone(),
            material: materials.add(nav_materials.cube_side_corner.clone()),
            transform: Transform::from_xyz(0.4, -0.4, -0.4),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 0.75);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI * 1.75);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // bottom-back-left corner
    commands.spawn((
        PbrBundle {
            mesh: corner_mesh.clone(),
            material: materials.add(nav_materials.cube_side_corner.clone()),
            transform: Transform::from_xyz(-0.4, -0.4, -0.4),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 1.25);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI * 1.75);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    //--------------------------------------------------------------------------

    // top
    commands.spawn((
        PbrBundle {
            mesh: quad_mesh.clone(),
            material: materials.add(nav_materials.cube_side_top),
            transform: Transform::from_xyz(0.0, 0.5, 0.0)
                .with_rotation(Quat::from_rotation_x(consts::PI * 1.5)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha = get_closest_angle(nav_orbit.target_alpha, 0.0);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI / 2.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // bottom
    commands.spawn((
        PbrBundle {
            mesh: quad_mesh.clone(),
            material: materials.add(nav_materials.cube_side_bottom),
            transform: Transform::from_xyz(0.0, -0.5, 0.0)
                .with_rotation(Quat::from_rotation_x(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha = get_closest_angle(nav_orbit.target_alpha, 0.0);
                    nav_orbit.target_beta =
                        get_closest_angle(nav_orbit.target_beta, consts::PI * 1.5);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // front
    commands.spawn((
        PbrBundle {
            mesh: quad_mesh.clone(),
            material: materials.add(nav_materials.cube_side_front),
            transform: Transform::from_xyz(0.0, 0.0, 0.5),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha = get_closest_angle(nav_orbit.target_alpha, 0.0);
                    nav_orbit.target_beta = get_closest_angle(nav_orbit.target_beta, 0.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // back
    commands.spawn((
        PbrBundle {
            mesh: quad_mesh.clone(),
            material: materials.add(nav_materials.cube_side_back),
            transform: Transform::from_xyz(0.0, 0.0, -0.5)
                .with_rotation(Quat::from_rotation_y(consts::PI)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha = get_closest_angle(nav_orbit.target_alpha, consts::PI);
                    nav_orbit.target_beta = get_closest_angle(nav_orbit.target_beta, 0.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // right
    commands.spawn((
        PbrBundle {
            mesh: quad_mesh.clone(),
            material: materials.add(nav_materials.cube_side_right),
            transform: Transform::from_xyz(0.5, 0.0, 0.0)
                .with_rotation(Quat::from_rotation_y(consts::PI / 2.0)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI / 2.0);
                    nav_orbit.target_beta = get_closest_angle(nav_orbit.target_beta, 0.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));

    // left
    commands.spawn((
        PbrBundle {
            mesh: quad_mesh.clone(),
            material: materials.add(nav_materials.cube_side_left),
            transform: Transform::from_xyz(-0.5, 0.0, 0.0)
                .with_rotation(Quat::from_rotation_y(consts::PI * 1.5)),
            ..default()
        },
        On::<Pointer<Over>>::run(cube_color_highlight),
        On::<Pointer<Out>>::run(cube_color_reset),
        On::<Pointer<Click>>::run(
            |click: Listener<Pointer<Click>>,
             mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>| {
                if click.button == PointerButton::Primary {
                    let mut nav_orbit = nav_orbit_query.single_mut();
                    nav_orbit.target_alpha =
                        get_closest_angle(nav_orbit.target_alpha, consts::PI * 1.5);
                    nav_orbit.target_beta = get_closest_angle(nav_orbit.target_beta, 0.0);
                }
            },
        ),
        NAVIGATION_GIZMO_LAYER,
    ));
}

pub fn nav_gizmo_camera(mut commands: Commands) {
    commands
        .spawn((
            Camera3dBundle {
                transform: Transform::from_xyz(1.6, 1.6, 1.6).looking_at(Vec3::ZERO, Vec3::Y),
                // TODO: Use with orthographic mode in main camera
                // projection: OrthographicProjection {
                //     scaling_mode: ScalingMode::FixedVertical(2.0),
                //     ..default()
                // }
                // .into(),
                camera: Camera {
                    order: 1,
                    hdr: true,
                    ..Default::default()
                },
                camera_3d: Camera3d {
                    // NOTE: Don't clear on the NavGizmoCamera because the MainCamera already cleared the window
                    clear_color: ClearColorConfig::None,
                    ..default()
                },
                ..default()
            },
            UiCameraConfig { show_ui: false },
            NAVIGATION_GIZMO_LAYER,
            NavGizmoCamera,
        ))
        .insert(PanOrbitCamera {
            pan_sensitivity: 0.0,
            zoom_sensitivity: 0.0,
            orbit_sensitivity: 8.0,
            ..Default::default()
        });
}

pub fn sync_nav_camera(
    mut camera_state: ResMut<SharedCameraState>,
    mut main_orbit_query: Query<&mut PanOrbitCamera, Without<NavGizmoCamera>>,
    mut nav_orbit_query: Query<&mut PanOrbitCamera, With<NavGizmoCamera>>,
) {
    let mut main_orbit = main_orbit_query.single_mut();
    let mut nav_orbit = nav_orbit_query.single_mut();

    if main_orbit.target_alpha != camera_state.target_alpha
        || main_orbit.target_beta != camera_state.target_beta
    {
        camera_state.target_alpha = main_orbit.target_alpha;
        camera_state.target_beta = main_orbit.target_beta;

        nav_orbit.target_alpha = camera_state.target_alpha;
        nav_orbit.target_beta = camera_state.target_beta;
    } else if nav_orbit.target_alpha != camera_state.target_alpha
        || nav_orbit.target_beta != camera_state.target_beta
    {
        camera_state.target_alpha = nav_orbit.target_alpha;
        camera_state.target_beta = nav_orbit.target_beta;

        main_orbit.target_alpha = camera_state.target_alpha;
        main_orbit.target_beta = camera_state.target_beta;
    }
}

pub fn set_camera_viewport(
    window: Query<&Window>,
    egui_settings: Res<bevy_egui::EguiSettings>,
    mut contexts: EguiContexts,
    mut nav_camera_query: Query<&mut Camera, With<NavGizmoCamera>>,
) {
    let margin = 16.0;
    let side_length = 128.0;

    let available_rect = contexts.ctx_mut().available_rect();

    let window = window.single();
    let scale_factor = (window.scale_factor() * egui_settings.scale_factor) as f32;

    let margin = margin * scale_factor;
    let side_length = side_length * scale_factor;

    let viewport_pos = available_rect.left_top().to_vec2() * scale_factor;
    let viewport_size = available_rect.size() * scale_factor;

    let nav_viewport_pos = Vec2::new(
        (viewport_pos.x + viewport_size.x) - (side_length + margin),
        viewport_pos.y + margin,
    );

    let mut nav_camera = nav_camera_query.single_mut();
    nav_camera.viewport = Some(Viewport {
        physical_position: UVec2::new(nav_viewport_pos.x as u32, nav_viewport_pos.y as u32),
        physical_size: UVec2::new(side_length as u32, side_length as u32),
        depth: 0.0..1.0,
    });
}
