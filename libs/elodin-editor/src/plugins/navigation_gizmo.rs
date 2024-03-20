use bevy::ecs::entity::Entity;
use bevy::math::primitives::Sphere;
use bevy::pbr::AlphaMode;
use bevy::{
    app::{App, Plugin, PostUpdate, Startup},
    asset::{Assets, Handle},
    core_pipeline::core_3d::{Camera3d, Camera3dBundle},
    ecs::{
        component::Component,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut},
    },
    hierarchy::BuildChildren,
    math::{Mat3, Quat, UVec2, Vec2, Vec3},
    pbr::{PbrBundle, StandardMaterial},
    prelude::ClearColorConfig,
    render::{
        camera::{Camera, Viewport},
        color::Color,
        mesh::Mesh,
    },
    transform::components::{GlobalTransform, Transform},
    utils::default,
    window::Window,
};
use bevy_tweening::{EaseFunction, Lens, Tween};
use std::{f32::consts, time::Duration};

use crate::{MainCamera, NAVIGATION_GIZMO_LAYER};
use bevy_egui::EguiContexts;
use bevy_mod_picking::prelude::*;

pub struct NavigationGizmoPlugin;

impl Plugin for NavigationGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, nav_gizmo)
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
            material.base_color *= 0.75;
        }
    }
}

fn cube_color_reset(
    reset_color: Color,
) -> impl Fn(Listener<Pointer<Out>>, ResMut<Assets<StandardMaterial>>, Query<&Handle<StandardMaterial>>)
{
    move |event: Listener<Pointer<Out>>,
          mut materials: ResMut<Assets<StandardMaterial>>,
          target_query: Query<&Handle<StandardMaterial>>| {
        if let Ok(material_asset) = target_query.get(event.target) {
            if let Some(material) = materials.get_mut(material_asset) {
                material.base_color = reset_color;
            }
        }
    }
}

#[derive(Component)]
pub struct NavGizmo;

pub fn nav_gizmo(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let sphere = meshes.add(Mesh::from(Sphere::new(0.15)));

    let nav_gizmo = commands
        .spawn((
            NavGizmo,
            Transform::from_xyz(0.0, 0.0, 0.0),
            GlobalTransform::default(),
            On::<Pointer<Drag>>::run(drag_nav_gizmo),
        ))
        .id();

    let distance = 0.4;

    let edges = [
        // top
        (
            crate::ui::colors::bevy::BLUE,
            Transform::from_xyz(0.0, distance, 0.0)
                .with_rotation(Quat::from_rotation_x(consts::PI * 1.5)),
            side_clicked_cb(0.0, consts::PI / 2.0),
        ),
        // bottom
        (
            crate::ui::colors::bevy::BLUE.with_a(0.2),
            Transform::from_xyz(0.0, -distance, 0.0)
                .with_rotation(Quat::from_rotation_x(consts::PI / 2.0)),
            side_clicked_cb(0.0, consts::PI * 1.5),
        ),
        // front
        (
            crate::ui::colors::bevy::RED,
            Transform::from_xyz(0.0, 0.0, distance),
            side_clicked_cb(0.0, 0.0),
        ),
        // back
        (
            crate::ui::colors::bevy::RED.with_a(0.2),
            Transform::from_xyz(0.0, 0.0, -distance)
                .with_rotation(Quat::from_rotation_y(consts::PI)),
            side_clicked_cb(consts::PI, 0.0),
        ),
        // right
        (
            crate::ui::colors::bevy::GREEN,
            Transform::from_xyz(distance, 0.0, 0.0)
                .with_rotation(Quat::from_rotation_y(consts::PI / 2.0)),
            side_clicked_cb(consts::PI / 2.0, 0.0),
        ),
        // left
        (
            crate::ui::colors::bevy::GREEN.with_a(0.2),
            Transform::from_xyz(-distance, 0.0, 0.0)
                .with_rotation(Quat::from_rotation_y(consts::PI * 1.5)),
            side_clicked_cb(consts::PI * 1.5, 0.0),
        ),
    ];

    for (color, transform, cb) in edges {
        let material = materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        });
        commands
            .spawn((
                PbrBundle {
                    mesh: sphere.clone(),
                    material,
                    transform,
                    ..default()
                },
                On::<Pointer<Over>>::run(cube_color_highlight),
                On::<Pointer<Out>>::run(cube_color_reset(color)),
                On::<Pointer<Click>>::run(cb),
                NAVIGATION_GIZMO_LAYER,
            ))
            .set_parent(nav_gizmo);
    }
}

pub fn nav_gizmo_camera(mut commands: Commands) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
            camera: Camera {
                order: 1,
                hdr: true,
                // NOTE: Don't clear on the NavGizmoCamera because the MainCamera already cleared the window
                clear_color: ClearColorConfig::None,

                ..Default::default()
            },
            camera_3d: Camera3d { ..default() },
            ..default()
        },
        //UiCameraConfig { show_ui: false },
        NAVIGATION_GIZMO_LAYER,
        NavGizmoCamera,
    ));
}

#[derive(Component)]
pub struct DraggedMarker;

pub fn drag_nav_gizmo(
    drag: Listener<Pointer<Drag>>,
    mut query: Query<&mut Transform, With<MainCamera>>,
    mut commands: Commands,
) {
    if drag.delta.length() > 0.1 {
        commands.entity(drag.target).insert(DraggedMarker);
    } else {
        commands.entity(drag.target).remove::<DraggedMarker>();
    }
    let mut transform = query.single_mut();
    let delta_x = drag.delta.x / 75.0 * std::f32::consts::PI;
    let delta_y = drag.delta.y / 75.0 * std::f32::consts::PI;
    let yaw = Quat::from_rotation_y(-delta_x);
    let pitch = Quat::from_rotation_x(-delta_y);
    set_orbit_rotation(Vec3::ZERO, yaw, pitch, transform.as_mut())
}

fn set_orbit_rotation(anchor: Vec3, yaw: Quat, pitch: Quat, transform: &mut Transform) {
    let radius = (transform.translation - anchor).length();
    transform.rotation = yaw * transform.rotation;
    transform.rotation *= pitch;
    let rot_matrix = Mat3::from_quat(transform.rotation);
    transform.translation = anchor + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, radius));
}

fn side_clicked_cb(
    yaw: f32,
    pitch: f32,
) -> impl Fn(
    Listener<Pointer<Click>>,
    Query<(Entity, &Transform), With<MainCamera>>,
    Query<&DraggedMarker>,
    Commands,
) {
    move |click: Listener<Pointer<Click>>,
          query: Query<(Entity, &Transform), With<MainCamera>>,
          drag_query: Query<&DraggedMarker>,
          mut commands: Commands| {
        let (entity, old_transform) = query.single();
        if drag_query.get(click.target).is_ok() {
            return;
        }
        if click.button == PointerButton::Primary {
            let mut new_transform = *old_transform;
            let anchor = Vec3::ZERO;
            let yaw = Quat::from_rotation_y(yaw);
            let pitch = Quat::from_rotation_x(-pitch);
            let radius = (new_transform.translation - anchor).length();
            new_transform.rotation = yaw * pitch;
            let tween = Tween::new(
                EaseFunction::SineInOut,
                Duration::from_millis(250),
                OrbitLens {
                    start: *old_transform,
                    end: new_transform,
                    radius,
                    anchor,
                },
            );
            commands
                .entity(entity)
                .insert(bevy_tweening::Animator::new(tween));
        }
    }
}

pub fn sync_nav_camera(
    mut main_transform_query: Query<&Transform, (With<MainCamera>, Without<NavGizmo>)>,
    mut nav_transform_query: Query<&mut Transform, With<NavGizmo>>,
) {
    let main = main_transform_query.single_mut();
    let mut nav = nav_transform_query.single_mut();
    nav.rotation = main.rotation.conjugate();
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

    if available_rect.size().x > window.width() || available_rect.size().y > window.height() {
        return;
    }

    let scale_factor = window.scale_factor() * egui_settings.scale_factor;

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

struct OrbitLens {
    start: Transform,
    end: Transform,
    radius: f32,
    anchor: Vec3,
}

impl Lens<Transform> for OrbitLens {
    fn lerp(&mut self, target: &mut Transform, ratio: f32) {
        target.rotation = self.start.rotation.slerp(self.end.rotation, ratio);
        let rot_matrix = Mat3::from_quat(target.rotation);
        target.translation = self.anchor + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, self.radius));
    }
}
