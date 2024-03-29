use crate::MainCamera;
use bevy::prelude::*;
use bevy::render::camera::Viewport;
use bevy::render::view::RenderLayers;
use bevy_egui::EguiContexts;
use bevy_mod_picking::prelude::*;
use bevy_tweening::lens::TransformScaleLens;
use bevy_tweening::{EaseFunction, Lens, Tween};
use std::{f32::consts, time::Duration};

pub struct NavigationGizmoPlugin;

impl Plugin for NavigationGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderLayerAlloc>()
            .add_systems(PostUpdate, set_camera_viewport)
            .add_systems(PostUpdate, sync_nav_camera);
    }
}

#[derive(Component)]
pub struct NavGizmoCamera;

fn cube_color_highlight(
    event: Listener<Pointer<Over>>,
    mut target_query: Query<Entity>,
    mut commands: Commands,
) {
    if let Ok(entity) = target_query.get_mut(event.target) {
        let tween = Tween::new(
            EaseFunction::SineInOut,
            Duration::from_millis(100),
            TransformScaleLens {
                start: Vec3::splat(1.0),
                end: Vec3::splat(1.3),
            },
        );
        commands
            .entity(entity)
            .insert(bevy_tweening::Animator::new(tween));
    }
}

fn cube_color_reset(
    event: Listener<Pointer<Out>>,
    mut target_query: Query<Entity>,
    mut commands: Commands,
) {
    if let Ok(entity) = target_query.get_mut(event.target) {
        let tween = Tween::new(
            EaseFunction::SineInOut,
            Duration::from_millis(100),
            TransformScaleLens {
                start: Vec3::splat(1.3),
                end: Vec3::splat(1.0),
            },
        );
        commands
            .entity(entity)
            .insert(bevy_tweening::Animator::new(tween));
    }
}

#[derive(Resource, Debug)]
pub struct RenderLayerAlloc(u32);

impl Default for RenderLayerAlloc {
    fn default() -> Self {
        Self(!1u32)
    }
}

impl RenderLayerAlloc {
    fn alloc(&mut self) -> Option<u32> {
        let bits = self.0;
        let mut mask = 1;
        for i in 0..32 {
            if (bits & mask) != 0 {
                self.0 &= !mask;
                return Some(i);
            }
            mask <<= 1;
        }
        None
    }

    #[allow(dead_code)]
    pub fn free(&mut self, layer: u32) {
        self.0 |= 1 << layer;
    }
}

#[derive(Component, Debug)]
pub struct NavGizmo;

#[derive(Component, Debug)]
pub struct NavGizmoParent {
    main_camera: Entity,
}

pub fn spawn_gizmo(
    main_camera: Entity,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    render_layer_alloc: &mut ResMut<RenderLayerAlloc>,
) {
    let Some(render_layer) = render_layer_alloc.alloc() else {
        return;
    };
    let render_layers = RenderLayers::layer(render_layer as u8);
    let sphere = meshes.add(Mesh::from(Sphere::new(0.075)));

    let nav_gizmo = commands
        .spawn((
            NavGizmo,
            NavGizmoParent { main_camera },
            Transform::from_xyz(0.0, 0.0, 0.0),
            GlobalTransform::default(),
            On::<Pointer<Drag>>::run(drag_nav_gizmo),
        ))
        .id();

    let distance = 0.35;

    let edges = [
        // top
        (
            crate::ui::colors::bevy::GREEN,
            Transform::from_xyz(0.0, distance, 0.0)
                .with_rotation(Quat::from_rotation_x(consts::PI * 1.5)),
            side_clicked_cb(0.0, consts::PI / 2.0),
        ),
        // bottom
        (
            crate::ui::colors::bevy::GREY_900,
            Transform::from_xyz(0.0, -distance, 0.0)
                .with_rotation(Quat::from_rotation_x(consts::PI / 2.0)),
            side_clicked_cb(0.0, consts::PI * 1.5),
        ),
        // front
        (
            crate::ui::colors::bevy::BLUE,
            Transform::from_xyz(0.0, 0.0, distance),
            side_clicked_cb(0.0, 0.0),
        ),
        // back
        (
            crate::ui::colors::bevy::GREY_900,
            Transform::from_xyz(0.0, 0.0, -distance)
                .with_rotation(Quat::from_rotation_y(consts::PI)),
            side_clicked_cb(consts::PI, 0.0),
        ),
        // right
        (
            crate::ui::colors::bevy::RED,
            Transform::from_xyz(distance, 0.0, 0.0)
                .with_rotation(Quat::from_rotation_y(consts::PI / 2.0)),
            side_clicked_cb(consts::PI / 2.0, 0.0),
        ),
        // left
        (
            crate::ui::colors::bevy::GREY_900,
            Transform::from_xyz(-distance, 0.0, 0.0)
                .with_rotation(Quat::from_rotation_y(consts::PI * 1.5)),
            side_clicked_cb(consts::PI * 1.5, 0.0),
        ),
    ];

    for (color, transform, cb) in edges {
        let material = StandardMaterial {
            base_color: color,
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        };
        commands
            .spawn((
                PbrBundle {
                    mesh: meshes.add(Mesh::from(Cuboid::new(0.03, 0.03, distance))),
                    material: materials.add(material.clone()),
                    transform: Transform {
                        translation: transform.translation / 2.0,
                        rotation: transform.rotation,
                        ..Default::default()
                    },
                    ..default()
                },
                NavGizmoParent { main_camera },
                render_layers,
            ))
            .set_parent(nav_gizmo);
        commands
            .spawn((
                PbrBundle {
                    mesh: sphere.clone(),
                    material: materials.add(material),
                    transform,
                    ..default()
                },
                NavGizmoParent { main_camera },
                On::<Pointer<Over>>::run(cube_color_highlight),
                On::<Pointer<Out>>::run(cube_color_reset),
                On::<Pointer<Click>>::run(cb),
                render_layers,
            ))
            .set_parent(nav_gizmo);
    }

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
            camera: Camera {
                order: 2,
                hdr: true,
                // NOTE: Don't clear on the NavGizmoCamera because the MainCamera already cleared the window
                clear_color: ClearColorConfig::None,

                ..Default::default()
            },
            camera_3d: Camera3d { ..default() },
            ..default()
        },
        render_layers,
        NavGizmoParent { main_camera },
        NavGizmoCamera,
    ));
}

#[derive(Component)]
pub struct DraggedMarker;

pub fn drag_nav_gizmo(
    drag: Listener<Pointer<Drag>>,
    nav_gizmo: Query<&NavGizmoParent>,
    mut query: Query<&mut Transform, With<MainCamera>>,
    mut commands: Commands,
) {
    let Ok(nav_gizmo) = nav_gizmo.get(drag.target) else {
        return;
    };
    let Ok(mut transform) = query.get_mut(nav_gizmo.main_camera) else {
        return;
    };
    if drag.delta.length() > 0.1 {
        commands.entity(drag.target).insert(DraggedMarker);
    } else {
        commands.entity(drag.target).remove::<DraggedMarker>();
    }
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
    Query<&NavGizmoParent>,
    Query<&DraggedMarker>,
    Commands,
) {
    move |click: Listener<Pointer<Click>>,
          query: Query<(Entity, &Transform), With<MainCamera>>,
          nav_gizmo: Query<&NavGizmoParent>,
          drag_query: Query<&DraggedMarker>,
          mut commands: Commands| {
        let Ok(nav_gizmo) = nav_gizmo.get(click.target) else {
            return;
        };
        let Ok((entity, old_transform)) = query.get(nav_gizmo.main_camera) else {
            return;
        };

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
    main_transform_query: Query<&Transform, (With<MainCamera>, Without<NavGizmoParent>)>,
    mut nav_transform_query: Query<(Entity, &NavGizmoParent, &mut Transform), With<NavGizmo>>,
    mut commands: Commands,
) {
    for (entity, nav_gizmo, mut nav_transform) in nav_transform_query.iter_mut() {
        let Ok(main) = main_transform_query.get(nav_gizmo.main_camera) else {
            commands.entity(entity).despawn_recursive();
            continue;
        };
        nav_transform.rotation = main.rotation.conjugate();
    }
}

pub fn set_camera_viewport(
    window: Query<&Window>,
    egui_settings: Res<bevy_egui::EguiSettings>,
    _contexts: EguiContexts,
    mut nav_camera_query: Query<(&mut Camera, &NavGizmoParent)>,
    main_camera_query: Query<&mut Camera, Without<NavGizmoParent>>,
) {
    let margin = 8.0;
    let side_length = 128.0;
    let Some(window) = window.iter().next() else {
        return;
    };
    let scale_factor = window.scale_factor() * egui_settings.scale_factor;

    let margin = margin * scale_factor;
    let side_length = side_length * scale_factor;
    for (mut nav_camera, parent) in nav_camera_query.iter_mut() {
        let Ok(main) = main_camera_query.get(parent.main_camera) else {
            continue;
        };
        let Some(viewport) = &main.viewport else {
            continue;
        };
        let viewport_pos = viewport.physical_position.as_vec2();
        let viewport_size = viewport.physical_size.as_vec2();
        let nav_viewport_pos = Vec2::new(
            (viewport_pos.x + viewport_size.x) - (side_length + margin),
            viewport_pos.y,
        );
        let physical_size = if main.is_active {
            UVec2::new(side_length as u32, side_length as u32)
        } else {
            UVec2::new(1, 1)
        };

        nav_camera.viewport = Some(Viewport {
            physical_position: UVec2::new(nav_viewport_pos.x as u32, nav_viewport_pos.y as u32),
            physical_size,
            depth: 0.0..1.0,
        });
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_layer_alloc() {
        let mut default = RenderLayerAlloc::default();
        assert_eq!(default.alloc().unwrap(), 1);
        assert_eq!(default.alloc().unwrap(), 2);
    }
}
