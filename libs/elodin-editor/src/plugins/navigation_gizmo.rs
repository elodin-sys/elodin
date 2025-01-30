use crate::MainCamera;
use bevy::animation::{animated_field, AnimationTarget, AnimationTargetId};
use bevy::math::{DVec3, Dir3};
use bevy::prelude::*;
use bevy::render::camera::Viewport;
use bevy::render::view::RenderLayers;
use bevy_editor_cam::controller::component::EditorCam;
use bevy_editor_cam::extensions::look_to::LookToTrigger;
use bevy_egui::EguiContexts;
use std::f32::consts;

pub struct NavigationGizmoPlugin;

impl Plugin for NavigationGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderLayerAlloc>()
            .add_systems(PostUpdate, set_camera_viewport)
            .add_systems(PostUpdate, sync_nav_camera)
            .add_plugins(MeshPickingPlugin);
    }
}

#[derive(Component)]
pub struct NavGizmoCamera;

fn cube_color_highlight(
    event: Trigger<Pointer<Over>>,
    mut target_query: Query<Entity>,
    mut animations: ResMut<Assets<AnimationClip>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut commands: Commands,
) {
    if let Ok(entity) = target_query.get_mut(event.target) {
        let target = AnimationTargetId::from_name(&Name::new(entity.to_string()));
        let mut animation = AnimationClip::default();

        animation.add_curve_to_target(
            target,
            AnimatableCurve::new(
                animated_field![Transform::scale],
                AnimatableKeyframeCurve::new([(0.0, Vec3::splat(1.0)), (0.1, Vec3::splat(1.3))])
                    .expect("bad curve"),
            ),
        );
        let (graph, animation_index) = AnimationGraph::from_clip(animations.add(animation));

        let mut player = AnimationPlayer::default();

        player.play(animation_index);
        commands
            .entity(entity)
            .insert(AnimationGraphHandle(graphs.add(graph)))
            .insert(player)
            .insert(AnimationTarget {
                id: target,
                player: entity,
            });
    }
}

fn cube_color_reset(
    event: Trigger<Pointer<Out>>,
    mut target_query: Query<Entity>,
    mut animations: ResMut<Assets<AnimationClip>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut commands: Commands,
) {
    if let Ok(entity) = target_query.get_mut(event.target) {
        let target = AnimationTargetId::from_name(&Name::new(entity.to_string()));
        let mut animation = AnimationClip::default();

        animation.add_curve_to_target(
            target,
            AnimatableCurve::new(
                animated_field![Transform::scale],
                AnimatableKeyframeCurve::new([(0.0, Vec3::splat(1.3)), (0.1, Vec3::splat(1.0))])
                    .expect("bad curve"),
            ),
        );
        let (graph, animation_index) = AnimationGraph::from_clip(animations.add(animation));

        let mut player = AnimationPlayer::default();

        player.play(animation_index);
        commands
            .entity(entity)
            .insert(AnimationGraphHandle(graphs.add(graph)))
            .insert(player)
            .insert(AnimationTarget {
                id: target,
                player: entity,
            });
    }
}

#[derive(Resource, Debug)]
pub struct RenderLayerAlloc(usize);

impl Default for RenderLayerAlloc {
    fn default() -> Self {
        Self(!1usize)
    }
}

impl RenderLayerAlloc {
    pub fn alloc(&mut self) -> Option<usize> {
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
    pub fn free(&mut self, layer: usize) {
        self.0 |= 1 << layer;
    }

    pub fn free_all(&mut self) {
        self.0 = !1;
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
) -> (Option<Entity>, Option<Entity>) {
    let Some(render_layer) = render_layer_alloc.alloc() else {
        return (None, None);
    };
    let render_layers = RenderLayers::layer(render_layer);
    let sphere = meshes.add(Mesh::from(Sphere::new(0.075)));

    let nav_gizmo = commands
        .spawn((
            NavGizmo,
            NavGizmoParent { main_camera },
            Transform::from_xyz(0.0, 0.0, 0.0),
            GlobalTransform::default(),
        ))
        .observe(drag_nav_gizmo)
        .id();

    let distance = 0.35;

    let edges = [
        // top
        (
            crate::ui::colors::bevy::BLUE,
            Transform::from_xyz(0.0, distance, 0.0)
                .with_rotation(Quat::from_rotation_x(consts::PI * 1.5)),
            side_clicked_cb(Dir3::NEG_Y),
        ),
        // bottom
        (
            crate::ui::colors::bevy::GREY_900,
            Transform::from_xyz(0.0, -distance, 0.0)
                .with_rotation(Quat::from_rotation_x(consts::PI / 2.0)),
            side_clicked_cb(Dir3::Y),
        ),
        // front
        (
            crate::ui::colors::bevy::GREY_900,
            Transform::from_xyz(0.0, 0.0, distance),
            side_clicked_cb(Dir3::NEG_Z),
        ),
        // back
        (
            crate::ui::colors::bevy::GREEN,
            Transform::from_xyz(0.0, 0.0, -distance)
                .with_rotation(Quat::from_rotation_y(consts::PI)),
            side_clicked_cb(Dir3::Z),
        ),
        // right
        (
            crate::ui::colors::bevy::RED,
            Transform::from_xyz(distance, 0.0, 0.0)
                .with_rotation(Quat::from_rotation_y(consts::PI / 2.0)),
            side_clicked_cb(Dir3::NEG_X),
        ),
        // left
        (
            crate::ui::colors::bevy::GREY_900,
            Transform::from_xyz(-distance, 0.0, 0.0)
                .with_rotation(Quat::from_rotation_y(consts::PI * 1.5)),
            side_clicked_cb(Dir3::X),
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
                Mesh3d(meshes.add(Mesh::from(Cuboid::new(0.03, 0.03, distance)))),
                MeshMaterial3d(materials.add(material.clone())),
                Transform {
                    translation: transform.translation / 2.0,
                    rotation: transform.rotation,
                    ..Default::default()
                },
                NavGizmoParent { main_camera },
                render_layers.clone(),
            ))
            .set_parent(nav_gizmo);
        commands
            .spawn((
                Mesh3d(sphere.clone()),
                MeshMaterial3d(materials.add(material)),
                transform,
                NavGizmoParent { main_camera },
                render_layers.clone(),
            ))
            .observe(cube_color_highlight)
            .observe(cube_color_reset)
            .observe(cb)
            .set_parent(nav_gizmo);
    }

    let nav_gizmo_camera = commands
        .spawn((
            Transform::from_xyz(0.0, 0.0, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
            Camera {
                order: 3,
                hdr: false,
                // NOTE: Don't clear on the NavGizmoCamera because the MainCamera already cleared the window
                clear_color: ClearColorConfig::None,
                ..Default::default()
            },
            Camera3d::default(),
            render_layers.clone(),
            NavGizmoParent { main_camera },
            NavGizmoCamera,
        ))
        .id();

    (Some(nav_gizmo), Some(nav_gizmo_camera))
}

#[derive(Component)]
pub struct DraggedMarker;

pub fn drag_nav_gizmo(
    drag: Trigger<Pointer<Drag>>,
    nav_gizmo: Query<&NavGizmoParent>,
    mut query: Query<(&mut Transform, &mut EditorCam, &Camera), With<MainCamera>>,
    mut commands: Commands,
) {
    let Ok(nav_gizmo) = nav_gizmo.get(drag.target) else {
        return;
    };
    let Ok((transform, mut editor_cam, cam)) = query.get_mut(nav_gizmo.main_camera) else {
        return;
    };
    if drag.delta.length() > 0.1 {
        commands.entity(drag.target).insert(DraggedMarker);
    } else {
        commands.entity(drag.target).remove::<DraggedMarker>();
    }
    let delta = drag.delta
        * cam
            .physical_viewport_size()
            .unwrap_or_else(|| UVec2::new(256, 256))
            .as_vec2()
        / 75.0;
    let anchor = transform
        .compute_matrix()
        .as_dmat4()
        .inverse()
        .transform_point3(DVec3::ZERO);
    editor_cam.end_move();
    editor_cam.start_orbit(Some(anchor));
    editor_cam.send_screenspace_input(delta);
}

#[allow(clippy::type_complexity)]
fn side_clicked_cb(
    direction: Dir3,
) -> impl Fn(
    Trigger<Pointer<Click>>,
    Query<(Entity, &Transform, &EditorCam), With<MainCamera>>,
    Query<&NavGizmoParent>,
    Query<&DraggedMarker>,
    EventWriter<LookToTrigger>,
) {
    move |click: Trigger<Pointer<Click>>,
          query: Query<(Entity, &Transform, &EditorCam), With<MainCamera>>,
          nav_gizmo: Query<&NavGizmoParent>,
          drag_query: Query<&DraggedMarker>,
          mut look_to: EventWriter<LookToTrigger>| {
        let Ok(nav_gizmo) = nav_gizmo.get(click.target) else {
            return;
        };
        let Ok((entity, transform, editor_cam)) = query.get(nav_gizmo.main_camera) else {
            return;
        };

        if drag_query.get(click.target).is_ok() {
            return;
        }
        if click.button == PointerButton::Primary {
            look_to.send(LookToTrigger::auto_snap_up_direction(
                direction, entity, transform, editor_cam,
            ));
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
    window: Query<(&Window, &bevy_egui::EguiSettings)>,
    _contexts: EguiContexts,
    mut nav_camera_query: Query<(&mut Camera, &NavGizmoParent)>,
    main_camera_query: Query<&mut Camera, Without<NavGizmoParent>>,
) {
    let margin = 8.0;
    let side_length = 128.0;
    let Some((window, egui_settings)) = window.iter().next() else {
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
        nav_camera.is_active = main.is_active;

        nav_camera.viewport = Some(Viewport {
            physical_position: UVec2::new(nav_viewport_pos.x as u32, nav_viewport_pos.y as u32),
            physical_size,
            depth: 0.0..1.0,
        });
    }
}

// struct OrbitLens {
//     start: Transform,
//     end: Transform,
//     radius: f32,
//     anchor: Vec3,
// }

// impl Lens<Transform> for OrbitLens {
//     fn lerp(&mut self, target: &mut Transform, ratio: f32) {
//         target.rotation = self.start.rotation.slerp(self.end.rotation, ratio);
//         let rot_matrix = Mat3::from_quat(target.rotation);
//         target.translation = self.anchor + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, self.radius));
//     }
// }

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
