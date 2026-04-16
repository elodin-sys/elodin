use crate::{MainCamera, plugins::camera_anchor::camera_anchor_from_transform, ui::ViewportRect};
use bevy::animation::{AnimationTarget, AnimationTargetId, animated_field};
use bevy::camera::{RenderTarget, Viewport};
use bevy::math::Dir3;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowRef};
use bevy_editor_cam::controller::component::EditorCam;
use bevy_editor_cam::extensions::look_to::LookToTrigger;
use bevy_editor_cam::prelude::EnabledMotion;
use bevy_egui::EguiContexts;
use std::{collections::HashMap, f32::consts};

use super::render_layer_alloc::{self, RenderLayerLease, RenderLayerAllocator};

pub struct NavigationGizmoPlugin;

impl Plugin for NavigationGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NavGizmoAnchorState>()
            .add_systems(PostUpdate, set_camera_viewport)
            .add_systems(PostUpdate, sync_nav_camera)
            .add_plugins(MeshPickingPlugin)
            .add_plugins(render_layer_alloc::plugin);
    }
}

#[derive(Component)]
pub struct NavGizmoCamera;

fn cube_color_highlight(
    event: On<Pointer<Over>>,
    mut target_query: Query<Entity>,
    mut animations: ResMut<Assets<AnimationClip>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut commands: Commands,
) {
    let event_target = event.event().event_target();
    if let Ok(entity) = target_query.get_mut(event_target) {
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
    event: On<Pointer<Out>>,
    mut target_query: Query<Entity>,
    mut animations: ResMut<Assets<AnimationClip>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut commands: Commands,
) {
    let event_target = event.event().event_target();
    if let Ok(entity) = target_query.get_mut(event_target) {
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

#[derive(Component, Debug)]
pub struct NavGizmo;

#[derive(Component, Debug)]
pub struct NavGizmoParent {
    pub main_camera: Entity,
}

#[derive(Debug)]
struct NavGizmoDrag {
    main_camera: Entity,
    pointer_press: Vec2,
    grab_offset: Vec2,
    dragging: bool,
}

#[derive(Resource, Default, Debug)]
struct NavGizmoAnchorState {
    offsets: HashMap<Entity, Vec2>,
    active_drag: Option<NavGizmoDrag>,
}

pub fn spawn_gizmo(
    main_camera: Entity,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    render_layer_alloc: &mut ResMut<RenderLayerAllocator>,
) -> (Option<Entity>, Option<Entity>) {
    let Some(lease) = render_layer_alloc.alloc() else {
        return (None, None);
    };
    let render_layers = lease.render_layers();
    let sphere = meshes.add(Mesh::from(Sphere::new(0.075)));

    let nav_gizmo = commands
        .spawn((
            NavGizmo,
            NavGizmoParent { main_camera },
            Transform::from_xyz(0.0, 0.0, 0.0),
            GlobalTransform::default(),
            Name::new("nav gizmo"),
        ))
        .observe(drag_nav_gizmo)
        .observe(drag_nav_gizmo_end)
        .insert(lease)
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
            .insert(ChildOf(nav_gizmo));
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
            .insert(ChildOf(nav_gizmo));
    }

    let nav_gizmo_camera = commands
        .spawn((
            Transform::from_xyz(0.0, 0.0, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
            Camera {
                order: 3,
                // NOTE: Don't clear on the NavGizmoCamera because the
                // MainCamera already cleared the window.
                clear_color: ClearColorConfig::None,
                ..Default::default()
            },
            Camera3d::default(),
            render_layers.clone(),
            NavGizmoParent { main_camera },
            NavGizmoCamera,
            Name::new("nav gizmo camera"),
        ))
        .id();

    (Some(nav_gizmo), Some(nav_gizmo_camera))
}

#[derive(Component)]
pub struct DraggedMarker;

pub fn drag_nav_gizmo(
    drag: On<Pointer<Drag>>,
    nav_gizmo: Query<&NavGizmoParent>,
    mut query: Query<(&Transform, &mut EditorCam, &Camera), With<MainCamera>>,
    dragged_query: Query<(), With<DraggedMarker>>,
    mut commands: Commands,
) {
    let drag_target = drag.event().event_target();

    let Ok(nav_gizmo) = nav_gizmo.get(drag_target) else {
        return;
    };
    let Ok((transform, mut editor_cam, cam)) = query.get_mut(nav_gizmo.main_camera) else {
        return;
    };
    let first_drag = dragged_query.get(drag_target).is_err();
    if first_drag {
        commands.entity(drag_target).insert(DraggedMarker);
        editor_cam.end_move();
        let anchor = camera_anchor_from_transform(transform);
        editor_cam.start_orbit(anchor);
    }
    let delta = drag.delta
        * cam
            .physical_viewport_size()
            .unwrap_or_else(|| UVec2::new(256, 256))
            .as_vec2()
        / 75.0;
    editor_cam.send_screenspace_input(delta);
}

pub fn drag_nav_gizmo_end(
    drag_end: On<Pointer<DragEnd>>,
    nav_gizmo: Query<&NavGizmoParent>,
    mut query: Query<&mut EditorCam, With<MainCamera>>,
    mut commands: Commands,
) {
    let drag_end_target = drag_end.event().event_target();

    let Ok(nav_gizmo) = nav_gizmo.get(drag_end_target) else {
        return;
    };
    if let Ok(mut editor_cam) = query.get_mut(nav_gizmo.main_camera) {
        editor_cam.end_move();
    }
    commands.entity(drag_end_target).remove::<DraggedMarker>();
}

#[allow(clippy::type_complexity)]
fn side_clicked_cb(
    direction: Dir3,
) -> impl Fn(
    On<Pointer<Click>>,
    Query<(Entity, &Transform, &EditorCam), With<MainCamera>>,
    Query<&NavGizmoParent>,
    Query<&DraggedMarker>,
    MessageWriter<LookToTrigger>,
) {
    move |click: On<Pointer<Click>>,
          query: Query<(Entity, &Transform, &EditorCam), With<MainCamera>>,
          nav_gizmo: Query<&NavGizmoParent>,
          drag_query: Query<&DraggedMarker>,
          mut look_to: MessageWriter<LookToTrigger>| {
        let target = click.event().event_target();

        let Ok(nav_gizmo) = nav_gizmo.get(target) else {
            return;
        };
        let Ok((entity, transform, editor_cam)) = query.get(nav_gizmo.main_camera) else {
            return;
        };

        if drag_query.get(target).is_ok() {
            return;
        }
        if click.button == PointerButton::Primary {
            look_to.write(LookToTrigger::auto_snap_up_direction(
                direction, entity, transform, editor_cam,
            ));
        }
    }
}

pub fn sync_nav_camera(
    main_transform_query: Query<&GlobalTransform, (With<MainCamera>, Without<NavGizmoParent>)>,
    mut nav_transform_query: Query<(Entity, &NavGizmoParent, &mut Transform), With<NavGizmo>>,
    mut commands: Commands,
) {
    for (entity, nav_gizmo, mut nav_transform) in nav_transform_query.iter_mut() {
        let Ok(main) = main_transform_query.get(nav_gizmo.main_camera) else {
            commands.entity(entity).despawn();
            continue;
        };
        // What does this do, actually? Is this a way to have a child that does
        // not rotate with its parent? I believe there are better ways to
        // achieve this with Bevy's generic parent-child relationships.
        nav_transform.rotation = main.rotation().conjugate();
    }
}

fn clamp_overlay_position(position: Vec2, side_length: f32, window_size: Vec2) -> Vec2 {
    let max_x = (window_size.x - side_length).max(0.0);
    let max_y = (window_size.y - side_length).max(0.0);
    Vec2::new(position.x.clamp(0.0, max_x), position.y.clamp(0.0, max_y))
}

#[allow(clippy::too_many_arguments)]
fn set_camera_viewport(
    windows: Query<(Entity, &Window, &bevy_egui::EguiContextSettings)>,
    _contexts: EguiContexts,
    mut nav_camera_query: Query<(&mut Camera, &NavGizmoParent)>,
    main_camera_query: Query<(&Camera, Option<&ViewportRect>), Without<NavGizmoParent>>,
    mut main_editor_cam_query: Query<&mut EditorCam, (With<MainCamera>, Without<NavGizmoParent>)>,
    primary_query: Query<Entity, With<PrimaryWindow>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut anchor_state: ResMut<NavGizmoAnchorState>,
) {
    let margin = 8.0;
    let top_offset = 10.0;
    let preferred_side_length = 128.0;
    let max_viewport_fraction = 0.45;
    let min_side_length = 64.0;
    let min_viewport_for_gizmo = 100.0;
    let drag_start_threshold = 4.0;

    if !mouse_buttons.pressed(MouseButton::Left) {
        anchor_state.active_drag = None;
    }

    for (mut nav_camera, parent) in nav_camera_query.iter_mut() {
        let Ok((main, viewport_rect)) = main_camera_query.get(parent.main_camera) else {
            continue;
        };
        let target_window = match &nav_camera.target {
            RenderTarget::Window(WindowRef::Primary) => primary_query.iter().next(),
            RenderTarget::Window(WindowRef::Entity(entity)) => Some(*entity),
            _ => None,
        };
        let Some(window_entity) = target_window else {
            continue;
        };
        let Ok((_, window, egui_settings)) = windows.get(window_entity) else {
            continue;
        };
        let scale_factor = window.scale_factor() * egui_settings.scale_factor;
        let margin = margin * scale_factor;
        let top_offset = top_offset * scale_factor;
        let (viewport_pos, viewport_size) = if let Some(rect) = viewport_rect.and_then(|r| r.0) {
            let pos = rect.left_top().to_vec2() * scale_factor;
            let size = rect.size() * scale_factor;
            (Vec2::new(pos.x, pos.y), Vec2::new(size.x, size.y))
        } else {
            let Some(viewport) = &main.viewport else {
                continue;
            };
            (
                viewport.physical_position.as_vec2(),
                viewport.physical_size.as_vec2(),
            )
        };

        let min_viewport_dim = viewport_size.x.min(viewport_size.y);
        if min_viewport_dim < min_viewport_for_gizmo * scale_factor {
            nav_camera.is_active = false;
            nav_camera.viewport = Some(Viewport {
                physical_position: UVec2::ZERO,
                physical_size: UVec2::new(1, 1),
                depth: 0.0..1.0,
            });
            continue;
        }

        let side_length = (preferred_side_length * scale_factor)
            .min(min_viewport_dim * max_viewport_fraction)
            .max(min_side_length * scale_factor);
        let right_offset = 20.0 * scale_factor; // Slight left offset to avoid overlap with right panel
        let default_nav_viewport_pos = Vec2::new(
            (viewport_pos.x + viewport_size.x) - (side_length + margin + right_offset),
            viewport_pos.y + top_offset,
        );

        let window_size = window.physical_size().as_vec2();
        let current_offset = anchor_state
            .offsets
            .get(&parent.main_camera)
            .copied()
            .unwrap_or(Vec2::ZERO);
        let mut nav_viewport_pos = clamp_overlay_position(
            default_nav_viewport_pos + current_offset,
            side_length,
            window_size,
        );

        if mouse_buttons.just_pressed(MouseButton::Left)
            && anchor_state.active_drag.is_none()
            && let Some(cursor_pos) = window.physical_cursor_position()
        {
            let drag_rect = Rect::from_corners(
                nav_viewport_pos,
                nav_viewport_pos + Vec2::splat(side_length),
            );
            if drag_rect.contains(cursor_pos) {
                anchor_state.active_drag = Some(NavGizmoDrag {
                    main_camera: parent.main_camera,
                    pointer_press: cursor_pos,
                    grab_offset: cursor_pos - nav_viewport_pos,
                    dragging: false,
                });
            }
        }

        if let Some(active_drag) = anchor_state.active_drag.as_mut()
            && active_drag.main_camera == parent.main_camera
            && let Some(cursor_pos) = window.physical_cursor_position()
        {
            if !active_drag.dragging
                && cursor_pos.distance(active_drag.pointer_press) >= drag_start_threshold
            {
                active_drag.dragging = true;
            }

            if active_drag.dragging {
                let dragged_pos = clamp_overlay_position(
                    cursor_pos - active_drag.grab_offset,
                    side_length,
                    window_size,
                );
                anchor_state
                    .offsets
                    .insert(parent.main_camera, dragged_pos - default_nav_viewport_pos);
                nav_viewport_pos = dragged_pos;

                if let Ok(mut editor_cam) = main_editor_cam_query.get_mut(parent.main_camera) {
                    // Prevent the main viewport from orbiting while we drag the overlay.
                    editor_cam.enabled_motion = EnabledMotion {
                        pan: false,
                        orbit: false,
                        zoom: false,
                    };
                    editor_cam.end_move();
                }
            }
        }

        // Clamp the gizmo viewport to the actual window surface to avoid invalid wgpu viewports
        // when the target window is smaller than the desired overlay.
        let window_size = window.physical_size();
        let pos_x = nav_viewport_pos.x.max(0.0) as u32;
        let pos_y = nav_viewport_pos.y.max(0.0) as u32;
        let max_w = window_size.x.saturating_sub(pos_x);
        let max_h = window_size.y.saturating_sub(pos_y);
        let (physical_size, is_active) = if main.is_active && max_w > 0 && max_h > 0 {
            (
                UVec2::new(
                    side_length.min(max_w as f32) as u32,
                    side_length.min(max_h as f32) as u32,
                ),
                true,
            )
        } else {
            (UVec2::new(1, 1), false)
        };
        nav_camera.is_active = is_active;

        let new_viewport = Viewport {
            physical_position: UVec2::new(pos_x, pos_y),
            physical_size,
            depth: 0.0..1.0,
        };
        let unchanged = nav_camera
            .viewport
            .as_ref()
            .map(|vp| {
                vp.physical_position == new_viewport.physical_position
                    && vp.physical_size == new_viewport.physical_size
            })
            .unwrap_or(false);
        if !unchanged {
            nav_camera.viewport = Some(new_viewport);
        }
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
