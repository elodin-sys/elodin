use std::collections::HashMap;

use bevy::{
    app::{App, Plugin},
    camera::{Camera, RenderTarget},
    input::{
        ButtonInput,
        mouse::{MouseButton, MouseScrollUnit, MouseWheel},
    },
    prelude::{Entity, MessageReader, MessageWriter, Query, Res, ResMut, Vec2, With},
    window::{PrimaryWindow, Window},
};
use bevy_editor_cam::{
    controller::component::EditorCam,
    input::{CameraPointerMap, EditorCamInputMessage, MotionKind},
};
use bevy_picking::pointer::{PointerAction, PointerId, PointerInput};

use crate::ui::input_owner::UiInputOwners;

/// Registers message/resources only. Input systems run in [`crate::ui`] after
/// [`crate::ui::set_camera_viewport`] so hit tests use the current viewport rect.
pub struct EditorCamInputPlugin;

impl Plugin for EditorCamInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<EditorCamInputMessage>()
            .init_resource::<CameraPointerMap>();
    }
}

fn cursor_in_camera_view(camera: &Camera, cursor: Vec2) -> bool {
    camera.is_active
        && camera
            .logical_viewport_rect()
            .is_some_and(|rect| rect.contains(cursor))
}

#[allow(clippy::too_many_arguments)]
pub fn gated_camera_inputs(
    pointer_map: Res<CameraPointerMap>,
    mut controller: MessageWriter<EditorCamInputMessage>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    cameras: Query<(Entity, &Camera, &RenderTarget, &EditorCam)>,
    primary_window: Query<(Entity, &Window), With<PrimaryWindow>>,
    input_owners: Res<UiInputOwners>,
) {
    let orbit_start = MouseButton::Right;
    let pan_start = MouseButton::Left;
    let zoom_stop = 0.0;
    let wheel_windows = mouse_wheel_windows(&mut mouse_wheel);

    let Ok((window_entity, window)) = primary_window.single() else {
        return;
    };
    let cursor = window.cursor_position();
    let pointer_pos = cursor.map(|pos| bevy_egui::egui::pos2(pos.x, pos.y));

    if let Some(&camera) = pointer_map.get(&PointerId::Mouse) {
        let owner_allows_camera = pointer_pos
            .map(|pos| input_owners.permits_viewport_at(window_entity, camera, pos))
            .unwrap_or_default();
        let camera_query = cameras.get(camera).ok();
        let is_in_zoom_mode = camera_query
            .map(|(_, _, _, editor_cam)| editor_cam.current_motion.is_zooming_only())
            .unwrap_or_default();
        let zoom_amount_abs = camera_query
            .and_then(|(_, _, _, editor_cam)| {
                editor_cam
                    .current_motion
                    .inputs()
                    .map(|inputs| inputs.zoom_velocity_abs(editor_cam.smoothing.zoom.mul_f32(2.0)))
            })
            .unwrap_or(0.0);
        let should_zoom_end = is_in_zoom_mode && zoom_amount_abs <= zoom_stop;

        if !owner_allows_camera
            || mouse_input.any_just_released([orbit_start, pan_start])
            || should_zoom_end
        {
            controller.write(EditorCamInputMessage::End { camera });
        }
    }

    let Some(cursor) = cursor else {
        return;
    };
    let Some(pointer_pos) = pointer_pos else {
        return;
    };

    let Some((camera, _camera_component, render_target, _)) =
        cameras.iter().find(|(entity, camera, _, _)| {
            cursor_in_camera_view(camera, cursor)
                && input_owners.permits_viewport_at(window_entity, *entity, pointer_pos)
        })
    else {
        return;
    };

    let Some(render_window_entity) = camera_window_entity(render_target, &primary_window) else {
        return;
    };

    let pointer = PointerId::Mouse;

    if mouse_input.just_pressed(orbit_start) {
        controller.write(EditorCamInputMessage::Start {
            kind: MotionKind::OrbitZoom,
            camera,
            pointer,
        });
    } else if mouse_input.just_pressed(pan_start) {
        controller.write(EditorCamInputMessage::Start {
            kind: MotionKind::PanZoom,
            camera,
            pointer,
        });
    } else if wheel_windows.contains_key(&render_window_entity) {
        controller.write(EditorCamInputMessage::Start {
            kind: MotionKind::Zoom,
            camera,
            pointer,
        });
    }
}

pub fn send_gated_pointer_inputs(
    mut camera_map: ResMut<CameraPointerMap>,
    mut camera_controllers: Query<(Entity, &mut EditorCam, &Camera, &RenderTarget)>,
    primary_window: Query<(Entity, &Window), With<PrimaryWindow>>,
    input_owners: Res<UiInputOwners>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mut moves: MessageReader<PointerInput>,
) {
    let moves_list: Vec<_> = moves.read().collect();
    let wheel_by_window = mouse_wheel_offsets_by_window(&mut mouse_wheel);
    let active_pointers: Vec<_> = camera_map
        .iter()
        .map(|(&pointer, &camera)| (pointer, camera))
        .collect();
    let mut ended_pointers = Vec::new();

    let Ok((_, window)) = primary_window.single() else {
        return;
    };
    let cursor = window.cursor_position();
    let pointer_pos = cursor.map(|pos| bevy_egui::egui::pos2(pos.x, pos.y));

    for (pointer, camera) in active_pointers {
        let Ok((entity, mut camera_controller, camera_component, render_target)) =
            camera_controllers.get_mut(camera)
        else {
            ended_pointers.push(pointer);
            continue;
        };
        let Some(window_entity) = camera_window_entity(render_target, &primary_window) else {
            camera_controller.end_move();
            ended_pointers.push(pointer);
            continue;
        };
        let viewport_ok = cursor.is_some_and(|cursor| {
            cursor_in_camera_view(camera_component, cursor)
                && pointer_pos
                    .is_some_and(|pos| input_owners.permits_viewport_at(window_entity, entity, pos))
        });
        if !viewport_ok {
            camera_controller.end_move();
            ended_pointers.push(pointer);
            continue;
        }

        let screenspace_input: Vec2 = moves_list
            .iter()
            .filter(|m| m.pointer_id.eq(&pointer))
            .filter_map(|m| match m.action {
                PointerAction::Move { delta } => Some(delta),
                PointerAction::Press { .. } => None,
                PointerAction::Cancel => None,
                _ => None,
            })
            .sum();

        let zoom_amount = match pointer {
            PointerId::Mouse => wheel_by_window
                .get(&window_entity)
                .copied()
                .unwrap_or_default(),
            _ => 0.0,
        };

        camera_controller.send_screenspace_input(screenspace_input);
        camera_controller.send_zoom_input(zoom_amount);
    }

    for pointer in ended_pointers {
        camera_map.remove(&pointer);
    }
}

fn camera_window_entity(
    render_target: &RenderTarget,
    primary_window: &Query<(Entity, &Window), With<PrimaryWindow>>,
) -> Option<Entity> {
    let (primary_entity, _) = primary_window.single().ok()?;
    crate::ui::window::window_entity_from_target(render_target, primary_entity)
}

fn mouse_wheel_windows(mouse_wheel: &mut MessageReader<MouseWheel>) -> HashMap<Entity, ()> {
    let mut windows = HashMap::new();
    for ev in mouse_wheel.read() {
        if ev.y.abs() > 0.0 {
            windows.insert(ev.window, ());
        }
    }
    windows
}

fn mouse_wheel_offsets_by_window(
    mouse_wheel: &mut MessageReader<MouseWheel>,
) -> HashMap<Entity, f32> {
    let mut offsets = HashMap::new();
    for ev in mouse_wheel.read() {
        let scroll_multiplier = match ev.unit {
            MouseScrollUnit::Line => 150.0,
            MouseScrollUnit::Pixel => 1.0,
        };
        offsets
            .entry(ev.window)
            .and_modify(|offset| *offset += ev.y * scroll_multiplier)
            .or_insert(ev.y * scroll_multiplier);
    }
    offsets
}
