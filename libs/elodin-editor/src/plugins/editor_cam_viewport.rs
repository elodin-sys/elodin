//! Viewport-local mouse input for [`EditorCam`] on [`MainCamera`].
//!
//! Bevy's default editor-cam input runs in `PreUpdate`, before the UI assigns camera
//! viewports for the current frame. That stale viewport breaks orbit/pan when the
//! pointer is over the 3D pane. These systems run after [`crate::ui::set_camera_viewport`].

use std::collections::HashMap;

use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_editor_cam::{
    controller::component::EditorCam, input::CameraPointerMap, prelude::EnabledMotion,
};

use crate::{
    MainCamera,
    plugins::{camera_anchor::camera_anchor_from_transform, navigation_gizmo::NavGizmoAnchorState},
    ui::tiles::ViewportContainsPointer,
};

#[derive(Resource, Default)]
pub struct ActiveViewportCamDrag {
    camera: Option<Entity>,
    orbit: bool,
}

#[derive(Resource, Default)]
pub struct DefaultEditorCamMotionOverride {
    previous: HashMap<Entity, EnabledMotion>,
}

fn cursor_in_camera_view(camera: &Camera, cursor: Vec2) -> bool {
    camera.is_active
        && camera
            .logical_viewport_rect()
            .is_some_and(|rect| rect.contains(cursor))
}

/// Prevent [`bevy_editor_cam::input::default_camera_inputs`] from starting motions with a stale viewport.
pub fn disable_default_editor_cam_motion(
    mut override_state: ResMut<DefaultEditorCamMotionOverride>,
    mut editor_cams: Query<(Entity, &mut EditorCam), With<MainCamera>>,
) {
    override_state.previous.clear();
    for (entity, mut editor_cam) in &mut editor_cams {
        override_state
            .previous
            .insert(entity, editor_cam.enabled_motion.clone());
        editor_cam.enabled_motion = EnabledMotion {
            pan: false,
            orbit: false,
            zoom: false,
        };
    }
}

/// Restore normal `EditorCam` motion flags after the default input system has had a chance to no-op.
pub fn restore_default_editor_cam_motion(
    mut override_state: ResMut<DefaultEditorCamMotionOverride>,
    mut editor_cams: Query<&mut EditorCam, With<MainCamera>>,
    mut camera_pointer_map: ResMut<CameraPointerMap>,
) {
    camera_pointer_map.clear();
    for (entity, previous) in override_state.previous.drain() {
        if let Ok(mut editor_cam) = editor_cams.get_mut(entity) {
            editor_cam.enabled_motion = previous;
        }
    }
}

pub fn update_viewport_contains_pointer(
    mut viewport_contains: ResMut<ViewportContainsPointer>,
    window: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<&Camera, With<MainCamera>>,
) {
    let Some(cursor) = window.single().ok().and_then(|w| w.cursor_position()) else {
        viewport_contains.0 = false;
        return;
    };
    viewport_contains.0 = cameras
        .iter()
        .any(|camera| cursor_in_camera_view(camera, cursor));
}

pub fn viewport_editor_cam_mouse_input(
    mouse: Res<ButtonInput<MouseButton>>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mut mouse_motion: MessageReader<MouseMotion>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<(Entity, &Camera, &mut EditorCam, &Transform), With<MainCamera>>,
    mut drag: ResMut<ActiveViewportCamDrag>,
    nav: Res<NavGizmoAnchorState>,
) {
    if nav.is_suppressing_camera_motion() {
        if let Some(entity) = drag.camera.take()
            && let Ok((_, _, mut editor_cam, _)) = cameras.get_mut(entity)
        {
            editor_cam.end_move();
        }
        return;
    }

    let Ok(window) = window.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };

    let motion: Vec2 = mouse_motion.read().map(|event| event.delta).sum();
    let zoom_amount: f32 = mouse_wheel
        .read()
        .map(|wheel| {
            let multiplier = match wheel.unit {
                MouseScrollUnit::Line => 150.0,
                MouseScrollUnit::Pixel => 1.0,
            };
            wheel.y * multiplier
        })
        .sum();

    let target = cameras
        .iter()
        .find_map(|(entity, camera, _, _)| cursor_in_camera_view(camera, cursor).then_some(entity));

    if mouse.just_pressed(MouseButton::Right) {
        if let Some(entity) = target
            && let Ok((_, _, mut editor_cam, transform)) = cameras.get_mut(entity)
        {
            editor_cam.end_move();
            editor_cam.start_orbit(camera_anchor_from_transform(transform));
            drag.camera = Some(entity);
            drag.orbit = true;
        }
    } else if mouse.just_pressed(MouseButton::Left)
        && let Some(entity) = target
        && let Ok((_, _, mut editor_cam, transform)) = cameras.get_mut(entity)
    {
        editor_cam.end_move();
        editor_cam.start_pan(camera_anchor_from_transform(transform));
        drag.camera = Some(entity);
        drag.orbit = false;
    }

    if zoom_amount.abs() > f32::EPSILON {
        if let Some(entity) = target
            && let Ok((_, _, mut editor_cam, transform)) = cameras.get_mut(entity)
        {
            if !editor_cam.is_actively_controlled() {
                editor_cam.start_zoom(camera_anchor_from_transform(transform));
            }
            editor_cam.send_zoom_input(zoom_amount);
        }
        mouse_wheel.clear();
    }

    let Some(entity) = drag.camera else {
        return;
    };
    let button = if drag.orbit {
        MouseButton::Right
    } else {
        MouseButton::Left
    };
    if mouse.pressed(button) {
        if let Ok((_, _, mut editor_cam, _)) = cameras.get_mut(entity) {
            editor_cam.send_screenspace_input(motion);
        }
    } else if let Ok((_, _, mut editor_cam, _)) = cameras.get_mut(entity) {
        editor_cam.end_move();
        drag.camera = None;
    }
}
