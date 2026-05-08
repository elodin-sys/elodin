use std::collections::HashMap;

use bevy::{
    app::{App, Plugin, PreUpdate},
    camera::Camera,
    ecs::schedule::IntoScheduleConfigs,
    input::{
        ButtonInput,
        mouse::{MouseButton, MouseScrollUnit, MouseWheel},
    },
    prelude::{Entity, MessageReader, MessageWriter, Query, Res, ResMut, Vec2, With},
    window::PrimaryWindow,
};
use bevy_editor_cam::{
    controller::component::EditorCam,
    input::{CameraPointerMap, EditorCamInputMessage, MotionKind},
};
use bevy_picking::{
    PickingSystems,
    pointer::{PointerAction, PointerId, PointerInput, PointerLocation},
};

use crate::ui::{input_owner::UiInputOwners, window::window_entity_from_target};

pub struct EditorCamInputPlugin;

impl Plugin for EditorCamInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<EditorCamInputMessage>()
            .init_resource::<CameraPointerMap>()
            .add_systems(
                PreUpdate,
                (
                    gated_camera_inputs,
                    EditorCamInputMessage::receive_messages,
                    send_gated_pointer_inputs,
                )
                    .chain()
                    .after(PickingSystems::Last)
                    .before(EditorCam::update_camera_positions),
            );
    }
}

pub fn gated_camera_inputs(
    pointers: Query<(&PointerId, &PointerLocation)>,
    pointer_map: Res<CameraPointerMap>,
    mut controller: MessageWriter<EditorCamInputMessage>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    cameras: Query<(Entity, &Camera, &EditorCam)>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    input_owners: Res<UiInputOwners>,
) {
    let orbit_start = MouseButton::Right;
    let pan_start = MouseButton::Left;
    let zoom_stop = 0.0;
    let wheel_windows = mouse_wheel_windows(&mut mouse_wheel);

    if let Some(&camera) = pointer_map.get(&PointerId::Mouse) {
        let camera_query = cameras.get(camera).ok();
        let owner_allows_camera = camera_query
            .map(|(entity, camera, _)| {
                input_owners_permit_camera(&input_owners, &primary_window, camera, entity)
            })
            .unwrap_or_default();
        let is_in_zoom_mode = camera_query
            .map(|(.., editor_cam)| editor_cam.current_motion.is_zooming_only())
            .unwrap_or_default();
        let zoom_amount_abs = camera_query
            .and_then(|(.., editor_cam)| {
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

    for (&pointer, pointer_location) in pointers
        .iter()
        .filter_map(|(id, loc)| loc.location().map(|loc| (id, loc)))
    {
        match pointer {
            PointerId::Mouse => {
                let Some((camera, camera_component, _)) =
                    cameras.iter().find(|(entity, camera, _)| {
                        pointer_location.is_in_viewport(camera, &primary_window)
                            && input_owners_permit_camera(
                                &input_owners,
                                &primary_window,
                                camera,
                                *entity,
                            )
                    })
                else {
                    continue;
                };

                let Some(window_entity) = camera_window_entity(camera_component, &primary_window)
                else {
                    continue;
                };

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
                } else if wheel_windows.contains_key(&window_entity) {
                    controller.write(EditorCamInputMessage::Start {
                        kind: MotionKind::Zoom,
                        camera,
                        pointer,
                    });
                }
            }
            PointerId::Touch(_) => continue,
            PointerId::Custom(_) => continue,
        }
    }
}

pub fn send_gated_pointer_inputs(
    mut camera_map: ResMut<CameraPointerMap>,
    mut camera_controllers: Query<(Entity, &mut EditorCam, &Camera)>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
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

    for (pointer, camera) in active_pointers {
        let Ok((entity, mut camera_controller, camera_component)) =
            camera_controllers.get_mut(camera)
        else {
            ended_pointers.push(pointer);
            continue;
        };
        let Some(window_entity) = camera_window_entity(camera_component, &primary_window) else {
            camera_controller.end_move();
            ended_pointers.push(pointer);
            continue;
        };
        if !input_owners.permits_viewport(window_entity, entity) {
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

fn input_owners_permit_camera(
    input_owners: &UiInputOwners,
    primary_window: &Query<Entity, With<PrimaryWindow>>,
    camera: &Camera,
    camera_entity: Entity,
) -> bool {
    camera_window_entity(camera, primary_window)
        .map(|window| input_owners.permits_viewport(window, camera_entity))
        .unwrap_or_default()
}

fn camera_window_entity(
    camera: &Camera,
    primary_window: &Query<Entity, With<PrimaryWindow>>,
) -> Option<Entity> {
    let primary_entity = primary_window.single().ok()?;
    window_entity_from_target(&camera.target, primary_entity)
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
