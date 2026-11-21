use std::collections::HashMap;

use bevy::{
    app::{App, Plugin},
    camera::{Camera, Projection, RenderTarget},
    input::{
        ButtonInput,
        mouse::{MouseButton, MouseScrollUnit, MouseWheel},
    },
    prelude::{Entity, MessageReader, MessageWriter, Query, Res, ResMut, Vec2, With},
    window::{PrimaryWindow, Window},
};
use bevy_editor_cam::{
    controller::{component::EditorCam, motion::CurrentMotion},
    input::{CameraPointerMap, EditorCamInputMessage, MotionKind},
};
use bevy_picking::pointer::{PointerAction, PointerButton, PointerId, PointerInput};

use crate::ui::{input_owner::UiInputOwners, inspector::viewport::Viewport};

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

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gated_camera_inputs(
    mut pointer_map: ResMut<CameraPointerMap>,
    mut controller: MessageWriter<EditorCamInputMessage>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut cameras: Query<(
        Entity,
        &Camera,
        &RenderTarget,
        &Projection,
        &mut EditorCam,
        Option<&Viewport>,
    )>,
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
            .map(|(_, _, _, _, editor_cam, _)| editor_cam.current_motion.is_zooming_only())
            .unwrap_or_default();
        let zoom_amount_abs = camera_query
            .and_then(|(_, _, _, _, editor_cam, _)| {
                editor_cam
                    .current_motion
                    .inputs()
                    .map(|inputs| inputs.zoom_velocity_abs(editor_cam.smoothing.zoom.mul_f32(2.0)))
            })
            .unwrap_or(0.0);
        // New wheel input may arrive before the zoom queue is populated.
        // Keep that gesture alive instead of ending it and replacing its anchor.
        let should_zoom_end = is_in_zoom_mode
            && zoom_amount_abs <= zoom_stop
            && !wheel_windows.contains_key(&window_entity);

        if !owner_allows_camera
            || mouse_input.any_just_released([orbit_start, pan_start])
            || should_zoom_end
        {
            // End before starting another gesture in this frame. Follow-view
            // pans start directly below instead of using the picked surface.
            if let Ok((_, _, _, _, mut editor_cam, _)) = cameras.get_mut(camera) {
                editor_cam.end_move();
            }
            pointer_map.remove(&PointerId::Mouse);
        }
    }

    let Some(cursor) = cursor else {
        return;
    };
    let pointer_pos = bevy_egui::egui::pos2(cursor.x, cursor.y);

    let Some((camera, camera_component, render_target, projection, mut editor_cam, viewport)) =
        cameras.iter_mut().find(|(entity, camera, _, _, _, _)| {
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
    // Press and release can both arrive in one frame. Do not start a drag
    // that has already ended: there would be no later release to stop it.
    let start_orbit = mouse_input.just_pressed(orbit_start) && mouse_input.pressed(orbit_start);
    let start_pan = mouse_input.just_pressed(pan_start) && mouse_input.pressed(pan_start);

    // A new mouse press must take control immediately, even during the
    // previous drag's momentum debounce. Keep its depth, discard its velocity.
    if (start_orbit || start_pan)
        && matches!(editor_cam.current_motion, CurrentMotion::Momentum { .. })
    {
        editor_cam.current_motion = CurrentMotion::Stationary;
    }

    if start_orbit {
        controller.write(EditorCamInputMessage::Start {
            kind: MotionKind::OrbitZoom,
            camera,
            pointer,
        });
    } else if start_pan {
        if viewport.is_some_and(|viewport| !viewport.look_at.eql.trim().is_empty()) {
            if !editor_cam.is_actively_controlled() {
                // A second drag can hit terrain kilometres behind the followed
                // subject. Keep the existing depth so pan sensitivity does not
                // jump, while retaining the cursor's position on that plane.
                let anchor = pan_anchor(
                    camera_component,
                    projection,
                    cursor,
                    editor_cam.last_anchor_depth,
                );
                editor_cam.start_pan(anchor);
                pointer_map.insert(pointer, camera);
            }
        } else {
            controller.write(EditorCamInputMessage::Start {
                kind: MotionKind::PanZoom,
                camera,
                pointer,
            });
        }
    } else if wheel_windows.contains_key(&render_window_entity)
        && !editor_cam.current_motion.is_zooming_only()
    {
        controller.write(EditorCamInputMessage::Start {
            kind: MotionKind::Zoom,
            camera,
            pointer,
        });
    }
}

fn pan_anchor(
    camera: &Camera,
    projection: &Projection,
    cursor: Vec2,
    depth: f64,
) -> Option<bevy::math::DVec3> {
    let rect = camera.logical_viewport_rect()?;
    let ndc = ((cursor - rect.min) / rect.size() * 2.0 - Vec2::ONE) * Vec2::new(1.0, -1.0);
    let near = projection
        .get_clip_from_view()
        .as_dmat4()
        .inverse()
        .project_point3(ndc.extend(1.0).as_dvec3());
    let anchor = match projection {
        Projection::Orthographic(_) => bevy::math::DVec3::new(near.x, near.y, depth),
        _ => near * (depth / near.z),
    };
    anchor.is_finite().then_some(anchor)
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
            .fold(Vec2::ZERO, |input, m| match m.action {
                // A frame can contain motion from before the drag started.
                // Only forward movement following the camera button press.
                PointerAction::Press(PointerButton::Primary | PointerButton::Secondary) => {
                    Vec2::ZERO
                }
                PointerAction::Move { delta } => input + delta,
                _ => input,
            });

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        object_3d::EditableEQL,
        ui::{input_owner::PointerOwner, inspector::viewport::Viewport},
    };
    use bevy::{
        camera::{ComputedCameraValues, Projection, RenderTargetInfo},
        prelude::*,
    };
    use bevy_picking::{
        backend::HitData,
        hover::{HoverMap, PreviousHoverMap, update_interactions},
        pointer::{Location, PointerButton, PointerMap, update_pointer_map},
    };

    fn camera_input_app(follow: bool) -> (App, Entity, Entity, Entity) {
        let mut app = App::new();
        app.add_plugins(EditorCamInputPlugin)
            .add_message::<MouseWheel>()
            .add_message::<PointerInput>()
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<UiInputOwners>()
            .init_resource::<PointerMap>()
            .init_resource::<HoverMap>()
            .init_resource::<PreviousHoverMap>()
            .add_systems(
                Update,
                (
                    update_pointer_map,
                    update_interactions,
                    gated_camera_inputs,
                    EditorCamInputMessage::receive_messages,
                    send_gated_pointer_inputs,
                )
                    .chain(),
            );
        let mut window = Window::default();
        window.resolution.set(1280., 720.);
        window.set_cursor_position(Some(Vec2::new(640., 360.)));
        let window = app.world_mut().spawn((window, PrimaryWindow)).id();
        app.world_mut().spawn(PointerId::Mouse);
        let subject = app.world_mut().spawn_empty().id();
        let camera = app
            .world_mut()
            .spawn((
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: UVec2::new(1280, 720),
                            scale_factor: 1.,
                        }),
                        ..default()
                    },
                    ..default()
                },
                Projection::default(),
                GlobalTransform::IDENTITY,
                EditorCam {
                    last_anchor_depth: -200.,
                    input_debounce: std::time::Duration::ZERO,
                    ..default()
                },
                Viewport::new(
                    subject,
                    EditableEQL::default(),
                    EditableEQL {
                        eql: if follow {
                            "booster.world_pos".into()
                        } else {
                            String::new()
                        },
                        compiled_expr: None,
                    },
                    EditableEQL::default(),
                    None,
                    0.,
                ),
            ))
            .id();
        app.world_mut()
            .resource_mut::<UiInputOwners>()
            .register_content_rect(
                window,
                bevy_egui::egui::Rect::from_min_max(
                    bevy_egui::egui::pos2(0., 0.),
                    bevy_egui::egui::pos2(1280., 720.),
                ),
                PointerOwner::Viewport { camera },
            );
        (app, camera, subject, window)
    }

    fn gesture_depths(follow: bool, button: MouseButton) -> Vec<f64> {
        let (mut app, camera, subject, _) = camera_input_app(follow);
        let mut depths = Vec::new();
        for depth in [200., 10_000.] {
            app.world_mut().resource_mut::<HoverMap>().insert(
                PointerId::Mouse,
                [(
                    subject,
                    HitData::new(camera, depth, Some(Vec3::new(0., 0., -depth)), None),
                )]
                .into_iter()
                .collect(),
            );
            app.world_mut()
                .resource_mut::<ButtonInput<MouseButton>>()
                .press(button);
            app.update();
            depths.push(
                app.world()
                    .get::<EditorCam>(camera)
                    .unwrap()
                    .anchor_view_space()
                    .unwrap()
                    .z,
            );
            {
                let mut buttons = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
                buttons.clear();
                buttons.release(button);
            }
            app.update();
            assert!(app.world().resource::<CameraPointerMap>().is_empty());
            app.world_mut()
                .resource_mut::<ButtonInput<MouseButton>>()
                .clear();
        }
        depths
    }

    #[test]
    fn second_follow_pan_keeps_subject_depth_when_pointer_hits_distant_terrain() {
        assert_eq!(gesture_depths(true, MouseButton::Left), [-200., -200.]);
    }

    #[test]
    fn free_pan_and_orbit_still_use_the_picked_surface() {
        assert_eq!(gesture_depths(false, MouseButton::Left), [-200., -10_000.]);
        assert_eq!(gesture_depths(true, MouseButton::Right), [-200., -10_000.]);
    }

    #[test]
    fn camera_drag_excludes_pointer_motion_before_each_press() {
        use bevy_editor_cam::controller::inputs::MotionInputs;

        fn latest_drag(app: &App, camera: Entity) -> Vec2 {
            let controller = app.world().get::<EditorCam>(camera).unwrap();
            let queue = match controller.current_motion.inputs().unwrap() {
                MotionInputs::PanZoom {
                    screenspace_inputs, ..
                }
                | MotionInputs::OrbitZoom {
                    screenspace_inputs, ..
                } => screenspace_inputs,
                _ => panic!("expected a drag"),
            };
            queue.iter_unsmoothed().next().unwrap().1
        }

        for (button, pointer_button) in [
            (MouseButton::Left, PointerButton::Primary),
            (MouseButton::Right, PointerButton::Secondary),
        ] {
            let (mut app, camera, _, window) = camera_input_app(true);
            let location = Location {
                target: RenderTarget::Window(bevy::window::WindowRef::Entity(window))
                    .normalize(Some(window))
                    .unwrap(),
                position: Vec2::new(640., 360.),
            };
            for _ in 0..2 {
                app.world_mut()
                    .resource_mut::<ButtonInput<MouseButton>>()
                    .press(button);
                for action in [
                    PointerAction::Move {
                        delta: Vec2::new(-800., 300.),
                    },
                    PointerAction::Press(pointer_button),
                    PointerAction::Move {
                        delta: Vec2::new(12., -4.),
                    },
                ] {
                    app.world_mut().write_message(PointerInput::new(
                        PointerId::Mouse,
                        location.clone(),
                        action,
                    ));
                }
                app.update();
                assert_eq!(latest_drag(&app, camera), Vec2::new(12., -4.));

                // Subsequent frames have no press: retain all held-button motion.
                app.world_mut()
                    .resource_mut::<ButtonInput<MouseButton>>()
                    .clear();
                for delta in [Vec2::new(-5., 2.), Vec2::new(-3., 1.)] {
                    app.world_mut().write_message(PointerInput::new(
                        PointerId::Mouse,
                        location.clone(),
                        PointerAction::Move { delta },
                    ));
                }
                app.update();
                assert_eq!(latest_drag(&app, camera), Vec2::new(-8., 3.));
                let mut buttons = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
                buttons.clear();
                buttons.release(button);
                app.update();
                app.world_mut()
                    .resource_mut::<ButtonInput<MouseButton>>()
                    .clear();
            }
        }
    }

    #[test]
    fn explicit_mouse_press_restarts_pan_and_orbit_during_momentum_debounce() {
        for follow in [false, true] {
            for button in [MouseButton::Left, MouseButton::Right] {
                let (mut app, camera, _, _) = camera_input_app(follow);
                // Keep the debounce active regardless of test runner speed.
                app.world_mut()
                    .get_mut::<EditorCam>(camera)
                    .unwrap()
                    .input_debounce = std::time::Duration::from_secs(60);
                app.world_mut()
                    .resource_mut::<ButtonInput<MouseButton>>()
                    .press(button);
                app.update();
                assert_eq!(
                    app.world()
                        .resource::<CameraPointerMap>()
                        .get(&PointerId::Mouse),
                    Some(&camera)
                );
                {
                    let mut buttons = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
                    buttons.clear();
                    buttons.release(button);
                }
                app.update();
                assert!(app.world().resource::<CameraPointerMap>().is_empty());
                assert!(
                    app.world()
                        .get::<EditorCam>(camera)
                        .unwrap()
                        .is_actively_controlled()
                );
                {
                    let mut buttons = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
                    buttons.clear();
                    buttons.press(button);
                }
                app.update();
                assert_eq!(
                    app.world()
                        .resource::<CameraPointerMap>()
                        .get(&PointerId::Mouse),
                    Some(&camera)
                );
                assert!(
                    app.world()
                        .get::<EditorCam>(camera)
                        .unwrap()
                        .current_motion
                        .is_user_controlled()
                );
            }
        }
    }

    #[test]
    fn click_released_in_one_frame_does_not_leave_a_drag_active() {
        for follow in [false, true] {
            for button in [MouseButton::Left, MouseButton::Right] {
                let (mut app, camera, _, _) = camera_input_app(follow);
                {
                    let mut buttons = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
                    buttons.press(button);
                    buttons.release(button);
                }
                app.update();
                assert!(app.world().resource::<CameraPointerMap>().is_empty());
                assert!(
                    !app.world()
                        .get::<EditorCam>(camera)
                        .unwrap()
                        .current_motion
                        .is_user_controlled()
                );

                {
                    let mut buttons = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
                    buttons.clear();
                    buttons.press(button);
                }
                app.update();
                assert_eq!(
                    app.world()
                        .resource::<CameraPointerMap>()
                        .get(&PointerId::Mouse),
                    Some(&camera)
                );
            }
        }
    }

    #[test]
    fn pan_anchor_stays_under_cursor_in_an_offset_scaled_viewport() {
        let camera = Camera {
            viewport: Some(bevy::camera::Viewport {
                physical_position: UVec2::new(100, 50),
                physical_size: UVec2::new(800, 500),
                ..default()
            }),
            computed: ComputedCameraValues {
                target_info: Some(RenderTargetInfo {
                    physical_size: UVec2::new(1280, 720),
                    scale_factor: 2.0,
                }),
                ..default()
            },
            ..default()
        };
        let rect = camera.logical_viewport_rect().unwrap();
        let cursor = rect.center() + Vec2::new(70., -40.);
        for projection in [
            Projection::default(),
            Projection::Orthographic(OrthographicProjection::default_3d()),
        ] {
            let anchor = pan_anchor(&camera, &projection, cursor, -200.).unwrap();
            let ndc = projection
                .get_clip_from_view()
                .as_dmat4()
                .project_point3(anchor);
            let projected_cursor = rect.min
                + (ndc.truncate().as_vec2() * Vec2::new(1., -1.) + Vec2::ONE) * rect.size() / 2.;
            assert!(projected_cursor.distance(cursor) < 0.001);
            assert!((anchor.z + 200.).abs() < 1e-9);
        }
    }
}
