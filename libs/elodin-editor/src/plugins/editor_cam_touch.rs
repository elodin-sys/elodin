use bevy::app::{App, Plugin, Update};
use bevy::ecs::schedule::IntoSystemConfigs;
use bevy::ecs::system::Query;
use bevy::input::touch::Touch;
use bevy::math::{DVec3, Vec2};
use bevy::prelude::{Res, ResMut, Resource, Touches};
use bevy::render::camera::Camera;
use bevy::transform::components::Transform;
use bevy_editor_cam::controller::component::EditorCam;
use std::ops::{Add, Sub};

pub struct EditorCamTouchPlugin;

impl Plugin for EditorCamTouchPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TouchTracker>()
            .add_systems(Update, touch_tracker)
            .add_systems(Update, touch_editor_cam.after(touch_tracker));
    }
}

// source: https://github.com/Plonq/bevy_panorbit_camera/blob/master/src/touch.rs

/// Holds information about current mobile gestures
#[derive(Debug, Clone)]
pub enum TouchGestures {
    /// No mobile gestures
    None,
    /// One finger mobile gestures
    OneFinger(OneFingerGestures),
    /// Two finger mobile gestures
    TwoFinger(TwoFingerGestures),
}

/// Holds information pertaining to one finger gestures
#[derive(Debug, Clone, Copy)]
pub struct OneFingerGestures {
    /// The delta movement of the mobile
    pub motion: Vec2,
    pub midpoint: Vec2,
}

/// Holds information pertaining to two finger gestures
#[derive(Debug, Clone, Copy)]
pub struct TwoFingerGestures {
    /// The delta movement of both touches.
    /// Uses the midpoint between the touches to calculate movement. Thus, if the midpoint doesn't
    /// move then this will be zero (or close to zero), like when pinching.
    pub motion: Vec2,
    /// The delta distance between both touches.
    /// Use this to implement pinch gestures.
    pub pinch: f32,
    /// The delta angle of the two touches.
    /// Positive values correspond to rotating clockwise.
    #[allow(dead_code)]
    pub rotation: f32,

    pub midpoint: Vec2,
}

/// Stores current and previous frame mobile data, and provides a method to get mobile gestures
#[derive(Resource, Default, Debug)]
pub struct TouchTracker {
    curr_pressed: (Option<Touch>, Option<Touch>),
    prev_pressed: (Option<Touch>, Option<Touch>),
}

impl TouchTracker {
    /// Calculate and return mobile gesture data for this frame
    pub fn get_touch_gestures(&self) -> TouchGestures {
        // The below matches only match when the previous and current frames have the same number
        // of touches. This means that when the number of touches changes, there's one frame
        // where this will return `TouchGestures::None`. From my testing, this does not result
        // in any adverse effects.
        match (self.curr_pressed, self.prev_pressed) {
            // Zero fingers
            ((None, None), (None, None)) => TouchGestures::None,
            // One finger
            ((Some(curr), None), (Some(prev), None)) => {
                let curr_pos = curr.position();
                let prev_pos = prev.position();

                let motion = curr_pos - prev_pos;

                TouchGestures::OneFinger(OneFingerGestures {
                    motion,
                    midpoint: curr_pos,
                })
            }
            // Two fingers
            ((Some(curr1), Some(curr2)), (Some(prev1), Some(prev2))) => {
                let curr1_pos = curr1.position();
                let curr2_pos = curr2.position();
                let prev1_pos = prev1.position();
                let prev2_pos = prev2.position();

                // Move
                let curr_midpoint = curr1_pos.midpoint(curr2_pos);
                let prev_midpoint = prev1_pos.midpoint(prev2_pos);
                let motion = curr_midpoint - prev_midpoint;

                // Pinch
                let curr_dist = curr1_pos.distance(curr2_pos);
                let prev_dist = prev1_pos.distance(prev2_pos);
                let pinch = curr_dist - prev_dist;

                // Rotate
                let prev_vec = prev2_pos - prev1_pos;
                let curr_vec = curr2_pos - curr1_pos;
                let prev_angle_negy = prev_vec.angle_between(Vec2::NEG_Y);
                let curr_angle_negy = curr_vec.angle_between(Vec2::NEG_Y);
                let prev_angle_posy = prev_vec.angle_between(Vec2::Y);
                let curr_angle_posy = curr_vec.angle_between(Vec2::Y);
                let rotate_angle_negy = curr_angle_negy - prev_angle_negy;
                let rotate_angle_posy = curr_angle_posy - prev_angle_posy;
                // The angle between -1deg and +1deg is 358deg according to Vec2::angle_between,
                // but we want the answer to be +2deg (or -2deg if swapped). Therefore, we calculate
                // two angles - one from UP and one from DOWN, and use the one with the smallest
                // absolute value. This is necessary to get a predictable result when the two touches
                // swap sides (i.e mobile 1's X position being less than the other, to the other way
                // round).
                let rotation = if rotate_angle_negy.abs() < rotate_angle_posy.abs() {
                    rotate_angle_negy
                } else {
                    rotate_angle_posy
                };

                TouchGestures::TwoFinger(TwoFingerGestures {
                    motion,
                    pinch,
                    rotation,
                    midpoint: curr_midpoint,
                })
            }
            // Three fingers and more not currently supported
            _ => TouchGestures::None,
        }
    }
}

/// Read touch input and save it in TouchTracker resource for easy consumption by the main system
pub fn touch_tracker(touches: Res<Touches>, mut touch_tracker: ResMut<TouchTracker>) {
    let pressed: Vec<&Touch> = touches.iter().collect();

    match pressed.len() {
        0 => {
            touch_tracker.curr_pressed = (None, None);
            touch_tracker.prev_pressed = (None, None);
        }
        1 => {
            let touch: &Touch = pressed.first().unwrap();
            touch_tracker.prev_pressed = touch_tracker.curr_pressed;
            touch_tracker.curr_pressed = (Some(*touch), None);
        }
        2 => {
            let touch1: &Touch = pressed.first().unwrap();
            let touch2: &Touch = pressed.last().unwrap();
            touch_tracker.prev_pressed = touch_tracker.curr_pressed;
            touch_tracker.curr_pressed = (Some(*touch1), Some(*touch2));
        }
        _ => {}
    }
}

pub trait Midpoint {
    type V: Add + Sub;

    /// Return the value exact halfway between two values
    fn midpoint(&self, other: Self::V) -> Self::V;
}

impl Midpoint for Vec2 {
    type V = Vec2;

    /// Return the vector exact halfway between two vectors
    fn midpoint(&self, other: Self::V) -> Self::V {
        let vec_to_other = other - *self;
        let half = vec_to_other / 2.0;
        *self + half
    }
}

pub fn touch_editor_cam(
    touch_tracker: Res<TouchTracker>,
    mut cams: Query<(&mut EditorCam, &Transform, &Camera)>,
) {
    let touch_gestures = touch_tracker.get_touch_gestures();
    let midpoint = match touch_gestures {
        TouchGestures::OneFinger(one_finger) => one_finger.midpoint,
        TouchGestures::TwoFinger(two_finger) => two_finger.midpoint,
        _ => return,
    };
    for (mut editor_cam, transform, cam) in cams.iter_mut() {
        let Some(viewport_rect) = cam.logical_viewport_rect() else {
            continue;
        };
        if !viewport_rect.contains(midpoint) {
            continue;
        }
        match touch_gestures {
            // orbit
            TouchGestures::OneFinger(gesture) => {
                editor_cam.end_move();
                let anchor = transform
                    .compute_matrix()
                    .as_dmat4()
                    .inverse()
                    .transform_point3(DVec3::ZERO);
                editor_cam.start_orbit(Some(anchor));
                editor_cam.send_screenspace_input(gesture.motion);
            }
            TouchGestures::TwoFinger(gesture) => {
                editor_cam.end_move();
                let anchor = transform
                    .compute_matrix()
                    .as_dmat4()
                    .inverse()
                    .transform_point3(DVec3::ZERO);
                editor_cam.start_pan(Some(anchor));
                editor_cam.send_screenspace_input(gesture.motion);
                editor_cam.send_zoom_input(gesture.pinch * 10.0);
            }
            TouchGestures::None => {}
        }
    }
}
