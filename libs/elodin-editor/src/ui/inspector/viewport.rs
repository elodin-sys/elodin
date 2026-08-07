#![allow(non_snake_case)]
use std::collections::HashSet;

use bevy::ecs::system::{SystemParam, SystemState};
use bevy::math::{DQuat, DVec3};
use bevy::picking::prelude::Pickable;
use bevy::prelude::*;
use bevy::{
    camera::Projection,
    camera::visibility::RenderLayers,
    ecs::{entity::Entity, system::Query},
};
use bevy_editor_cam::prelude::EditorCam;
use bevy_egui::egui::{self, Align};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoRotation, OrDefault};
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue, QueryType, WorldPos};
use nox::ArrayBuf;

use crate::EqlContext;
use crate::WorldPosExt;
use crate::object_3d::{ComponentArrayExt, EditableEQL, Object3DState, compile_eql_expr};
use crate::ui::button::EButton;
use crate::ui::colors::{EColor, get_scheme};
use crate::ui::theme::configure_input_with_border;
use crate::ui::widgets::WidgetSystem;
use crate::ui::{CameraQuery, ViewportRect};
use crate::{
    GridHandle, MainCamera,
    ui::tiles::ViewportConfig,
    ui::{label::ELabel, theme, utils::MarginSides},
};

use super::{color_popup, empty_inspector, eql_autocomplete, query};
use crate::ui::widgets::SystemStateExt;

const DEFAULT_EDITOR_CAM_ANCHOR_DEPTH: f64 = -2.0;
const ANCHOR_DEPTH_EPSILON: f64 = 1.0e-9;

#[derive(Component)]
pub struct ViewportFocusPickTarget;

/// Extract a 3-vector from a ComponentValue (e.g. F64 array of length >= 3). Returns None if not a numeric array or length < 3.
fn extract_vec3(val: &ComponentValue) -> Option<DVec3> {
    let ComponentValue::F64(array) = val else {
        return None;
    };
    let data = array.buf.as_buf();
    if data.len() < 3 {
        return None;
    }
    Some(DVec3::new(data[0], data[1], data[2]))
}

#[derive(Component)]
pub struct Viewport {
    parent_entity: Entity,
    pub pos: EditableEQL,
    pub look_at: EditableEQL,
    /// Optional camera up vector in world frame. EQL that evaluates to a 3-vector (e.g. "(0,0,1)" or "pose.direction(0,1,1)").
    pub up: EditableEQL,
    /// Optional geo frame for interpreting position and rotation.
    pub frame: Option<GeoFrame>,
    /// Follow low-pass time constant in seconds; 0 disables smoothing.
    pub smoothing: f32,
    smoothing_state: Option<FollowSmoothingState>,
}

/// Last smoothed follow pose, kept across frames to low-pass the raw telemetry
/// targets.
///
/// - `raw_target` is the previous unsmoothed target and `step_ema` a smoothed
///   *magnitude* of recent per-frame motion, used only for seek detection so a
///   timeline scrub can be told apart from ordinary — even fast — motion
///   without a single noisy frame tripping a false snap.
/// - `vel_ema` / `look_vel_ema` are low-passed *signed* velocities (m/s) of the
///   pos and look-at targets. They drive the One Euro adaptive cutoff: because
///   they are signed, zero-mean telemetry noise averages toward zero and does
///   not inflate the speed estimate the way a magnitude EMA would, so the
///   filter stays heavy while the vehicle is merely jittering in place. Both
///   endpoints are tracked so a fixed-`pos` viewport still opens its cutoff (and
///   snaps on a scrub) when only the look-at moves.
/// - `raw_att` / `att_step_ema` play the same seek-detection role for the
///   attitude: with a fixed `pos` and no `look_at` the camera rides the pose
///   orientation, so a scrub that mainly changes attitude (hover, pad dwell)
///   must still snap instead of slowly slerping across the jump.
struct FollowSmoothingState {
    pos: DVec3,
    look_at: Option<DVec3>,
    att: Option<DQuat>,
    raw_target: DVec3,
    raw_look_at: Option<DVec3>,
    raw_att: Option<DQuat>,
    step_ema: f64,
    att_step_ema: f64,
    vel_ema: DVec3,
    look_vel_ema: DVec3,
}

impl Viewport {
    pub fn new(
        parent_entity: Entity,
        pos: EditableEQL,
        look_at: EditableEQL,
        up: EditableEQL,
        frame: Option<GeoFrame>,
        smoothing: f32,
    ) -> Self {
        Self {
            parent_entity,
            pos,
            look_at,
            up,
            frame,
            smoothing,
            smoothing_state: None,
        }
    }

    /// Low-pass the raw follow targets with a One Euro filter. Returns the pose
    /// to apply this frame and updates the internal state.
    ///
    /// The filter time constant is speed-adaptive: heavy smoothing when the
    /// target is slow or still (so telemetry noise — and the floating-origin
    /// jitter it causes — is strongly rejected, collapsing a stationary noisy
    /// target to nearly a point), and progressively lighter as it moves faster
    /// so the camera keeps up during flight. Concretely the steady-state follow
    /// lag is *bounded* by a small fraction of the framing distance instead of
    /// growing without limit like a fixed-`smoothing` first-order filter (`lag =
    /// smoothing * speed`), which is what made `smoothing=1.0` trail visibly
    /// during a fast climb. The bound is a fraction rather than the framing
    /// distance itself because the subject is drawn from raw telemetry, so any
    /// lag reads as a framing error: trailing by a whole framing distance throws
    /// the subject ~45 degrees off centre, which looks far worse than the noise
    /// being filtered. At zero speed it reduces exactly to a fixed first-order
    /// low-pass with time constant `smoothing`, so the parameter's meaning at
    /// rest is unchanged.
    ///
    /// Snaps straight to the target on the first frame, on non-finite values,
    /// and on a *seek* — a jump both larger than the framing distance and a
    /// sudden discontinuity versus recent per-frame motion (timeline scrub) —
    /// so a scrub doesn't slowly glide across the world.
    fn smooth_follow(
        &mut self,
        target_pos: DVec3,
        target_look_at: Option<DVec3>,
        target_att: Option<DQuat>,
        dt: f64,
    ) -> (DVec3, Option<DVec3>, Option<DQuat>) {
        if self.smoothing <= 0.0 {
            self.smoothing_state = None;
            return (target_pos, target_look_at, target_att);
        }
        let att_finite = target_att.is_some_and(|q| {
            q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite()
        });
        if !target_pos.is_finite()
            || target_look_at.is_some_and(|t| !t.is_finite())
            || target_att.is_some_and(|_| !att_finite)
        {
            self.smoothing_state = None;
            return (target_pos, target_look_at, target_att);
        }
        // The framing distance is the natural scale for the seek test. Use the
        // *previous* frame's raw framing rather than the current one: a look-at
        // that jumps on its own inflates the current framing, so a genuine
        // scrub would measure smaller than the floor and never snap.
        let framing = match &self.smoothing_state {
            Some(prev) => prev
                .raw_look_at
                .map(|look| (look - prev.raw_target).length()),
            None => target_look_at.map(|look| (look - target_pos).length()),
        }
        .filter(|d| d.is_finite() && *d > 0.0);
        let max_lag = framing.unwrap_or(DEFAULT_MAX_LAG);
        // Cap on the follow lag. With a `look_at` the lag *is* a framing error:
        // the subject is drawn at its raw position, so trailing it by the whole
        // framing distance throws it ~45 degrees off centre. Bound the lag to a
        // small fraction of the framing so the subject stays near the centre —
        // `atan(FRAMING_LAG_RATIO)` is the worst-case angle. It doubles as the
        // scale that fades the filter out with speed, so a tight budget makes it
        // transparent in flight and full-strength only when the target is slow.
        // Riding the pose has no framing to protect, so the absolute default
        // stands.
        let lag_cap = match framing {
            Some(d) => d * FRAMING_LAG_RATIO,
            None => DEFAULT_MAX_LAG,
        };

        // Seek detection watches *both* endpoints against the previous framing:
        // a viewport with a fixed pos and a telemetry look-at (or vice versa)
        // must still snap when its moving end jumps on a scrub. Steady motion
        // has consistent per-frame steps (bounded acceleration); a scrub jumps
        // far in a single frame. Comparing against a smoothed step magnitude
        // (not just the last frame) keeps smoothing active throughout fast
        // flight while still snapping on a real scrub, and stops a single noisy
        // frame from tripping a false snap.
        let (pos_step, look_step, att_step) = match &self.smoothing_state {
            Some(prev) => {
                let pos_step = (target_pos - prev.raw_target).length();
                let look_step = match (prev.raw_look_at, target_look_at) {
                    (Some(a), Some(b)) => (b - a).length(),
                    _ => 0.0,
                };
                let att_step = match (prev.raw_att, target_att) {
                    (Some(a), Some(b)) => a.angle_between(b),
                    _ => 0.0,
                };
                (pos_step, look_step, att_step)
            }
            None => (0.0, 0.0, 0.0),
        };
        let step = pos_step.max(look_step);
        // Attitude jumps get their own seek test: translation may be exactly
        // zero on a scrub (fixed pos, no look_at, vehicle hovering) yet the
        // orientation still jumps with the playhead. Rotation has an absolute
        // scale, so a fixed angular floor replaces the framing distance.
        let seek = match &self.smoothing_state {
            Some(prev) => {
                let pos_seek = step > max_lag && step > prev.step_ema * SEEK_STEP_RATIO;
                // Attitude seek only applies when the camera actually rides the
                // pose: with a `look_at` the orientation comes from the viewing
                // vector, not `target_att`, so a telemetry attitude jump is
                // irrelevant and must not hitch a chase camera that was merely
                // lagging in translation. Its scrubs trip `pos_seek` via
                // `look_step` instead.
                let att_seek = target_look_at.is_none()
                    && att_step > ATT_SEEK_FLOOR
                    && att_step > prev.att_step_ema * SEEK_STEP_RATIO;
                pos_seek || att_seek
            }
            None => true,
        };

        let (state, out_pos, out_look_at, out_att) = match (&self.smoothing_state, seek) {
            (Some(prev), false) => {
                // The One Euro speed estimate, taken from the *eased* pose of
                // the previous frames rather than the raw telemetry. Raw
                // per-frame differences are dominated by noise while the
                // target is still — 0.5 m of jitter at 60 Hz reads as 30 m/s
                // — which would open the cutoff exactly where the heaviest
                // filtering is wanted, and it is only the huge lag budget of
                // an unbounded framing that hid it. The eased pose has that
                // noise already removed and travels at the true speed in
                // steady motion, so the cutoff tracks real motion only.
                // Tracking the look-at too lets a fixed-pos viewport open its
                // cutoff for a fast look-at.
                let speed = prev.vel_ema.length().max(prev.look_vel_ema.length());

                // One Euro adaptive time constant, expressed via inverse time
                // constants ("rates") so the 2*pi cancels: at rest the rate
                // is 1/smoothing (unchanged); adding speed/lag_cap raises it
                // so the steady-state lag `speed * tau_eff` approaches
                // `lag_cap` as speed grows.
                let rate = 1.0 / self.smoothing as f64 + speed / lag_cap;
                let tau_eff = 1.0 / rate;
                let alpha = smoothing_alpha(dt, tau_eff);

                // The adaptive cutoff only *approaches* the bounded lag
                // asymptotically, so right after a snap — while the speed
                // estimate is still rebuilding — a sustained scrub would
                // trail far behind. Clamp the eased result so each endpoint
                // trails its raw target by at most a small multiple of the
                // lag cap, making the documented bound hold every frame
                // instead of only in the limit.
                let max_catchup = lag_cap * CATCHUP_LAG_RATIO;
                let pos = clamp_lag(prev.pos.lerp(target_pos, alpha), target_pos, max_catchup);
                let look_at = match (prev.look_at, target_look_at) {
                    (Some(prev_look), Some(target)) => Some(clamp_lag(
                        prev_look.lerp(target, alpha),
                        target,
                        max_catchup,
                    )),
                    _ => target_look_at,
                };
                // With no `look_at` the camera rides the telemetry attitude,
                // so it must be eased too or the view keeps jittering while
                // the position settles. `slerp` takes the shortest arc; the
                // same adaptive alpha keeps translation and rotation in step.
                let att = match (prev.att, target_att) {
                    (Some(prev_att), Some(target)) => Some(prev_att.slerp(target, alpha)),
                    _ => target_att,
                };
                // Feed the speed estimate from this frame's eased increments,
                // so the next frame sees the smoothed trajectory's velocity.
                let alpha_v = smoothing_alpha(dt, DERIV_TAU);
                let inst_vel = if dt > 0.0 {
                    (pos - prev.pos) / dt
                } else {
                    DVec3::ZERO
                };
                let inst_look_vel = match (prev.look_at, look_at) {
                    (Some(a), Some(b)) if dt > 0.0 => (b - a) / dt,
                    _ => DVec3::ZERO,
                };
                let vel_ema = prev.vel_ema.lerp(inst_vel, alpha_v);
                let look_vel_ema = prev.look_vel_ema.lerp(inst_look_vel, alpha_v);
                let step_ema = prev.step_ema + STEP_EMA_ALPHA * (step - prev.step_ema);
                let att_step_ema =
                    prev.att_step_ema + STEP_EMA_ALPHA * (att_step - prev.att_step_ema);

                (
                    FollowSmoothingState {
                        pos,
                        look_at,
                        att,
                        raw_target: target_pos,
                        raw_look_at: target_look_at,
                        raw_att: target_att,
                        step_ema,
                        att_step_ema,
                        vel_ema,
                        look_vel_ema,
                    },
                    pos,
                    look_at,
                    att,
                )
            }
            // First frame or a seek: snap, and seed the step estimates with
            // this jump so the frames right after a scrub aren't misread as
            // more seeks (the estimates decay back toward the real motion).
            // The velocity estimates reset to zero so a scrub can't leave a
            // stale high speed that would under-smooth the frames after it.
            _ => (
                FollowSmoothingState {
                    pos: target_pos,
                    look_at: target_look_at,
                    att: target_att,
                    raw_target: target_pos,
                    raw_look_at: target_look_at,
                    raw_att: target_att,
                    step_ema: step,
                    att_step_ema: att_step,
                    vel_ema: DVec3::ZERO,
                    look_vel_ema: DVec3::ZERO,
                },
                target_pos,
                target_look_at,
                target_att,
            ),
        };

        self.smoothing_state = Some(state);
        (out_pos, out_look_at, out_att)
    }
}

/// A raw target jump counts as a seek (and snaps) only when it exceeds this
/// multiple of the previous frame's step, i.e. a sudden discontinuity rather
/// than continuous — even if fast — motion.
const SEEK_STEP_RATIO: f64 = 8.0;

/// Blend factor for the smoothed per-frame step magnitude used by seek
/// detection. Small enough that one noisy frame barely moves the estimate.
const STEP_EMA_ALPHA: f64 = 0.2;

/// Minimum single-frame attitude jump (radians) that can count as a seek —
/// about 20 degrees. Telemetry attitude noise and smoothed flight rotation stay
/// far below this in one frame; a timeline scrub lands far above it.
const ATT_SEEK_FLOOR: f64 = 0.35;

/// Time constant (seconds) of the One Euro derivative low-pass. It measures the
/// already-smoothed pose, so it needs no extra margin against telemetry noise
/// and can stay short enough to pick up a launch within a few frames — the
/// cutoff only opens once this estimate rises, and the catch-up clamp covers
/// the transient meanwhile.
///
/// It also sets how far the estimate trails under acceleration
/// (`DERIV_TAU * accel`), and the cutoff opens late by that much, so a hard
/// launch wants it short. Shortening it also makes the speed estimate noisier,
/// which opens the cutoff while parked and lets scene jitter back in, so this is
/// a balance point rather than a minimum.
const DERIV_TAU: f64 = 0.10;

/// Fraction of the framing distance the eased pose may trail its raw target by.
/// The subject is drawn from raw telemetry, so this lag reads directly as a
/// framing error of at most `atan(FRAMING_LAG_RATIO)` — under 2 degrees — which
/// keeps a chase camera pointed at its subject during fast flight.
///
/// It is also the knob that decides *how fast* the adaptive cutoff opens, since
/// the speed term of the rate is `speed / lag_cap`: a tight budget means the
/// filter fades out early in the speed range and is all but transparent in
/// flight, which is deliberate. Smoothing cannot steady the world and hold the
/// subject centred at the same time — the two are the same lag seen from either
/// end — and in flight framing wins: the scene is streaming past anyway, so the
/// jitter it hides is hardly noticeable, whereas a subject sliding off centre is
/// glaring. At a wider budget (0.15) the filter kept working through a climb and
/// steadied the background 3-4x, but parked the subject several degrees off
/// centre, which read as worse than no smoothing at all during liftoff.
///
/// It cannot be tightened much further, because sensor noise has a speed of its
/// own and a small budget lets that speed open the cutoff too, which costs the
/// parked case — the one that needs the filter most. Measured on a
/// degraded-sensor recording at 35 m framing, parked scene motion runs 1.2 px per
/// frame here against 23 px unsmoothed, but 2.4 px at 0.02 and 9.4 px at 0.008.
///
/// The bound is also what ties a usable `smoothing` to a framing distance:
/// filtering a fast subject needs a lag budget of at least a frame of its
/// motion, so a 250 m/s vehicle needs tens of metres of framing, not a close-up.
const FRAMING_LAG_RATIO: f64 = 0.03;

/// Fallback cap (metres) on the adaptive follow lag when there is no `look_at`
/// to define a framing distance.
const DEFAULT_MAX_LAG: f64 = 5.0;

/// The eased pose may trail its raw target by at most this multiple of the
/// framing distance. Bounds the transient lag of a sustained timeline scrub —
/// before the speed estimate rebuilds after a snap — while staying wide enough
/// that ordinary at-rest noise and slow tracking never reach it.
const CATCHUP_LAG_RATIO: f64 = 2.0;

/// Framerate-independent exponential low-pass factor for a first-order
/// filter with time constant `time_constant` (seconds).
fn smoothing_alpha(dt: f64, time_constant: f64) -> f64 {
    if time_constant <= 0.0 {
        return 1.0;
    }
    1.0 - (-dt.max(0.0) / time_constant).exp()
}

/// Clamp `smoothed` so it trails `target` by at most `max_lag` metres, holding
/// the follow lag bounded even before the adaptive cutoff has caught up.
fn clamp_lag(smoothed: DVec3, target: DVec3, max_lag: f64) -> DVec3 {
    let delta = smoothed - target;
    let len = delta.length();
    if len > max_lag && len > 0.0 {
        target + delta / len * max_lag
    } else {
        smoothed
    }
}

#[derive(SystemParam)]
pub struct InspectorViewport<'w, 's> {
    camera_query: Query<'w, 's, CameraQuery, With<MainCamera>>,
    viewports: Query<'w, 's, &'static mut Viewport>,
    viewport_configs: Query<'w, 's, &'static mut ViewportConfig>,
    viewport_rects: Query<'w, 's, &'static ViewportRect, With<MainCamera>>,
    editor_cams: Query<'w, 's, &'static mut EditorCam>,
    object_3d_states: Query<'w, 's, &'static Object3DState>,
    eql_ctx: ResMut<'w, EqlContext>,
}

impl WidgetSystem for InspectorViewport<'_, '_> {
    type Args = (Entity, String);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let scheme = get_scheme();
        let state_mut = state.params_mut(world);

        let (camera, title) = args;

        let InspectorViewport {
            mut camera_query,
            mut viewports,
            mut viewport_configs,
            viewport_rects,
            mut editor_cams,
            object_3d_states,
            eql_ctx,
        } = state_mut;

        let Ok(mut cam) = camera_query.get_mut(camera) else {
            ui.add(empty_inspector());
            return;
        };

        let Ok(mut viewport) = viewports.get_mut(camera) else {
            return;
        };
        let Ok(mut viewport_config) = viewport_configs.get_mut(camera) else {
            return;
        };
        let has_detected_ellipsoid = object_3d_states.iter().any(|object_state| {
            matches!(
                &object_state.data.mesh,
                impeller2_wkt::Object3DMesh::Ellipsoid { .. }
            )
        });

        ui.spacing_mut().item_spacing.y = 8.0;
        let title = title.trim();
        let title = if title.is_empty() { "Viewport" } else { title };
        ui.add(
            ELabel::new(title)
                .padding(egui::Margin::same(8).bottom(24.0))
                .bottom_stroke(egui::Stroke {
                    color: get_scheme().border_primary,
                    width: 1.0,
                })
                .margin(egui::Margin::same(0).bottom(16.0)),
        );

        ui.label(egui::RichText::new("POSITION").color(get_scheme().text_secondary));
        eql_input(ui, &mut viewport.pos, &eql_ctx.0);
        ui.separator();
        ui.label(egui::RichText::new("LOOK AT").color(get_scheme().text_secondary));
        eql_input(ui, &mut viewport.look_at, &eql_ctx.0);
        ui.separator();
        ui.label(egui::RichText::new("UP").color(get_scheme().text_secondary));
        eql_input(ui, &mut viewport.up, &eql_ctx.0);
        ui.separator();

        if ui.add(EButton::highlight("Reset Pos")).clicked() {
            *cam.transform = <bevy::prelude::Transform as std::default::Default>::default();
        }

        if let Projection::Perspective(persp) = cam.projection.as_mut() {
            ui.separator();
            let mut configured_clip_planes = None;
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(8, 8))
                .show(ui, |ui| {
                    let mut fov = persp.fov.to_degrees();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FOV").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            if ui.add(egui::DragValue::new(&mut fov).speed(0.1)).changed() {
                                persp.fov = fov.to_radians();
                            }
                        });
                    });
                    ui.add_space(8.0);
                    ui.style_mut().spacing.slider_width = ui.available_size().x;
                    ui.style_mut().visuals.widgets.inactive.bg_fill = scheme.bg_secondary;
                    if ui
                        .add(egui::Slider::new(&mut fov, 5.0..=120.0).show_value(false))
                        .changed()
                    {
                        persp.fov = fov.to_radians();
                    }

                    ui.add_space(8.0);
                    let mut near = persp.near;
                    let mut far = persp.far;
                    let mut near_changed = false;
                    let mut far_changed = false;

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("NEAR").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            near_changed |= ui
                                .add(egui::DragValue::new(&mut near).speed(0.001))
                                .changed();
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FAR").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            far_changed |=
                                ui.add(egui::DragValue::new(&mut far).speed(0.01)).changed();
                        });
                    });

                    if near_changed || far_changed {
                        near = near.max(0.0001);
                        if far <= near + 0.0001 {
                            far = near + 0.0001;
                        }
                        persp.near = near;
                        persp.far = far;
                        configured_clip_planes = Some((near, far));

                        if let Ok(mut editor_cam) = editor_cams.get_mut(camera) {
                            if near_changed {
                                editor_cam.perspective.near_clip_limits = near..near;
                            }
                            if far_changed {
                                let (min_size_per_pixel, max_size_per_pixel) =
                                    crate::ui::tiles::zoom_limits_for_far(far);
                                editor_cam.zoom_limits.min_size_per_pixel = min_size_per_pixel;
                                editor_cam.zoom_limits.max_size_per_pixel = max_size_per_pixel;
                            }
                        }
                    }

                    ui.add_space(8.0);
                    let derived_aspect = viewport_rects
                        .get(camera)
                        .ok()
                        .and_then(|rect| rect.0)
                        .and_then(|rect| {
                            let size = rect.size();
                            if size.x > 0.0 && size.y > 0.0 {
                                Some((size.x / size.y, size.x, size.y))
                            } else {
                                None
                            }
                        });

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("REAL ASPECT").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            if let Some((aspect, width, height)) = derived_aspect {
                                ui.label(format!("{aspect:.3} ({width:.0}x{height:.0})"));
                            } else {
                                ui.label("N/A");
                            }
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("ASPECT MODE").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            let fixed_selected = viewport_config.aspect.is_some();
                            if ui.selectable_label(fixed_selected, "FIXED").clicked()
                                && viewport_config.aspect.is_none()
                            {
                                viewport_config.aspect = Some(
                                    derived_aspect
                                        .map(|(aspect, _, _)| aspect)
                                        .unwrap_or(persp.aspect_ratio.max(0.0001)),
                                );
                            }
                            ui.add_space(8.0);
                            if ui.selectable_label(!fixed_selected, "AUTO").clicked() {
                                viewport_config.aspect = None;
                            }
                        });
                    });

                    if let Some(aspect) = viewport_config.aspect.as_mut() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("ASPECT").color(scheme.text_secondary));
                            ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                                if ui.add(egui::DragValue::new(aspect).speed(0.01)).changed() {
                                    *aspect = (*aspect).max(0.0001);
                                }
                            });
                        });
                    }
                });
            if let Some((near, far)) = configured_clip_planes {
                viewport_config.configured_near = Some(near);
                viewport_config.configured_far = Some(far);
            }
        }

        if let Some(&GridHandle { layer }) = cam.grid_handle {
            ui.separator();
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(8, 8))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("SHOW GRID").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            let mut visible =
                                cam.render_layers.intersects(&RenderLayers::layer(layer));
                            theme::configure_input_with_border(ui.style_mut());
                            ui.checkbox(&mut visible, "");
                            if visible {
                                *cam.render_layers = cam.render_layers.clone().with(layer);
                            } else {
                                *cam.render_layers = cam.render_layers.clone().without(layer);
                            }
                        });
                    });
                });
        }

        ui.separator();
        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(8, 8))
            .show(ui, |ui| {
                let create_button_width = 88.0;
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("FRUSTUM").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        let frustum_created = viewport_config.create_frustum;
                        let button = if frustum_created {
                            EButton::red("DELETE")
                        } else {
                            EButton::highlight("CREATE")
                        };
                        if ui.add(button.width(create_button_width)).clicked() {
                            viewport_config.create_frustum = !frustum_created;
                        }
                    });
                });

                if viewport_config.create_frustum {
                    ui.add_space(8.0);

                    let mut frustums_color = viewport_config.frustums_color.into_color32();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FRUSTUM COLOR").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                            let swatch = ui.add(
                                egui::Button::new("")
                                    .fill(frustums_color)
                                    .stroke(egui::Stroke::new(1.0, scheme.border_primary))
                                    .corner_radius(egui::CornerRadius::same(10))
                                    .min_size(egui::vec2(20.0, 20.0)),
                            );
                            let color_id = ui.auto_id_with("frustums_color");
                            if swatch.clicked() {
                                egui::Popup::toggle_id(ui.ctx(), color_id);
                            }
                            color_popup(ui, &mut frustums_color, color_id, &swatch);
                        });
                    });
                    viewport_config.frustums_color =
                        impeller2_wkt::Color::from_color32(frustums_color);

                    ui.add_space(8.0);
                    let mut projection_color = viewport_config.projection_color.into_color32();
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("PROJ. 2D COLOR").color(scheme.text_secondary),
                        );
                        ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                            let swatch = ui.add(
                                egui::Button::new("")
                                    .fill(projection_color)
                                    .stroke(egui::Stroke::new(1.0, scheme.border_primary))
                                    .corner_radius(egui::CornerRadius::same(10))
                                    .min_size(egui::vec2(20.0, 20.0)),
                            );
                            let color_id = ui.auto_id_with("projection_color");
                            if swatch.clicked() {
                                egui::Popup::toggle_id(ui.ctx(), color_id);
                            }
                            color_popup(ui, &mut projection_color, color_id, &swatch);
                        });
                    });
                    viewport_config.projection_color =
                        impeller2_wkt::Color::from_color32(projection_color);

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("THICKNESS").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            let mut thickness = viewport_config.frustums_thickness;
                            if ui
                                .add(egui::DragValue::new(&mut thickness).speed(0.001))
                                .changed()
                            {
                                viewport_config.frustums_thickness = thickness.max(0.0001);
                            }
                        });
                    });
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("SHOW FRUSTUMS").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        theme::configure_input_with_border(ui.style_mut());
                        ui.checkbox(&mut viewport_config.show_frustums, "");
                    });
                });
                let show_intersection_options =
                    viewport_config.show_frustums && has_detected_ellipsoid;
                if !show_intersection_options {
                    viewport_config.show_coverage_in_viewport = false;
                    viewport_config.show_projection_2d = false;
                }

                if show_intersection_options {
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("COVERAGE").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            theme::configure_input_with_border(ui.style_mut());
                            ui.checkbox(&mut viewport_config.show_coverage_in_viewport, "");
                        });
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("PROJ. 2D").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            theme::configure_input_with_border(ui.style_mut());
                            ui.checkbox(&mut viewport_config.show_projection_2d, "");
                        });
                    });
                }
            });
    }
}

fn eql_input(ui: &mut egui::Ui, editable_expr: &mut EditableEQL, ctx: &eql::Context) {
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        configure_input_with_border(ui.style_mut());
        let query_res = ui.add(query(&mut editable_expr.eql, QueryType::EQL));
        eql_autocomplete(ui, ctx, &query_res, &mut editable_expr.eql);
        if query_res.changed() {
            if editable_expr.eql.is_empty() {
                editable_expr.compiled_expr = None;
                return;
            }
            match ctx.parse_str(&editable_expr.eql) {
                Ok(expr) => {
                    editable_expr.compiled_expr = compile_eql_expr(expr).ok();
                }
                Err(err) => {
                    ui.colored_label(get_scheme().error, err.to_string());
                }
            }
        }
    });
}

pub fn set_viewport_pos(
    mut viewports: Query<(&mut Viewport, &mut EditorCam)>,
    mut pos: Query<&mut WorldPos>,
    entity_map: Res<EntityMap>,
    values: Query<&'static ComponentValue>,
    geo_context: Res<GeoContext>,
    time: Res<Time>,
) {
    for (mut viewport, mut editor_cam) in viewports.iter_mut() {
        let Ok(mut pos) = pos.get_mut(viewport.parent_entity) else {
            continue;
        };
        let Some(executed) = viewport
            .pos
            .compiled_expr
            .as_ref()
            .map(|expr| expr.execute(&entity_map, &values))
        else {
            continue;
        };

        let look_at_executed = viewport
            .look_at
            .compiled_expr
            .as_ref()
            .map(|expr| expr.execute(&entity_map, &values));
        let target_look_at = look_at_executed
            .as_ref()
            .and_then(|executed| executed.as_ref().ok())
            .and_then(|val| val.as_world_pos())
            .map(|world_pos| world_pos.pos());
        // A configured look_at whose sample is missing this frame is a gap, not a
        // switch to body-attitude mode: a gap must freeze the last good pose like
        // a pos gap, not flip attitude, drop the smoothed look_at, and re-arm the
        // attitude seek.
        //
        // Only a failed *execution* counts. A value that resolves but is not a
        // 7-vector pose — a bare 3-vector, which `as_world_pos` rejects and the
        // reference docs still show — is not a gap: it has always meant "track
        // pos, don't aim", and freezing the camera for it would strand the
        // viewport on its first pose forever, `smoothing=0` included.
        let look_at_gap = matches!(look_at_executed, Some(Err(_)));
        // Only aim (below) when this frame's pos actually resolves. On a missing
        // sample we hold the last good pose rather than re-aiming a raw, unsmoothed
        // look_at against a frozen position, which would jump the view direction.
        let mut look_at_point = None;

        match executed {
            Ok(_) if look_at_gap => {
                // pos resolved but the configured look_at target has no sample
                // this frame. Hold the last good pose (leave `pos` untouched, skip
                // aiming) and reseed on resume, mirroring the pos-gap path, so a
                // chase camera doesn't hitch during normal missing-sample frames.
                viewport.smoothing_state = None;
            }
            Ok(val) => {
                if let Some(world_pos) = val.as_world_pos() {
                    // Low-pass pos, look_at and attitude together so noisy
                    // telemetry doesn't shake the whole view; the viewing
                    // direction stays intact. With a `look_at` the attitude is
                    // recomputed from it below, so the smoothed attitude only
                    // matters when the camera rides the pose directly.
                    // `smoothing == 0` is a plain passthrough.
                    let (smoothed_pos, smoothed_look_at, smoothed_att) = viewport.smooth_follow(
                        world_pos.pos(),
                        target_look_at,
                        Some(world_pos.att()),
                        time.delta_secs_f64(),
                    );
                    // Preserve the last good attitude before overwriting the
                    // pose: with a `look_at`, pos and look_at ease independently,
                    // so the smoothed viewing vector can null out for a frame even
                    // when the raw targets never do (e.g. the subject crossing the
                    // camera). The look_at attitude write below is then skipped,
                    // and without this the view would flash to the body pose
                    // attitude for that frame.
                    let prev_att = pos.att;
                    *pos = world_pos;
                    pos.pos = nox::Vector3::new(smoothed_pos.x, smoothed_pos.y, smoothed_pos.z);
                    if target_look_at.is_some() {
                        // Attitude is driven by `look_at` below; hold the previous
                        // value so a transient collapse keeps the last good
                        // orientation instead of the body pose attitude.
                        pos.att = prev_att;
                    } else if let Some(att) = smoothed_att {
                        pos.att =
                            nox::Quaternion(nox::Tensor::from_buf([att.x, att.y, att.z, att.w]));
                    }
                    look_at_point = smoothed_look_at;
                } else {
                    bevy::log::warn!("viewport pos expression didn't produce a WorldPos");
                    viewport.smoothing_state = None;
                }
            }
            Err(e) => {
                // Missing ComponentValue is normal while connecting / before the
                // series has samples at the playhead — not an actionable error.
                // Hold the last good pose (leave `pos` frozen, skip aiming) and
                // drop the smoothing state so a resumed stream reseeds cleanly
                // instead of reading the gap as one huge frame, which would
                // inflate the speed estimate or trip a false snap.
                bevy::log::debug!("viewport pos formula execution error: {e}");
                viewport.smoothing_state = None;
            }
        }

        if let Some(look_at) = look_at_point {
            let frame = viewport.frame.or_default().unwrap_or(GeoFrame::ENU);
            // Everything stays in the viewport's frame: direction and up
            // are frame coordinates, and `GeoRotation::look_at` yields the
            // attitude expressed in that frame. `sync_pos` carries it into
            // the entity's `GeoRotation` unchanged.
            let dir = look_at - pos.pos();
            let target_distance = dir.length();

            if !is_valid_viewport_target_distance(target_distance) {
                continue;
            }
            refresh_default_anchor_depth(&mut editor_cam, target_distance);
            let up = viewport
                .up
                .compiled_expr
                .as_ref()
                .and_then(|up_expr| up_expr.execute(&entity_map, &values).ok())
                .and_then(|v| extract_vec3(&v))
                .filter(|v| v.length_squared() > 1e-20);
            pos.att = GeoRotation::look_at(frame, dir, up, &geo_context).1.into();
        }
    }
}

fn refresh_default_anchor_depth(editor_cam: &mut EditorCam, target_distance: f64) {
    if !is_valid_viewport_target_distance(target_distance) {
        return;
    }
    if (editor_cam.last_anchor_depth - DEFAULT_EDITOR_CAM_ANCHOR_DEPTH).abs() > ANCHOR_DEPTH_EPSILON
    {
        return;
    }
    editor_cam.last_anchor_depth = -target_distance;
}

fn is_valid_viewport_target_distance(target_distance: f64) -> bool {
    target_distance.is_finite() && target_distance > f32::EPSILON as f64
}

pub fn sync_viewport_focus_pick_targets(
    mut commands: Commands,
    viewports: Query<&Viewport>,
    objects: Query<(Entity, &Object3DState)>,
    children: Query<&Children>,
    mesh_entities: Query<(), With<Mesh3d>>,
    current_targets: Query<Entity, With<ViewportFocusPickTarget>>,
) {
    let focus_eqls = viewport_focus_eqls(&viewports);
    let mut desired_targets = HashSet::new();
    let current_targets = current_targets.iter().collect::<HashSet<_>>();

    for (entity, object) in &objects {
        if is_focus_object_eql(&focus_eqls, &object.data.eql) {
            collect_mesh_descendants(entity, &children, &mesh_entities, &mut desired_targets);
        }
    }

    // `try_*` variants silence the command if the entity was despawned between
    // building the target set and applying these deferred commands (e.g. a
    // schematic reload triggered by skybox generation despawns Object3D meshes).
    for entity in current_targets.difference(&desired_targets) {
        commands
            .entity(*entity)
            .try_remove::<(Pickable, ViewportFocusPickTarget)>();
    }

    for entity in desired_targets.difference(&current_targets) {
        commands
            .entity(*entity)
            .try_insert((Pickable::default(), ViewportFocusPickTarget));
    }
}

fn viewport_focus_eqls(viewports: &Query<&Viewport>) -> HashSet<String> {
    viewports
        .iter()
        .filter_map(|viewport| normalized_focus_eql(&viewport.look_at.eql))
        .map(ToOwned::to_owned)
        .collect()
}

fn normalized_focus_eql(eql: &str) -> Option<&str> {
    let eql = eql.trim();
    (!eql.is_empty()).then_some(eql)
}

fn is_focus_object_eql(focus_eqls: &HashSet<String>, object_eql: &str) -> bool {
    normalized_focus_eql(object_eql).is_some_and(|eql| focus_eqls.contains(eql))
}

fn collect_mesh_descendants(
    entity: Entity,
    children: &Query<&Children>,
    mesh_entities: &Query<(), With<Mesh3d>>,
    output: &mut HashSet<Entity>,
) {
    if mesh_entities.contains(entity) {
        output.insert(entity);
    }
    if let Ok(child_list) = children.get(entity) {
        for child in child_list.iter() {
            collect_mesh_descendants(child, children, mesh_entities, output);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorldPosExt;
    use bevy::math::{Mat3, Quat, Vec3};

    fn smoothing_viewport(smoothing: f32) -> Viewport {
        Viewport::new(
            Entity::PLACEHOLDER,
            EditableEQL::default(),
            EditableEQL::default(),
            EditableEQL::default(),
            None,
            smoothing,
        )
    }

    #[test]
    fn smoothing_alpha_is_bounded() {
        for dt in [0.0, 1.0 / 240.0, 1.0 / 30.0, 0.5, 10.0] {
            for tau in [0.05, 0.3, 2.0] {
                let alpha = smoothing_alpha(dt, tau);
                assert!((0.0..=1.0).contains(&alpha), "alpha={alpha} out of bounds");
            }
        }
        assert_eq!(smoothing_alpha(1.0 / 60.0, 0.0), 1.0);
    }

    #[test]
    fn smoothing_alpha_is_framerate_independent() {
        // Two half-steps must land exactly where one full step does.
        let tau = 0.3;
        let dt = 1.0 / 30.0;
        let full = smoothing_alpha(dt, tau);
        let half = smoothing_alpha(dt / 2.0, tau);
        let two_halves = 1.0 - (1.0 - half) * (1.0 - half);
        assert!((full - two_halves).abs() < 1e-12);
    }

    #[test]
    fn smooth_follow_zero_smoothing_is_passthrough() {
        let mut viewport = smoothing_viewport(0.0);
        let target = DVec3::new(1.0, 2.0, 3.0);
        let (pos, look_at, _) = viewport.smooth_follow(target, Some(DVec3::ZERO), None, 1.0 / 60.0);
        assert_eq!(pos, target);
        assert_eq!(look_at, Some(DVec3::ZERO));
    }

    #[test]
    fn smooth_follow_snaps_on_first_frame_then_lags() {
        let mut viewport = smoothing_viewport(0.3);
        let look_at = DVec3::ZERO;
        let start = DVec3::new(10.0, 0.0, 0.0);
        let (pos, _, _) = viewport.smooth_follow(start, Some(look_at), None, 1.0 / 60.0);
        assert_eq!(pos, start, "first frame must snap to the target");

        // A small (sub-framing-distance) jump is eased, not applied 1:1.
        let target = start + DVec3::new(1.0, 0.0, 0.0);
        let (pos, _, _) = viewport.smooth_follow(target, Some(look_at), None, 1.0 / 60.0);
        assert!(pos.x > start.x && pos.x < target.x, "pos={pos} should lag");
    }

    #[test]
    fn smooth_follow_snaps_on_large_jump() {
        let mut viewport = smoothing_viewport(0.3);
        // Camera 10 m behind the vehicle (framing distance 10 m).
        viewport.smooth_follow(
            DVec3::new(10.0, 0.0, 0.0),
            Some(DVec3::ZERO),
            None,
            1.0 / 60.0,
        );

        // Seek: vehicle and camera jump together, far beyond the framing distance.
        let look_at = DVec3::new(500.0, 0.0, 0.0);
        let target = DVec3::new(510.0, 0.0, 0.0);
        let (pos, smoothed_look_at, _) =
            viewport.smooth_follow(target, Some(look_at), None, 1.0 / 60.0);
        assert_eq!(pos, target, "large jumps must snap, not lag");
        assert_eq!(smoothed_look_at, Some(look_at));
    }

    #[test]
    fn smooth_follow_converges_to_static_target() {
        let mut viewport = smoothing_viewport(0.3);
        let target = DVec3::new(5.0, -2.0, 1.0);
        let mut pos = DVec3::ZERO;
        viewport.smooth_follow(
            DVec3::new(4.5, -2.0, 1.0),
            Some(DVec3::ZERO),
            None,
            1.0 / 60.0,
        );
        for _ in 0..600 {
            (pos, _, _) = viewport.smooth_follow(target, Some(DVec3::ZERO), None, 1.0 / 60.0);
        }
        assert!((pos - target).length() < 1e-3, "pos={pos} did not converge");
    }

    /// Continuous fast motion — steps far larger than the framing distance on
    /// every frame — must keep being eased, not snapped. Before the smoothing
    /// fix the framing-distance snap fired every frame during flight so both
    /// viewports jittered identically.
    #[test]
    fn smooth_follow_eases_fast_continuous_motion() {
        let mut viewport = smoothing_viewport(0.3);
        let dt = 1.0 / 60.0;
        // 3 m per frame with a 2 m framing distance: every step exceeds the snap
        // threshold, yet these frames must go through the easing path rather than
        // being snapped. At this speed relative to the framing the filter is
        // deliberately transparent — the lag budget is a couple of centimetres —
        // so the pose itself cannot distinguish the two. The state can: a snap
        // resets the velocity estimate, easing keeps building it.
        let step = DVec3::new(3.0, 0.0, 0.0);
        let mut pos = DVec3::ZERO;
        let mut target = DVec3::ZERO;
        for _ in 0..120 {
            target += step;
            let look_at = target - DVec3::new(2.0, 0.0, 0.0);
            (pos, _, _) = viewport.smooth_follow(target, Some(look_at), None, dt);
        }
        assert!(pos.x > 0.0, "pos should advance");
        assert!(
            pos.x <= target.x,
            "pos={pos} must not overshoot the raw target={target}"
        );
        let state = viewport.smoothing_state.as_ref().expect("state kept");
        assert!(
            state.vel_ema.length() > 0.0,
            "the easing path must have run every frame, not the snap path"
        );
    }

    /// Liftoff regression: through a sustained *acceleration* the subject must
    /// stay near the centre of frame. This is the case a bounded-but-generous lag
    /// budget got wrong — the filter kept working through the climb and parked
    /// the subject several degrees off centre, which read as worse than no
    /// smoothing at all. Fading the filter out early in the speed range is what
    /// keeps it framed.
    #[test]
    fn smooth_follow_keeps_the_subject_framed_through_acceleration() {
        let mut viewport = smoothing_viewport(1.0);
        let dt = 1.0 / 60.0;
        let framing = 35.0;
        let offset = DVec3::new(0.0, -framing, 0.0);
        let mut target = DVec3::ZERO;
        let mut speed = 0.0f64;
        let mut worst = 0.0f64;
        for i in 0..600 {
            speed = (speed + 30.0 * dt).min(120.0); // 3 g climb, then hold
            target += DVec3::new(0.0, 0.0, speed * dt);
            let (pos, look_at, _) = viewport.smooth_follow(target + offset, Some(target), None, dt);
            // Angle between the view axis and the raw subject: the framing error
            // a viewer actually sees. Skip the first frames while the velocity
            // estimate builds.
            if i > 30 {
                let axis = (look_at.unwrap() - pos).normalize();
                let subject = (target - pos).normalize();
                worst = worst.max(axis.dot(subject).clamp(-1.0, 1.0).acos());
            }
        }
        // With a 0.15 budget this sat around `atan(0.15)` ~ 8.5 degrees for the
        // whole climb. The residual now comes from the cutoff opening a little
        // late, by `DERIV_TAU * accel`, so it is worst in the first moments.
        assert!(
            worst.to_degrees() < 3.0,
            "subject drifted {:.2} degrees off centre during acceleration",
            worst.to_degrees()
        );
    }

    /// One Euro win: at high steady speed the follow lag is *bounded* by a
    /// fraction of the framing distance instead of growing like a fixed
    /// first-order filter (`lag = smoothing * speed`). Here `smoothing=0.3` at
    /// 50 m/s would lag a fixed filter by ~15 m; the adaptive filter must stay
    /// far inside the 2 m framing.
    #[test]
    fn smooth_follow_bounds_lag_at_high_speed() {
        let mut viewport = smoothing_viewport(0.3);
        let dt = 1.0 / 60.0;
        let speed = 50.0;
        let framing = 2.0;
        let mut pos = DVec3::ZERO;
        let mut target = DVec3::ZERO;
        for _ in 0..600 {
            target += DVec3::new(speed * dt, 0.0, 0.0);
            let look_at = target - DVec3::new(framing, 0.0, 0.0);
            (pos, _, _) = viewport.smooth_follow(target, Some(look_at), None, dt);
        }
        let lag = target.x - pos.x;
        let fixed_lag = 0.3 * speed; // what the old fixed filter would trail by
        assert!(lag > 0.0, "must still lag (be eased), got {lag}");
        assert!(
            lag < fixed_lag * 0.5,
            "adaptive lag {lag} should be far under the fixed-filter lag {fixed_lag}"
        );
        let bound = framing * FRAMING_LAG_RATIO * CATCHUP_LAG_RATIO;
        assert!(
            lag < bound,
            "adaptive lag {lag} should stay within the framing-error bound {bound}"
        );
    }

    /// A chase camera keeps its subject framed during fast flight. The subject
    /// is drawn from raw telemetry while the camera is eased, so follow lag
    /// shows up directly as an angular framing error — bounding the lag by the
    /// whole framing distance let the subject slide ~30 degrees off centre,
    /// which reads as far worse than the noise being filtered.
    #[test]
    fn smooth_follow_keeps_the_subject_framed_at_high_speed() {
        let dt = 1.0 / 60.0;
        let speed = 250.0;
        // The rig a follow viewport describes: camera offset from the subject in
        // world coordinates, aimed back at the subject.
        let offset = DVec3::new(12.0, 12.0, -8.0);
        let mut viewport = smoothing_viewport(1.0);
        let mut worst = 0.0_f64;
        for i in 0..1200 {
            let subject = DVec3::new(0.0, 0.0, speed * i as f64 * dt);
            let (pos, look_at, _) =
                viewport.smooth_follow(subject + offset, Some(subject), None, dt);
            if i > 600 {
                let axis = (look_at.unwrap() - pos).normalize();
                let to_subject = (subject - pos).normalize();
                worst = worst.max(to_subject.dot(axis).clamp(-1.0, 1.0).acos());
            }
        }
        let limit = FRAMING_LAG_RATIO.atan() * 1.5;
        assert!(
            worst < limit,
            "subject drifted {} deg off centre, past the {} deg framing bound",
            worst.to_degrees(),
            limit.to_degrees()
        );
    }

    /// High-frequency noise on a *stationary* target is strongly damped — the
    /// primary goal, since the floating-origin jitter that shimmers rigid
    /// geometry is worst at rest. The adaptive filter must not read zero-mean
    /// noise as speed and lighten up.
    #[test]
    fn smooth_follow_damps_noise_at_rest() {
        let dt = 1.0 / 60.0;
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut white = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 40) as f64 / (1u64 << 23) as f64) - 1.0
        };
        let mut viewport = smoothing_viewport(0.3);
        let center = DVec3::new(100.0, 0.0, 0.0);
        // The framing distance has to be large compared to the noise: the lag
        // budget is a fraction of it, and a camera cannot filter jitter it is not
        // allowed to trail by. Sub-metre noise at 20 m of framing is the real
        // ratio; the same noise at 2 m would be 16 degrees of angular jitter,
        // which no bounded-framing filter can absorb.
        let look_at = center - DVec3::new(20.0, 0.0, 0.0);
        let mut input_noise = 0.0;
        let mut residual_noise = 0.0;
        for i in 0..2400 {
            let n = DVec3::new(white(), white(), white());
            let (pos, _, _) = viewport.smooth_follow(center + n, Some(look_at + n), None, dt);
            if i > 1200 {
                input_noise += n.length();
                residual_noise += (pos - center).length();
            }
        }
        assert!(
            residual_noise < input_noise * 0.4,
            "residual noise {residual_noise} should be far under the input {input_noise}"
        );
    }

    /// Noise on a slowly moving target is still strongly damped: at low speed
    /// the adaptive cutoff stays near its resting (heavy) value. Isolated from
    /// the velocity lag by differencing the clean and noisy runs of the same
    /// filter — the surviving difference is the noise it let through.
    #[test]
    fn smooth_follow_damps_noise_on_slow_target() {
        let dt = 1.0 / 60.0;
        let speed = 2.0;
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut white = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 40) as f64 / (1u64 << 23) as f64) - 1.0
        };
        let mut vp_noisy = smoothing_viewport(0.3);
        let mut vp_clean = smoothing_viewport(0.3);
        let mut input_noise = 0.0;
        let mut residual_noise = 0.0;
        for i in 0..2400 {
            let clean = DVec3::new(speed * i as f64 * dt, 0.0, 0.0);
            // Framing large compared to the noise, as above.
            let look_clean = clean - DVec3::new(20.0, 0.0, 0.0);
            let n = DVec3::new(white(), white(), white());
            let (p_noisy, _, _) = vp_noisy.smooth_follow(clean + n, Some(look_clean + n), None, dt);
            let (p_clean, _, _) = vp_clean.smooth_follow(clean, Some(look_clean), None, dt);
            if i > 1200 {
                input_noise += n.length();
                residual_noise += (p_noisy - p_clean).length();
            }
        }
        assert!(
            residual_noise < input_noise * 0.4,
            "residual noise {residual_noise} should be far under the input {input_noise}"
        );
    }

    /// With no `look_at` the camera rides the telemetry attitude directly, so
    /// a noisy attitude must be eased rather than passed through raw. Before the
    /// fix `*pos = world_pos` copied the raw attitude while only translation was
    /// smoothed, so the view kept jittering.
    #[test]
    fn smooth_follow_eases_attitude_without_look_at() {
        let mut viewport = smoothing_viewport(0.3);
        let dt = 1.0 / 60.0;
        let pos = DVec3::new(100.0, 0.0, 0.0);
        // First frame snaps to the identity attitude.
        let (_, _, att) = viewport.smooth_follow(pos, None, Some(DQuat::IDENTITY), dt);
        assert_eq!(att, Some(DQuat::IDENTITY));

        // A rotated pose one frame later (below the attitude seek floor) must
        // be eased between the previous and target attitude, not applied 1:1.
        let target = DQuat::from_rotation_z(0.2);
        let (_, look_at, att) = viewport.smooth_follow(pos, None, Some(target), dt);
        assert_eq!(look_at, None, "no look_at stays absent");
        let att = att.expect("attitude is smoothed");
        let eased = att.angle_between(DQuat::IDENTITY);
        let full = target.angle_between(DQuat::IDENTITY);
        assert!(
            eased > 0.0 && eased < full,
            "attitude angle {eased} should lag between identity and target {full}"
        );
    }

    /// A timeline scrub must snap even without a `look_at`: previously the seek
    /// floor defaulted to infinity, so a huge single-frame jump was eased and
    /// the camera glided across the world instead of jumping.
    #[test]
    fn smooth_follow_snaps_on_large_jump_without_look_at() {
        let mut viewport = smoothing_viewport(0.3);
        let dt = 1.0 / 60.0;
        // Seed with slow steady motion so the smoothed step estimate is small.
        let mut p = DVec3::ZERO;
        for _ in 0..10 {
            p += DVec3::new(0.01, 0.0, 0.0);
            viewport.smooth_follow(p, None, Some(DQuat::IDENTITY), dt);
        }
        let target = DVec3::new(1000.0, 0.0, 0.0);
        let (pos, _, _) = viewport.smooth_follow(target, None, Some(DQuat::IDENTITY), dt);
        assert_eq!(pos, target, "a large jump must snap even without a look_at");
    }

    /// A viewport with a fixed `pos` and a telemetry `look_at` must still snap
    /// when the look-at scrubs far in one frame. Seek detection used to watch
    /// only `pos`, so such a scrub slowly panned instead of snapping.
    #[test]
    fn smooth_follow_snaps_on_look_at_jump_with_fixed_pos() {
        let mut viewport = smoothing_viewport(0.3);
        let dt = 1.0 / 60.0;
        let pos = DVec3::ZERO;
        // Settle a stable, nearby look-at (framing ~10 m).
        for _ in 0..15 {
            viewport.smooth_follow(pos, Some(DVec3::new(10.0, 0.0, 0.0)), None, dt);
        }
        // The look-at scrubs far away in one frame while pos stays put.
        let target_look = DVec3::new(1000.0, 0.0, 0.0);
        let (_, look_at, _) = viewport.smooth_follow(pos, Some(target_look), None, dt);
        assert_eq!(
            look_at,
            Some(target_look),
            "a look_at scrub must snap even with a fixed pos"
        );
    }

    /// With a fixed `pos` and no `look_at` the camera rides the pose attitude,
    /// so a scrub that mainly changes orientation (hover, pad dwell) must snap
    /// too. Seek detection used to watch only translation, so the view slowly
    /// slerped across the jump instead.
    #[test]
    fn smooth_follow_snaps_on_attitude_jump_with_fixed_pos() {
        let mut viewport = smoothing_viewport(0.3);
        let dt = 1.0 / 60.0;
        let pos = DVec3::ZERO;
        // Settle on a steady attitude so the angular step estimate is small.
        for _ in 0..15 {
            viewport.smooth_follow(pos, None, Some(DQuat::IDENTITY), dt);
        }
        // Scrub: the orientation flips far in one frame while pos stays put.
        let target = DQuat::from_rotation_z(2.0);
        let (_, _, att) = viewport.smooth_follow(pos, None, Some(target), dt);
        let att = att.expect("attitude present");
        assert!(
            att.angle_between(target) < 1e-9,
            "an attitude scrub must snap, got {} rad short of the target",
            att.angle_between(target)
        );

        // And steady rotation right after must go back to being eased.
        let next = DQuat::from_rotation_z(2.1);
        let (_, _, att) = viewport.smooth_follow(pos, None, Some(next), dt);
        let att = att.expect("attitude present");
        let eased = att.angle_between(target);
        let full = next.angle_between(target);
        assert!(
            eased > 0.0 && eased < full,
            "steady rotation must ease ({eased} of {full} rad), not snap"
        );
    }

    /// A chase camera driven by a `look_at` must ignore telemetry attitude
    /// jumps: its orientation comes from the viewing vector, not the pose
    /// attitude, so an attitude scrub must not force a full snap and hitch a
    /// camera that was only lagging in translation.
    #[test]
    fn smooth_follow_attitude_seek_ignored_with_look_at() {
        let mut viewport = smoothing_viewport(0.3);
        let dt = 1.0 / 60.0;
        let pos = DVec3::new(10.0, 0.0, 0.0);
        let look = DVec3::ZERO;
        // Settle a stable framing (~10 m) with a steady attitude.
        for _ in 0..15 {
            viewport.smooth_follow(pos, Some(look), Some(DQuat::IDENTITY), dt);
        }
        // Attitude flips far in one frame while pos/look_at nudge sub-framing.
        let next_pos = pos + DVec3::new(0.05, 0.0, 0.0);
        let next_look = look + DVec3::new(0.05, 0.0, 0.0);
        let (out_pos, _, _) = viewport.smooth_follow(
            next_pos,
            Some(next_look),
            Some(DQuat::from_rotation_z(2.0)),
            dt,
        );
        let pos_lag = (out_pos - next_pos).length();
        assert!(
            pos_lag > 1e-6,
            "attitude jump must not snap a look_at camera; pos should ease (lag={pos_lag})"
        );
    }

    /// Dragging the playhead makes the target jump every frame. After the first
    /// snap the speed estimate rebuilds over `DERIV_TAU`, so a fixed first-order
    /// filter would trail far behind meanwhile. The catch-up clamp must keep the
    /// lag within `CATCHUP_LAG_RATIO * FRAMING_LAG_RATIO * framing` every frame.
    #[test]
    fn smooth_follow_bounds_lag_during_sustained_scrub() {
        let mut viewport = smoothing_viewport(0.3);
        let dt = 1.0 / 60.0;
        let framing = 2.0;
        let mut target = DVec3::new(100.0, 0.0, 0.0);
        // Settle at rest so the smoothed step estimate is small.
        for _ in 0..30 {
            viewport.smooth_follow(
                target,
                Some(target - DVec3::new(framing, 0.0, 0.0)),
                None,
                dt,
            );
        }
        // Sustained scrub: steps several times the framing distance every frame,
        // consistent enough that the ratio test stops flagging them as seeks.
        let mut max_lag = 0.0_f64;
        for _ in 0..120 {
            target += DVec3::new(10.0, 0.0, 0.0);
            let (pos, _, _) = viewport.smooth_follow(
                target,
                Some(target - DVec3::new(framing, 0.0, 0.0)),
                None,
                dt,
            );
            max_lag = max_lag.max(target.x - pos.x);
        }
        assert!(max_lag > 0.0, "the camera should trail during the scrub");
        let bound = framing * FRAMING_LAG_RATIO * CATCHUP_LAG_RATIO;
        assert!(
            max_lag <= bound + 1e-6,
            "sustained scrub lag {max_lag} must stay within the catch-up bound {bound}"
        );
    }

    #[test]
    fn refresh_default_anchor_depth_uses_viewport_look_at_distance() {
        let mut editor_cam = EditorCam::default();

        refresh_default_anchor_depth(&mut editor_cam, 14.696_938);

        assert!((editor_cam.last_anchor_depth + 14.696_938).abs() < 1.0e-9);
    }

    #[test]
    fn refresh_default_anchor_depth_keeps_user_adjusted_depth() {
        let mut editor_cam = EditorCam {
            last_anchor_depth: -8.0,
            ..Default::default()
        };

        refresh_default_anchor_depth(&mut editor_cam, 14.696_938);

        assert_eq!(editor_cam.last_anchor_depth, -8.0);
    }

    #[test]
    fn refresh_default_anchor_depth_rejects_invalid_distance() {
        for target_distance in [0.0, f64::NAN, f64::INFINITY] {
            let mut editor_cam = EditorCam::default();

            refresh_default_anchor_depth(&mut editor_cam, target_distance);

            assert_eq!(
                editor_cam.last_anchor_depth,
                DEFAULT_EDITOR_CAM_ANCHOR_DEPTH
            );
        }
    }

    /// Full pipeline test with real EQL: `set_viewport_pos` -> `sync_pos` ->
    /// `apply_geo_rotation`. A NED viewport with an explicit `up="(0,0,-1)"`
    /// (up, away from the ground in NED) must produce a right-side-up rig
    /// looking at the target; `up="(0,0,1)"` (down) must produce an inverted
    /// one.
    #[test]
    fn ned_viewport_explicit_up_through_pipeline() {
        use crate::object_3d::{EditableEQL, compile_eql_expr};
        use bevy::math::{DQuat, DVec3};
        use bevy::prelude::{IntoScheduleConfigs, Transform};
        use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};

        let eql_ctx = eql::Context::default();
        let editable = |s: &str| EditableEQL {
            eql: s.to_string(),
            compiled_expr: Some(
                compile_eql_expr(
                    eql_ctx
                        .parse_str(s)
                        .unwrap_or_else(|e| panic!("parse {s:?}: {e}")),
                )
                .unwrap_or_else(|e| panic!("compile {s:?}: {e}")),
            ),
        };

        for (up_eql, expect_up_y) in [("(0,0,-1)", 1.0f32), ("(0,0,1)", -1.0f32)] {
            let mut app = bevy::app::App::new();
            app.insert_resource(GeoContext::default());
            app.init_resource::<super::EntityMap>();
            app.init_resource::<Time>();
            crate::register_world_pos_components(&mut app);
            app.add_systems(
                bevy::app::Update,
                (
                    super::set_viewport_pos,
                    crate::sync_pos,
                    bevy_geo_frames::apply_geo_rotation,
                )
                    .chain(),
            );

            let frame = GeoFrame::NED;
            let parent = app
                .world_mut()
                .spawn((
                    super::WorldPos::default(),
                    GeoPosition(frame, DVec3::ZERO),
                    GeoRotation::relative(frame, DQuat::IDENTITY),
                ))
                .id();
            // Values from the failing ball.kdl viewport.
            let viewport_entity = app
                .world_mut()
                .spawn((
                    super::Viewport::new(
                        parent,
                        editable("(0,0,0,0, 0,0,0)"),
                        editable("(0,0,0,0, 0,-3,0)"),
                        editable(up_eql),
                        Some(frame),
                        0.0,
                    ),
                    EditorCam::default(),
                ))
                .id();
            app.update();

            let editor_cam = app.world().get::<EditorCam>(viewport_entity).unwrap();
            assert!(
                (editor_cam.last_anchor_depth + 3.0).abs() < 1e-9,
                "up={up_eql}: viewport positioning system did not run"
            );
            let transform = *app.world().get::<Transform>(parent).unwrap();
            let up = transform.rotation * Vec3::Y;
            assert!(
                up.y * expect_up_y > 0.5,
                "up={up_eql}: rig up {up:?}, expected y sign {expect_up_y}"
            );
            // NED (0,-3,0) is 3 m west of the origin => bevy -X.
            let fwd = transform.rotation * Vec3::NEG_Z;
            assert!(
                (fwd.x - -1.0).abs() < 1e-5 && fwd.y.abs() < 1e-5,
                "up={up_eql}: camera fwd = {fwd:?}, expected -X"
            );
        }
    }

    /// A `look_at` that resolves to something other than a 7-vector pose — a
    /// bare 3-vector, which the reference docs still show — must not be mistaken
    /// for a missing sample. Gap handling freezes the pose, so treating it as one
    /// would strand the viewport on its first position forever, whatever the
    /// `smoothing` value.
    #[test]
    fn non_pose_look_at_still_tracks_pos() {
        use crate::object_3d::{EditableEQL, compile_eql_expr};
        use bevy::math::{DQuat, DVec3};
        use bevy::prelude::IntoScheduleConfigs;
        use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};

        let eql_ctx = eql::Context::default();
        let editable = |s: &str| EditableEQL {
            eql: s.to_string(),
            compiled_expr: Some(
                compile_eql_expr(
                    eql_ctx
                        .parse_str(s)
                        .unwrap_or_else(|e| panic!("parse {s:?}: {e}")),
                )
                .unwrap_or_else(|e| panic!("compile {s:?}: {e}")),
            ),
        };

        for smoothing in [0.0, 1.0] {
            let mut app = bevy::app::App::new();
            app.insert_resource(GeoContext::default());
            app.init_resource::<super::EntityMap>();
            app.init_resource::<Time>();
            crate::register_world_pos_components(&mut app);
            app.add_systems(
                bevy::app::Update,
                (super::set_viewport_pos, crate::sync_pos).chain(),
            );

            let frame = GeoFrame::ENU;
            let parent = app
                .world_mut()
                .spawn((
                    super::WorldPos::default(),
                    GeoPosition(frame, DVec3::ZERO),
                    GeoRotation::relative(frame, DQuat::IDENTITY),
                ))
                .id();
            let viewport_entity = app
                .world_mut()
                .spawn((
                    super::Viewport::new(
                        parent,
                        editable("(0,0,0,1, 1,2,3)"),
                        // Three elements: executes fine, but is not a pose.
                        editable("(0,0,0)"),
                        editable("(0,0,1)"),
                        Some(frame),
                        smoothing,
                    ),
                    EditorCam::default(),
                ))
                .id();
            app.update();

            let pos = app.world().get::<super::WorldPos>(parent).unwrap().pos();
            assert!(
                (pos - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-9,
                "smoothing={smoothing}: pos should follow the expression, got {pos:?}"
            );
            assert!(
                app.world().get::<EditorCam>(viewport_entity).is_some(),
                "viewport should still exist"
            );
        }
    }

    macro_rules! assert_eq_mat {
        ($a:expr, $b:expr $(,)?) => {{
            assert_eq_mat!($a, $b, "");
        }};
        ($a:expr, $b:expr, $($arg:tt)+) => {{
            let a = $a;
            let b = $b;

            for i in 0..3 {
                let aa = a.col(i);
                let bb = b.col(i);
                for j in 0..3 {
                    let delta = aa[j] - bb[j];
                    if delta.abs() > 1e-5 {
                        panic!("First mismatch on column {}:\nleft:  {}\nright: {}: {}",
                                i + 1, a, b, format_args!($($arg)+));
                    }
                }
            }
        }};
    }
    macro_rules! assert_eq_vec {
        ($a:expr, $b:expr $(,)?) => {{
            assert_eq_vec!($a, $b, "");
        }};
        ($a:expr, $b:expr, $($arg:tt)+) => {{
            let a = $a;
            let b = $b;

            for i in 0..3 {
                let delta = a[i] - b[i];
                if delta.abs() > 1e-5 {
                    panic!("First mismatch on index {}:\nleft:  {}\nright: {}: {}",
                           i, a, b, format_args!($($arg)+));
                }
            }
        }};
    }
    macro_rules! assert_eq_quat {
        ($a:expr, $b:expr $(,)?) => {{
            assert_eq_quat!($a, $b, "");
        }};

        ($a:expr, $b:expr, $($arg:tt)+) => {{

            let a = $a;
            let b = $b;


            let dot = a.dot(b).abs();

            assert!(
                (1.0 - dot) <= 1e-5,
                "Quat mismatch:\nleft:  {:?}\nright: {:?}: {}",
                a,
                b,
                format_args!($($arg)+)
            );
        }};
    }

    #[inline]
    fn are_collinear(a: Vec3, b: Vec3) -> bool {
        a.cross(b).length_squared() < 1e-6
    }

    /// Constructs a look_at rotation matrix that matches
    /// [nox::Matrix3::look_at_rh].
    fn glam_look_at_rh(dir: Vec3, up: Vec3) -> (Mat3, Vec3) {
        let up_candidates = [up, Vec3::Y, Vec3::X, Vec3::Z];
        let up = up_candidates
            .into_iter()
            .find(|up| !are_collinear(*up, dir))
            .expect("it can't be collinear with everyone");
        // Constructs a look_at rotation matrix using the same algorithm as
        // nox::Matrix3::look_at_rh.
        //
        // let f = dir.normalize();
        // let s = f.cross(up).normalize();
        // let u = s.cross(f);
        // // nox uses from_rows then transpose, which equals from_cols
        // Mat3::from_cols(s, f, u)
        (Mat3::look_to_rh(dir, up), up)
    }

    /// This function converts an Elodin rotation matrix to an EUS/Bevy rotation
    /// matrix and vice versa. It behaves as though one right-multiplied M by
    /// bevy_R_enu and transposed, i.e., (M * bevy_R_elodin)^T but no actual matrix
    /// multiplication happens because column re-ordering is faster.
    ///
    ///
    /// ```ignore
    ///   elodin_R_bevy =  [ 1  0  0 ]
    ///                    [ 0  0 -1 ]
    ///                    [ 0  1  0 ]
    /// ```
    ///
    /// Note: It's orthonormal, so its transpose is its inverse.
    #[inline]
    fn elodin_R_bevy(M: Mat3) -> Mat3 {
        // Bevy +X -> ENU East
        // Bevy +Y -> ENU Up
        // Bevy +Z -> -ENU North
        Mat3::from_cols(M.x_axis, M.z_axis, -M.y_axis).transpose()
    }

    // #[inline]
    fn bevy_R_elodin(M: Mat3) -> Mat3 {
        // ENU East  -> Bevy +X
        // ENU North -> Bevy -Z
        // ENU Up    -> Bevy +Y
        let M = M.transpose();
        Mat3::from_cols(M.x_axis, -M.z_axis, M.y_axis)
    }

    #[test]
    fn test_inverses() {
        let A = Mat3::from_cols(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        let B = elodin_R_bevy(A);
        let C = bevy_R_elodin(B);
        assert_eq_mat!(A, C, "b to e to b");
        let B = bevy_R_elodin(A);
        let C = elodin_R_bevy(B);
        assert_eq_mat!(A, C, "e to b to e");
    }

    /// Compare against elodin's ENU.
    #[test]
    fn test_look_at_rh_nox_vs_glam_elodin() {
        test_look_at_rh_nox_vs_glam(
            |glam_mat, nox_mat| (elodin_R_bevy(glam_mat), nox_mat),
            |M| M,
        );
    }

    /// Compare against Bevy's EUS.
    #[test]
    fn test_look_at_rh_nox_vs_glam_bevy() {
        test_look_at_rh_nox_vs_glam(
            |glam_mat, nox_mat| (glam_mat, bevy_R_elodin(nox_mat)),
            elodin_R_bevy,
        );
    }

    /// `WorldPosExt::bevy_att` must match `GeoRotation::to_bevy` in plane mode.
    #[test]
    fn test_bevy_att_vs_geo_frames_plane() {
        use bevy_geo_frames::{GeoContext, GeoFrame, GeoRotation, Present};

        let ctx = GeoContext::default().with_present(Present::Plane);

        for (i, (dir, up)) in look_at_test_cases().into_iter().enumerate() {
            let nox_dir = nox::Vec3::from(dir.as_dvec3());
            let nox_up = nox::Vec3::from(up.as_dvec3());
            let (nox_mat, _) = nox::Matrix3::look_at_rh_up(nox_dir, nox_up);
            let nox_quat = nox::Quaternion::from_rot_mat(nox_mat);
            let world_pos = super::WorldPos {
                att: nox_quat,
                pos: nox::Vec3::new(0.0, 0.0, 0.0),
            };

            let elodin_bevy = world_pos.bevy_att();
            let geo_frames_bevy =
                GeoRotation::relative(GeoFrame::ENU, world_pos.att()).to_bevy(&ctx);
            assert_eq_quat!(
                elodin_bevy.as_quat(),
                geo_frames_bevy.as_quat(),
                "case {i} dir {dir} up {up}"
            );
        }
    }

    #[test]
    fn focus_object_eql_matches_trimmed_viewport_look_at() {
        let focus_eqls = HashSet::from(["lander.world_pos".to_string()]);

        assert!(is_focus_object_eql(&focus_eqls, " lander.world_pos "));
        assert!(!is_focus_object_eql(&focus_eqls, "lander_truth.world_pos"));
        assert!(!is_focus_object_eql(&focus_eqls, ""));
    }
    #[test]
    fn test_from_mat3() {
        let q = Quat::from_mat3(&Mat3::IDENTITY);
        assert_eq_quat!(q, Quat::IDENTITY);

        let dir = Vec3::Y;
        let up = Vec3::Z;
        let (M, _) = glam_look_at_rh(dir, up);
        assert_eq_mat!(M, Mat3::from_cols(Vec3::X, Vec3::NEG_Z, Vec3::Y));
        let q = Quat::from_mat3(&Mat3::IDENTITY);
        assert_eq_quat!(q, Quat::IDENTITY);
        let S = elodin_R_bevy(M).transpose();
        assert_eq_mat!(S, Mat3::IDENTITY);

        assert_eq_mat!(
            Mat3::from_cols(Vec3::NEG_X, Vec3::NEG_Y, Vec3::Z),
            glam_look_at_rh(Vec3::NEG_Z, Vec3::NEG_Y).0,
            "trial 0"
        );
        // assert_eq_mat!(Mat3::from_cols(Vec3::NEG_X, Vec3::NEG_Y, Vec3::Z), glam_look_at_rh(Vec3::Z, Vec3::Y).0, "current");
    }

    fn look_at_test_cases() -> [(Vec3, Vec3); 10] {
        [
            (Vec3::new(0.0, 1.0, 0.0), Vec3::Z), // 0: identity for Elodin ENU
            (Vec3::NEG_Z, Vec3::Y),              // 1: identity for Bevy
            (Vec3::new(0.0, 1.0, 0.0), Vec3::Z),
            (Vec3::new(0.0, 0.0, 1.0), Vec3::Y),
            (Vec3::new(1.0, 2.0, 3.0).normalize(), Vec3::Y),
            (Vec3::new(-1.0, 0.5, 0.3).normalize(), Vec3::Y),
            (Vec3::new(0.0, 0.0, -1.0), Vec3::Y),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::Y),
            (Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
            (Vec3::new(0.0, 0.0, 1.0), Vec3::Z),
        ]
    }

    fn test_look_at_rh_nox_vs_glam(f: fn(Mat3, Mat3) -> (Mat3, Mat3), g: fn(Mat3) -> Mat3) {
        for (i, (dir, up)) in look_at_test_cases().into_iter().enumerate() {
            let nox_dir = nox::Vec3::from(dir.as_dvec3());
            let nox_up = nox::Vec3::from(up.as_dvec3());

            let (nox_mat, nox_up_actual) = nox::Matrix3::look_at_rh_up(nox_dir, nox_up);
            let (glam_mat, up_actual) = glam_look_at_rh(dir, up);
            assert_eq_vec!(
                up_actual,
                bevy::math::DVec3::from(nox_up_actual).as_vec3(),
                "case {i} nox and bevy up vector don't match"
            );
            // let glam_mat = elodin_R_bevy(glam_mat);
            let nox_mat_bevy: bevy::math::Mat3 = bevy::math::DMat3::from(nox_mat).as_mat3();
            // let nox_mat_bevy = bevy_R_elodin(nox_mat_bevy);
            let (glam_mat, _nox_mat_bevy) = f(glam_mat, nox_mat_bevy);

            // Weird thing. The matrices are not always the same but the
            // quaternions are.
            // assert_eq_mat!(nox_mat_bevy, glam_mat, "\ncase {i} dir {dir} up {up}");

            // Also compare resulting quaternions
            let nox_quat_look_at = nox::Quaternion::look_at_rh(nox_dir, nox_up);
            let nox_quat = nox::Quaternion::from_rot_mat(nox_mat);
            assert_eq_quat!(
                bevy::math::DQuat::from(nox_quat),
                bevy::math::DQuat::from(nox_quat_look_at),
                "case {i} look_at_rh vs mat"
            );
            let world_pos = super::WorldPos {
                att: nox_quat,
                pos: nox::Vec3::new(0.0, 0.0, 0.0),
            };
            let nox_quat_bevy = world_pos.bevy_att();
            // let glam_quat = Quat::from_mat3(&elodin_R_bevy(glam_mat).transpose());
            let glam_quat = Quat::from_mat3(&g(glam_mat));
            assert_eq_quat!(nox_quat_bevy.as_quat(), glam_quat, "case {i} second");
        }
    }

    fn focus_object_state(eql: &str) -> Object3DState {
        use impeller2_wkt::{Object3D, Object3DMesh};
        Object3DState {
            compiled_expr: None,
            scale_expr: None,
            scale_error: None,
            error_covariance_cholesky_expr: None,
            error_covariance_expr: None,
            joint_animations: Vec::new(),
            data: Object3D {
                eql: eql.to_string(),
                mesh: Object3DMesh::glb("model.glb"),
                frame: None,
                frame_orientation: None,
                orientation: Default::default(),
                icon: None,
                thrusters: Vec::new(),
                mesh_visibility_range: None,
                node_id: Default::default(),
            },
        }
    }

    fn focus_viewport(eql: &str, parent_entity: Entity) -> Viewport {
        Viewport {
            parent_entity,
            pos: EditableEQL {
                eql: String::new(),
                compiled_expr: None,
            },
            look_at: EditableEQL {
                eql: eql.to_string(),
                compiled_expr: None,
            },
            up: EditableEQL {
                eql: String::new(),
                compiled_expr: None,
            },
            frame: None,
            smoothing: 0.0,
            smoothing_state: None,
        }
    }

    /// Regression for the panic where a focus mesh entity is despawned between
    /// `sync_viewport_focus_pick_targets` queueing its `insert` and the deferred
    /// commands applying (e.g. skybox generation reloads the schematic). The
    /// queued command must be silenced rather than panic on the dead entity.
    #[test]
    #[allow(clippy::type_complexity)]
    fn sync_viewport_focus_pick_targets_survives_despawned_target() {
        let mut world = World::new();
        let parent = world.spawn(focus_object_state("e.world_pos")).id();
        let mesh = world.spawn(Mesh3d(Handle::default())).id();
        world.entity_mut(parent).add_child(mesh);
        let viewport_parent = world.spawn_empty().id();
        world.spawn(focus_viewport("e.world_pos", viewport_parent));

        let mut state: SystemState<(
            Commands,
            Query<&Viewport>,
            Query<(Entity, &Object3DState)>,
            Query<&Children>,
            Query<(), With<Mesh3d>>,
            Query<Entity, With<ViewportFocusPickTarget>>,
        )> = SystemState::new(&mut world);

        {
            let (commands, viewports, objects, children, mesh_entities, current_targets) =
                state.params_mut(&mut world);
            sync_viewport_focus_pick_targets(
                commands,
                viewports,
                objects,
                children,
                mesh_entities,
                current_targets,
            );
        }

        // Despawn the target after the insert is queued but before it applies.
        world.entity_mut(mesh).despawn();
        state.apply(&mut world);

        assert!(world.get_entity(mesh).is_err());
    }
}
