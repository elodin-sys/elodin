//! Horizon gauge: a classic artificial horizon (attitude indicator), driven by
//! an EQL-bound pose. Unlike the [orientation gauge](super::orientation), whose
//! markings co-rotate with the body, this is a cockpit view — the world
//! counter-rotates behind a screen-fixed aircraft symbol.
//!
//! The instrument only reads when a **local vertical** exists: NED/ENU are
//! tangent frames that carry their own vertical, whereas ECEF is fixed to the
//! whole body and needs the pose's position to know which way is up. Anything
//! else draws an explicit inactive face — never a fake wings-level.

use bevy::{
    ecs::system::SystemParam,
    math::{DQuat, DVec3},
    prelude::*,
};
use bevy_egui::egui::{self, Align2, Color32, FontId, Pos2, Sense, Shape, Stroke, Vec2};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoOrigin, approx_radius, ecef_to_lla_deg};
use impeller2_bevy::{EntityMap, TelemetryCache};
use impeller2_wkt::{ComponentValue, CurrentTimestamp};

use super::{BARE_QUAT_UNIT_TOLERANCE, EqlBinding, GaugePane, gauge_title, text_with_halo};
use crate::ui::{
    colors::get_scheme,
    widgets::{SystemStateExt, WidgetSystem},
};

/// Backing data for a horizon gauge pane; the EQL lives in the sibling
/// [`EqlBinding`] component.
///
/// There is deliberately no `display` field: a horizon is intrinsically local,
/// so the frame choice only matters as the `source` of the incoming attitude.
#[derive(Component)]
pub struct HorizonGaugeData {
    /// Frame the EQL pose is expressed in: the quaternion rotates body→source,
    /// and a 7-vector's position tail is read in that same frame — as in the
    /// [geo-position gauge](super::geo_position), `source` describes the data,
    /// not just the rotation. Overriding it to ECEF therefore asserts the
    /// positions are ECEF, whatever the scene's `coordinate` is.
    ///
    /// `None` means inherit the schematic global [`crate::Coordinate`] (same as
    /// omitting `source` in KDL), resolved via [`Self::effective_source`].
    pub source: Option<GeoFrame>,
    /// Attitude (body→source) the gauge reads as level: it renders
    /// `q · reference⁻¹`. Identity means the raw component attitude.
    pub reference: DQuat,
}

impl HorizonGaugeData {
    pub fn new(source: Option<GeoFrame>) -> Self {
        Self {
            source,
            reference: DQuat::IDENTITY,
        }
    }

    /// Set the level-attitude quaternion from its KDL form (`[x, y, z, w]`,
    /// normalized here so hand-written schematics don't need to be exact).
    pub fn with_reference(mut self, reference: Option<[f64; 4]>) -> Self {
        if let Some([x, y, z, w]) = reference {
            let q = DQuat::from_xyzw(x, y, z, w);
            if q.length() > 1e-9 {
                self.reference = q.normalize();
            }
        }
        self
    }

    /// KDL form of [`Self::reference`]: `None` when it is (numerically) the
    /// identity, so the common case serializes to nothing.
    pub fn reference_kdl(&self) -> Option<[f64; 4]> {
        let q = self.reference;
        (q.dot(DQuat::IDENTITY).abs() < 1.0 - 1e-9).then_some([q.x, q.y, q.z, q.w])
    }

    /// Concrete source frame: explicit override, else schematic `coordinate`,
    /// else ENU (same fallback as the sibling gauges).
    pub fn effective_source(&self, coordinate: Option<GeoFrame>) -> GeoFrame {
        self.source.or(coordinate).unwrap_or(GeoFrame::ENU)
    }
}

/// Why the instrument cannot show an attitude. Each variant is surfaced to the
/// user, so a blank gauge is always explained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InactiveReason {
    /// No telemetry sample at the playhead, or the EQL does not resolve.
    NoSample,
    /// The value carries no attitude (position-only, or a non-unit 4-vector).
    NoAttitude,
    /// ECEF attitude without a position: no way to know which way is up.
    EcefWithoutPosition,
    /// Too close to the body's centre for a vertical to be meaningful.
    NearBodyCenter,
}

impl InactiveReason {
    /// Short explanation drawn under the empty face.
    pub fn label(self) -> &'static str {
        match self {
            InactiveReason::NoSample => "no sample",
            InactiveReason::NoAttitude => "no attitude",
            InactiveReason::EcefWithoutPosition => "ECEF needs a pose",
            InactiveReason::NearBodyCenter => "no local vertical",
        }
    }
}

/// Attitude relative to the local level plane, in radians. Positive pitch is
/// nose-up, positive roll is right-wing-down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attitude {
    pub pitch: f64,
    pub roll: f64,
}

/// A pose sample: attitude plus, for `world_pos`-style 7-vectors, position.
struct Pose {
    q: DQuat,
    pos: Option<DVec3>,
}

/// Extract a pose from a component value, accepting (in `F32` or `F64`) a
/// `SpatialTransform`/[`WorldPos`](impeller2_wkt::WorldPos) 7-vector (quaternion
/// head + position tail) or a bare, near-unit-length 4-vector quaternion.
///
/// The unit-length gate on bare 4-vectors mirrors the orientation gauge: an
/// arbitrary 4-element component (e.g. fin deflections) must not be normalized
/// into a plausible-looking attitude.
fn component_value_to_pose(value: &ComponentValue) -> Option<Pose> {
    let data = super::component_buf_f64(value)?;
    if data.len() >= 7 {
        let q = DQuat::from_xyzw(data[0], data[1], data[2], data[3]);
        return (q.length_squared() > 1e-12).then(|| Pose {
            q: q.normalize(),
            pos: Some(DVec3::new(data[4], data[5], data[6])),
        });
    }
    if data.len() == 4 {
        let q = DQuat::from_xyzw(data[0], data[1], data[2], data[3]);
        return ((q.length_squared() - 1.0).abs() <= BARE_QUAT_UNIT_TOLERANCE).then(|| Pose {
            q: q.normalize(),
            pos: None,
        });
    }
    None
}

/// The body's own "up" axis, numerically, for a body whose identity attitude
/// aligns it with `frame` — i.e. Z-up frames give a Z-up body, NED gives the
/// aerospace Z-down body. The nose is always the frame's `+X`.
///
/// Distinct from [`super::orientation`]'s display triad: that describes the
/// axes the gimbal *draws*, this one the vehicle's own axes.
fn body_up_axis(frame: GeoFrame) -> DVec3 {
    match frame {
        GeoFrame::NED => DVec3::NEG_Z,
        GeoFrame::ENU | GeoFrame::ECEF => DVec3::Z,
    }
}

/// Unit "up" of the local level plane, expressed in `source` coordinates.
///
/// NED/ENU are local tangent frames, so their vertical is a frame constant. In
/// ECEF the vertical is the geodetic ellipsoid normal **at the vehicle's own
/// position** — using the schematic origin instead would tilt the horizon by
/// the angle travelled over the body.
fn local_vertical(
    source: GeoFrame,
    pos: Option<DVec3>,
    ctx: &GeoContext,
) -> Result<DVec3, InactiveReason> {
    match source {
        GeoFrame::NED => Ok(DVec3::NEG_Z),
        GeoFrame::ENU => Ok(DVec3::Z),
        GeoFrame::ECEF => {
            let pos = pos.ok_or(InactiveReason::EcefWithoutPosition)?;
            // Deep inside the body the geodetic normal is meaningless (and
            // numerically unstable); refuse rather than invent an orientation.
            if pos.length() < 0.5 * approx_radius(&ctx.origin.ellipsoid) {
                return Err(InactiveReason::NearBodyCenter);
            }
            let (lat, lon, _alt) = ecef_to_lla_deg(pos, &ctx.origin.ellipsoid);
            let here =
                GeoOrigin::new_from_degrees(lat, lon, 0.0).with_ellipsoid(ctx.origin.ellipsoid);
            // The ENU basis at `here`, expressed in ECEF: its +U column is the
            // geodetic normal.
            Ok(GeoFrame::ecef_R_(&GeoFrame::ENU, &here) * DVec3::Z)
        }
    }
}

/// Resolve a sample into pitch/roll against the local level plane, or the
/// reason the instrument stays blank.
fn solve_attitude(
    value: Option<&ComponentValue>,
    source: GeoFrame,
    reference: DQuat,
    ctx: &GeoContext,
) -> Result<Attitude, InactiveReason> {
    let value = value.ok_or(InactiveReason::NoSample)?;
    let pose = component_value_to_pose(value).ok_or(InactiveReason::NoAttitude)?;
    let vertical = local_vertical(source, pose.pos, ctx)?;

    // Rotation of the body in source space since the reference pose, so a
    // vehicle sitting at its reference attitude reads level.
    let delta = pose.q * reference.inverse();
    Ok(attitude_against(delta, source, vertical))
}

/// Pitch and roll of a body rotated by `delta` against a given local vertical.
///
/// Both come from resolving the vertical onto the body axes rather than from
/// Euler extraction, so they stay well-conditioned and sign-correct: positive
/// pitch is nose-up, positive roll is right-wing-down.
fn attitude_against(delta: DQuat, source: GeoFrame, vertical: DVec3) -> Attitude {
    let nose = delta * DVec3::X;
    let up = delta * body_up_axis(source);
    let right = nose.cross(up);
    Attitude {
        pitch: nose.dot(vertical).clamp(-1.0, 1.0).asin(),
        roll: (-right.dot(vertical)).atan2(up.dot(vertical)),
    }
}

/// Half-angle of pitch visible between the face centre and its rim. Small
/// attitudes are only legible as a symbol-to-horizon offset, so the scale is
/// chosen (rather than inherited from the rung spacing) to keep a few degrees
/// clearly visible; the numeric readout carries the precision.
const PITCH_HALF_RANGE_DEG: f64 = 35.0;

/// Screen-space (y-down) unit directions of the rolled world group: its "up"
/// and its "right" along the horizon.
///
/// A right bank (positive roll) tilts the world's up to the *left*, so the
/// horizon's right end rises — the world counter-rotates behind the fixed
/// aircraft symbol.
fn world_dirs(roll: f64) -> (Vec2, Vec2) {
    let (s, c) = (roll.sin() as f32, roll.cos() as f32);
    (Vec2::new(-s, -c), Vec2::new(c, -s))
}

/// Which part of a disc a half-plane covers.
enum DiscSplit {
    /// The half-plane covers the whole disc.
    All,
    /// It covers none of it.
    Empty,
    /// It covers this circular segment (convex, screen coords).
    Segment(Vec<Pos2>),
}

/// Split a disc by the line through `line_point`, keeping the `normal` side.
fn half_disc(center: Pos2, radius: f32, line_point: Pos2, normal: Vec2) -> DiscSplit {
    // Signed distance of the centre from the line, along `normal`.
    let h = (center - line_point).dot(normal);
    if h >= radius {
        return DiscSplit::All;
    }
    if h <= -radius {
        return DiscSplit::Empty;
    }
    // Circle points are `center + radius * (normal cos t + tangent sin t)`, and
    // sit on the kept side where `cos t >= -h / radius`.
    let tangent = Vec2::new(-normal.y, normal.x);
    let t0 = (-h / radius).clamp(-1.0, 1.0).acos();
    let steps = ((t0 / 0.08).ceil() as usize).max(2);
    let pts = (0..=2 * steps)
        .map(|i| {
            let t = -t0 + 2.0 * t0 * (i as f32 / (2 * steps) as f32);
            center + (normal * t.cos() + tangent * t.sin()) * radius
        })
        .collect();
    DiscSplit::Segment(pts)
}

/// Chord where the line through `line_point` with the given `normal` crosses
/// the circle, or `None` when it misses.
fn line_chord(center: Pos2, radius: f32, line_point: Pos2, normal: Vec2) -> Option<(Pos2, Pos2)> {
    let h = (center - line_point).dot(normal);
    if h.abs() >= radius {
        return None;
    }
    let half = (radius * radius - h * h).max(0.0).sqrt();
    let tangent = Vec2::new(-normal.y, normal.x);
    let mid = center - normal * h;
    Some((mid - tangent * half, mid + tangent * half))
}

/// Clip a segment to a circle, or `None` when it stays entirely outside.
fn clip_segment(center: Pos2, radius: f32, a: Pos2, b: Pos2) -> Option<(Pos2, Pos2)> {
    let d = b - a;
    let f = a - center;
    let aa = d.dot(d);
    if aa <= f32::EPSILON {
        return (f.length() <= radius).then_some((a, b));
    }
    let disc = d.dot(f) * d.dot(f) - aa * (f.dot(f) - radius * radius);
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t0 = ((-d.dot(f) - sq) / aa).max(0.0);
    let t1 = ((-d.dot(f) + sq) / aa).min(1.0);
    (t0 < t1).then(|| (a + d * t0, a + d * t1))
}

/// Sky, ground and marking tones for the current theme.
fn horizon_palette() -> (Color32, Color32, Color32) {
    if crate::ui::colors::is_light_mode() {
        (
            Color32::from_rgb(0x8F, 0xA3, 0xB8),
            Color32::from_rgb(0xA8, 0x96, 0x8A),
            Color32::from_rgb(0x1A, 0x1A, 0x1A),
        )
    } else {
        (
            Color32::from_rgb(0x3C, 0x4A, 0x5A),
            Color32::from_rgb(0x4A, 0x40, 0x38),
            Color32::from_rgb(0xFF, 0xFB, 0xF0),
        )
    }
}

/// Amber of the screen-fixed symbology, shared with the gimbal's reticle.
const SYMBOL: Color32 = Color32::from_rgb(255, 179, 0);

/// Bank angles ticked on the roll scale (mirrored either side of zero).
const BANK_TICKS_DEG: [f64; 5] = [10.0, 20.0, 30.0, 45.0, 60.0];

/// Type size of the readout under the face, and the gap above it.
const READOUT_FONT_SIZE: f32 = 10.0;
const READOUT_GAP: f32 = 3.0;

/// Largest face we draw. The instrument follows its pane — pitch and bank are
/// only as readable as the face is big — but a maximised tile should not turn
/// into a wall of sky.
const FACE_MAX: f32 = 320.0;
/// Below this the ladder and roll scale collapse into noise, so the face is
/// dropped and the pane keeps the numbers alone.
const FACE_MIN: f32 = 60.0;
/// Face the symbology's type and line weights are quoted for; they scale from
/// here so a bigger instrument reads bigger instead of just emptier.
const REFERENCE_FACE: f32 = 170.0;

/// Side of the square face for a pane of `avail`, once `readout_room` is set
/// aside for the numbers underneath. `None` when what's left is too small to
/// draw a legible instrument — the readout outranks the face, since a horizon
/// nobody can read is worth less than the two figures it stands for.
fn face_size(avail: Vec2, readout_room: f32) -> Option<f32> {
    let size = avail.x.min(avail.y - readout_room).min(FACE_MAX);
    (size >= FACE_MIN).then_some(size)
}

/// Scale of type and line weights for a given face, damped and bounded so the
/// extremes stay legible rather than tracking the face exactly.
fn symbol_scale(size: f32) -> f32 {
    (size / REFERENCE_FACE).clamp(0.85, 1.5)
}

/// Paint the instrument face: two-tone sky/ground split by the horizon, a pitch
/// ladder and roll scale, and the fixed aircraft symbol. An inactive state
/// draws a dimmed empty rim with its reason.
///
/// Draws nothing at all when the pane is too short to hold both the face and
/// the readout the caller adds underneath.
fn paint_horizon(ui: &mut egui::Ui, state: Result<Attitude, InactiveReason>) {
    let scheme = get_scheme();
    let readout_room =
        READOUT_GAP + ui.fonts_mut(|fonts| fonts.row_height(&FontId::monospace(READOUT_FONT_SIZE)));
    let Some(size) = face_size(ui.available_size(), readout_room) else {
        return;
    };
    let k = symbol_scale(size);
    let avail = ui.available_width();
    let (full_rect, _response) =
        ui.allocate_exact_size(Vec2::new(avail.max(size), size), Sense::hover());
    let rect = egui::Rect::from_center_size(full_rect.center(), Vec2::splat(size));
    let painter = ui.painter_at(rect);
    let center = rect.center();
    let radius = size * 0.42;

    let attitude = match state {
        Ok(attitude) => attitude,
        Err(reason) => {
            painter.circle_stroke(
                center,
                radius,
                Stroke::new(1.5 * k, scheme.border_primary.gamma_multiply(0.5)),
            );
            painter.text(
                center,
                Align2::CENTER_CENTER,
                "—",
                FontId::monospace(15.0 * k),
                scheme.text_secondary,
            );
            painter.text(
                Pos2::new(center.x, center.y + radius * 0.55),
                Align2::CENTER_CENTER,
                reason.label(),
                FontId::monospace(9.0 * k),
                scheme.text_secondary,
            );
            return;
        }
    };

    let (sky, ground, marking) = horizon_palette();
    let (up_dir, right_dir) = world_dirs(attitude.roll);
    let px_per_deg = radius as f64 / PITCH_HALF_RANGE_DEG;
    let pitch_deg = attitude.pitch.to_degrees();
    // Nose-up pushes the world down, so the horizon sits below the symbol.
    let horizon_point = center - up_dir * (pitch_deg * px_per_deg) as f32;

    painter.circle_filled(center, radius, ground);
    match half_disc(center, radius, horizon_point, up_dir) {
        DiscSplit::All => {
            painter.circle_filled(center, radius, sky);
        }
        DiscSplit::Segment(pts) => {
            painter.add(Shape::convex_polygon(pts, sky, Stroke::NONE));
        }
        DiscSplit::Empty => {}
    }

    // Pitch ladder: rungs every 5°, labelled every 10°, all parallel to the
    // horizon because the whole world group is rigid.
    for step in -18..=18 {
        let rung_deg = f64::from(step) * 5.0;
        let major = step % 2 == 0;
        if major && step == 0 {
            continue; // the horizon line itself is drawn below
        }
        let offset = ((rung_deg - pitch_deg) * px_per_deg) as f32;
        let mid = center + up_dir * offset;
        let half = radius * if major { 0.34 } else { 0.16 };
        let (a, b) = (mid - right_dir * half, mid + right_dir * half);
        let Some((a, b)) = clip_segment(center, radius, a, b) else {
            continue;
        };
        painter.line_segment([a, b], Stroke::new(1.0 * k, marking.gamma_multiply(0.9)));
        if !major {
            continue;
        }
        let label = format!("{}", rung_deg.abs() as i32);
        for end in [a, b] {
            let outward = (end - mid).normalized();
            let pos = end + outward * 10.0 * k;
            if (pos - center).length() > radius * 0.98 {
                continue;
            }
            text_with_halo(
                &painter,
                pos,
                &label,
                FontId::monospace(9.0 * k),
                marking,
                Color32::BLACK.gamma_multiply(0.7),
            );
        }
    }

    // Horizon line, brighter than the ladder.
    if let Some((a, b)) = line_chord(center, radius, horizon_point, up_dir) {
        painter.line_segment([a, b], Stroke::new(1.8 * k, marking));
    }

    painter.circle_stroke(center, radius, Stroke::new(1.5 * k, scheme.border_primary));

    // Roll scale: fixed to the screen, ticked either side of top-dead-centre.
    // The bank index belongs to the rolling world, so a right bank moves it
    // left of zero (sky-pointer convention).
    // Ticks and index share `world_dirs`' up, so the scale and the pointer
    // cannot drift apart.
    for (bank_deg, len) in std::iter::once((0.0, 0.13))
        .chain(BANK_TICKS_DEG.iter().flat_map(|&b| [(b, 0.09), (-b, 0.09)]))
    {
        let (dir, _) = world_dirs(bank_deg.to_radians());
        painter.line_segment(
            [
                center + dir * radius * 1.02,
                center + dir * radius * (1.02 + len),
            ],
            Stroke::new(1.0 * k, scheme.text_secondary),
        );
    }
    let tip = center + up_dir * radius * 1.05;
    let side = Vec2::new(-up_dir.y, up_dir.x);
    painter.add(Shape::convex_polygon(
        vec![
            tip,
            tip + (up_dir * 7.0 + side * 4.0) * k,
            tip + (up_dir * 7.0 - side * 4.0) * k,
        ],
        SYMBOL,
        Stroke::NONE,
    ));

    // Screen-fixed aircraft symbol: wings plus a centre dot.
    for sign in [-1.0_f32, 1.0] {
        painter.line_segment(
            [
                Pos2::new(center.x + sign * radius * 0.10, center.y),
                Pos2::new(center.x + sign * radius * 0.34, center.y),
            ],
            Stroke::new(2.5 * k, SYMBOL),
        );
    }
    painter.rect_filled(
        egui::Rect::from_center_size(center, Vec2::splat(4.0 * k)),
        0.0,
        SYMBOL,
    );
}

/// `PITCH … ROLL …` readout, or placeholders when the gauge is inactive.
fn attitude_readout(state: Result<Attitude, InactiveReason>) -> String {
    match state {
        Ok(a) => format!(
            "PITCH {:+.1}°  ROLL {:+.1}°",
            a.pitch.to_degrees(),
            a.roll.to_degrees()
        ),
        Err(_) => "PITCH —  ROLL —".to_string(),
    }
}

#[derive(SystemParam)]
pub struct HorizonGaugeWidget<'w, 's> {
    gauges: Query<'w, 's, (&'static HorizonGaugeData, &'static EqlBinding)>,
    entity_map: Res<'w, EntityMap>,
    values: Query<'w, 's, &'static ComponentValue>,
    telemetry_cache: Res<'w, TelemetryCache>,
    current_timestamp: Res<'w, CurrentTimestamp>,
    geo_context: Res<'w, GeoContext>,
    coordinate: Res<'w, crate::Coordinate>,
}

impl WidgetSystem for HorizonGaugeWidget<'_, '_> {
    type Args = GaugePane;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        pane: Self::Args,
    ) -> Self::Output {
        let HorizonGaugeWidget {
            gauges,
            entity_map,
            values,
            telemetry_cache,
            current_timestamp,
            geo_context,
            coordinate,
        } = state.params_mut(world);
        let Ok((data, binding)) = gauges.get(pane.entity) else {
            return;
        };

        let ts = current_timestamp.0;
        let value = binding.resolve(&entity_map, &values, &telemetry_cache, ts);
        let title = gauge_title(&binding.eql, &pane.name);
        // Keep inherit (`source = None`) live against `Coordinate` changes.
        let source = data.effective_source(coordinate.0);
        let solved = solve_attitude(value.as_ref(), source, data.reference, &geo_context);

        egui::Frame::NONE
            .inner_margin(egui::Margin::same(super::GAUGE_PANEL_MARGIN))
            .show(ui, |ui| {
                super::gauge_header(ui, &title);
                // The face yields its space to the readout, never the reverse:
                // the numbers are what the pane is for.
                paint_horizon(ui, solved);
                ui.add_space(READOUT_GAP);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new(attitude_readout(solved))
                            .monospace()
                            .size(READOUT_FONT_SIZE)
                            .color(get_scheme().text_secondary),
                    );
                });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::DMat3;
    use bevy_geo_frames::Ellipsoid;
    use nox::{Array, Dyn};
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    const MOON: Ellipsoid = Ellipsoid::Sphere {
        radius: 1_737_400.0,
    };

    fn ctx_at(lat_deg: f64, lon_deg: f64) -> GeoContext {
        GeoContext::from(GeoOrigin::new_from_degrees(lat_deg, lon_deg, 0.0))
    }

    fn f64_value(values: &[f64]) -> ComponentValue {
        ComponentValue::F64(
            Array::<f64, Dyn>::from_shape_vec(smallvec::smallvec![values.len()], values.to_vec())
                .expect("f64 buffer"),
        )
    }

    fn f32_value(values: &[f32]) -> ComponentValue {
        ComponentValue::F32(
            Array::<f32, Dyn>::from_shape_vec(smallvec::smallvec![values.len()], values.to_vec())
                .expect("f32 buffer"),
        )
    }

    /// A 7-vec pose value from a quaternion and position.
    fn pose_value(q: DQuat, pos: DVec3) -> ComponentValue {
        f64_value(&[q.x, q.y, q.z, q.w, pos.x, pos.y, pos.z])
    }

    fn solve(value: &ComponentValue, source: GeoFrame, ctx: &GeoContext) -> Attitude {
        solve_attitude(Some(value), source, DQuat::IDENTITY, ctx).expect("active horizon")
    }

    #[test]
    fn level_pose_reads_wings_level_in_tangent_frames() {
        let ctx = ctx_at(34.72, -86.64);
        // Identity attitude in a tangent frame is level by construction, with
        // or without a position (a bare quaternion is enough here).
        for source in [GeoFrame::ENU, GeoFrame::NED] {
            let a = solve(&f64_value(&[0.0, 0.0, 0.0, 1.0]), source, &ctx);
            assert!(a.pitch.abs() < 1e-12, "{source:?} pitch {}", a.pitch);
            assert!(a.roll.abs() < 1e-12, "{source:?} roll {}", a.roll);
        }
    }

    #[test]
    fn pitch_and_roll_signs_follow_aerospace_convention() {
        let ctx = ctx_at(0.0, 0.0);
        // ENU body: +X nose (East), +Z up. Rotating about −Y lifts the nose.
        let up30 = DQuat::from_rotation_y(-30f64.to_radians());
        let a = solve(&pose_value(up30, DVec3::ZERO), GeoFrame::ENU, &ctx);
        assert!((a.pitch.to_degrees() - 30.0).abs() < 1e-9, "{a:?}");
        assert!(a.roll.abs() < 1e-9, "{a:?}");

        // Rotating about the nose (+X) by a positive angle drops the right
        // wing: a right bank must read positive roll.
        let bank45 = DQuat::from_rotation_x(45f64.to_radians());
        let a = solve(&pose_value(bank45, DVec3::ZERO), GeoFrame::ENU, &ctx);
        assert!((a.roll.to_degrees() - 45.0).abs() < 1e-9, "{a:?}");
        assert!(a.pitch.abs() < 1e-9, "{a:?}");

        // Nose-down and left bank invert both signs.
        let down = DQuat::from_rotation_y(20f64.to_radians());
        assert!(solve(&pose_value(down, DVec3::ZERO), GeoFrame::ENU, &ctx).pitch < 0.0);
        let left = DQuat::from_rotation_x(-20f64.to_radians());
        assert!(solve(&pose_value(left, DVec3::ZERO), GeoFrame::ENU, &ctx).roll < 0.0);
    }

    #[test]
    fn ned_pitch_and_roll_match_enu_for_the_same_physical_attitude() {
        let ctx = ctx_at(10.0, 20.0);
        // NED body: +X nose (North), +Z down. Nose-up is a rotation about +Y
        // (East); right bank is about +X with the wing going toward +Z (down).
        let a = solve(
            &pose_value(DQuat::from_rotation_y(25f64.to_radians()), DVec3::ZERO),
            GeoFrame::NED,
            &ctx,
        );
        assert!((a.pitch.to_degrees() - 25.0).abs() < 1e-9, "{a:?}");
        let a = solve(
            &pose_value(DQuat::from_rotation_x(30f64.to_radians()), DVec3::ZERO),
            GeoFrame::NED,
            &ctx,
        );
        assert!((a.roll.to_degrees() - 30.0).abs() < 1e-9, "{a:?}");
    }

    /// Attitude of a vehicle parked level at `(lat, lon)`, nose north, as a
    /// body→ECEF quaternion: body X→North, Y→West, Z→Up.
    fn level_in_ecef(lat_deg: f64, lon_deg: f64, ellipsoid: Ellipsoid) -> (DQuat, DVec3) {
        let here = GeoOrigin::new_from_degrees(lat_deg, lon_deg, 0.0).with_ellipsoid(ellipsoid);
        let ecef_r_enu = GeoFrame::ecef_R_(&GeoFrame::ENU, &here);
        let (east, north, up) = (
            ecef_r_enu * DVec3::X,
            ecef_r_enu * DVec3::Y,
            ecef_r_enu * DVec3::Z,
        );
        let q = DQuat::from_mat3(&DMat3::from_cols(north, -east, up));
        let pos = GeoFrame::ECEF
            ._M_(&GeoFrame::ENU, &GeoContext::from(here))
            .transform_point3(DVec3::ZERO);
        (q, pos)
    }

    #[test]
    fn ecef_horizon_follows_the_vehicle_position() {
        // The schematic origin stays at (0, 0) while the vehicle sits level
        // elsewhere on the ellipsoid: the horizon must come from the vehicle's
        // own position, so it still reads level.
        let ctx = ctx_at(0.0, 0.0);
        for (lat, lon) in [(0.0, 0.0), (45.0, 90.0), (-33.9, 151.2), (89.0, 0.0)] {
            let (q, pos) = level_in_ecef(lat, lon, ctx.origin.ellipsoid);
            let a = solve(&pose_value(q, pos), GeoFrame::ECEF, &ctx);
            assert!(
                a.pitch.abs() < 1e-6 && a.roll.abs() < 1e-6,
                "level at ({lat}, {lon}) read {a:?}"
            );
        }
    }

    #[test]
    fn ecef_vertical_at_the_origin_would_be_wrong_far_away() {
        // Guards the reason the vertical is rebuilt per sample: reusing the
        // schematic origin's vertical tilts the attitude by the arc travelled.
        let ctx = ctx_at(0.0, 0.0);
        let origin_vertical = GeoFrame::ecef_R_(&GeoFrame::ENU, &ctx.origin) * DVec3::Z;
        for (lat, lon) in [(45.0, 0.0), (45.0, 90.0), (0.0, 60.0)] {
            let (q, pos) = level_in_ecef(lat, lon, ctx.origin.ellipsoid);
            let bogus = attitude_against(q, GeoFrame::ECEF, origin_vertical);
            let worst = bogus
                .pitch
                .to_degrees()
                .abs()
                .max(bogus.roll.to_degrees().abs());
            assert!(
                worst > 30.0,
                "origin-based vertical should be badly wrong at ({lat}, {lon}), got {bogus:?}"
            );
            // The real solver stays level for the same sample.
            let a = solve(&pose_value(q, pos), GeoFrame::ECEF, &ctx);
            assert!(a.pitch.abs() < 1e-6 && a.roll.abs() < 1e-6, "{a:?}");
        }
    }

    #[test]
    fn ecef_horizon_works_on_a_lunar_sphere() {
        let mut ctx = ctx_at(0.0, 0.0);
        ctx.origin = ctx.origin.with_ellipsoid(MOON);
        for (lat, lon) in [(0.0, 0.0), (45.0, 90.0), (-70.0, -30.0)] {
            let (q, pos) = level_in_ecef(lat, lon, MOON);
            let a = solve(&pose_value(q, pos), GeoFrame::ECEF, &ctx);
            assert!(
                a.pitch.abs() < 1e-6 && a.roll.abs() < 1e-6,
                "lunar level at ({lat}, {lon}) read {a:?}"
            );
        }
        // An Earth-radius position is far outside the Moon, but still has a
        // well-defined normal — only the body centre is refused.
        let (q, _) = level_in_ecef(0.0, 0.0, MOON);
        assert!(
            solve_attitude(
                Some(&pose_value(q, DVec3::new(6_378_137.0, 0.0, 0.0))),
                GeoFrame::ECEF,
                DQuat::IDENTITY,
                &ctx
            )
            .is_ok()
        );
    }

    #[test]
    fn inactive_states_are_explicit() {
        let ctx = ctx_at(0.0, 0.0);
        let solve =
            |v: Option<&ComponentValue>, source| solve_attitude(v, source, DQuat::IDENTITY, &ctx);

        // No sample at the playhead.
        assert_eq!(solve(None, GeoFrame::ENU), Err(InactiveReason::NoSample));

        // A bare quaternion has no position, so ECEF has no vertical — while
        // the very same value is fine in a tangent frame.
        let bare = f64_value(&[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            solve(Some(&bare), GeoFrame::ECEF),
            Err(InactiveReason::EcefWithoutPosition)
        );
        assert!(solve(Some(&bare), GeoFrame::ENU).is_ok());

        // Position-only and non-unit 4-vectors carry no attitude.
        for v in [
            f64_value(&[1.0, 2.0, 3.0]),
            f64_value(&[0.1, 0.2, 0.3, 0.4]),
            f64_value(&[0.0, 0.0, 0.0, 0.0]),
            f64_value(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ] {
            assert_eq!(
                solve(Some(&v), GeoFrame::ENU),
                Err(InactiveReason::NoAttitude),
                "{v:?} must not produce an attitude"
            );
        }

        // At the body centre there is no vertical to speak of.
        assert_eq!(
            solve(
                Some(&pose_value(DQuat::IDENTITY, DVec3::ZERO)),
                GeoFrame::ECEF
            ),
            Err(InactiveReason::NearBodyCenter)
        );

        // Every reason is user-visible.
        for reason in [
            InactiveReason::NoSample,
            InactiveReason::NoAttitude,
            InactiveReason::EcefWithoutPosition,
            InactiveReason::NearBodyCenter,
        ] {
            assert!(!reason.label().is_empty());
        }
    }

    #[test]
    fn f32_poses_are_accepted_like_f64() {
        let ctx = ctx_at(0.0, 0.0);
        let a = solve(
            &f32_value(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 100.0]),
            GeoFrame::ENU,
            &ctx,
        );
        assert!(a.pitch.abs() < 1e-6 && a.roll.abs() < 1e-6);
    }

    #[test]
    fn inverted_flight_puts_the_ground_on_top() {
        let ctx = ctx_at(0.0, 0.0);
        let a = solve(
            &pose_value(DQuat::from_rotation_x(PI), DVec3::ZERO),
            GeoFrame::ENU,
            &ctx,
        );
        assert!(a.roll.abs().to_degrees() > 179.0, "{a:?}");
        // The world's up now points down the screen, so the sky fills the
        // bottom of the face.
        let (up_dir, _) = world_dirs(a.roll);
        assert!(
            up_dir.y > 0.9,
            "world up should point screen-down: {up_dir:?}"
        );
    }

    #[test]
    fn roll_sweep_never_jumps() {
        // A full roll must rotate the rendered world continuously — the failure
        // mode of deriving the horizon from clamped `asin` scalars.
        let ctx = ctx_at(0.0, 0.0);
        let mut prev: Option<Vec2> = None;
        for i in 0..=720 {
            let theta = TAU * (i as f64 / 720.0);
            let a = solve(
                &pose_value(DQuat::from_rotation_x(theta), DVec3::ZERO),
                GeoFrame::ENU,
                &ctx,
            );
            let (up_dir, _) = world_dirs(a.roll);
            if let Some(p) = prev {
                assert!(
                    (up_dir - p).length() < 0.05,
                    "world up jumped at step {i}: {p:?} -> {up_dir:?}"
                );
            }
            prev = Some(up_dir);
        }
    }

    #[test]
    fn pitch_sweep_through_vertical_stays_bounded() {
        // Pitching up through the vertical and over the top: pitch folds back
        // (as on any attitude indicator) but never leaves ±90°, and roll flips
        // to keep the picture correct instead of producing NaNs.
        let ctx = ctx_at(0.0, 0.0);
        for i in 0..=720 {
            let theta = TAU * (i as f64 / 720.0);
            let a = solve(
                &pose_value(DQuat::from_rotation_y(-theta), DVec3::ZERO),
                GeoFrame::ENU,
                &ctx,
            );
            assert!(a.pitch.is_finite() && a.roll.is_finite(), "{a:?}");
            assert!(a.pitch.abs() <= FRAC_PI_2 + 1e-9, "{a:?}");
        }
    }

    #[test]
    fn reference_attitude_reads_level_and_round_trips() {
        // Identity reference stays implicit in KDL.
        let data = HorizonGaugeData::new(None);
        assert_eq!(data.reference_kdl(), None);

        // A nose-up-modelled body: its own attitude is the reference, so it
        // must read level rather than permanently pitched.
        let raw = [0.0, 2.0, 0.0, 2.0]; // unnormalized 90° about Y
        let data = data.with_reference(Some(raw));
        assert!((data.reference.length() - 1.0).abs() < 1e-12);
        assert!(data.reference_kdl().is_some());

        let ctx = ctx_at(0.0, 0.0);
        let a = solve_attitude(
            Some(&pose_value(data.reference, DVec3::ZERO)),
            GeoFrame::ENU,
            data.reference,
            &ctx,
        )
        .expect("active");
        assert!(a.pitch.abs() < 1e-9 && a.roll.abs() < 1e-9, "{a:?}");

        // Without the reference the same pose reads steeply pitched.
        let raw_read = solve(
            &pose_value(data.reference, DVec3::ZERO),
            GeoFrame::ENU,
            &ctx,
        );
        assert!(raw_read.pitch.to_degrees().abs() > 80.0, "{raw_read:?}");
    }

    #[test]
    fn effective_source_inherits_coordinate_then_enu() {
        let inherit = HorizonGaugeData::new(None);
        assert_eq!(inherit.effective_source(Some(GeoFrame::NED)), GeoFrame::NED);
        assert_eq!(inherit.effective_source(None), GeoFrame::ENU);
        let explicit = HorizonGaugeData::new(Some(GeoFrame::ECEF));
        assert_eq!(
            explicit.effective_source(Some(GeoFrame::ENU)),
            GeoFrame::ECEF
        );
    }

    #[test]
    fn right_bank_raises_the_horizons_right_end() {
        // The cockpit convention: a right bank tilts the world's up to the
        // left, so the horizon's right end rises on screen (y grows downward).
        let (up_dir, right_dir) = world_dirs(15f64.to_radians());
        assert!(up_dir.x < 0.0, "world up should lean left: {up_dir:?}");
        assert!(
            right_dir.y < 0.0,
            "horizon should rise to the right: {right_dir:?}"
        );
        // Level flight keeps it flat and upright.
        let (up_dir, right_dir) = world_dirs(0.0);
        assert!((up_dir - Vec2::new(0.0, -1.0)).length() < 1e-6);
        assert!((right_dir - Vec2::new(1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn half_disc_covers_all_none_or_a_segment_inside_the_face() {
        let center = Pos2::new(50.0, 50.0);
        let radius = 20.0;
        let up = Vec2::new(0.0, -1.0);

        // Horizon far below the face: all sky. Far above: none.
        assert!(matches!(
            half_disc(center, radius, center + Vec2::new(0.0, 100.0), up),
            DiscSplit::All
        ));
        assert!(matches!(
            half_disc(center, radius, center - Vec2::new(0.0, 100.0), up),
            DiscSplit::Empty
        ));

        // Horizon through the centre: a half-disc, entirely inside the face and
        // entirely on the sky side.
        let DiscSplit::Segment(pts) = half_disc(center, radius, center, up) else {
            panic!("expected a segment");
        };
        assert!(pts.len() >= 3);
        for p in &pts {
            assert!(
                (*p - center).length() <= radius + 1e-3,
                "segment left the face: {p:?}"
            );
            assert!(
                (*p - center).dot(up) >= -1e-3,
                "segment crossed the horizon"
            );
        }
    }

    #[test]
    fn line_chord_and_clip_segment_stay_inside_the_face() {
        let center = Pos2::new(0.0, 0.0);
        let radius = 10.0;
        let up = Vec2::new(0.0, -1.0);

        let (a, b) = line_chord(center, radius, center, up).expect("chord through centre");
        assert!((a - center).length() <= radius + 1e-3);
        assert!((b - center).length() <= radius + 1e-3);
        assert!(((b - a).length() - 2.0 * radius).abs() < 1e-3, "diameter");
        // A line that misses the disc has no chord.
        assert!(line_chord(center, radius, center + Vec2::new(0.0, 50.0), up).is_none());

        // A rung crossing the rim is truncated to the disc.
        let (a, b) = clip_segment(
            center,
            radius,
            Pos2::new(-100.0, 0.0),
            Pos2::new(100.0, 0.0),
        )
        .expect("crossing segment");
        assert!((a.x + radius).abs() < 1e-3 && (b.x - radius).abs() < 1e-3);
        // A rung fully outside is dropped.
        assert!(
            clip_segment(
                center,
                radius,
                Pos2::new(-100.0, 50.0),
                Pos2::new(100.0, 50.0)
            )
            .is_none()
        );
    }

    /// Room the readout takes at the sizes egui actually reports for a 10 px
    /// monospace row.
    const READOUT_ROOM: f32 = READOUT_GAP + 14.0;

    #[test]
    fn the_face_never_eats_the_readouts_room() {
        for height in [0.0, 40.0, 77.0, 120.0, 200.0, 400.0, 2000.0] {
            let avail = Vec2::new(400.0, height);
            if let Some(size) = face_size(avail, READOUT_ROOM) {
                assert!(
                    size + READOUT_ROOM <= height + 1e-3,
                    "face {size} + readout leaves nothing of {height}"
                );
            }
        }
    }

    #[test]
    fn the_face_grows_with_the_pane_up_to_the_cap() {
        let at = |h: f32| face_size(Vec2::new(1000.0, h), READOUT_ROOM);
        // Monotone in the pane's height, so dragging a tile larger never
        // shrinks the instrument.
        let mut prev = 0.0;
        for step in 0..200u16 {
            let Some(size) = at(50.0 + f32::from(step) * 10.0) else {
                continue;
            };
            assert!(size >= prev - 1e-3, "shrank from {prev} to {size}");
            prev = size;
        }
        // It does grow past the old fixed 200 px, and stops at the cap.
        assert!(at(300.0).expect("drawable") > 200.0);
        assert_eq!(at(4000.0), Some(FACE_MAX));
        // A narrow pane is bounded by its width instead.
        assert_eq!(
            face_size(Vec2::new(120.0, 4000.0), READOUT_ROOM),
            Some(120.0)
        );
    }

    #[test]
    fn a_pane_too_short_for_both_keeps_the_readout() {
        // Anything that would leave a face below the legibility floor draws no
        // face at all, so the caller's readout is all that remains.
        for height in [0.0, 20.0, FACE_MIN, FACE_MIN + READOUT_ROOM - 1.0] {
            assert_eq!(
                face_size(Vec2::new(400.0, height), READOUT_ROOM),
                None,
                "height {height} should drop the face"
            );
        }
        assert_eq!(
            face_size(Vec2::new(400.0, FACE_MIN + READOUT_ROOM), READOUT_ROOM),
            Some(FACE_MIN)
        );
    }

    #[test]
    fn symbology_scales_with_the_face_within_bounds() {
        assert!((symbol_scale(REFERENCE_FACE) - 1.0).abs() < 1e-6);
        assert!(symbol_scale(FACE_MAX) > 1.0);
        assert!(symbol_scale(FACE_MIN) < 1.0);
        // Bounded at both ends: type stays readable, line weights stay thin.
        for size in [0.0, FACE_MIN, 200.0, FACE_MAX, 10_000.0] {
            let k = symbol_scale(size);
            assert!((0.85..=1.5).contains(&k), "scale {k} at face {size}");
        }
    }

    #[test]
    fn readout_shows_degrees_or_placeholders() {
        let a = Attitude {
            pitch: 14f64.to_radians(),
            roll: -15f64.to_radians(),
        };
        assert_eq!(attitude_readout(Ok(a)), "PITCH +14.0°  ROLL -15.0°");
        assert_eq!(
            attitude_readout(Err(InactiveReason::NoSample)),
            "PITCH —  ROLL —"
        );
    }
}
