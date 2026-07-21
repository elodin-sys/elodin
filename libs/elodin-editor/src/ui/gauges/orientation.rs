//! Orientation gauge: a 3D attitude gimbal driven by an EQL-bound quaternion
//! (bare `[x,y,z,w]` 4-vector or the head of a `world_pos`-style 7-vector).
//! The sphere shows the `display` frame's triad rotated by the body attitude,
//! with a two-tone up/down shading — no position values here.

use bevy::{
    ecs::system::SystemParam,
    math::{DQuat, DVec3},
    prelude::*,
};
use bevy_egui::egui::{self, Align2, Color32, FontId, Pos2, Sense, Shape, Stroke, Vec2};
use bevy_geo_frames::{GeoContext, GeoFrame};
use impeller2_bevy::{EntityMap, TelemetryCache};
use impeller2_wkt::{ComponentValue, CurrentTimestamp};
use std::f32::consts::TAU;

use super::{EqlBinding, GaugePane, gauge_title};
use crate::ui::{
    colors::get_scheme,
    theme,
    widgets::{SystemStateExt, WidgetSystem},
};

/// Backing data for an orientation gauge pane; the EQL lives in the sibling
/// [`EqlBinding`] component.
#[derive(Component)]
pub struct OrientationGaugeData {
    /// Frame the EQL quaternion rotates from (body→source).
    ///
    /// `None` means inherit the schematic global [`crate::Coordinate`] (same
    /// as omitting `source` in KDL). Resolved at display/export via
    /// [`Self::effective_source`].
    pub source: Option<GeoFrame>,
    /// Frame whose triad the gimbal draws. `None` defaults to NED.
    pub display: Option<GeoFrame>,
    /// Attitude (body→source) the gimbal shows as neutral: the sphere renders
    /// `q · reference⁻¹`. Identity means the raw component attitude.
    pub reference: DQuat,
}

impl OrientationGaugeData {
    pub fn new(source: Option<GeoFrame>, display: Option<GeoFrame>) -> Self {
        Self {
            source,
            display,
            reference: DQuat::IDENTITY,
        }
    }

    /// Set the neutral-attitude quaternion from its KDL form (`[x, y, z, w]`,
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
    /// else ENU (same fallback as viewport / view-cube when both are unset).
    pub fn effective_source(&self, coordinate: Option<GeoFrame>) -> GeoFrame {
        self.source.or(coordinate).unwrap_or(GeoFrame::ENU)
    }

    /// Concrete display frame (default NED).
    pub fn effective_display(&self) -> GeoFrame {
        self.display.unwrap_or(GeoFrame::NED)
    }
}

#[derive(SystemParam)]
pub struct OrientationGaugeWidget<'w, 's> {
    gauges: Query<'w, 's, (&'static mut OrientationGaugeData, &'static EqlBinding)>,
    entity_map: Res<'w, EntityMap>,
    values: Query<'w, 's, &'static ComponentValue>,
    telemetry_cache: Res<'w, TelemetryCache>,
    current_timestamp: Res<'w, CurrentTimestamp>,
    geo_context: Res<'w, GeoContext>,
    coordinate: Res<'w, crate::Coordinate>,
}

impl WidgetSystem for OrientationGaugeWidget<'_, '_> {
    type Args = GaugePane;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        pane: Self::Args,
    ) -> Self::Output {
        let OrientationGaugeWidget {
            mut gauges,
            entity_map,
            values,
            telemetry_cache,
            current_timestamp,
            geo_context,
            coordinate,
        } = state.params_mut(world);
        let Ok((mut data, binding)) = gauges.get_mut(pane.entity) else {
            return;
        };

        let ts = current_timestamp.0;
        let value = binding.resolve(&entity_map, &values, &telemetry_cache, ts);
        let title = gauge_title(&binding.eql, &pane.name);
        // Keep inherit (`source = None`) live against `Coordinate` changes.
        let source = data.effective_source(coordinate.0);
        let combo_id = egui::Id::new(("orientation_gauge_display", pane.entity));

        egui::Frame::NONE
            .inner_margin(egui::Margin::same(4))
            .show(ui, |ui| {
                let scheme = get_scheme();
                ui.label(
                    egui::RichText::new(title)
                        .monospace()
                        .size(10.0)
                        .color(scheme.text_secondary),
                );
                ui.add_space(3.0);

                // In-panel display-frame dropdown (source stays in the inspector).
                {
                    let style = ui.style_mut();
                    theme::configure_input_with_border(style);
                    style
                        .text_styles
                        .iter_mut()
                        .for_each(|(_, font)| font.size = 10.0);
                }
                let display = data.effective_display();
                egui::ComboBox::from_id_salt(combo_id)
                    .selected_text(frame_label(display))
                    .width(86.0)
                    .show_ui(ui, |ui| {
                        for frame in [GeoFrame::NED, GeoFrame::ENU, GeoFrame::ECEF] {
                            ui.selectable_value(&mut data.display, Some(frame), frame_label(frame));
                        }
                    });

                // Read `display` after the ComboBox so a change applies immediately.
                let display = data.effective_display();
                let reference = data.reference;
                let att_source = value
                    .as_ref()
                    .and_then(component_value_to_quat)
                    // Attitude change since the neutral pose (identity
                    // reference ⇒ raw component attitude).
                    .map(|q| q * reference.inverse());
                // Missing / non-quaternion samples draw a muted empty rim —
                // never treat missing attitude as identity / wings-level.
                paint_frame_sphere(ui, display, source, &geo_context, att_source);

                // Numeric attitude below the gimbal, 2 dp (or "—" when there's
                // no valid quaternion sample).
                ui.add_space(3.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new(quat_readout(att_source))
                            .monospace()
                            .size(10.0)
                            .color(scheme.text_secondary),
                    );
                });
            });
    }
}

fn frame_label(frame: GeoFrame) -> &'static str {
    match frame {
        GeoFrame::ECEF => "ECEF",
        GeoFrame::NED => "NED",
        GeoFrame::ENU => "ENU",
    }
}

/// Two-decimal `x y z w` readout of the drawn attitude, or a placeholder when
/// there is no valid quaternion sample.
///
/// `q` and `-q` are the same rotation, so the sample's raw sign can flip between
/// ticks even during smooth motion. Canonicalize to a single hemisphere so the
/// printed numbers stay continuous.
fn quat_readout(att: Option<DQuat>) -> String {
    match att.map(canonical_hemisphere) {
        Some(q) => format!("x {:+.2}  y {:+.2}  z {:+.2}  w {:+.2}", q.x, q.y, q.z, q.w),
        None => "x —  y —  z —  w —".to_string(),
    }
}

/// Pick the representative of `{q, -q}` with a non-negative leading component
/// (`w`, then `z`, `y`, `x`), so equivalent attitudes always print identically.
fn canonical_hemisphere(q: DQuat) -> DQuat {
    for c in [q.w, q.z, q.y, q.x] {
        if c > 0.0 {
            return q;
        }
        if c < 0.0 {
            return DQuat::from_xyzw(-q.x, -q.y, -q.z, -q.w);
        }
    }
    q
}

/// Max deviation of a bare 4-vector's length² from 1 to still count as a
/// quaternion. Loose enough for telemetry drift / un-renormalized integration,
/// tight enough that arbitrary 4-vectors (e.g. fin deflections) are rejected.
const BARE_QUAT_UNIT_TOLERANCE: f64 = 0.1;

/// Extract an attitude quaternion from a component value.
///
/// Accepts only (in `F32` or `F64`):
/// - a SpatialTransform / [`WorldPos`](impeller2_wkt::WorldPos) (≥7 elements
///   whose head 4 are the quaternion `[x, y, z, w]`), or
/// - a bare, (approximately) unit-length 4-vector `[x, y, z, w]`.
///
/// A bare 4-vector must already be near unit length: a genuine attitude
/// quaternion is normalized, whereas an arbitrary 4-vector (e.g. fin
/// deflections) is not — blindly normalizing one would invent a misleading
/// gimbal instead of the empty "—" state. This mirrors the geo-position
/// gauge, which likewise refuses to treat a non-pose 4-vector as data.
///
/// Telemetry quats can still drift slightly off unit length (and
/// `DQuat::inverse` assumes normalized), so accepted quats are re-normalized.
fn component_value_to_quat(value: &ComponentValue) -> Option<DQuat> {
    let data = super::component_buf_f64(value)?;
    // A world_pos-style pose: the head 4 elements are the quaternion. Genuine
    // poses are ≥7 elements, so only these (never a bare 4-vector) take this path.
    if data.len() >= 7 {
        let q = DQuat::from_xyzw(data[0], data[1], data[2], data[3]);
        return (q.length_squared() > 1e-12).then(|| q.normalize());
    }
    if data.len() == 4 {
        let q = DQuat::from_xyzw(data[0], data[1], data[2], data[3]);
        return ((q.length_squared() - 1.0).abs() <= BARE_QUAT_UNIT_TOLERANCE)
            .then(|| q.normalize());
    }
    None
}

/// Unit axes of a frame's display triad: up-ish first (the horizon pole, used
/// for camera framing and sky/ground shading), then the two rim axes. For the
/// labelled axes the gauge *draws*, see [`frame_axes`].
fn sphere_axis_dirs(frame: GeoFrame) -> [DVec3; 3] {
    match frame {
        GeoFrame::NED => [
            DVec3::new(0.0, 0.0, -1.0), // U = -D
            DVec3::new(0.0, 1.0, 0.0),  // E
            DVec3::new(1.0, 0.0, 0.0),  // N
        ],
        GeoFrame::ENU => [
            DVec3::Z, // U
            DVec3::X, // E
            DVec3::Y, // N
        ],
        GeoFrame::ECEF => [DVec3::Z, DVec3::Y, DVec3::X],
    }
}

/// Canonical positive axes of a frame with their labels, in the frame's own
/// numeric coordinates. Unlike [`sphere_axis_dirs`] (which lists the vertical
/// pole first, for camera framing and shading), these are the axes the gauge
/// actually *draws*: a NED gauge shows **Down** pointing down (not Up), ENU
/// shows Up, and ECEF shows X/Y/Z.
fn frame_axes(frame: GeoFrame) -> [(DVec3, &'static str); 3] {
    match frame {
        // NED coords: X=N, Y=E, Z=D — Down (+Z) projects toward screen-bottom.
        GeoFrame::NED => [(DVec3::X, "N"), (DVec3::Y, "E"), (DVec3::Z, "D")],
        // ENU coords: X=E, Y=N, Z=U.
        GeoFrame::ENU => [(DVec3::X, "E"), (DVec3::Y, "N"), (DVec3::Z, "U")],
        GeoFrame::ECEF => [(DVec3::X, "X"), (DVec3::Y, "Y"), (DVec3::Z, "Z")],
    }
}

/// The display triad expressed in `source` coordinates — the *physical* axes
/// the gauge draws. Keeping everything in source coordinates (instead of
/// conjugating the attitude into the display frame) means an ECEF gauge shows
/// the true tilt between the body and the Earth axes (~lat-dependent), rather
/// than treating "aligned with ECEF" as level.
fn display_triad_in_source(display: GeoFrame, source: GeoFrame, ctx: &GeoContext) -> [DVec3; 3] {
    let r = source._R_(&display, ctx);
    sphere_axis_dirs(display).map(|d| r * d)
}

/// The drawn labelled axes ([`frame_axes`]) expressed in `source` coordinates,
/// so a NED gauge draws its Down axis downward while the sphere stays anchored
/// to the source frame. Same source-coordinate rationale as
/// [`display_triad_in_source`].
fn display_axes_in_source(
    display: GeoFrame,
    source: GeoFrame,
    ctx: &GeoContext,
) -> [(DVec3, &'static str); 3] {
    let r = source._R_(&display, ctx);
    frame_axes(display).map(|(d, label)| (r * d, label))
}

/// Fixed orthographic camera for the gauge sphere, built from a frame's
/// numeric triad so U is near screen-top (ENU's up is +Z but NED's is −Z —
/// a hardcoded world-up would render NED upside down).
///
/// The eye sits at an elevated diagonal so all three axes are visible at
/// identity (U near top, the other two on the rim).
struct SphereCamera {
    right: DVec3,
    up: DVec3,
    fwd: DVec3,
}

impl SphereCamera {
    fn new(triad: [DVec3; 3]) -> Self {
        let [u, e, n] = triad;
        let eye = (e * 0.75 + n * 0.55 + u * 0.35).normalize();
        let right = (-eye).cross(u).normalize();
        let up = right.cross(-eye).normalize();
        Self {
            right,
            up,
            fwd: eye,
        }
    }

    /// Camera-space coordinates of a body-frame vector: `x` right, `y` up,
    /// `z` depth toward the viewer (`z > 0` = front hemisphere).
    fn project(&self, v: DVec3) -> DVec3 {
        DVec3::new(v.dot(self.right), v.dot(self.up), v.dot(self.fwd))
    }
}

/// Two unit vectors spanning the plane perpendicular to `u` (assumed unit).
fn basis_perp(u: DVec3) -> (DVec3, DVec3) {
    let helper = if u.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let e1 = u.cross(helper).normalize();
    let e2 = u.cross(e1);
    (e1, e2)
}

/// True when the front-hemisphere sphere point projecting to `p` (unit-disk
/// coords, `+y` up) is on the sky side of the horizon plane ⊥ `u_cam`.
/// Test-only sky/ground oracle (the fill uses the convex-cap boundary).
#[cfg(test)]
fn front_point_is_sky(p: Vec2, u_cam: DVec3) -> bool {
    let r2 = (p.x * p.x + p.y * p.y) as f64;
    let z = (1.0 - r2).max(0.0).sqrt();
    p.x as f64 * u_cam.x + p.y as f64 * u_cam.y + z * u_cam.z >= 0.0
}

/// Great circle ⊥ `normal_cam` split into the front and back polylines, in
/// unit-disk coords (`+y` up). `None` when `normal_cam` is (anti)parallel to
/// the view axis — the circle then coincides with the rim.
fn great_circle_arcs(normal_cam: DVec3) -> Option<(Vec<Vec2>, Vec<Vec2>)> {
    if (normal_cam.x * normal_cam.x + normal_cam.y * normal_cam.y).sqrt() < 1e-6 {
        return None;
    }
    let (e1, e2) = basis_perp(normal_cam);
    const N: usize = 96;
    let pts: Vec<DVec3> = (0..N)
        .map(|i| {
            let a = std::f64::consts::TAU * (i as f64 / N as f64);
            e1 * a.cos() + e2 * a.sin()
        })
        .collect();
    // A great circle crosses the silhouette exactly twice; the front samples
    // form one contiguous run (mod N). Find its start.
    let front = |v: &DVec3| v.z >= 0.0;
    let start = (0..N).find(|&i| front(&pts[i]) && !front(&pts[(i + N - 1) % N]))?;
    let mut front_arc = Vec::new();
    let mut back_arc = Vec::new();
    for k in 0..N {
        let v = &pts[(start + k) % N];
        let p = Vec2::new(v.x as f32, v.y as f32);
        if front(v) {
            front_arc.push(p);
        } else {
            back_arc.push(p);
        }
    }
    Some((front_arc, back_arc))
}

/// Closed boundary (unit-disk coords) of the convex screen region covered by
/// the front hemisphere around `pole` (unit, camera space, `pole.z >= 0`):
/// front horizon arc plus the rim arc on the pole side.
fn horizon_cap_boundary(pole: DVec3, front_arc: &[Vec2]) -> Vec<Vec2> {
    let mut boundary: Vec<Vec2> = front_arc.to_vec();
    let (Some(first), Some(last)) = (front_arc.first(), front_arc.last()) else {
        return Vec::new();
    };
    // Rim arc from the arc's end back to its start, through the pole's screen
    // direction (the side where rim points satisfy r·pole > 0).
    let a_start = first.y.atan2(first.x);
    let a_end = last.y.atan2(last.x);
    let phi_pole = (pole.y as f32).atan2(pole.x as f32);
    let mut delta = (a_start - a_end).rem_euclid(TAU);
    let mid = a_end + delta * 0.5;
    if (mid - phi_pole).cos() < 0.0 {
        delta -= TAU; // go the other way round, through the pole side
    }
    let steps = ((delta.abs() / 0.08).ceil() as usize).max(1);
    for i in 1..steps {
        let a = a_end + delta * (i as f32 / steps as f32);
        boundary.push(Vec2::new(a.cos(), a.sin()));
    }
    boundary
}

/// Two-tone fill plan for the attitude disc: the whole disc is painted `base`,
/// then (optionally) the pole-facing hemisphere is painted on top as a convex
/// `cap`. Only the pole-facing hemisphere projects to a convex region, so the
/// base/cap roles swap with `u_cam.z` — the swap is paired with the rim-closure
/// side in [`horizon_cap_boundary`] so the rendered split stays continuous as
/// `u_cam` crosses the silhouette (`u_cam.z = 0`). `fill_is_sky` verifies this
/// against the ground-truth oracle.
struct HorizonFill {
    /// Colour under the whole disc (true = sky, false = ground).
    base_is_sky: bool,
    /// Convex pole-facing cap drawn over the base: `(boundary, cap_is_sky)`.
    /// `None` when up is (anti)parallel to the view axis (all one tone).
    cap: Option<(Vec<Vec2>, bool)>,
}

/// Resolve the two-tone fill for camera-space up `u_cam`, reusing the horizon
/// `arcs` already computed by the caller.
fn horizon_fill(u_cam: DVec3, arcs: &Option<(Vec<Vec2>, Vec<Vec2>)>) -> HorizonFill {
    match arcs {
        // Up (anti)parallel to the view axis: the whole disc is one tone.
        None => HorizonFill {
            base_is_sky: u_cam.z >= 0.0,
            cap: None,
        },
        Some((front_arc, _)) => {
            // The pole facing the camera projects to the convex cap; the other
            // hemisphere wraps around the rim and is painted as the base.
            let (base_is_sky, cap_is_sky, pole) = if u_cam.z >= 0.0 {
                (false, true, u_cam)
            } else {
                (true, false, -u_cam)
            };
            let boundary = horizon_cap_boundary(pole, front_arc);
            HorizonFill {
                base_is_sky,
                cap: (boundary.len() >= 3).then_some((boundary, cap_is_sky)),
            }
        }
    }
}

/// Paint a circular attitude sphere: two-tone up/down shading and three axis
/// labels. When `att_source` is set (body→source), the shading and the
/// display-frame triad track attitude (back-facing tips stay visible, dimmed,
/// through the sphere). All math stays in source coordinates: the drawn axes
/// are the display triad's *physical* directions, so an ECEF gauge shows the
/// body's absolute tilt against the Earth axes while NED/ENU (whose triads
/// are physically identical) render the same local-level view. Without
/// attitude (no sample, or a non-quaternion value), draw a muted empty rim —
/// never treat missing attitude as identity / wings-level.
fn paint_frame_sphere(
    ui: &mut egui::Ui,
    display: GeoFrame,
    source: GeoFrame,
    geo_context: &GeoContext,
    att_source: Option<DQuat>,
) {
    let scheme = get_scheme();
    // Respect the tile's actual available space — do not invent a floor that
    // can overflow short panels. Centre the sphere in the remaining width.
    let size = ui
        .available_width()
        .min(ui.available_height())
        .clamp(0.0, 150.0);
    let avail = ui.available_width();
    let (full_rect, _response) =
        ui.allocate_exact_size(Vec2::new(avail.max(size), size), Sense::hover());
    let rect = egui::Rect::from_center_size(full_rect.center(), Vec2::splat(size));
    let painter = ui.painter_at(rect);

    let center = rect.center();
    let radius = size * 0.42;

    let Some(q) = att_source else {
        painter.circle_stroke(
            center,
            radius,
            Stroke::new(1.5, scheme.border_primary.gamma_multiply(0.5)),
        );
        painter.text(
            center,
            Align2::CENTER_CENTER,
            "—",
            FontId::monospace(15.0),
            scheme.text_secondary,
        );
        return;
    };

    // The markings co-rotate with the body (like the 3D cube in the viewport):
    // we apply the attitude `q` directly to the source-frame triad, so a
    // right-hand spin turns the gimbal's top the same way as the vehicle's.
    // (Applying `q.inverse()` instead would draw the world as seen from the
    // body — an ADI cockpit view whose top counter-rotates.) The shading
    // boundary is the true 3D great circle ⊥ up, so any attitude renders
    // continuously; the camera is anchored to the source frame, so "level in
    // source" reads U-near-top while the triad keeps its physical direction.
    //
    // Shading tracks the display frame's *up* (so the sky stays up), but the
    // drawn axes are the display frame's canonical positive axes — hence NED
    // shows Down pointing down, decoupled from the up pole.
    let up_source = display_triad_in_source(display, source, geo_context)[0];
    let axes = display_axes_in_source(display, source, geo_context);
    let cam = SphereCamera::new(sphere_axis_dirs(source));
    let q_draw = q;
    let u_cam = cam.project(q_draw * up_source);
    let to_screen = |p: Vec2| Pos2::new(center.x + radius * p.x, center.y - radius * p.y);

    // Two-tone hemispheres: light for the up half, a muted gray for the down
    // half. Kept deliberately flat (no hatching / ground texture) so it reads
    // as "which way is up", not a simulated artificial-horizon instrument.
    let light_mode = crate::ui::colors::is_light_mode();
    let (sky, ground, horizon) = if light_mode {
        (
            Color32::from_gray(210),
            Color32::from_gray(140),
            Color32::from_gray(120),
        )
    } else {
        (
            Color32::from_gray(225),
            Color32::from_gray(120),
            Color32::from_gray(150),
        )
    };

    // Fill: the hemisphere whose pole faces the camera projects to a convex
    // cap; the other hemisphere wraps around the rim. Paint the wrapping one
    // over the whole disc, then the convex cap on top.
    let arcs = great_circle_arcs(u_cam);
    let tone = |is_sky: bool| if is_sky { sky } else { ground };
    let fill = horizon_fill(u_cam, &arcs);
    painter.circle_filled(center, radius, tone(fill.base_is_sky));
    if let Some((boundary, cap_is_sky)) = &fill.cap {
        let pts: Vec<Pos2> = boundary.iter().map(|&p| to_screen(p)).collect();
        painter.add(Shape::convex_polygon(pts, tone(*cap_is_sky), Stroke::NONE));
    }

    // The display frame's coordinate planes as attitude-driven great circles;
    // back halves stay visible, dimmed, "through" the sphere.
    let curve = Color32::from_gray(128).gamma_multiply(0.8);
    for (axis, _) in axes {
        match great_circle_arcs(cam.project(q_draw * axis)) {
            Some((front_c, back_c)) => {
                let back: Vec<Pos2> = back_c.iter().map(|&p| to_screen(p)).collect();
                painter.add(Shape::line(
                    back,
                    Stroke::new(1.0, curve.gamma_multiply(0.4)),
                ));
                let front: Vec<Pos2> = front_c.iter().map(|&p| to_screen(p)).collect();
                painter.add(Shape::line(front, Stroke::new(1.0, curve)));
            }
            // Plane parallel to the screen: the circle coincides with the rim.
            // Stroke it there instead of blinking out for a frame.
            None => {
                painter.circle_stroke(center, radius, Stroke::new(1.0, curve));
            }
        }
    }

    // Up/down boundary: bright front arc, dimmed back arc through the sphere.
    if let Some((front_arc, back_arc)) = &arcs {
        let back: Vec<Pos2> = back_arc.iter().map(|&p| to_screen(p)).collect();
        painter.add(Shape::line(
            back,
            Stroke::new(1.0, horizon.gamma_multiply(0.35)),
        ));
        let front: Vec<Pos2> = front_arc.iter().map(|&p| to_screen(p)).collect();
        painter.add(Shape::line(front, Stroke::new(1.5, horizon)));
    }

    // Outer rim, over the fills so the disc has a crisp edge.
    painter.circle_stroke(center, radius, Stroke::new(1.5, scheme.border_primary));

    // Fixed centre reticle (wings + dot): a screen-fixed reference the
    // co-rotating markings move against. Amber so it reads on both tones.
    let wing = Color32::from_rgb(255, 179, 0);
    painter.line_segment(
        [
            Pos2::new(center.x - radius * 0.22, center.y),
            Pos2::new(center.x - radius * 0.06, center.y),
        ],
        Stroke::new(2.0, wing),
    );
    painter.line_segment(
        [
            Pos2::new(center.x + radius * 0.06, center.y),
            Pos2::new(center.x + radius * 0.22, center.y),
        ],
        Stroke::new(2.0, wing),
    );
    painter.circle_stroke(center, 3.5, Stroke::new(1.5, wing));

    // Display-triad axes rotated by the body attitude (source identity ⇒
    // physical directions). Drawn last so back-facing tips remain visible
    // "through" the sphere. Unit vectors project inside the disc, no clamp.
    let mut tips: Vec<(f32, &'static str, Pos2, f32)> = axes
        .into_iter()
        .map(|(dir, label)| {
            let c = cam.project(q_draw * dir);
            let (sx, sy, depth) = (c.x as f32, c.y as f32, c.z as f32);
            let pos = Pos2::new(center.x + radius * sx, center.y - radius * sy);
            let alpha = if depth >= 0.0 { 1.0 } else { 0.45 };
            (depth, label, pos, alpha)
        })
        .collect();
    tips.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for &(_depth, label, pos, alpha) in &tips {
        painter.line_segment([center, pos], Stroke::new(1.0, curve.gamma_multiply(alpha)));
        painter.circle_filled(pos, 3.0, Color32::from_gray(160).gamma_multiply(alpha));
        // Axis toward the camera projects onto the centre: keep the label
        // offset finite instead of normalizing a zero vector.
        let dir = pos - center;
        let offset = if dir.length() > 1e-3 {
            dir.normalized() * 9.0
        } else {
            Vec2::new(0.0, -9.0)
        };
        // White text with a dark halo stays readable over both tones.
        text_with_halo(
            &painter,
            pos + offset,
            label,
            FontId::monospace(10.0),
            Color32::WHITE.gamma_multiply(alpha),
            Color32::BLACK.gamma_multiply(alpha * 0.9),
        );
    }
}

/// Draw `text` with a 1px halo so it reads over both hemisphere tones.
fn text_with_halo(
    painter: &egui::Painter,
    pos: Pos2,
    text: &str,
    font: FontId,
    color: Color32,
    halo: Color32,
) {
    for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
        painter.text(
            pos + Vec2::new(dx, dy),
            Align2::CENTER_CENTER,
            text,
            font.clone(),
            halo,
        );
    }
    painter.text(pos, Align2::CENTER_CENTER, text, font, color);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_geo_frames::GeoOrigin;
    use nox::{Array, Dyn};

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

    #[test]
    fn f32_bare_quat_and_pose_parse_like_f64() {
        // Bare F32 [x,y,z,w] quaternion.
        let q = component_value_to_quat(&f32_value(&[0.0, 0.0, 0.0, 1.0])).unwrap();
        assert!(q.abs_diff_eq(DQuat::IDENTITY, 1e-6));

        // F32 world_pos-style pose: head 4 elements are the quaternion.
        let q = component_value_to_quat(&f32_value(&[0.0, 0.0, 2.0, 2.0, 10.0, 20.0, 30.0]))
            .expect("f32 pose head quat");
        assert!((q.length() - 1.0).abs() < 1e-6);
        assert!((q.z - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);

        // Non-pose F32 lengths are still rejected.
        assert_eq!(component_value_to_quat(&f32_value(&[1.0, 2.0, 3.0])), None);
    }

    #[test]
    fn quat_from_bare_4_vector_and_pose_head() {
        // Bare [x,y,z,w] quaternion.
        let q = component_value_to_quat(&f64_value(&[0.0, 0.0, 0.0, 1.0])).unwrap();
        assert!(q.abs_diff_eq(DQuat::IDENTITY, 1e-12));

        // WorldPos / SpatialTransform: head 4 elements, normalized on read.
        let q = component_value_to_quat(&f64_value(&[0.0, 0.0, 2.0, 2.0, 10.0, 20.0, 30.0]))
            .expect("pose head quat");
        assert!((q.length() - 1.0).abs() < 1e-12);
        assert!((q.z - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn non_quat_values_have_no_attitude() {
        // Position-only 3-vector must not drive the sphere via identity quat.
        assert_eq!(component_value_to_quat(&f64_value(&[1.0, 2.0, 3.0])), None);
        assert_eq!(
            component_value_to_quat(&f64_value(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            None
        );
        // Zero-length "quaternion" is telemetry garbage, not identity.
        assert_eq!(
            component_value_to_quat(&f64_value(&[0.0, 0.0, 0.0, 0.0])),
            None
        );
    }

    #[test]
    fn non_unit_4_vector_is_not_an_attitude() {
        // A 4-element component that is not near unit length (e.g. fin
        // deflections) must not be normalized into a misleading gimbal — it
        // should read as the empty "—" state, matching the position gauge's
        // refusal to treat a non-pose 4-vector as data.
        assert_eq!(
            component_value_to_quat(&f64_value(&[0.1, 0.2, 0.3, 0.4])),
            None
        );
        assert_eq!(
            component_value_to_quat(&f64_value(&[2.0, 0.0, 0.0, 0.0])),
            None
        );

        // A genuine unit quaternion that has drifted slightly is still accepted
        // and re-normalized.
        let drifted = component_value_to_quat(&f64_value(&[0.0, 0.0, 0.02, 1.0]))
            .expect("near-unit quat accepted");
        assert!((drifted.length() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn identity_attitude_puts_up_near_top() {
        // Each frame's own camera puts its triad "up" near screen-top.
        for frame in [GeoFrame::ENU, GeoFrame::NED, GeoFrame::ECEF] {
            let cam = SphereCamera::new(sphere_axis_dirs(frame));
            let u = cam.project(sphere_axis_dirs(frame)[0]);
            assert!(
                u.y > 0.6,
                "{frame:?} up should project near screen-top, got {u:?}"
            );
        }
        // A yaw of 90° about Up should move East toward where North was.
        let cam = SphereCamera::new(sphere_axis_dirs(GeoFrame::ENU));
        let q = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let e0 = cam.project(DVec3::X);
        let e1 = cam.project(q.inverse() * DVec3::X);
        assert!(
            (e0.x - e1.x).abs() > 0.3 || (e0.y - e1.y).abs() > 0.3,
            "yaw should move the E tip on the sphere"
        );
    }

    #[test]
    fn quat_readout_two_decimals_and_placeholder() {
        let q = DQuat::from_xyzw(0.123, -0.456, 0.0, 0.881);
        assert_eq!(quat_readout(Some(q)), "x +0.12  y -0.46  z +0.00  w +0.88");
        assert_eq!(quat_readout(None), "x —  y —  z —  w —");
    }

    #[test]
    fn quat_readout_is_sign_flip_invariant() {
        // q and -q are the same rotation; the readout must not flip signs.
        let q = DQuat::from_xyzw(0.123, -0.456, 0.0, 0.881);
        assert_eq!(quat_readout(Some(q)), quat_readout(Some(-q)));
        // Tie on w falls through to the next component (here z) for the sign.
        let q = DQuat::from_xyzw(0.6, -0.8, 0.0, 0.0);
        assert_eq!(quat_readout(Some(q)), quat_readout(Some(-q)));
    }

    #[test]
    fn drawn_axes_follow_frame_convention() {
        let ctx = GeoContext::from(GeoOrigin::new_from_degrees(0.0, 0.0, 0.0));

        // NED draws N, E, D — with Down pointing toward screen-bottom, which is
        // the reported expectation (see the "D vector, not the up vector" note).
        let cam = SphereCamera::new(sphere_axis_dirs(GeoFrame::NED));
        let ned = display_axes_in_source(GeoFrame::NED, GeoFrame::NED, &ctx);
        assert_eq!(
            ned.iter().map(|(_, l)| *l).collect::<Vec<_>>(),
            ["N", "E", "D"]
        );
        let (d_dir, _) = ned.iter().find(|(_, l)| *l == "D").unwrap();
        assert!(
            cam.project(*d_dir).y < -0.5,
            "NED Down tip must project below centre, got {:?}",
            cam.project(*d_dir)
        );

        // ENU still draws Up upward.
        let cam = SphereCamera::new(sphere_axis_dirs(GeoFrame::ENU));
        let enu = display_axes_in_source(GeoFrame::ENU, GeoFrame::ENU, &ctx);
        assert_eq!(
            enu.iter().map(|(_, l)| *l).collect::<Vec<_>>(),
            ["E", "N", "U"]
        );
        let (u_dir, _) = enu.iter().find(|(_, l)| *l == "U").unwrap();
        assert!(
            cam.project(*u_dir).y > 0.5,
            "ENU Up tip must project above centre"
        );

        // ECEF keeps X/Y/Z labels.
        let ecef = display_axes_in_source(GeoFrame::ECEF, GeoFrame::ECEF, &ctx);
        assert_eq!(
            ecef.iter().map(|(_, l)| *l).collect::<Vec<_>>(),
            ["X", "Y", "Z"]
        );
    }

    /// Camera-space up for a body→source quaternion, `display == source` (what
    /// the painter uses for the same-frame gauges in these tests). Markings
    /// co-rotate with the body, so the attitude is applied directly (matches
    /// `q_draw` in `paint_frame_sphere`).
    fn up_cam(frame: GeoFrame, q: DQuat) -> DVec3 {
        let cam = SphereCamera::new(sphere_axis_dirs(frame));
        cam.project(q * sphere_axis_dirs(frame)[0])
    }

    #[test]
    fn shading_identity_sky_on_top() {
        let u = up_cam(GeoFrame::ENU, DQuat::IDENTITY);
        assert!(u.y > 0.6, "up should project near screen-top, got {u:?}");
        assert!(front_point_is_sky(Vec2::new(0.0, 0.9), u), "top is sky");
        assert!(
            !front_point_is_sky(Vec2::new(0.0, -0.9), u),
            "bottom is ground"
        );
    }

    #[test]
    fn shading_inverted_flight_flips_sky() {
        // 180° roll about body X: the old asin-based horizon rocked back to
        // level here; the quaternion-driven one must show ground on top.
        let q = DQuat::from_rotation_x(std::f64::consts::PI);
        let u = up_cam(GeoFrame::ENU, q);
        assert!(!front_point_is_sky(Vec2::new(0.0, 0.9), u), "top is ground");
        assert!(front_point_is_sky(Vec2::new(0.0, -0.9), u), "bottom is sky");
    }

    #[test]
    fn shading_continuous_through_full_roll() {
        // Sweep a full 360° roll: the projected sky direction must rotate
        // continuously (no snap-back at ±90° like the old asin clamp).
        let n = 360;
        let mut prev: Option<f64> = None;
        for i in 0..=n {
            let theta = std::f64::consts::TAU * (i as f64 / n as f64);
            let u = up_cam(GeoFrame::ENU, DQuat::from_rotation_x(theta));
            let angle = u.y.atan2(u.x);
            if let Some(p) = prev {
                let mut d = (angle - p).abs();
                if d > std::f64::consts::PI {
                    d = std::f64::consts::TAU - d;
                }
                assert!(d < 0.1, "sky direction jumped {d:.3} rad at step {i}");
            }
            prev = Some(angle);
        }
    }

    #[test]
    fn gimbal_markings_corotate_with_body() {
        // Screen angle of a projected point as it is spun about ENU up.
        let cam = SphereCamera::new(sphere_axis_dirs(GeoFrame::ENU));
        let screen_angle = |v: DVec3| {
            let c = cam.project(v);
            c.y.atan2(c.x)
        };
        let unwrap_delta = |a: f64, b: f64| {
            let mut d = b - a;
            while d > std::f64::consts::PI {
                d -= std::f64::consts::TAU;
            }
            while d < -std::f64::consts::PI {
                d += std::f64::consts::TAU;
            }
            d
        };

        // Witnesses in the rotation's equatorial plane (⊥ the ENU up spin
        // axis) so their projected screen angle tracks the rotation cleanly.
        let e = DVec3::X; // east marking
        let body_ref = DVec3::Y; // north: any other body-fixed in-plane point
        let q0 = DQuat::from_rotation_z(0.1);
        let q1 = DQuat::from_rotation_z(0.2);

        // New convention (q_draw = q): the marking co-rotates with a
        // body-fixed point — same sign of screen-angle change as the cube.
        let d_mark = unwrap_delta(screen_angle(q0 * e), screen_angle(q1 * e));
        let d_body = unwrap_delta(screen_angle(q0 * body_ref), screen_angle(q1 * body_ref));
        assert!(
            d_mark * d_body > 0.0,
            "marking must turn the same way as the body: {d_mark} vs {d_body}"
        );

        // The old world-relative convention (q.inverse()) turned the opposite
        // way — that was the reported anti-correlation.
        let d_old = unwrap_delta(
            screen_angle(q0.inverse() * e),
            screen_angle(q1.inverse() * e),
        );
        assert!(
            d_mark * d_old < 0.0,
            "conjugate must reverse the previous rotation sense: {d_mark} vs {d_old}"
        );
    }

    #[test]
    fn great_circle_arcs_split_front_and_back() {
        let normal = SphereCamera::new(sphere_axis_dirs(GeoFrame::ENU)).project(DVec3::Z);
        let (front, back) = great_circle_arcs(normal).expect("tilted normal has arcs");
        assert!(!front.is_empty() && !back.is_empty());
        // Arc endpoints sit on the silhouette (unit circle) and every sample
        // stays inside the disc.
        for p in front.iter().chain(back.iter()) {
            assert!(p.length() <= 1.0 + 1e-3, "sample outside disc: {p:?}");
        }
        assert!(front.first().unwrap().length() > 0.98);
        assert!(front.last().unwrap().length() > 0.98);
        // A view-axis-aligned normal has no arcs (circle == rim).
        assert!(great_circle_arcs(DVec3::new(0.0, 0.0, 1.0)).is_none());
    }

    /// Ray-cast point-in-polygon for a simple (here convex) boundary.
    fn point_in_polygon(p: Vec2, poly: &[Vec2]) -> bool {
        let n = poly.len();
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let a = poly[i];
            let b = poly[j];
            if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Sky/ground a disc point resolves to under the actual [`horizon_fill`]
    /// plan the painter uses (base tone, overpainted by the convex cap).
    fn fill_is_sky(p: Vec2, u_cam: DVec3) -> bool {
        let arcs = great_circle_arcs(u_cam);
        let fill = horizon_fill(u_cam, &arcs);
        match &fill.cap {
            Some((boundary, cap_is_sky)) if point_in_polygon(p, boundary) => *cap_is_sky,
            _ => fill.base_is_sky,
        }
    }

    #[test]
    fn fill_matches_sky_oracle_across_silhouette() {
        // The claim behind "sky/ground flips" and "rim jumps" is that the fill
        // becomes wrong as `u_cam.z` crosses 0. Sweep a full roll (so up passes
        // edge-on to the camera twice) and assert the painted fill matches the
        // ground-truth front-hemisphere oracle everywhere except a thin band
        // around the horizon (polygon discretisation) and the rim.
        for i in 0..1440 {
            let theta = std::f64::consts::TAU * (i as f64 / 1440.0);
            let u = up_cam(GeoFrame::ENU, DQuat::from_rotation_x(theta));
            for gy in -9..=9 {
                for gx in -9..=9 {
                    let p = Vec2::new(gx as f32 * 0.1, gy as f32 * 0.1);
                    let r2 = (p.x * p.x + p.y * p.y) as f64;
                    if r2 > 0.9 * 0.9 {
                        continue; // avoid rim ambiguity
                    }
                    let z = (1.0 - r2).sqrt();
                    let signed = p.x as f64 * u.x + p.y as f64 * u.y + z * u.z;
                    if signed.abs() < 0.08 {
                        continue; // horizon band: polygon vs exact arc differ
                    }
                    assert_eq!(
                        fill_is_sky(p, u),
                        signed >= 0.0,
                        "fill disagrees with oracle at theta={theta:.4}, p={p:?}, u={u:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn grid_circles_never_snap_to_rim_during_roll() {
        // Bugbot "Great circles snap to rim" claims a display-axis grid circle's
        // normal goes parallel to the view axis during roll, so
        // `great_circle_arcs` returns `None` and the painter strokes the full
        // rim (the alleged "pop"). But the camera eye keeps an East component
        // while a roll about East holds every triad axis in the
        // East-perpendicular plane, so no axis can align with the view axis:
        // `|n.z|` stays well under 1 and `great_circle_arcs` never returns
        // `None`. Sweep a full roll for the rotating-cube camera and assert it.
        let source = GeoFrame::ENU;
        let ctx = GeoContext::from(GeoOrigin::new_from_degrees(28.6084, -80.6043, 3.0));
        let cam = SphereCamera::new(sphere_axis_dirs(source));
        let triad = display_triad_in_source(GeoFrame::NED, source, &ctx);
        let mut max_depth = 0.0_f64;
        for i in 0..2000 {
            let theta = std::f64::consts::TAU * (i as f64 / 2000.0);
            let q = DQuat::from_rotation_x(theta); // roll about East (ENU +X)
            for axis in triad {
                let normal = cam.project(q * axis);
                max_depth = max_depth.max(normal.z.abs());
                assert!(
                    great_circle_arcs(normal).is_some(),
                    "grid circle collapsed to rim at roll {theta:.4} for axis {axis:?}"
                );
            }
        }
        // Nowhere near the view axis (|n.z| = 1), let alone the 1e-6 threshold.
        assert!(
            max_depth < 0.9,
            "grid normals approached the view axis: max |n.z| = {max_depth}"
        );
    }

    #[test]
    fn horizon_cap_boundary_stays_in_disc_and_closes() {
        let u = up_cam(GeoFrame::ENU, DQuat::from_rotation_x(0.7));
        let pole = if u.z >= 0.0 { u } else { -u };
        let (front, _) = great_circle_arcs(u).expect("arcs");
        let boundary = horizon_cap_boundary(pole, &front);
        assert!(boundary.len() > front.len(), "rim arc appended");
        for p in &boundary {
            assert!(p.length() <= 1.0 + 1e-3, "boundary outside disc: {p:?}");
        }
        // The rim closure passes on the pole's side of the screen.
        let rim_mid = boundary[front.len() + (boundary.len() - front.len()) / 2];
        assert!(
            rim_mid.x as f64 * pole.x + rim_mid.y as f64 * pole.y > 0.0,
            "rim arc must close on the sky side"
        );
    }

    #[test]
    fn reference_attitude_reads_neutral_and_round_trips() {
        // Identity reference stays implicit in KDL.
        let data = OrientationGaugeData::new(None, None);
        assert_eq!(data.reference_kdl(), None);

        // A non-identity reference is normalized on load and round-trips.
        let raw = [0.0, 2.0, 0.0, 2.0]; // unnormalized 90° about Y
        let data = data.with_reference(Some(raw));
        assert!((data.reference.length() - 1.0).abs() < 1e-12);
        let kdl = data.reference_kdl().expect("non-identity serializes");
        assert!((kdl[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);

        // At the reference pose the gimbal shows identity (level).
        let q_vehicle = data.reference;
        let q_rel = q_vehicle * data.reference.inverse();
        assert!(q_rel.abs_diff_eq(DQuat::IDENTITY, 1e-12));
    }

    #[test]
    fn ecef_gauge_shows_absolute_tilt_but_local_frames_stay_level() {
        let lat = 28.6084_f64.to_radians();
        let ctx = GeoContext::from(GeoOrigin::new_from_degrees(28.6084, -80.6043, 3.0));

        // NED display of an ENU source keeps the physical local-level triad:
        // same picture as the ENU gauge (up stays up).
        let ned = display_triad_in_source(GeoFrame::NED, GeoFrame::ENU, &ctx);
        let enu = display_triad_in_source(GeoFrame::ENU, GeoFrame::ENU, &ctx);
        for (a, b) in ned.iter().zip(enu.iter()) {
            assert!((*a - *b).length() < 1e-9, "NED triad {a:?} != ENU {b:?}");
        }

        // ECEF display: the Earth axis is tilted by the colatitude, so a
        // source-identity ("flat") body must NOT read as Earth-aligned.
        let ecef = display_triad_in_source(GeoFrame::ECEF, GeoFrame::ENU, &ctx);
        let up_component = ecef[0].dot(DVec3::Z); // Earth Z vs local up
        assert!(
            (up_component - lat.sin()).abs() < 1e-9,
            "Earth axis should tilt by colatitude, got dot {up_component}"
        );

        // Rendered: the Z tip of a flat body sits well off screen-top.
        let cam = SphereCamera::new(sphere_axis_dirs(GeoFrame::ENU));
        let z_tip = cam.project(DQuat::IDENTITY.inverse() * ecef[0]);
        let u_tip = cam.project(enu[0]);
        assert!(
            (z_tip - u_tip).length() > 0.3,
            "ECEF Z tip should be visibly tilted away from local up"
        );
    }
}
