use bevy::{
    ecs::system::SystemParam,
    math::{DQuat, DVec3},
    prelude::*,
};
use bevy_egui::egui::{self, Align2, Color32, FontId, Pos2, Sense, Shape, Stroke, Vec2};
use bevy_geo_frames::{GeoContext, GeoFrame, ecef_to_lla_deg};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::{EntityMap, TelemetryCache};
use impeller2_wkt::{ComponentValue, CurrentTimestamp, DisplayFrame};
use nox::ArrayBuf;
use std::f32::consts::TAU;

use crate::EqlContext;
use crate::WorldPosExt;
use crate::object_3d::{CompiledExpr, ComponentArrayExt, compile_eql_expr};

use super::{
    PaneName,
    colors::get_scheme,
    monitor::{CardStyle, render_value_cards},
    theme,
    widgets::WidgetSystem,
};

#[derive(Clone)]
pub struct SpatialGaugePane {
    pub entity: Entity,
    pub name: PaneName,
}

impl SpatialGaugePane {
    pub fn new(entity: Entity, name: PaneName) -> Self {
        Self { entity, name }
    }
}

/// Backing data for a [`SpatialGaugePane`]: an EQL-bound position, the frame
/// it is expressed in, and the frame/LLA to display it in. The compiled EQL is
/// kept in sync by [`compile_spatial_gauge_exprs`].
#[derive(Component)]
pub struct SpatialGaugeData {
    pub eql: String,
    /// Frame the EQL position is expressed in.
    ///
    /// `None` means inherit the schematic global [`crate::Coordinate`] (same
    /// as omitting `source` in KDL). Resolved at display/export via
    /// [`Self::effective_source`].
    pub source: Option<GeoFrame>,
    pub display: DisplayFrame,
    /// Attitude (body→source) the gimbal shows as neutral: the sphere renders
    /// `reference⁻¹ · q`. Identity means the raw component attitude.
    pub reference: DQuat,
    pub compiled_expr: Option<CompiledExpr>,
    /// Component IDs referenced by `compiled_expr`, used to resolve playhead
    /// samples from [`TelemetryCache`] (same path as the component monitor).
    component_ids: Vec<ComponentId>,
    /// When the EQL is a bare component (no formulas/ops), resolve that id
    /// directly from the cache — same as [`super::monitor::MonitorWidget`].
    plain_component_id: Option<ComponentId>,
    /// The `eql` string `compiled_expr` was built from, so recompilation only
    /// happens when the text actually changes (or a prior compile failed).
    compiled_for: Option<String>,
}

impl SpatialGaugeData {
    pub fn new(eql: String, source: Option<GeoFrame>, display: DisplayFrame) -> Self {
        Self {
            eql,
            source,
            display,
            reference: DQuat::IDENTITY,
            compiled_expr: None,
            component_ids: Vec::new(),
            plain_component_id: None,
            compiled_for: None,
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
}

/// Recompile each spatial gauge's EQL when its text changes or a previous
/// compile failed (e.g. the referenced component only became known later).
pub fn compile_spatial_gauge_exprs(
    mut gauges: Query<&mut SpatialGaugeData>,
    eql_context: Res<EqlContext>,
) {
    for mut data in gauges.iter_mut() {
        // Empty EQL is a settled state (`compiled_expr = None`). Non-empty must
        // have a successful compile; failures retry when the context catches up.
        let up_to_date = data.compiled_for.as_deref() == Some(data.eql.as_str())
            && (data.eql.trim().is_empty() || data.compiled_expr.is_some());
        if up_to_date {
            continue;
        }
        let eql = data.eql.clone();
        let (compiled, component_ids, plain_component_id) = if eql.trim().is_empty() {
            (None, Vec::new(), None)
        } else {
            match eql_context.0.parse_str(&eql) {
                Ok(expr) => {
                    let plain_component_id = match &expr {
                        eql::Expr::ComponentPart(part) => Some(part.id),
                        _ => None,
                    };
                    let mut ids = Vec::new();
                    collect_component_ids(&expr, &mut ids);
                    (compile_eql_expr(expr).ok(), ids, plain_component_id)
                }
                Err(_) => (None, Vec::new(), None),
            }
        };
        data.compiled_expr = compiled;
        data.component_ids = component_ids;
        data.plain_component_id = plain_component_id;
        data.compiled_for = Some(eql);
    }
}

/// Walk an EQL AST and collect every referenced component id.
fn collect_component_ids(expr: &eql::Expr, out: &mut Vec<ComponentId>) {
    match expr {
        eql::Expr::ComponentPart(part) => out.push(part.id),
        eql::Expr::Time(component) => out.push(component.id),
        eql::Expr::ArrayAccess(inner, _)
        | eql::Expr::Formula(_, inner)
        | eql::Expr::Last(inner, _)
        | eql::Expr::First(inner, _) => collect_component_ids(inner, out),
        eql::Expr::Tuple(exprs) => {
            for e in exprs {
                collect_component_ids(e, out);
            }
        }
        eql::Expr::BinaryOp(left, right, _) => {
            collect_component_ids(left, out);
            collect_component_ids(right, out);
        }
        eql::Expr::FloatLiteral(_) | eql::Expr::StringLiteral(_) => {}
    }
}

/// Returns true when any referenced component has cached history but no sample
/// at/before the playhead — the same gap where `apply_cached_data` leaves a
/// stale entity `ComponentValue` behind.
fn playhead_sample_missing(
    component_ids: &[ComponentId],
    telemetry_cache: &TelemetryCache,
    ts: Timestamp,
) -> bool {
    component_ids.iter().any(|id| {
        telemetry_cache.has_series(id) && telemetry_cache.get_at_or_before(id, ts).is_none()
    })
}

/// Resolve a bare component at the playhead — identical to [`super::monitor::MonitorWidget`].
fn resolve_plain_component(
    id: ComponentId,
    entity_map: &EntityMap,
    values: &Query<&ComponentValue>,
    telemetry_cache: &TelemetryCache,
    ts: Timestamp,
) -> Option<ComponentValue> {
    if let Some(cached) = telemetry_cache.get_at_or_before(&id, ts) {
        return Some(cached.clone());
    }
    if !telemetry_cache.has_series(&id) {
        let entity = entity_map.get(&id)?;
        return values.get(*entity).ok().cloned();
    }
    None
}

#[derive(SystemParam)]
pub struct SpatialGaugeWidget<'w, 's> {
    gauges: Query<'w, 's, &'static mut SpatialGaugeData>,
    entity_map: Res<'w, EntityMap>,
    values: Query<'w, 's, &'static ComponentValue>,
    telemetry_cache: Res<'w, TelemetryCache>,
    current_timestamp: Res<'w, CurrentTimestamp>,
    geo_context: Res<'w, GeoContext>,
    coordinate: Res<'w, crate::Coordinate>,
}

impl WidgetSystem for SpatialGaugeWidget<'_, '_> {
    type Args = SpatialGaugePane;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        pane: Self::Args,
    ) -> Self::Output {
        let SpatialGaugeWidget {
            mut gauges,
            entity_map,
            values,
            telemetry_cache,
            current_timestamp,
            geo_context,
            coordinate,
        } = state.get_mut(world);
        let Ok(mut data) = gauges.get_mut(pane.entity) else {
            return;
        };

        // Evaluate the pose before the ComboBox mutates `data.display`; convert
        // cards / attitude after the dropdown so a frame change applies immediately.
        let ts = current_timestamp.0;
        let value = if playhead_sample_missing(&data.component_ids, &telemetry_cache, ts) {
            None
        } else if let Some(id) = data.plain_component_id {
            resolve_plain_component(id, &entity_map, &values, &telemetry_cache, ts)
        } else {
            // Formula / multi-component EQL: entity values are synced by
            // `apply_cached_data` when samples exist; the gate above rejects gaps.
            data.compiled_expr
                .as_ref()
                .and_then(|expr| expr.execute(&entity_map, &values).ok())
        };

        let title = if data.eql.trim().is_empty() {
            pane.name.as_str().to_ascii_uppercase()
        } else {
            data.eql.to_ascii_uppercase()
        };
        // Keep inherit (`source = None`) live against `Coordinate` changes.
        let source = data.effective_source(coordinate.0);
        let combo_id = egui::Id::new(("spatial_gauge_display", pane.entity));

        egui::Frame::NONE
            .inner_margin(egui::Margin::same(4))
            .show(ui, |ui| {
                // Title — component / EQL path, matching the sketch header.
                let scheme = get_scheme();
                ui.label(
                    egui::RichText::new(title)
                        .monospace()
                        .size(10.0)
                        .color(scheme.text_secondary),
                );
                ui.add_space(3.0);

                // Two columns: dropdown + stacked value cards on the left, the
                // gimbal filling the remaining space on the right.
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(CARD_COLUMN_WIDTH);
                        // In-panel display-frame dropdown (source stays in the inspector).
                        theme::configure_input_with_border(ui.style_mut());
                        ui.style_mut()
                            .text_styles
                            .iter_mut()
                            .for_each(|(_, font)| font.size = 10.0);
                        egui::ComboBox::from_id_salt(combo_id)
                            .selected_text(data.display.as_str())
                            .width(CARD_COLUMN_WIDTH - 12.0)
                            .show_ui(ui, |ui| {
                                for frame in [
                                    DisplayFrame::NED,
                                    DisplayFrame::ENU,
                                    DisplayFrame::ECEF,
                                    DisplayFrame::LLA,
                                ] {
                                    ui.selectable_value(&mut data.display, frame, frame.as_str());
                                }
                            });

                        let display = data.display;
                        let labels = display_labels(display);
                        let out = value
                            .as_ref()
                            .and_then(component_value_to_position)
                            .map(|pos_src| convert(pos_src, source, display, &geo_context));
                        let cards: Vec<(String, String)> = labels
                            .iter()
                            .enumerate()
                            .map(|(i, label)| {
                                let value = out
                                    .map(|v| fmt_val(v[i]))
                                    .unwrap_or_else(|| "—".to_string());
                                ((*label).to_string(), value)
                            })
                            .collect();

                        ui.add_space(2.0);
                        // The column fits exactly one card, so the wrapped
                        // cards stack vertically.
                        render_value_cards(ui, &cards, &CardStyle::COMPACT);
                    });

                    // Attitude from the same SpatialTransform (quat head); optional if the
                    // EQL is a bare 3-vector.
                    // Read `display` after the ComboBox so a change applies immediately.
                    let display = data.display;
                    let reference = data.reference;
                    let att_source = value
                        .as_ref()
                        .and_then(|v| v.as_world_pos())
                        .and_then(|wp| {
                            let q = wp.att();
                            // Telemetry quats can drift off unit length, and
                            // `DQuat::inverse` assumes normalized.
                            (q.length_squared() > 1e-12).then(|| {
                                // Attitude change since the neutral pose
                                // (identity reference ⇒ raw component attitude).
                                q.normalize() * reference.inverse()
                            })
                        });
                    // Frame sphere needs a WorldPos quaternion. Position-only EQL
                    // (bare 3-vector) must not paint identity as wings-level.
                    paint_frame_sphere(ui, display, source, &geo_context, att_source);
                });
            });
    }
}

/// Width of the left column: one compact value card plus a small gutter.
const CARD_COLUMN_WIDTH: f32 = 110.0;

/// Extract a position (metres) from a component value for the spatial gauge.
///
/// Accepts only:
/// - a bare 3-vector (`F32`/`F64` with exactly three elements), or
/// - a SpatialTransform / [`WorldPos`](impeller2_wkt::WorldPos) (`F64`, ≥7
///   elements: quat + position).
///
/// Rejects other lengths (e.g. 4-element fin deflections) so the gauge does not
/// treat arbitrary trailing floats as coordinates and invent NED/LLA values.
fn component_value_to_position(value: &ComponentValue) -> Option<DVec3> {
    if let Some(wp) = value.as_world_pos() {
        return Some(wp.pos());
    }
    match value {
        ComponentValue::F32(array) => {
            let data = array.buf.as_buf();
            (data.len() == 3).then(|| DVec3::new(data[0] as f64, data[1] as f64, data[2] as f64))
        }
        ComponentValue::F64(array) => {
            let data = array.buf.as_buf();
            (data.len() == 3).then(|| DVec3::new(data[0], data[1], data[2]))
        }
        _ => None,
    }
}

/// Convert a position from `source` into the `display` coordinate system.
/// For LLA the result is `(lat_deg, lon_deg, alt_m)`.
fn convert(pos_src: DVec3, source: GeoFrame, display: DisplayFrame, ctx: &GeoContext) -> DVec3 {
    match display.geo_frame() {
        Some(frame) => frame._M_(&source, ctx).transform_point3(pos_src),
        None => {
            let ecef = GeoFrame::ECEF._M_(&source, ctx).transform_point3(pos_src);
            let (lat, lon, alt) = ecef_to_lla_deg(ecef, &ctx.origin.ellipsoid);
            DVec3::new(lat, lon, alt)
        }
    }
}

/// Axis labels for the three value cards, matching the display frame.
fn display_labels(display: DisplayFrame) -> [&'static str; 3] {
    match display {
        DisplayFrame::NED => ["N", "E", "D"],
        DisplayFrame::ENU => ["E", "N", "U"],
        DisplayFrame::ECEF => ["X", "Y", "Z"],
        DisplayFrame::LLA => ["Lat", "Lon", "Alt"],
    }
}

fn fmt_val(v: f64) -> String {
    let mut s = format!("{v:.8}");
    s.truncate(10);
    s
}

/// Cartesian frame used for the attitude sphere / AI horizon.
///
/// LLA is geodetic (lat/lon/alt), not a body→frame rotation basis — use NED so
/// the local-level triad and pitch/roll match the attitude quaternion.
fn attitude_frame(display: DisplayFrame) -> GeoFrame {
    display.geo_frame().unwrap_or(GeoFrame::NED)
}

/// Labels drawn on the sphere rim for the selected display frame.
fn sphere_axis_labels(display: DisplayFrame) -> [&'static str; 3] {
    match display {
        // Sketch: U at top, E / N on the rim; hatched half = "down".
        // LLA shares NED's local-level triad (see [`attitude_frame`]).
        DisplayFrame::NED | DisplayFrame::LLA => ["U", "E", "N"],
        DisplayFrame::ENU => ["U", "E", "N"],
        DisplayFrame::ECEF => ["Z", "Y", "X"],
    }
}

/// Unit axes in the display frame matching [`sphere_axis_labels`].
fn sphere_axis_dirs(display: DisplayFrame) -> [DVec3; 3] {
    match display {
        // LLA attitude uses NED — keep dirs in lockstep with [`attitude_frame`].
        DisplayFrame::NED | DisplayFrame::LLA => [
            DVec3::new(0.0, 0.0, -1.0), // U = -D
            DVec3::new(0.0, 1.0, 0.0),  // E
            DVec3::new(1.0, 0.0, 0.0),  // N
        ],
        DisplayFrame::ENU => [
            DVec3::Z, // U
            DVec3::X, // E
            DVec3::Y, // N
        ],
        DisplayFrame::ECEF => [DVec3::Z, DVec3::Y, DVec3::X],
    }
}

/// Numeric triad (up, east, north — or Z, Y, X for ECEF) of a source frame,
/// expressed in that frame's own coordinates. Used to build the gauge camera.
fn frame_triad(frame: GeoFrame) -> [DVec3; 3] {
    match frame {
        GeoFrame::NED => sphere_axis_dirs(DisplayFrame::NED),
        GeoFrame::ENU => sphere_axis_dirs(DisplayFrame::ENU),
        GeoFrame::ECEF => sphere_axis_dirs(DisplayFrame::ECEF),
    }
}

/// The display triad expressed in `source` coordinates — the *physical* axes
/// the gauge draws. Keeping everything in source coordinates (instead of
/// conjugating the attitude into the display frame) means an ECEF gauge shows
/// the true tilt between the body and the Earth axes (~lat-dependent), rather
/// than treating "aligned with ECEF" as level.
fn display_triad_in_source(
    display: DisplayFrame,
    source: GeoFrame,
    ctx: &GeoContext,
) -> [DVec3; 3] {
    let r = source._R_(&attitude_frame(display), ctx);
    sphere_axis_dirs(display).map(|d| r * d)
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

/// Paint a circular frame sphere with an AI-style tilting horizon and three
/// axis labels. When `att_source` is set (body→source), the black ground
/// banks with roll and slides with pitch under the white sky, and the
/// display-frame triad tracks attitude (back-facing tips stay visible,
/// dimmed, through the sphere). All math stays in source coordinates: the
/// drawn axes are the display triad's *physical* directions, so an ECEF gauge
/// shows the body's absolute tilt against the Earth axes while NED/ENU/LLA
/// (whose triads are physically identical) render the same local-level view.
/// Without attitude (no sample, or position-only 3-vector), draw a muted
/// empty rim — never treat missing attitude as identity / wings-level.
fn paint_frame_sphere(
    ui: &mut egui::Ui,
    display: DisplayFrame,
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
    // body — an ADI cockpit view whose top counter-rotates.) The horizon is
    // still the true 3D great circle ⊥ up, so any attitude renders
    // continuously; the camera is anchored to the source frame, so "level in
    // source" reads U-near-top while the triad keeps its physical direction.
    let triad = display_triad_in_source(display, source, geo_context);
    let cam = SphereCamera::new(frame_triad(source));
    let q_draw = q;
    let u_cam = cam.project(q_draw * triad[0]);
    let to_screen = |p: Vec2| Pos2::new(center.x + radius * p.x, center.y - radius * p.y);

    // Classic artificial-horizon shading: light above the horizon (sky/up),
    // dark below (ground/down). In light mode the sky is toned down so the disc
    // still separates from the pale panel background.
    let light_mode = crate::ui::colors::is_light_mode();
    let (sky, ground, hatch, horizon) = if light_mode {
        (
            Color32::from_gray(205),
            Color32::from_gray(30),
            Color32::from_gray(90),
            Color32::from_gray(120),
        )
    } else {
        (
            Color32::from_gray(232),
            Color32::from_gray(14),
            Color32::from_gray(110),
            Color32::from_gray(150),
        )
    };

    // Fill: the hemisphere whose pole faces the camera projects to a convex
    // cap; the other hemisphere wraps around the rim. Paint the wrapping one
    // over the whole disc, then the convex cap on top.
    let arcs = great_circle_arcs(u_cam);
    match &arcs {
        // Up is (anti)parallel to the view axis: all sky or all ground.
        None => {
            let fill = if u_cam.z >= 0.0 { sky } else { ground };
            painter.circle_filled(center, radius, fill);
        }
        Some((front_arc, _)) => {
            let (base, cap, pole) = if u_cam.z >= 0.0 {
                (ground, sky, u_cam)
            } else {
                (sky, ground, -u_cam)
            };
            painter.circle_filled(center, radius, base);
            let boundary = horizon_cap_boundary(pole, front_arc);
            if boundary.len() >= 3 {
                let pts: Vec<Pos2> = boundary.iter().map(|&p| to_screen(p)).collect();
                painter.add(Shape::convex_polygon(pts, cap, Stroke::NONE));
            }
        }
    }

    // Hatch the ground, lines parallel to the projected horizon. Each sample
    // is tested against the true front-hemisphere surface, so the hatching
    // hugs the curved horizon instead of a straight approximation.
    if !(arcs.is_none() && u_cam.z >= 0.0) {
        let s = Vec2::new(u_cam.x as f32, u_cam.y as f32);
        let (s_hat, t_hat) = if s.length() > 1e-6 {
            let sh = s.normalized();
            (sh, Vec2::new(-sh.y, sh.x))
        } else {
            (Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0))
        };
        const HATCH_LINES: usize = 9;
        const SAMPLES: usize = 32;
        for i in 0..HATCH_LINES {
            let d = -1.0 + 2.0 * (i as f32 + 0.5) / HATCH_LINES as f32;
            let half = (1.0 - d * d).max(0.0).sqrt() * 0.98;
            let mut run_start: Option<Vec2> = None;
            let mut run_end = Vec2::ZERO;
            for k in 0..=SAMPLES {
                let t = -half + 2.0 * half * (k as f32 / SAMPLES as f32);
                let p = s_hat * d + t_hat * t;
                let on_ground = !front_point_is_sky(p, u_cam);
                if on_ground {
                    if run_start.is_none() {
                        run_start = Some(p);
                    }
                    run_end = p;
                }
                if (!on_ground || k == SAMPLES)
                    && let Some(a) = run_start.take()
                    && (run_end - a).length() > 1e-3
                {
                    painter
                        .line_segment([to_screen(a), to_screen(run_end)], Stroke::new(1.0, hatch));
                }
            }
        }
    }

    // The display frame's coordinate planes as attitude-driven great circles;
    // back halves stay visible, dimmed, "through" the sphere.
    let curve = Color32::from_gray(128).gamma_multiply(0.8);
    for axis in triad {
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

    // Horizon line: bright front arc, dimmed back arc through the sphere.
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
    // co-rotating markings move against. Amber so it reads on white and black.
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
    let labels = sphere_axis_labels(display);
    let mut tips: Vec<(f32, &'static str, Pos2, f32)> = triad
        .into_iter()
        .zip(labels)
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
        // White text with a black halo stays readable over both halves.
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

/// Draw `text` with a 1px halo so it reads over both the white sky and the
/// black ground of the horizon.
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

    #[test]
    fn position_from_bare_xyz_vector() {
        let v = f64_value(&[1.0, 2.0, 3.0]);
        assert_eq!(
            component_value_to_position(&v),
            Some(DVec3::new(1.0, 2.0, 3.0))
        );
    }

    #[test]
    fn bare_position_vector_has_no_attitude() {
        // Regression: position-only must not drive the sphere via identity quat.
        let v = f64_value(&[1.0, 2.0, 3.0]);
        assert!(component_value_to_position(&v).is_some());
        assert!(
            v.as_world_pos().is_none(),
            "bare XYZ has position cards but no attitude for the sphere"
        );
    }

    #[test]
    fn position_from_spatial_transform() {
        // quat (x,y,z,w) + pos — same layout as WorldPos / SpatialTransform.
        let v = f64_value(&[0.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0]);
        assert_eq!(
            component_value_to_position(&v),
            Some(DVec3::new(10.0, 20.0, 30.0))
        );
        assert!(
            v.as_world_pos().is_some(),
            "SpatialTransform supplies attitude for the sphere"
        );
    }

    #[test]
    fn non_pose_multi_element_is_not_a_position() {
        // e.g. fin deflections: last three must not become fake metres.
        let fins = f64_value(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(component_value_to_position(&fins), None);
        assert_eq!(component_value_to_position(&f64_value(&[1.0, 2.0])), None);
        assert_eq!(
            component_value_to_position(&f64_value(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            None
        );
    }

    #[test]
    fn convert_ecef_to_ned_and_lla_at_origin() {
        let ctx = GeoContext::from(GeoOrigin::new_from_degrees(34.72, -86.64, 180.0));

        // ECEF position of the NED origin itself.
        let ecef = GeoFrame::ECEF
            ._M_(&GeoFrame::NED, &ctx)
            .transform_point3(DVec3::ZERO);

        // ECEF -> NED at the origin collapses to ~0.
        let ned = convert(ecef, GeoFrame::ECEF, DisplayFrame::NED, &ctx);
        assert!(ned.length() < 1e-6, "expected ~0 NED, got {ned:?}");

        // ECEF -> LLA recovers the origin's geodetic coordinates (deg/deg/m).
        let lla = convert(ecef, GeoFrame::ECEF, DisplayFrame::LLA, &ctx);
        assert!((lla.x - 34.72).abs() < 1e-6, "lat {}", lla.x);
        assert!((lla.y + 86.64).abs() < 1e-6, "lon {}", lla.y);
        assert!((lla.z - 180.0).abs() < 1e-3, "alt {}", lla.z);
    }

    #[test]
    fn convert_ned_offset_round_trips_through_ecef() {
        let ctx = GeoContext::from(GeoOrigin::new_from_degrees(0.0, 0.0, 0.0));
        let ned_in = DVec3::new(100.0, -50.0, 25.0);
        // NED -> ECEF, then display ECEF back as NED should recover the input.
        let ecef = GeoFrame::ECEF
            ._M_(&GeoFrame::NED, &ctx)
            .transform_point3(ned_in);
        let ned_back = convert(ecef, GeoFrame::ECEF, DisplayFrame::NED, &ctx);
        assert!(
            (ned_back - ned_in).length() < 1e-6,
            "round-trip mismatch: {ned_back:?} vs {ned_in:?}"
        );
    }

    #[test]
    fn display_labels_match_frame_axes() {
        assert_eq!(display_labels(DisplayFrame::NED), ["N", "E", "D"]);
        assert_eq!(display_labels(DisplayFrame::ENU), ["E", "N", "U"]);
        assert_eq!(display_labels(DisplayFrame::ECEF), ["X", "Y", "Z"]);
        assert_eq!(display_labels(DisplayFrame::LLA), ["Lat", "Lon", "Alt"]);
    }

    #[test]
    fn effective_source_inherits_coordinate_then_enu() {
        let inherit = SpatialGaugeData::new("a.pos".into(), None, DisplayFrame::NED);
        assert_eq!(inherit.effective_source(Some(GeoFrame::NED)), GeoFrame::NED);
        assert_eq!(inherit.effective_source(None), GeoFrame::ENU);

        let explicit =
            SpatialGaugeData::new("a.pos".into(), Some(GeoFrame::ECEF), DisplayFrame::NED);
        assert_eq!(
            explicit.effective_source(Some(GeoFrame::ENU)),
            GeoFrame::ECEF
        );
    }

    #[test]
    fn identity_attitude_puts_up_near_top() {
        // Each frame's own camera puts its triad "up" near screen-top.
        for frame in [GeoFrame::ENU, GeoFrame::NED, GeoFrame::ECEF] {
            let cam = SphereCamera::new(frame_triad(frame));
            let u = cam.project(frame_triad(frame)[0]);
            assert!(
                u.y > 0.6,
                "{frame:?} up should project near screen-top, got {u:?}"
            );
        }
        // A yaw of 90° about Up should move East toward where North was.
        let cam = SphereCamera::new(frame_triad(GeoFrame::ENU));
        let q = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let e0 = cam.project(DVec3::X);
        let e1 = cam.project(q.inverse() * DVec3::X);
        assert!(
            (e0.x - e1.x).abs() > 0.3 || (e0.y - e1.y).abs() > 0.3,
            "yaw should move the E tip on the sphere"
        );
    }

    /// Camera-space up for a body→source quaternion, `display == source` (what
    /// the painter uses for the same-frame gauges in these tests). Markings
    /// co-rotate with the body, so the attitude is applied directly (matches
    /// `q_draw` in `paint_frame_sphere`).
    fn up_cam(display: DisplayFrame, q: DQuat) -> DVec3 {
        let cam = SphereCamera::new(frame_triad(attitude_frame(display)));
        cam.project(q * sphere_axis_dirs(display)[0])
    }

    #[test]
    fn ai_horizon_identity_sky_on_top() {
        let u = up_cam(DisplayFrame::ENU, DQuat::IDENTITY);
        assert!(u.y > 0.6, "up should project near screen-top, got {u:?}");
        assert!(front_point_is_sky(Vec2::new(0.0, 0.9), u), "top is sky");
        assert!(
            !front_point_is_sky(Vec2::new(0.0, -0.9), u),
            "bottom is ground"
        );
    }

    #[test]
    fn ai_horizon_inverted_flight_flips_sky() {
        // 180° roll about body X: the old asin-based horizon rocked back to
        // level here; the quaternion-driven one must show ground on top.
        let q = DQuat::from_rotation_x(std::f64::consts::PI);
        let u = up_cam(DisplayFrame::ENU, q);
        assert!(!front_point_is_sky(Vec2::new(0.0, 0.9), u), "top is ground");
        assert!(front_point_is_sky(Vec2::new(0.0, -0.9), u), "bottom is sky");
    }

    #[test]
    fn ai_horizon_continuous_through_full_roll() {
        // Sweep a full 360° roll: the projected sky direction must rotate
        // continuously (no snap-back at ±90° like the old asin clamp).
        let n = 360;
        let mut prev: Option<f64> = None;
        for i in 0..=n {
            let theta = std::f64::consts::TAU * (i as f64 / n as f64);
            let u = up_cam(DisplayFrame::ENU, DQuat::from_rotation_x(theta));
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
        let cam = SphereCamera::new(frame_triad(GeoFrame::ENU));
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
        let normal = SphereCamera::new(frame_triad(GeoFrame::ENU)).project(DVec3::Z);
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

    #[test]
    fn horizon_cap_boundary_stays_in_disc_and_closes() {
        let u = up_cam(DisplayFrame::ENU, DQuat::from_rotation_x(0.7));
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
        let data = SpatialGaugeData::new("a.pos".into(), None, DisplayFrame::NED);
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
    fn lla_attitude_sphere_matches_ned_local_level() {
        // Regression: LLA used to transform attitude in NED while the sphere
        // treated up/axes as ECEF (+Z / Z,Y,X), so identity looked banked.
        assert_eq!(attitude_frame(DisplayFrame::LLA), GeoFrame::NED);
        assert_eq!(
            sphere_axis_labels(DisplayFrame::LLA),
            sphere_axis_labels(DisplayFrame::NED)
        );
        assert_eq!(
            sphere_axis_dirs(DisplayFrame::LLA),
            sphere_axis_dirs(DisplayFrame::NED)
        );
        // Identity in the NED triad still puts local-level sky at screen top.
        let u = up_cam(DisplayFrame::LLA, DQuat::IDENTITY);
        assert!(u.y > 0.6, "LLA identity up should be near top, got {u:?}");
    }

    #[test]
    fn ecef_gauge_shows_absolute_tilt_but_local_frames_stay_level() {
        let lat = 28.6084_f64.to_radians();
        let ctx = GeoContext::from(GeoOrigin::new_from_degrees(28.6084, -80.6043, 3.0));

        // NED display of an ENU source keeps the physical local-level triad:
        // same picture as the ENU gauge (up stays up).
        let ned = display_triad_in_source(DisplayFrame::NED, GeoFrame::ENU, &ctx);
        let enu = display_triad_in_source(DisplayFrame::ENU, GeoFrame::ENU, &ctx);
        for (a, b) in ned.iter().zip(enu.iter()) {
            assert!((*a - *b).length() < 1e-9, "NED triad {a:?} != ENU {b:?}");
        }

        // ECEF display: the Earth axis is tilted by the colatitude, so a
        // source-identity ("flat") body must NOT read as Earth-aligned.
        let ecef = display_triad_in_source(DisplayFrame::ECEF, GeoFrame::ENU, &ctx);
        let up_component = ecef[0].dot(DVec3::Z); // Earth Z vs local up
        assert!(
            (up_component - lat.sin()).abs() < 1e-9,
            "Earth axis should tilt by colatitude, got dot {up_component}"
        );

        // Rendered: the Z tip of a flat body sits well off screen-top.
        let cam = SphereCamera::new(frame_triad(GeoFrame::ENU));
        let z_tip = cam.project(DQuat::IDENTITY.inverse() * ecef[0]);
        let u_tip = cam.project(enu[0]);
        assert!(
            (z_tip - u_tip).length() > 0.3,
            "ECEF Z tip should be visibly tilted away from local up"
        );
    }
}
