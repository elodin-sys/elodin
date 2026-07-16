use bevy::{
    ecs::system::SystemParam,
    math::{DQuat, DVec3},
    prelude::*,
};
use bevy_egui::egui::{self, Align2, Color32, FontId, Pos2, Sense, Shape, Stroke, Vec2};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoRotation, ecef_to_lla_deg};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::{EntityMap, TelemetryCache};
use impeller2_wkt::{ComponentValue, CurrentTimestamp, DisplayFrame};
use nox::ArrayBuf;
use std::f32::consts::{FRAC_PI_2, TAU};

use crate::EqlContext;
use crate::WorldPosExt;
use crate::object_3d::{CompiledExpr, ComponentArrayExt, compile_eql_expr};

use super::{
    PaneName, colors::get_scheme, monitor::render_value_cards, theme, widgets::WidgetSystem,
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
    pub source: GeoFrame,
    pub display: DisplayFrame,
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
    pub fn new(eql: String, source: GeoFrame, display: DisplayFrame) -> Self {
        Self {
            eql,
            source,
            display,
            compiled_expr: None,
            component_ids: Vec::new(),
            plain_component_id: None,
            compiled_for: None,
        }
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
        let source = data.source;
        let combo_id = egui::Id::new(("spatial_gauge_display", pane.entity));

        egui::Frame::NONE
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                // Title — component / EQL path, matching the sketch header.
                let scheme = get_scheme();
                ui.label(
                    egui::RichText::new(title)
                        .monospace()
                        .size(13.0)
                        .color(scheme.text_secondary),
                );
                ui.add_space(8.0);

                // In-panel display-frame dropdown (source stays in the inspector).
                theme::configure_input_with_border(ui.style_mut());
                egui::ComboBox::from_id_salt(combo_id)
                    .selected_text(data.display.as_str())
                    .width(96.0)
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
                // Attitude from the same SpatialTransform (quat head); optional if the
                // EQL is a bare 3-vector.
                let att_display = value.as_ref().and_then(|v| v.as_world_pos()).map(|wp| {
                    let frame = display.geo_frame().unwrap_or(GeoFrame::NED);
                    GeoRotation::relative(source, wp.att())
                        .as_frame(frame, &geo_context)
                        .1
                });

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

                ui.add_space(4.0);
                render_value_cards(ui, &cards);

                ui.add_space(8.0);
                // Frame sphere: display-frame U/E/N triad in body view, driven
                // by the SpatialTransform attitude when available.
                paint_frame_sphere(ui, display, att_display, out.is_some());
            });
    }
}

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

/// Labels drawn on the sphere rim for the selected display frame.
fn sphere_axis_labels(display: DisplayFrame) -> [&'static str; 3] {
    match display {
        // Sketch: U at top, E / N on the rim; hatched half = "down".
        DisplayFrame::NED => ["U", "E", "N"],
        DisplayFrame::ENU => ["U", "E", "N"],
        DisplayFrame::ECEF => ["Z", "Y", "X"],
        DisplayFrame::LLA => ["Z", "E", "N"],
    }
}

/// Unit axes in the display frame matching [`sphere_axis_labels`].
fn sphere_axis_dirs(display: DisplayFrame) -> [DVec3; 3] {
    match display {
        DisplayFrame::NED => [
            DVec3::new(0.0, 0.0, -1.0), // U = -D
            DVec3::new(0.0, 1.0, 0.0),  // E
            DVec3::new(1.0, 0.0, 0.0),  // N
        ],
        DisplayFrame::ENU => [
            DVec3::Z, // U
            DVec3::X, // E
            DVec3::Y, // N
        ],
        DisplayFrame::ECEF | DisplayFrame::LLA => [DVec3::Z, DVec3::Y, DVec3::X],
    }
}

/// Local "up" in the display frame (sky direction for the AI horizon).
fn display_up(display: DisplayFrame) -> DVec3 {
    sphere_axis_dirs(display)[0]
}

/// Pitch / roll of the body relative to display-frame level, looking along body +X.
/// Pitch nose-up and roll right-wing-down are positive (aircraft convention).
///
/// Uses display-frame sky (`up_display`) so NED (−Z up) and ENU (+Z up) both
/// read wings-level at identity, unlike a body-+Z atan2 that flips for NED.
fn ai_pitch_roll(q_body_to_display: DQuat, up_display: DVec3) -> (f32, f32) {
    let forward = q_body_to_display * DVec3::X;
    let right = q_body_to_display * DVec3::Y;
    let pitch = forward.dot(up_display).asin() as f32;
    let roll = (-right.dot(up_display)).asin() as f32;
    (pitch, roll)
}

/// Project a body-frame unit vector onto the gauge sphere view.
/// Returns `(screen_x, screen_y, depth)` with `+y` up and `depth > 0` in front.
fn project_body_axis(v: DVec3) -> (f32, f32, f32) {
    // Camera looks from a south-west elevated angle so U / E / N are all visible
    // at identity (U near top, E/N on the rim).
    let look = -DVec3::new(0.75, 0.55, 0.35).normalize();
    let world_up = DVec3::Z;
    let right = look.cross(world_up).normalize();
    let up = right.cross(look).normalize();
    (v.dot(right) as f32, v.dot(up) as f32, v.dot(-look) as f32)
}

/// Paint a circular frame sphere with an AI-style tilting horizon and three
/// axis labels. When `att_display` is set (body→display), the hatched ground
/// banks with roll and slides with pitch, and the U/E/N triad tracks attitude.
/// With no playhead sample (`has_sample == false`), draw a muted empty rim so
/// the sphere does not read as wings-level while the cards show "—".
fn paint_frame_sphere(
    ui: &mut egui::Ui,
    display: DisplayFrame,
    att_display: Option<DQuat>,
    has_sample: bool,
) {
    let scheme = get_scheme();
    // Respect the tile's actual available height — do not invent a 140px floor
    // that can overflow short panels.
    let size = ui
        .available_width()
        .min(ui.available_height())
        .clamp(0.0, 180.0);
    let (rect, _response) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    let painter = ui.painter_at(rect);

    let center = rect.center();
    let radius = size * 0.42;

    if !has_sample {
        painter.circle_stroke(
            center,
            radius,
            Stroke::new(1.5, scheme.border_primary.gamma_multiply(0.5)),
        );
        painter.text(
            center,
            Align2::CENTER_CENTER,
            "—",
            FontId::monospace(22.0),
            scheme.text_secondary,
        );
        return;
    }

    let q = att_display.unwrap_or(DQuat::IDENTITY);
    let (pitch, roll) = ai_pitch_roll(q, display_up(display));

    // Outer rim.
    painter.circle_stroke(center, radius, Stroke::new(1.5, scheme.border_primary));

    let hatch = scheme.border_primary.gamma_multiply(0.55);
    let ground = scheme.bg_secondary.gamma_multiply(0.85);
    let sky = scheme.bg_primary.gamma_multiply(0.4);

    paint_ai_horizon(
        &painter,
        center,
        radius,
        pitch,
        roll,
        [ground, sky, hatch, scheme.text_secondary],
    );

    // Meridian / equator curves for a bit of sphere depth (drawn lightly over horizon).
    let curve = scheme.border_primary.gamma_multiply(0.7);
    draw_ellipse_arc(
        &painter,
        center,
        radius * 0.95,
        radius * 0.35,
        -FRAC_PI_2 * 0.2,
        Stroke::new(1.0, curve),
    );
    draw_ellipse_arc(
        &painter,
        center,
        radius * 0.35,
        radius * 0.95,
        0.0,
        Stroke::new(1.0, curve),
    );

    // Fixed aircraft reference (wings + center) — body-fixed, horizon moves under it.
    let wing = scheme.text_primary;
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

    // Display-frame axes in body coordinates (identity ⇒ body aligns with frame).
    let q_inv = q.inverse();
    let labels = sphere_axis_labels(display);
    let mut tips: Vec<(f32, &'static str, Pos2, Color32, Color32)> = sphere_axis_dirs(display)
        .into_iter()
        .zip(labels)
        .map(|(dir, label)| {
            let (sx, sy, depth) = project_body_axis(q_inv * dir);
            let len = (sx * sx + sy * sy).sqrt().max(1e-4);
            let scale = (1.0_f32).min(1.0 / len);
            let pos = Pos2::new(
                center.x + radius * sx * scale,
                center.y - radius * sy * scale,
            );
            let front = depth >= 0.0;
            let tip = if front {
                scheme.text_secondary
            } else {
                scheme.text_secondary.gamma_multiply(0.35)
            };
            let text = if front {
                scheme.text_primary
            } else {
                scheme.text_primary.gamma_multiply(0.4)
            };
            (depth, label, pos, tip, text)
        })
        .collect();
    tips.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for &(_depth, label, pos, tip_color, label_color) in &tips {
        painter.line_segment([center, pos], Stroke::new(1.0, curve));
        painter.circle_filled(pos, 3.0, tip_color);
        let offset = (pos - center).normalized() * 12.0;
        painter.text(
            pos + offset,
            Align2::CENTER_CENTER,
            label,
            FontId::monospace(13.0),
            label_color,
        );
    }
}

/// Artificial-horizon fill: hatched ground / sky banked by `roll`, shifted by `pitch`.
/// `colors` = `[ground, sky, hatch, horizon]`.
fn paint_ai_horizon(
    painter: &egui::Painter,
    center: Pos2,
    radius: f32,
    pitch: f32,
    roll: f32,
    colors: [Color32; 4],
) {
    let [ground, sky, hatch, horizon_stroke] = colors;
    let (sr, cr) = roll.sin_cos();
    // Sky direction in math coords (+y up); egui y grows downward.
    let sky_math = Vec2::new(sr, cr);
    let sky_egui = Vec2::new(sky_math.x, -sky_math.y);
    let along_egui = Vec2::new(cr, sr); // horizon tangent in egui

    let pitch_c = pitch.clamp(-FRAC_PI_2 * 0.95, FRAC_PI_2 * 0.95);
    // Nose up → horizon moves down (opposite sky).
    let offset = -pitch_c.sin() * radius;
    let h_center = center + sky_egui * offset;

    let ground_poly = clip_disk_half_plane(
        center, radius, h_center, sky_egui, /*keep_sky=*/ false, 64,
    );
    let sky_poly = clip_disk_half_plane(
        center, radius, h_center, sky_egui, /*keep_sky=*/ true, 64,
    );

    if sky_poly.len() >= 3 {
        painter.add(Shape::convex_polygon(sky_poly, sky, Stroke::NONE));
    }
    if ground_poly.len() >= 3 {
        painter.add(Shape::convex_polygon(
            ground_poly.clone(),
            ground,
            Stroke::NONE,
        ));
        // Hatch lines parallel to the horizon, on the ground side only.
        for i in 1..8 {
            let t = i as f32 / 8.0;
            let mid = h_center - sky_egui * (t * radius * 1.15);
            let half = radius * 1.2;
            let a = mid - along_egui * half;
            let b = mid + along_egui * half;
            if let Some((ca, cb)) = clip_segment_to_disk(a, b, center, radius) {
                // Keep segments that sit on the ground side of the horizon.
                let mid_seg = Pos2::new((ca.x + cb.x) * 0.5, (ca.y + cb.y) * 0.5);
                let toward_sky = (mid_seg - h_center).dot(sky_egui);
                if toward_sky < 0.0 {
                    painter.line_segment([ca, cb], Stroke::new(1.0, hatch));
                }
            }
        }
    }

    // Horizon line clipped to the rim.
    let h0 = h_center - along_egui * radius * 1.2;
    let h1 = h_center + along_egui * radius * 1.2;
    if let Some((a, b)) = clip_segment_to_disk(h0, h1, center, radius) {
        painter.line_segment([a, b], Stroke::new(1.5, horizon_stroke));
    }
}

/// Polygon covering the disk ∩ half-plane on the sky (`keep_sky`) or ground side
/// of the horizon through `h_center` with sky normal `sky_egui`.
fn clip_disk_half_plane(
    center: Pos2,
    radius: f32,
    h_center: Pos2,
    sky_egui: Vec2,
    keep_sky: bool,
    n: usize,
) -> Vec<Pos2> {
    let mut pts = Vec::with_capacity(n + 3);
    let mut prev_in: Option<bool> = None;
    let mut prev_p = Pos2::ZERO;
    for i in 0..=n {
        let a = TAU * (i as f32 / n as f32);
        let p = Pos2::new(center.x + radius * a.cos(), center.y + radius * a.sin());
        let toward_sky = (p - h_center).dot(sky_egui);
        let inside = if keep_sky {
            toward_sky >= 0.0
        } else {
            toward_sky < 0.0
        };
        #[allow(clippy::collapsible_if)]
        if let Some(was_in) = prev_in {
            if was_in != inside {
                // Edge crosses the horizon — insert intersection.
                if let Some(x) = line_intersect_horizon(prev_p, p, h_center, sky_egui) {
                    pts.push(x);
                }
            }
        }
        if inside {
            pts.push(p);
        }
        prev_in = Some(inside);
        prev_p = p;
    }
    pts
}

fn line_intersect_horizon(a: Pos2, b: Pos2, h_center: Pos2, sky_egui: Vec2) -> Option<Pos2> {
    let da = (a - h_center).dot(sky_egui);
    let db = (b - h_center).dot(sky_egui);
    if (da - db).abs() < 1e-6 {
        return None;
    }
    let t = da / (da - db);
    Some(Pos2::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t))
}

fn clip_segment_to_disk(a: Pos2, b: Pos2, center: Pos2, radius: f32) -> Option<(Pos2, Pos2)> {
    // Liang–Barsky style in radial form via quadratic intersection.
    let d = b - a;
    let f = a - center;
    let a_c = d.dot(d);
    let b_c = 2.0 * f.dot(d);
    let c_c = f.dot(f) - radius * radius;
    let disc = b_c * b_c - 4.0 * a_c * c_c;
    if disc < 0.0 || a_c < 1e-12 {
        let mid = Pos2::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
        return ((mid - center).length() <= radius).then_some((a, b));
    }
    let s = disc.sqrt();
    let t0 = ((-b_c - s) / (2.0 * a_c)).clamp(0.0, 1.0);
    let t1 = ((-b_c + s) / (2.0 * a_c)).clamp(0.0, 1.0);
    if t1 < t0 {
        return None;
    }
    let p0 = Pos2::new(a.x + d.x * t0, a.y + d.y * t0);
    let p1 = Pos2::new(a.x + d.x * t1, a.y + d.y * t1);
    Some((p0, p1))
}

fn draw_ellipse_arc(
    painter: &egui::Painter,
    center: Pos2,
    rx: f32,
    ry: f32,
    rotation: f32,
    stroke: Stroke,
) {
    let n = 48;
    let mut pts = Vec::with_capacity(n + 1);
    let (sr, cr) = rotation.sin_cos();
    for i in 0..=n {
        let a = TAU * (i as f32 / n as f32);
        let (s, c) = a.sin_cos();
        let x = rx * c;
        let y = ry * s;
        pts.push(Pos2::new(
            center.x + x * cr - y * sr,
            center.y + x * sr + y * cr,
        ));
    }
    painter.add(Shape::line(pts, stroke));
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
    fn position_from_spatial_transform() {
        // quat (x,y,z,w) + pos — same layout as WorldPos / SpatialTransform.
        let v = f64_value(&[0.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0]);
        assert_eq!(
            component_value_to_position(&v),
            Some(DVec3::new(10.0, 20.0, 30.0))
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
    fn identity_attitude_puts_up_near_top() {
        let (_, sy, _) = project_body_axis(DVec3::Z);
        assert!(sy > 0.6, "ENU up should project near screen-top, sy={sy}");
        // A yaw of 90° about Up should move East toward where North was.
        let q = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let (e0x, e0y, _) = project_body_axis(DVec3::X);
        let (e1x, e1y, _) = project_body_axis(q.inverse() * DVec3::X);
        assert!(
            (e0x - e1x).abs() > 0.3 || (e0y - e1y).abs() > 0.3,
            "yaw should move the E tip on the sphere"
        );
    }

    #[test]
    fn ai_horizon_identity_is_level() {
        let (pitch, roll) = ai_pitch_roll(DQuat::IDENTITY, DVec3::Z);
        assert!(pitch.abs() < 1e-5, "pitch {pitch}");
        assert!(roll.abs() < 1e-5, "roll {roll}");
    }

    #[test]
    fn ai_horizon_nose_up_positive_pitch() {
        // Body→display = Ry(-θ): body +X (nose) tips toward display +Z (up).
        let q = DQuat::from_rotation_y(-0.4);
        let (pitch, roll) = ai_pitch_roll(q, DVec3::Z);
        assert!(pitch > 0.3, "expected nose-up pitch, got {pitch}");
        assert!(roll.abs() < 1e-4, "roll should stay ~0, got {roll}");
    }
}
