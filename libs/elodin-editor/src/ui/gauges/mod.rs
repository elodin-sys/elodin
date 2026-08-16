//! EQL-bound gauge panels sharing the same telemetry-resolution machinery
//! ([`EqlBinding`]). The position gauge shows three converted coordinates; the
//! orientation gauge shows an attitude gimbal (split so numbers next to a
//! gimbal are never mistaken for the orientation itself); the horizon gauge
//! shows a cockpit-view artificial horizon.

pub mod geo_position;
pub mod horizon;
pub mod orientation;

use bevy::{math::DVec3, prelude::*};
use bevy_egui::egui::{self, Align2, Color32, FontId, Pos2, Vec2};
use bevy_geo_frames::GeoContext;
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::{EntityMap, TelemetryCache};
use impeller2_wkt::ComponentValue;

use crate::EqlContext;
use crate::object_3d::{CompiledExpr, compile_eql_expr_with_geo};

use super::PaneName;

pub use geo_position::{GeoPositionGaugeData, GeoPositionGaugeWidget};
pub use horizon::{HorizonGaugeData, HorizonGaugeWidget};
pub use orientation::{OrientationGaugeData, OrientationGaugeWidget};

/// Max deviation of a bare 4-vector's length² from 1 to still count as a
/// quaternion. Loose enough for telemetry drift / un-renormalized integration,
/// tight enough that arbitrary 4-vectors (e.g. fin deflections) are rejected.
pub(crate) const BARE_QUAT_UNIT_TOLERANCE: f64 = 0.1;

/// Read a numeric component buffer as `f64`, accepting both `F32` and `F64`
/// telemetry so gauges treat single- and double-precision poses identically.
/// Returns `None` for non-float component types.
pub(crate) fn component_buf_f64(value: &ComponentValue) -> Option<Vec<f64>> {
    use nox::ArrayBuf;
    match value {
        ComponentValue::F32(array) => Some(array.buf.as_buf().iter().map(|&x| x as f64).collect()),
        ComponentValue::F64(array) => Some(array.buf.as_buf().to_vec()),
        _ => None,
    }
}

/// Extract a position (metres) from a component value.
///
/// Accepts only (in `F32` or `F64`):
/// - a bare 3-vector (exactly three elements), or
/// - a SpatialTransform / [`WorldPos`](impeller2_wkt::WorldPos) (≥7 elements:
///   quat `[x, y, z, w]` + position `[x, y, z]`).
///
/// Rejects other lengths (e.g. 4-element fin deflections) so a consumer does not
/// treat arbitrary trailing floats as coordinates and invent coordinate values.
///
/// Shared with the viewport's `look_at`, which wants the same rule: having two
/// definitions of what counts as a position is how they drift apart.
pub(crate) fn component_value_to_position(value: &ComponentValue) -> Option<DVec3> {
    let data = component_buf_f64(value)?;
    // world_pos-style pose: position is elements 4..7 (after the head quaternion).
    if data.len() >= 7 {
        return Some(DVec3::new(data[4], data[5], data[6]));
    }
    (data.len() == 3).then(|| DVec3::new(data[0], data[1], data[2]))
}

/// Tile pane for either gauge: points at the entity carrying the gauge data
/// and its [`EqlBinding`].
#[derive(Clone)]
pub struct GaugePane {
    pub entity: Entity,
    pub name: PaneName,
}

impl GaugePane {
    pub fn new(entity: Entity, name: PaneName) -> Self {
        Self { entity, name }
    }
}

/// An EQL expression bound to the telemetry cache, kept compiled by
/// [`compile_gauge_exprs`]. Spawned alongside each gauge's data component.
#[derive(Component)]
pub struct EqlBinding {
    pub eql: String,
    compiled_expr: Option<CompiledExpr>,
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

impl EqlBinding {
    pub fn new(eql: String) -> Self {
        Self {
            eql,
            compiled_expr: None,
            component_ids: Vec::new(),
            plain_component_id: None,
            compiled_for: None,
        }
    }

    /// Value at the playhead, or `None` while any referenced component has
    /// history but no sample at/before it (the gap where `apply_cached_data`
    /// leaves a stale entity `ComponentValue` behind).
    pub fn resolve<'b>(
        &self,
        entity_map: &EntityMap,
        values: &Query<'b, 'b, &'static ComponentValue>,
        telemetry_cache: &TelemetryCache,
        ts: Timestamp,
    ) -> Option<ComponentValue> {
        if self.component_ids.iter().any(|id| {
            telemetry_cache.has_series(id) && telemetry_cache.get_at_or_before(id, ts).is_none()
        }) {
            return None;
        }
        if let Some(id) = self.plain_component_id {
            if let Some(cached) = telemetry_cache.get_at_or_before(&id, ts) {
                return Some(cached.clone());
            }
            if !telemetry_cache.has_series(&id) {
                let entity = entity_map.get(&id)?;
                return values.get(*entity).ok().cloned();
            }
            return None;
        }
        // Formula / multi-component EQL: entity values are synced by
        // `apply_cached_data` when samples exist; the gate above rejects gaps.
        self.compiled_expr
            .as_ref()
            .and_then(|expr| expr.execute(entity_map, values).ok())
    }

    #[cfg(test)]
    fn take_compiled_expr(&mut self) -> Option<CompiledExpr> {
        self.compiled_expr.take()
    }
}

/// Recompile each gauge's EQL when its text changes or a previous compile
/// failed (e.g. the referenced component only became known later).
pub fn compile_gauge_exprs(
    mut bindings: Query<&mut EqlBinding>,
    eql_context: Res<EqlContext>,
    geo_context: Res<GeoContext>,
) {
    for mut binding in bindings.iter_mut() {
        // Empty EQL is a settled state (`compiled_expr = None`). Non-empty must
        // have a successful compile; failures retry when the context catches up.
        let up_to_date = binding.compiled_for.as_deref() == Some(binding.eql.as_str())
            && (binding.eql.trim().is_empty() || binding.compiled_expr.is_some());
        if up_to_date {
            continue;
        }
        let eql = binding.eql.clone();
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
                    (
                        compile_eql_expr_with_geo(expr, &geo_context).ok(),
                        ids,
                        plain_component_id,
                    )
                }
                Err(_) => (None, Vec::new(), None),
            }
        };
        binding.compiled_expr = compiled;
        binding.component_ids = component_ids;
        binding.plain_component_id = plain_component_id;
        binding.compiled_for = Some(eql);
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

/// Panel title: the EQL text, or the pane name while the EQL is empty.
fn gauge_title(eql: &str, name: &PaneName) -> String {
    if eql.trim().is_empty() {
        name.as_str().to_ascii_uppercase()
    } else {
        eql.to_ascii_uppercase()
    }
}

/// Uniform inner padding for both gauge panels, so sibling gauges line up.
pub(crate) const GAUGE_PANEL_MARGIN: i8 = 6;

/// Claims the whole gauge panel as a click target, so clicking it can put the
/// gauge in the inspector.
///
/// A gauge pane living directly in a split gets no tab bar, and the title it
/// draws is painted text, so without this there is nothing to click and the
/// inspector is unreachable. Call it *before* drawing the panel's contents:
/// widgets registered later keep their own clicks.
pub(crate) fn panel_select_target(ui: &mut egui::Ui, gauge: Entity) -> egui::Response {
    ui.interact(
        ui.max_rect(),
        ui.id().with(("gauge_panel", gauge)),
        egui::Sense::click(),
    )
}

/// Panel header shared by both gauges: the [`gauge_title`] in muted 10px mono,
/// followed by a small gap before the panel body.
pub(crate) fn gauge_header(ui: &mut egui::Ui, title: &str) {
    ui.label(
        egui::RichText::new(title)
            .monospace()
            .size(10.0)
            .color(crate::ui::colors::get_scheme().text_secondary),
    );
    ui.add_space(3.0);
}

/// Draw `text` with a 1px halo so it reads over both hemisphere tones.
pub(crate) fn text_with_halo(
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

/// Style the in-panel display ComboBox identically in both gauges: bordered
/// input styling with 10px text. Scoped to the calling `ui`, so it does not
/// leak into sibling panels.
pub(crate) fn style_gauge_combo(ui: &mut egui::Ui) {
    let style = ui.style_mut();
    crate::ui::theme::configure_input_with_border(style);
    style
        .text_styles
        .iter_mut()
        .for_each(|(_, font)| font.size = 10.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EqlContext;
    use crate::ui::widgets::SystemStateExt;
    use bevy::ecs::system::SystemState;
    use bevy_geo_frames::{GeoFrame, GeoOrigin};
    use impeller2::schema::Schema;
    use impeller2::types::PrimType;
    use impeller2_bevy::EntityMap;
    use nox::Array;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn compile_gauge_exprs_uses_schematic_origin() {
        let origin = GeoOrigin::new_from_degrees(28.5, -80.6, 0.0);
        let geo = GeoContext::from(origin);
        let origin_ecef = GeoFrame::ECEF
            ._M_(&GeoFrame::NED, &geo)
            .transform_point3(DVec3::ZERO);

        let component = Arc::new(eql::Component::new(
            "rocket.world_pos".to_string(),
            ComponentId::new("rocket.world_pos"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));
        let component_id = component.id;
        let eql_ctx = eql::Context::from_leaves([component], Timestamp(0), Timestamp(1000));

        let mut app = App::new();
        app.insert_resource(EqlContext(eql_ctx));
        app.insert_resource(geo);
        app.world_mut()
            .spawn(EqlBinding::new("rocket.world_pos.ecef_to_ned()".into()));

        app.world_mut()
            .run_system_cached(compile_gauge_exprs)
            .expect("compile_gauge_exprs");

        let compiled = {
            let mut query = app.world_mut().query::<&mut EqlBinding>();
            query
                .iter_mut(app.world_mut())
                .next()
                .expect("binding")
                .take_compiled_expr()
                .expect("compiled")
        };

        let mut world = World::new();
        let entity = world
            .spawn(ComponentValue::F64(
                Array::<f64, nox::Dyn>::from_shape_vec(
                    smallvec::smallvec![3],
                    vec![origin_ecef.x, origin_ecef.y, origin_ecef.z],
                )
                .unwrap(),
            ))
            .id();
        let entity_map = EntityMap(HashMap::from([(component_id, entity)]));
        let mut system_state: SystemState<(Query<'static, 'static, &ComponentValue>,)> =
            SystemState::new(&mut world);
        let (values,) = system_state.params(&world);
        let out = compiled.execute(&entity_map, &values).expect("execute");
        let pos = component_value_to_position(&out).expect("position");
        assert!(
            pos.length() < 1e-6,
            "ECEF of schematic origin must map to ~0 NED, got {pos:?}"
        );
    }
}
