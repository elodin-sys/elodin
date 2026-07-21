//! Geo-position and orientation gauges: two EQL-bound panels that share the
//! same telemetry-resolution machinery ([`EqlBinding`]). The position gauge
//! shows three converted coordinates; the orientation gauge shows an attitude
//! gimbal — split so numbers next to a gimbal are never mistaken for the
//! orientation itself.

pub mod geo_position;
pub mod orientation;

use bevy::prelude::*;
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::{EntityMap, TelemetryCache};
use impeller2_wkt::ComponentValue;

use crate::EqlContext;
use crate::object_3d::{CompiledExpr, compile_eql_expr};

use super::PaneName;

pub use geo_position::{GeoPositionGaugeData, GeoPositionGaugeWidget};
pub use orientation::{OrientationGaugeData, OrientationGaugeWidget};

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
}

/// Recompile each gauge's EQL when its text changes or a previous compile
/// failed (e.g. the referenced component only became known later).
pub fn compile_gauge_exprs(mut bindings: Query<&mut EqlBinding>, eql_context: Res<EqlContext>) {
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
                    (compile_eql_expr(expr).ok(), ids, plain_component_id)
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
