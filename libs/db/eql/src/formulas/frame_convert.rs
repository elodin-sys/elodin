//! Directed geo-frame conversions: `ecef_to_ned()`, `ned_to_ecef_direction()`, etc.

use crate::{Context, Error, Expr};
use bevy_geo_frames::{GeoFrame, GeoOrigin};
use bevy_math::{DMat3, DMat4};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameConvertKind {
    /// Affine point / pose position transform (`_M_`).
    Point,
    /// Rotation-only free vector / direction (`_R_`).
    Direction,
}

/// Metadata for editor runtime and SQL emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameConversion {
    pub from: GeoFrame,
    pub to: GeoFrame,
    pub kind: FrameConvertKind,
}

#[derive(Debug, Clone, Copy)]
pub struct FrameConvert {
    pub conversion: FrameConversion,
    name: &'static str,
}

impl FrameConvert {
    const fn new(name: &'static str, from: GeoFrame, to: GeoFrame, kind: FrameConvertKind) -> Self {
        Self {
            conversion: FrameConversion { from, to, kind },
            name,
        }
    }
}

pub const ECEF_TO_NED: FrameConvert = FrameConvert::new(
    "ecef_to_ned",
    GeoFrame::ECEF,
    GeoFrame::NED,
    FrameConvertKind::Point,
);
pub const NED_TO_ECEF: FrameConvert = FrameConvert::new(
    "ned_to_ecef",
    GeoFrame::NED,
    GeoFrame::ECEF,
    FrameConvertKind::Point,
);
pub const ENU_TO_NED: FrameConvert = FrameConvert::new(
    "enu_to_ned",
    GeoFrame::ENU,
    GeoFrame::NED,
    FrameConvertKind::Point,
);
pub const NED_TO_ENU: FrameConvert = FrameConvert::new(
    "ned_to_enu",
    GeoFrame::NED,
    GeoFrame::ENU,
    FrameConvertKind::Point,
);
pub const ECEF_TO_ENU: FrameConvert = FrameConvert::new(
    "ecef_to_enu",
    GeoFrame::ECEF,
    GeoFrame::ENU,
    FrameConvertKind::Point,
);
pub const ENU_TO_ECEF: FrameConvert = FrameConvert::new(
    "enu_to_ecef",
    GeoFrame::ENU,
    GeoFrame::ECEF,
    FrameConvertKind::Point,
);

pub const ECEF_TO_NED_DIRECTION: FrameConvert = FrameConvert::new(
    "ecef_to_ned_direction",
    GeoFrame::ECEF,
    GeoFrame::NED,
    FrameConvertKind::Direction,
);
pub const NED_TO_ECEF_DIRECTION: FrameConvert = FrameConvert::new(
    "ned_to_ecef_direction",
    GeoFrame::NED,
    GeoFrame::ECEF,
    FrameConvertKind::Direction,
);
pub const ENU_TO_NED_DIRECTION: FrameConvert = FrameConvert::new(
    "enu_to_ned_direction",
    GeoFrame::ENU,
    GeoFrame::NED,
    FrameConvertKind::Direction,
);
pub const NED_TO_ENU_DIRECTION: FrameConvert = FrameConvert::new(
    "ned_to_enu_direction",
    GeoFrame::NED,
    GeoFrame::ENU,
    FrameConvertKind::Direction,
);
pub const ECEF_TO_ENU_DIRECTION: FrameConvert = FrameConvert::new(
    "ecef_to_enu_direction",
    GeoFrame::ECEF,
    GeoFrame::ENU,
    FrameConvertKind::Direction,
);
pub const ENU_TO_ECEF_DIRECTION: FrameConvert = FrameConvert::new(
    "enu_to_ecef_direction",
    GeoFrame::ENU,
    GeoFrame::ECEF,
    FrameConvertKind::Direction,
);

pub fn all_frame_converts() -> [FrameConvert; 12] {
    [
        ECEF_TO_NED,
        NED_TO_ECEF,
        ENU_TO_NED,
        NED_TO_ENU,
        ECEF_TO_ENU,
        ENU_TO_ECEF,
        ECEF_TO_NED_DIRECTION,
        NED_TO_ECEF_DIRECTION,
        ENU_TO_NED_DIRECTION,
        NED_TO_ENU_DIRECTION,
        ECEF_TO_ENU_DIRECTION,
        ENU_TO_ECEF_DIRECTION,
    ]
}

/// Row-major `R` and translation `t` for `out = R * in + t` (SQL emission).
struct Affine3 {
    r: [[f64; 3]; 3],
    t: [f64; 3],
}

fn needs_geo_origin(from: GeoFrame, to: GeoFrame) -> bool {
    matches!((from, to), (GeoFrame::ECEF, _) | (_, GeoFrame::ECEF))
}

fn frame_name(frame: GeoFrame) -> &'static str {
    match frame {
        GeoFrame::ENU => "ENU",
        GeoFrame::NED => "NED",
        GeoFrame::ECEF => "ECEF",
    }
}

fn require_origin(conv: FrameConversion, origin: Option<GeoOrigin>) -> Result<GeoOrigin, Error> {
    if needs_geo_origin(conv.from, conv.to) {
        origin.ok_or_else(|| {
            Error::InvalidMethodCall(format!(
                "{}→{} conversion requires a geo origin (schematic `coordinate` lat/lon/alt)",
                frame_name(conv.from),
                frame_name(conv.to)
            ))
        })
    } else {
        Ok(origin.unwrap_or_default())
    }
}

fn affine_from_mat3(r: DMat3) -> Affine3 {
    Affine3 {
        r: [
            [r.x_axis.x, r.y_axis.x, r.z_axis.x],
            [r.x_axis.y, r.y_axis.y, r.z_axis.y],
            [r.x_axis.z, r.y_axis.z, r.z_axis.z],
        ],
        t: [0.0, 0.0, 0.0],
    }
}

fn affine_from_mat4(m: DMat4) -> Affine3 {
    let mut a = affine_from_mat3(DMat3::from_mat4(m));
    a.t = [m.w_axis.x, m.w_axis.y, m.w_axis.z];
    a
}

fn affine_for(conv: FrameConversion, origin: Option<GeoOrigin>) -> Result<Affine3, Error> {
    let origin = require_origin(conv, origin)?;
    Ok(match conv.kind {
        FrameConvertKind::Point => affine_from_mat4(conv.to.plane_M_(&conv.from, &origin)),
        FrameConvertKind::Direction => affine_from_mat3(conv.to.plane_R_(&conv.from, &origin)),
    })
}

fn fmt_sql_f64(v: f64) -> String {
    // Ensure a decimal so SQL parsers treat it as float.
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

fn sql_linear_combo(coeffs: [f64; 3], fields: [&str; 3], translation: f64) -> String {
    let mut parts = Vec::new();
    for (c, f) in coeffs.iter().zip(fields.iter()) {
        if c.abs() < 1e-18 {
            continue;
        }
        parts.push(format!("({} * {})", fmt_sql_f64(*c), f));
    }
    if translation.abs() >= 1e-18 || parts.is_empty() {
        parts.push(fmt_sql_f64(translation));
    }
    parts.join(" + ")
}

fn component_elem_count(part: &crate::ComponentPart) -> Result<usize, Error> {
    let comp = part.component.as_ref().ok_or_else(|| {
        Error::InvalidFieldAccess("frame conversion on non-leaf component".to_string())
    })?;
    let dims = comp.schema.dim();
    if dims.is_empty() {
        return Err(Error::InvalidMethodCall(
            "frame conversion on scalar component".to_string(),
        ));
    }
    Ok(dims.iter().copied().map(|d| d as usize).product())
}

fn position_start_index(n_elems: usize, kind: FrameConvertKind) -> Result<usize, Error> {
    match (n_elems, kind) {
        (3, _) => Ok(0),
        (7, FrameConvertKind::Point) => Ok(4),
        (7, FrameConvertKind::Direction) => Err(Error::InvalidMethodCall(
            "direction frame conversion expects a 3-vector, not a 7-element pose".to_string(),
        )),
        (n, _) => Err(Error::InvalidMethodCall(format!(
            "frame conversion expects a 3-vector or 7-element pose, got {n} elements"
        ))),
    }
}

/// Tuples parse as nested pairs (`(a, b, c)` → `((a, b), c)`), so flatten.
fn flatten_tuple<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    match expr {
        Expr::Tuple(elements) => elements.iter().for_each(|e| flatten_tuple(e, out)),
        e => out.push(e),
    }
}

fn tuple_elements(expr: &Expr) -> Vec<&Expr> {
    let mut out = Vec::new();
    flatten_tuple(expr, &mut out);
    out
}

/// The three source-frame SQL scalars a conversion consumes: either the xyz
/// slice of a component (`world_pos.ecef_to_ned()`) or an explicit 3-tuple
/// (`(world_pos[4], world_pos[5], world_pos[6]).ecef_to_ned()`).
fn source_fields(expr: &Expr, kind: FrameConvertKind) -> Result<[String; 3], Error> {
    match expr {
        Expr::ComponentPart(part) => {
            let start = position_start_index(component_elem_count(part)?, kind)?;
            let field = |i: usize| {
                Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), start + i)
                    .to_qualified_field()
            };
            Ok([field(0)?, field(1)?, field(2)?])
        }
        Expr::Tuple(_) => {
            let elements = tuple_elements(expr);
            let [x, y, z] = elements[..] else {
                return Err(tuple_arity_error(elements.len()));
            };
            Ok([
                x.to_qualified_field()?,
                y.to_qualified_field()?,
                z.to_qualified_field()?,
            ])
        }
        _ => Err(Error::InvalidMethodCall(
            "frame conversion SQL expects a component or 3-tuple receiver".to_string(),
        )),
    }
}

fn tuple_arity_error(got: usize) -> Error {
    Error::InvalidMethodCall(format!(
        "frame conversion on a tuple expects 3 elements, got {got}"
    ))
}

impl super::Formula for FrameConvert {
    fn name(&self) -> &'static str {
        self.name
    }

    fn frame_conversion(&self) -> Option<FrameConversion> {
        Some(self.conversion)
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if !args.is_empty() {
            return Err(Error::InvalidMethodCall(format!(
                "{}() takes no arguments",
                self.name
            )));
        }
        match &recv {
            Expr::ComponentPart(part) => {
                position_start_index(component_elem_count(part)?, self.conversion.kind)?;
            }
            Expr::Tuple(_) => {
                let n = tuple_elements(&recv).len();
                if n != 3 {
                    return Err(tuple_arity_error(n));
                }
            }
            _ => {}
        }
        Ok(Expr::Formula(Arc::new(*self), Box::new(recv)))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        expr.to_column_name()
            .map(|name| format!("{}({})", self.name, name))
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
        {
            let dims = component.schema.dim();
            if dims.is_empty() {
                return Vec::new();
            }
            let n: usize = dims.iter().copied().map(|d| d as usize).product();
            match (self.conversion.kind, n) {
                (FrameConvertKind::Point, 3 | 7) | (FrameConvertKind::Direction, 3) => {
                    return vec![format!("{}()", self.name)];
                }
                _ => {}
            }
        }
        Vec::new()
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        // Nested use would need the geo origin, which only `to_sql` receives.
        Err(Error::InvalidMethodCall(format!(
            "{}() must be the outermost formula in a query (needs the geo origin)",
            self.name
        )))
    }

    fn to_sql(&self, expr: &Expr, context: &Context) -> Result<String, Error> {
        let affine = affine_for(self.conversion, context.geo_origin)?;
        let fields = source_fields(expr, self.conversion.kind)?;
        let field_refs = [fields[0].as_str(), fields[1].as_str(), fields[2].as_str()];

        let t = match self.conversion.kind {
            FrameConvertKind::Point => affine.t,
            FrameConvertKind::Direction => [0.0, 0.0, 0.0],
        };

        let base_name = self
            .to_column_name(expr)
            .unwrap_or_else(|| self.name.to_string());
        let table = expr.to_table()?;
        let axes = ["x", "y", "z"];

        // For 7-element poses in SQL: emit only the three transformed position
        // components (attitude re-expression is editor-runtime only in v1).
        let mut selects = Vec::with_capacity(3);
        for (row, axis) in axes.iter().enumerate() {
            let expr_sql = sql_linear_combo(affine.r[row], field_refs, t[row]);
            selects.push(format!("{expr_sql} as '{base_name}.{axis}'"));
        }

        Ok(format!("select {} from {}", selects.join(", "), table))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Component, Context};
    use bevy_math::DVec3;
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType, Timestamp};
    use std::sync::Arc;

    fn ctx_with_pos(dim: u64) -> Context {
        let component = Arc::new(Component::new(
            "rocket.world_pos".to_string(),
            ComponentId::new("rocket.world_pos"),
            Schema::new(PrimType::F64, vec![dim]).unwrap(),
        ));
        let mut ctx = Context::from_leaves([component], Timestamp(0), Timestamp(1000));
        ctx.geo_origin = Some(GeoOrigin::new_from_degrees(28.5, -80.6, 0.0));
        ctx
    }

    #[test]
    fn parse_ecef_to_ned() {
        let ctx = ctx_with_pos(7);
        let expr = ctx.parse_str("rocket.world_pos.ecef_to_ned()").unwrap();
        match expr {
            Expr::Formula(f, _) => {
                assert_eq!(f.name(), "ecef_to_ned");
                assert_eq!(f.frame_conversion().unwrap().from, GeoFrame::ECEF);
            }
            _ => panic!("expected formula"),
        }
    }

    #[test]
    fn enu_to_ned_sql_permutes() {
        let ctx = ctx_with_pos(3);
        let expr = ctx.parse_str("rocket.world_pos.enu_to_ned()").unwrap();
        let sql = expr.to_sql(&ctx).unwrap();
        // ENU→NED: n=in_y, e=in_x, d=-in_z (SQL indices are 1-based).
        assert!(sql.contains("rocket_world_pos"), "{sql}");
        assert!(sql.contains("[2]"), "north from ENU y: {sql}");
        assert!(sql.contains("[1]"), "east from ENU x: {sql}");
        assert!(
            sql.contains("-1.0") || sql.contains("-("),
            "down = -up: {sql}"
        );
    }

    #[test]
    fn ecef_to_ned_requires_origin() {
        let component = Arc::new(Component::new(
            "rocket.world_pos".to_string(),
            ComponentId::new("rocket.world_pos"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));
        let ctx = Context::from_leaves([component], Timestamp(0), Timestamp(1000));
        let expr = ctx.parse_str("rocket.world_pos.ecef_to_ned()").unwrap();
        let err = expr.to_sql(&ctx).unwrap_err();
        assert!(
            err.to_string().contains("geo origin"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn ecef_to_ned_sql_from_element_tuple() {
        let ctx = ctx_with_pos(7);
        let expr = ctx
            .parse_str(
                "(rocket.world_pos[4], rocket.world_pos[5], rocket.world_pos[6]).ecef_to_ned()",
            )
            .unwrap();
        let sql = expr.to_sql(&ctx).unwrap();
        assert!(sql.starts_with("select "), "{sql}");
        for idx in ["[5]", "[6]", "[7]"] {
            assert!(sql.contains(idx), "missing {idx}: {sql}");
        }
        assert!(sql.contains(" as 'ecef_to_ned.x'"), "{sql}");
    }

    #[test]
    fn ecef_to_ned_sql_with_origin() {
        let ctx = ctx_with_pos(3);
        let expr = ctx.parse_str("rocket.world_pos.ecef_to_ned()").unwrap();
        let sql = expr.to_sql(&ctx).unwrap();
        assert!(sql.starts_with("select "), "{sql}");
        assert!(
            sql.contains(" as 'ecef_to_ned(rocket.world_pos).x'"),
            "{sql}"
        );
    }

    #[test]
    fn affine_matches_geo_frame_m() {
        let origin = GeoOrigin::new_from_degrees(28.5, -80.6, 0.0);
        let affine = affine_for(
            FrameConversion {
                from: GeoFrame::NED,
                to: GeoFrame::ECEF,
                kind: FrameConvertKind::Point,
            },
            Some(origin),
        )
        .unwrap();
        let ned = DVec3::new(10.0, -3.0, 2.0);
        let via_sql = [
            affine.r[0][0] * ned.x + affine.r[0][1] * ned.y + affine.r[0][2] * ned.z + affine.t[0],
            affine.r[1][0] * ned.x + affine.r[1][1] * ned.y + affine.r[1][2] * ned.z + affine.t[1],
            affine.r[2][0] * ned.x + affine.r[2][1] * ned.y + affine.r[2][2] * ned.z + affine.t[2],
        ];
        let via_m = GeoFrame::ECEF
            .plane_M_(&GeoFrame::NED, &origin)
            .transform_point3(ned);
        let err = ((via_sql[0] - via_m.x).powi(2)
            + (via_sql[1] - via_m.y).powi(2)
            + (via_sql[2] - via_m.z).powi(2))
        .sqrt();
        assert!(err < 1e-9, "sql affine vs plane_M_ err={err}");
    }
}
