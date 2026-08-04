//! Directed geo-frame conversions: `ecef_to_ned()`, `ned_to_ecef_direction()`, etc.

use crate::geo::{self, Affine3, FrameId, GeoOrigin};
use crate::{Context, Error, Expr};
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
    pub from: FrameId,
    pub to: FrameId,
    pub kind: FrameConvertKind,
}

#[derive(Debug, Clone, Copy)]
pub struct FrameConvert {
    pub conversion: FrameConversion,
    name: &'static str,
}

impl FrameConvert {
    const fn new(name: &'static str, from: FrameId, to: FrameId, kind: FrameConvertKind) -> Self {
        Self {
            conversion: FrameConversion { from, to, kind },
            name,
        }
    }
}

pub const ECEF_TO_NED: FrameConvert = FrameConvert::new(
    "ecef_to_ned",
    FrameId::Ecef,
    FrameId::Ned,
    FrameConvertKind::Point,
);
pub const NED_TO_ECEF: FrameConvert = FrameConvert::new(
    "ned_to_ecef",
    FrameId::Ned,
    FrameId::Ecef,
    FrameConvertKind::Point,
);
pub const ENU_TO_NED: FrameConvert = FrameConvert::new(
    "enu_to_ned",
    FrameId::Enu,
    FrameId::Ned,
    FrameConvertKind::Point,
);
pub const NED_TO_ENU: FrameConvert = FrameConvert::new(
    "ned_to_enu",
    FrameId::Ned,
    FrameId::Enu,
    FrameConvertKind::Point,
);
pub const ECEF_TO_ENU: FrameConvert = FrameConvert::new(
    "ecef_to_enu",
    FrameId::Ecef,
    FrameId::Enu,
    FrameConvertKind::Point,
);
pub const ENU_TO_ECEF: FrameConvert = FrameConvert::new(
    "enu_to_ecef",
    FrameId::Enu,
    FrameId::Ecef,
    FrameConvertKind::Point,
);

pub const ECEF_TO_NED_DIRECTION: FrameConvert = FrameConvert::new(
    "ecef_to_ned_direction",
    FrameId::Ecef,
    FrameId::Ned,
    FrameConvertKind::Direction,
);
pub const NED_TO_ECEF_DIRECTION: FrameConvert = FrameConvert::new(
    "ned_to_ecef_direction",
    FrameId::Ned,
    FrameId::Ecef,
    FrameConvertKind::Direction,
);
pub const ENU_TO_NED_DIRECTION: FrameConvert = FrameConvert::new(
    "enu_to_ned_direction",
    FrameId::Enu,
    FrameId::Ned,
    FrameConvertKind::Direction,
);
pub const NED_TO_ENU_DIRECTION: FrameConvert = FrameConvert::new(
    "ned_to_enu_direction",
    FrameId::Ned,
    FrameId::Enu,
    FrameConvertKind::Direction,
);
pub const ECEF_TO_ENU_DIRECTION: FrameConvert = FrameConvert::new(
    "ecef_to_enu_direction",
    FrameId::Ecef,
    FrameId::Enu,
    FrameConvertKind::Direction,
);
pub const ENU_TO_ECEF_DIRECTION: FrameConvert = FrameConvert::new(
    "enu_to_ecef_direction",
    FrameId::Enu,
    FrameId::Ecef,
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

fn affine_for(conv: FrameConversion, origin: Option<GeoOrigin>) -> Result<Affine3, Error> {
    let result = match conv.kind {
        FrameConvertKind::Point => geo::point_affine(conv.from, conv.to, origin),
        FrameConvertKind::Direction => geo::direction_affine(conv.from, conv.to, origin),
    };
    result.map_err(Error::InvalidMethodCall)
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

fn component_elem_count(expr: &Expr) -> Result<usize, Error> {
    let Expr::ComponentPart(part) = expr else {
        return Err(Error::InvalidMethodCall(
            "frame conversion SQL expects a component receiver".to_string(),
        ));
    };
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

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        // Used when nested; emit a bracket list of transformed components.
        let n = component_elem_count(expr)?;
        let start = position_start_index(n, self.conversion.kind)?;
        // Without Context we can't bake ECEF; require the select/to_sql path.
        let _ = start;
        Err(Error::InvalidMethodCall(format!(
            "{}() must be compiled via to_sql (needs geo origin for ECEF conversions)",
            self.name
        )))
    }

    fn to_sql(&self, expr: &Expr, context: &Context) -> Result<String, Error> {
        let n = component_elem_count(expr)?;
        let start = position_start_index(n, self.conversion.kind)?;
        let affine = affine_for(self.conversion, context.geo_origin)?;

        let Expr::ComponentPart(part) = expr else {
            return Err(Error::InvalidMethodCall(
                "frame conversion SQL expects a component receiver".to_string(),
            ));
        };

        let mut fields = Vec::with_capacity(3);
        for i in 0..3 {
            fields.push(
                Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), start + i)
                    .to_qualified_field()?,
            );
        }
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
        ctx.geo_origin = Some(GeoOrigin::new(28.5, -80.6, 0.0));
        ctx
    }

    #[test]
    fn parse_ecef_to_ned() {
        let ctx = ctx_with_pos(7);
        let expr = ctx.parse_str("rocket.world_pos.ecef_to_ned()").unwrap();
        match expr {
            Expr::Formula(f, _) => {
                assert_eq!(f.name(), "ecef_to_ned");
                assert_eq!(f.frame_conversion().unwrap().from, FrameId::Ecef);
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
}
