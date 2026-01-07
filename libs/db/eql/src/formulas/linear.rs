//! Linear formula for extracting position/velocity from spatial types.
//!
//! For SpatialTransform (7 elements): extracts indices 4-6 (position x, y, z)
//! For SpatialMotion (6 elements): extracts indices 3-5 (linear velocity x, y, z)

use crate::{Context, Error, Expr};
use std::sync::Arc;

/// Determines the start index for linear components based on array size.
/// SpatialTransform (7): position at indices 4-6
/// SpatialMotion (6): linear velocity at indices 3-5
fn linear_start_index(n_elems: usize) -> Result<usize, Error> {
    match n_elems {
        7 => Ok(4), // SpatialTransform: [qx, qy, qz, qw, px, py, pz]
        6 => Ok(3), // SpatialMotion: [ax, ay, az, vx, vy, vz]
        _ => Err(Error::InvalidMethodCall(format!(
            "linear() expects a spatial type (6 or 7 elements), got {} elements",
            n_elems
        ))),
    }
}

fn to_qualified_field(expr: &Expr) -> Result<String, Error> {
    let part = match expr {
        Expr::ComponentPart(p) => p.clone(),
        _ => {
            return Err(Error::InvalidMethodCall(
                "linear() expects a spatial component".to_string(),
            ));
        }
    };
    let comp = part
        .component
        .as_ref()
        .ok_or_else(|| Error::InvalidFieldAccess("linear() on non-leaf component".to_string()))?;
    let dims = comp.schema.dim();
    if dims.is_empty() {
        return Err(Error::InvalidMethodCall(
            "linear() on scalar component".to_string(),
        ));
    }
    let n_elems: usize = dims.iter().copied().map(|d| d as usize).product();
    let start_idx = linear_start_index(n_elems)?;

    // Extract 3 elements starting at start_idx
    let mut terms = Vec::with_capacity(3);
    for i in 0..3 {
        let qi = Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), start_idx + i)
            .to_qualified_field()?;
        terms.push(qi);
    }
    Ok(format!("[{}]", terms.join(", ")))
}

fn to_column_name(expr: &Expr) -> Option<String> {
    expr.to_column_name().map(|name| format!("linear({name})"))
}

fn parse(recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
    if args.is_empty() && matches!(recv, Expr::ComponentPart(_)) {
        return Ok(Expr::Formula(Arc::new(Linear), Box::new(recv)));
    }
    Err(Error::InvalidMethodCall("linear".to_string()))
}

#[derive(Debug, Clone)]
pub struct Linear;

impl super::Formula for Linear {
    fn name(&self) -> &'static str {
        "linear"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        parse(recv, args)
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        to_qualified_field(expr)
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        to_column_name(expr)
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
        {
            let dims = component.schema.dim();
            if !dims.is_empty() {
                let n_elems: usize = dims.iter().copied().map(|d| d as usize).product();
                // Only suggest for spatial types (6 or 7 elements)
                if n_elems == 6 || n_elems == 7 {
                    return vec!["linear()".to_string()];
                }
            }
        }
        Vec::new()
    }

    fn to_sql(&self, expr: &Expr, _context: &Context) -> Result<String, Error> {
        // Generate separate SELECT for each of the 3 linear components
        let part = match expr {
            Expr::ComponentPart(p) => p.clone(),
            _ => {
                return Err(Error::InvalidMethodCall(
                    "linear() expects a spatial component".to_string(),
                ));
            }
        };
        let comp = part.component.as_ref().ok_or_else(|| {
            Error::InvalidFieldAccess("linear() on non-leaf component".to_string())
        })?;
        let dims = comp.schema.dim();
        let n_elems: usize = dims.iter().copied().map(|d| d as usize).product();
        let start_idx = linear_start_index(n_elems)?;

        let base_name = expr.to_column_name().unwrap_or_default();
        let table = expr.to_table()?;

        let mut selects = Vec::with_capacity(3);
        let axes = ["x", "y", "z"];
        for (i, axis) in axes.iter().enumerate() {
            let qi = Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), start_idx + i)
                .to_qualified_field()?;
            selects.push(format!("{} as '{}.{}'", qi, base_name, axis));
        }

        Ok(format!("select {} from {}", selects.join(", "), table))
    }
}
