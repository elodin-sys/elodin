//! Angular formula for extracting quaternion/angular velocity from spatial types.
//!
//! For SpatialTransform (7 elements): extracts indices 0-3 (quaternion x, y, z, w)
//! For SpatialMotion (6 elements): extracts indices 0-2 (angular velocity x, y, z)

use crate::{Context, Error, Expr};
use std::sync::Arc;

/// Determines the number of angular components based on array size.
/// SpatialTransform (7): quaternion at indices 0-3 (4 elements)
/// SpatialMotion (6): angular velocity at indices 0-2 (3 elements)
fn angular_count(n_elems: usize) -> Result<usize, Error> {
    match n_elems {
        7 => Ok(4), // SpatialTransform: quaternion [qx, qy, qz, qw]
        6 => Ok(3), // SpatialMotion: angular velocity [ax, ay, az]
        _ => Err(Error::InvalidMethodCall(format!(
            "angular() expects a spatial type (6 or 7 elements), got {} elements",
            n_elems
        ))),
    }
}

fn to_qualified_field(expr: &Expr) -> Result<String, Error> {
    let part = match expr {
        Expr::ComponentPart(p) => p.clone(),
        _ => {
            return Err(Error::InvalidMethodCall(
                "angular() expects a spatial component".to_string(),
            ));
        }
    };
    let comp = part
        .component
        .as_ref()
        .ok_or_else(|| Error::InvalidFieldAccess("angular() on non-leaf component".to_string()))?;
    let dims = comp.schema.dim();
    if dims.is_empty() {
        return Err(Error::InvalidMethodCall(
            "angular() on scalar component".to_string(),
        ));
    }
    let n_elems: usize = dims.iter().copied().map(|d| d as usize).product();
    let count = angular_count(n_elems)?;

    // Extract elements starting at index 0
    let mut terms = Vec::with_capacity(count);
    for i in 0..count {
        let qi = Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), i)
            .to_qualified_field()?;
        terms.push(qi);
    }
    Ok(format!("[{}]", terms.join(", ")))
}

fn to_column_name(expr: &Expr) -> Option<String> {
    expr.to_column_name().map(|name| format!("angular({name})"))
}

fn parse(recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
    if args.is_empty() && matches!(recv, Expr::ComponentPart(_)) {
        return Ok(Expr::Formula(Arc::new(Angular), Box::new(recv)));
    }
    Err(Error::InvalidMethodCall("angular".to_string()))
}

#[derive(Debug, Clone)]
pub struct Angular;

impl super::Formula for Angular {
    fn name(&self) -> &'static str {
        "angular"
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
                    return vec!["angular()".to_string()];
                }
            }
        }
        Vec::new()
    }

    fn to_sql(&self, expr: &Expr, _context: &Context) -> Result<String, Error> {
        // Generate separate SELECT for each angular component
        let part = match expr {
            Expr::ComponentPart(p) => p.clone(),
            _ => {
                return Err(Error::InvalidMethodCall(
                    "angular() expects a spatial component".to_string(),
                ));
            }
        };
        let comp = part.component.as_ref().ok_or_else(|| {
            Error::InvalidFieldAccess("angular() on non-leaf component".to_string())
        })?;
        let dims = comp.schema.dim();
        let n_elems: usize = dims.iter().copied().map(|d| d as usize).product();
        let count = angular_count(n_elems)?;

        let base_name = expr.to_column_name().unwrap_or_default();
        let table = expr.to_table()?;

        let mut selects = Vec::with_capacity(count);
        let axes = if count == 4 {
            vec!["x", "y", "z", "w"] // quaternion
        } else {
            vec!["x", "y", "z"] // angular velocity
        };

        for (i, axis) in axes.iter().enumerate() {
            let qi = Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), i)
                .to_qualified_field()?;
            selects.push(format!("{} as '{}.{}'", qi, base_name, axis));
        }

        Ok(format!("select {} from {}", selects.join(", "), table))
    }
}

