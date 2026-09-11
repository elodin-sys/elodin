//! Formula `world_pos.direction(x, y, z)` returns the body-frame direction (x,y,z) transformed to world frame (editor runtime only).

use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Direction;

impl super::Formula for Direction {
    fn name(&self) -> &'static str {
        "direction"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        let flat = super::flatten_comma_args(args);
        let elems = match flat.as_slice() {
            [x, y, z] => vec![recv, x.clone(), y.clone(), z.clone()],
            [x, y, z, flag] => {
                super::orientation_flag(flag)?;
                vec![recv, x.clone(), y.clone(), z.clone(), flag.clone()]
            }
            _ => {
                return Err(Error::InvalidMethodCall(
                    "direction requires three arguments: x, y, z (optional true/false)".to_string(),
                ));
            }
        };
        Ok(Expr::Formula(
            Arc::new(Direction),
            Box::new(Expr::Tuple(elems)),
        ))
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "direction is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && (elements.len() == 4 || elements.len() == 5)
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("direction({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["direction(".to_string()];
        }
        Vec::new()
    }
}
