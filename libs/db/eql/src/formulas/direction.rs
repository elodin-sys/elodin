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
        // Syntax: pose.direction(x, y, z) or pose.direction((x, y, z))
        let (x_expr, y_expr, z_expr) = if args.len() == 1 {
            if let Expr::Tuple(outer_elements) = &args[0] {
                if outer_elements.len() == 3 {
                    (
                        outer_elements[0].clone(),
                        outer_elements[1].clone(),
                        outer_elements[2].clone(),
                    )
                } else if outer_elements.len() == 2 {
                    if let Expr::Tuple(inner_elements) = &outer_elements[0] {
                        if inner_elements.len() == 2 {
                            (
                                inner_elements[0].clone(),
                                inner_elements[1].clone(),
                                outer_elements[1].clone(),
                            )
                        } else {
                            return Err(Error::InvalidMethodCall(
                                "direction requires three arguments: x, y, z".to_string(),
                            ));
                        }
                    } else {
                        return Err(Error::InvalidMethodCall(
                            "direction requires three arguments: x, y, z".to_string(),
                        ));
                    }
                } else {
                    return Err(Error::InvalidMethodCall(
                        "direction requires three arguments: x, y, z".to_string(),
                    ));
                }
            } else {
                return Err(Error::InvalidMethodCall(
                    "direction requires three arguments: x, y, z".to_string(),
                ));
            }
        } else if args.len() == 3 {
            (args[0].clone(), args[1].clone(), args[2].clone())
        } else {
            return Err(Error::InvalidMethodCall(
                "direction requires three arguments: x, y, z".to_string(),
            ));
        };

        Ok(Expr::Formula(
            Arc::new(Direction),
            Box::new(Expr::Tuple(vec![recv, x_expr, y_expr, z_expr])),
        ))
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "direction is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 4
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
