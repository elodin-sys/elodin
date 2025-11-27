use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct RotateWorldX;

impl super::Formula for RotateWorldX {
    fn name(&self) -> &'static str {
        "rotate_world_x"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            // Store as tuple: (receiver, angle)
            Ok(Expr::Formula(
                Arc::new(RotateWorldX),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "rotate_world_x requires one argument: angle in degrees".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "rotate_world_x is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("rotate_world_x({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        // Suggest rotate_world_x for SpatialTransform components (7-element arrays)
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["rotate_world_x(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct RotateWorldY;

impl super::Formula for RotateWorldY {
    fn name(&self) -> &'static str {
        "rotate_world_y"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            Ok(Expr::Formula(
                Arc::new(RotateWorldY),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "rotate_world_y requires one argument: angle in degrees".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "rotate_world_y is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("rotate_world_y({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["rotate_world_y(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct RotateWorldZ;

impl super::Formula for RotateWorldZ {
    fn name(&self) -> &'static str {
        "rotate_world_z"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            Ok(Expr::Formula(
                Arc::new(RotateWorldZ),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "rotate_world_z requires one argument: angle in degrees".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "rotate_world_z is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("rotate_world_z({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["rotate_world_z(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct RotateWorld;

impl super::Formula for RotateWorld {
    fn name(&self) -> &'static str {
        "rotate_world"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        // Syntax: pos.rotate_world(x, y, z)
        // Parser creates nested tuples: Tuple(Tuple(a, b), c) for f(a, b, c)
        let (x_expr, y_expr, z_expr) = if args.len() == 1 {
            if let Expr::Tuple(outer_elements) = &args[0] {
                if outer_elements.len() == 2 {
                    // Nested tuple: Tuple(Tuple(x, y), z)
                    if let Expr::Tuple(inner_elements) = &outer_elements[0] {
                        if inner_elements.len() == 2 {
                            (
                                inner_elements[0].clone(),
                                inner_elements[1].clone(),
                                outer_elements[1].clone(),
                            )
                        } else {
                            return Err(Error::InvalidMethodCall(
                                "rotate_world requires three arguments: x, y, z angles in degrees".to_string(),
                            ));
                        }
                    } else {
                        return Err(Error::InvalidMethodCall(
                            "rotate_world requires three arguments: x, y, z angles in degrees".to_string(),
                        ));
                    }
                } else if outer_elements.len() == 3 {
                    // Flat tuple (in case parser changes)
                    (
                        outer_elements[0].clone(),
                        outer_elements[1].clone(),
                        outer_elements[2].clone(),
                    )
                } else {
                    return Err(Error::InvalidMethodCall(
                        "rotate_world requires three arguments: x, y, z angles in degrees".to_string(),
                    ));
                }
            } else {
                return Err(Error::InvalidMethodCall(
                    "rotate_world requires three arguments: x, y, z angles in degrees".to_string(),
                ));
            }
        } else if args.len() == 3 {
            (args[0].clone(), args[1].clone(), args[2].clone())
        } else {
            return Err(Error::InvalidMethodCall(
                "rotate_world requires three arguments: x, y, z angles in degrees".to_string(),
            ));
        };

        Ok(Expr::Formula(
            Arc::new(RotateWorld),
            Box::new(Expr::Tuple(vec![recv, x_expr, y_expr, z_expr])),
        ))
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "rotate_world is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 4
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("rotate_world({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["rotate_world(".to_string()];
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Component, Context};
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType, Timestamp};
    use std::sync::Arc;

    fn create_test_world_pos_component() -> Arc<Component> {
        Arc::new(Component::new(
            "bdx.world_pos".to_string(),
            ComponentId::new("bdx.world_pos"),
            Schema::new(PrimType::F64, vec![7u64]).unwrap(),
        ))
    }

    fn create_test_context() -> Context {
        Context::from_leaves(
            [create_test_world_pos_component()],
            Timestamp(0),
            Timestamp(1000),
        )
    }

    #[test]
    fn test_rotate_world_x_parse() {
        let context = create_test_context();
        let expr = context.parse_str("bdx.world_pos.rotate_world_x(45.0)").unwrap();

        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "rotate_world_x");
            if let Expr::Tuple(elements) = *inner {
                assert_eq!(elements.len(), 2);
                assert!(matches!(elements[0], Expr::ComponentPart(_)));
                assert!(matches!(elements[1], Expr::FloatLiteral(45.0)));
            } else {
                panic!("Expected Tuple in Formula");
            }
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_rotate_world_z_parse() {
        let context = create_test_context();
        let expr = context.parse_str("bdx.world_pos.rotate_world_z(90.0)").unwrap();

        if let Expr::Formula(formula, _) = expr {
            assert_eq!(formula.name(), "rotate_world_z");
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_rotate_world_parse() {
        let context = create_test_context();
        let expr = context
            .parse_str("bdx.world_pos.rotate_world(10.0, 20.0, 30.0)")
            .unwrap();

        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "rotate_world");
            if let Expr::Tuple(elements) = *inner {
                assert_eq!(elements.len(), 4); // receiver + 3 angles
                assert!(matches!(elements[0], Expr::ComponentPart(_)));
            }
        } else {
            panic!("Expected Formula expression");
        }
    }
}

