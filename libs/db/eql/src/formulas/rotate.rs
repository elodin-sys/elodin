use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct RotateX;

impl super::Formula for RotateX {
    fn name(&self) -> &'static str {
        "rotate_x"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            // Store as tuple: (receiver, angle)
            Ok(Expr::Formula(
                Arc::new(RotateX),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "rotate_x requires one argument: angle in degrees".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "rotate_x is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("rotate_x({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        // Suggest rotate_x for SpatialTransform components (7-element arrays)
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["rotate_x(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct RotateY;

impl super::Formula for RotateY {
    fn name(&self) -> &'static str {
        "rotate_y"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            Ok(Expr::Formula(
                Arc::new(RotateY),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "rotate_y requires one argument: angle in degrees".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "rotate_y is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("rotate_y({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["rotate_y(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct RotateZ;

impl super::Formula for RotateZ {
    fn name(&self) -> &'static str {
        "rotate_z"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            Ok(Expr::Formula(
                Arc::new(RotateZ),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "rotate_z requires one argument: angle in degrees".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "rotate_z is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("rotate_z({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["rotate_z(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct Rotate;

impl super::Formula for Rotate {
    fn name(&self) -> &'static str {
        "rotate"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        // Syntax: pos.rotate(x_angle, y_angle, z_angle)
        // Parser may give us a single Tuple or three separate args
        let (x_expr, y_expr, z_expr) = if args.len() == 1 {
            if let Expr::Tuple(tuple_elements) = &args[0] {
                if tuple_elements.len() == 3 {
                    (
                        tuple_elements[0].clone(),
                        tuple_elements[1].clone(),
                        tuple_elements[2].clone(),
                    )
                } else {
                    return Err(Error::InvalidMethodCall(
                        "rotate requires three arguments: x, y, z angles in degrees".to_string(),
                    ));
                }
            } else {
                return Err(Error::InvalidMethodCall(
                    "rotate requires three arguments: x, y, z angles in degrees".to_string(),
                ));
            }
        } else if args.len() == 3 {
            (args[0].clone(), args[1].clone(), args[2].clone())
        } else {
            return Err(Error::InvalidMethodCall(
                "rotate requires three arguments: x, y, z angles in degrees".to_string(),
            ));
        };

        Ok(Expr::Formula(
            Arc::new(Rotate),
            Box::new(Expr::Tuple(vec![recv, x_expr, y_expr, z_expr])),
        ))
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "rotate is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 4
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("rotate({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["rotate(".to_string()];
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Component, ComponentPart, Context};
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType, Timestamp};
    use std::collections::BTreeMap;
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
    fn test_rotate_x_parse() {
        let context = create_test_context();
        let expr = context.parse_str("bdx.world_pos.rotate_x(45.0)").unwrap();

        // Verify it creates a Formula expression
        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "rotate_x");
            // Inner should be a tuple with receiver and angle
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
    fn test_rotate_y_parse() {
        let context = create_test_context();
        let expr = context.parse_str("bdx.world_pos.rotate_y(90.0)").unwrap();

        if let Expr::Formula(formula, _) = expr {
            assert_eq!(formula.name(), "rotate_y");
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_rotate_z_parse() {
        let context = create_test_context();
        let expr = context.parse_str("bdx.world_pos.rotate_z(-90.0)").unwrap();

        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "rotate_z");
            if let Expr::Tuple(elements) = *inner {
                assert_eq!(elements.len(), 2);
                assert!(matches!(elements[1], Expr::FloatLiteral(-90.0)));
            }
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_rotate_parse() {
        let context = create_test_context();
        let expr = context
            .parse_str("bdx.world_pos.rotate(10.0, 20.0, 30.0)")
            .unwrap();

        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "rotate");
            if let Expr::Tuple(elements) = *inner {
                assert_eq!(elements.len(), 4); // receiver + 3 angles
                assert!(matches!(elements[0], Expr::ComponentPart(_)));
                assert!(matches!(elements[1], Expr::FloatLiteral(10.0)));
                assert!(matches!(elements[2], Expr::FloatLiteral(20.0)));
                assert!(matches!(elements[3], Expr::FloatLiteral(30.0)));
            }
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_rotate_z_chaining() {
        let context = create_test_context();
        // Test chaining multiple rotations (will test in editor compile later)
        let result = context.parse_str("bdx.world_pos.rotate_z(-90.0)");
        assert!(result.is_ok());
    }
}

