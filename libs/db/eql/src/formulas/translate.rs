use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct TranslateX;

impl super::Formula for TranslateX {
    fn name(&self) -> &'static str {
        "translate_x"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            // Store as tuple: (receiver, distance)
            Ok(Expr::Formula(
                Arc::new(TranslateX),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "translate_x requires one argument: distance".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "translate_x is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("translate_x({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        // Suggest translate_x for SpatialTransform components (7-element arrays)
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["translate_x(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct TranslateY;

impl super::Formula for TranslateY {
    fn name(&self) -> &'static str {
        "translate_y"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            Ok(Expr::Formula(
                Arc::new(TranslateY),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "translate_y requires one argument: distance".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "translate_y is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("translate_y({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["translate_y(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct TranslateZ;

impl super::Formula for TranslateZ {
    fn name(&self) -> &'static str {
        "translate_z"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            Ok(Expr::Formula(
                Arc::new(TranslateZ),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "translate_z requires one argument: distance".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "translate_z is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("translate_z({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["translate_z(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct Translate;

impl super::Formula for Translate {
    fn name(&self) -> &'static str {
        "translate"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        // Syntax: pos.translate(x, y, z)
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
                                "translate requires three arguments: x, y, z distances".to_string(),
                            ));
                        }
                    } else {
                        return Err(Error::InvalidMethodCall(
                            "translate requires three arguments: x, y, z distances".to_string(),
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
                        "translate requires three arguments: x, y, z distances".to_string(),
                    ));
                }
            } else {
                return Err(Error::InvalidMethodCall(
                    "translate requires three arguments: x, y, z distances".to_string(),
                ));
            }
        } else if args.len() == 3 {
            (args[0].clone(), args[1].clone(), args[2].clone())
        } else {
            return Err(Error::InvalidMethodCall(
                "translate requires three arguments: x, y, z distances".to_string(),
            ));
        };

        Ok(Expr::Formula(
            Arc::new(Translate),
            Box::new(Expr::Tuple(vec![recv, x_expr, y_expr, z_expr])),
        ))
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "translate is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 4
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("translate({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["translate(".to_string()];
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
    fn test_translate_x_parse() {
        let context = create_test_context();
        let expr = context.parse_str("bdx.world_pos.translate_x(1.0)").unwrap();

        // Verify it creates a Formula expression
        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "translate_x");
            // Inner should be a tuple with receiver and distance
            if let Expr::Tuple(elements) = *inner {
                assert_eq!(elements.len(), 2);
                assert!(matches!(elements[0], Expr::ComponentPart(_)));
                assert!(matches!(elements[1], Expr::FloatLiteral(1.0)));
            } else {
                panic!("Expected Tuple in Formula");
            }
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_translate_y_parse() {
        let context = create_test_context();
        let expr = context
            .parse_str("bdx.world_pos.translate_y(-2.0)")
            .unwrap();

        if let Expr::Formula(formula, _) = expr {
            assert_eq!(formula.name(), "translate_y");
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_translate_z_parse() {
        let context = create_test_context();
        let expr = context.parse_str("bdx.world_pos.translate_z(0.5)").unwrap();

        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "translate_z");
            if let Expr::Tuple(elements) = *inner {
                assert_eq!(elements.len(), 2);
                assert!(matches!(elements[1], Expr::FloatLiteral(0.5)));
            }
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_translate_parse() {
        let context = create_test_context();
        let expr = context
            .parse_str("bdx.world_pos.translate(1.0, 2.0, 3.0)")
            .unwrap();

        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "translate");
            if let Expr::Tuple(elements) = *inner {
                assert_eq!(elements.len(), 4); // receiver + 3 distances
                assert!(matches!(elements[0], Expr::ComponentPart(_)));
                assert!(matches!(elements[1], Expr::FloatLiteral(1.0)));
                assert!(matches!(elements[2], Expr::FloatLiteral(2.0)));
                assert!(matches!(elements[3], Expr::FloatLiteral(3.0)));
            }
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_translate_x_chaining() {
        let context = create_test_context();
        // Test parsing translation (will test chaining in editor compile later)
        let result = context.parse_str("bdx.world_pos.translate_x(1.0)");
        assert!(result.is_ok());
    }
}
