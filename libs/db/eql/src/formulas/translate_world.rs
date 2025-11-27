use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct TranslateWorldX;

impl super::Formula for TranslateWorldX {
    fn name(&self) -> &'static str {
        "translate_world_x"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            // Store as tuple: (receiver, distance)
            Ok(Expr::Formula(
                Arc::new(TranslateWorldX),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "translate_world_x requires one argument: distance".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "translate_world_x is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("translate_world_x({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["translate_world_x(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct TranslateWorldY;

impl super::Formula for TranslateWorldY {
    fn name(&self) -> &'static str {
        "translate_world_y"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            Ok(Expr::Formula(
                Arc::new(TranslateWorldY),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "translate_world_y requires one argument: distance".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "translate_world_y is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("translate_world_y({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["translate_world_y(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct TranslateWorldZ;

impl super::Formula for TranslateWorldZ {
    fn name(&self) -> &'static str {
        "translate_world_z"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.len() == 1 {
            Ok(Expr::Formula(
                Arc::new(TranslateWorldZ),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "translate_world_z requires one argument: distance".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "translate_world_z is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("translate_world_z({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["translate_world_z(".to_string()];
        }
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct TranslateWorld;

impl super::Formula for TranslateWorld {
    fn name(&self) -> &'static str {
        "translate_world"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        // Syntax: pos.translate_world(x, y, z)
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
                                "translate_world requires three arguments: x, y, z distances".to_string(),
                            ));
                        }
                    } else {
                        return Err(Error::InvalidMethodCall(
                            "translate_world requires three arguments: x, y, z distances".to_string(),
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
                        "translate_world requires three arguments: x, y, z distances".to_string(),
                    ));
                }
            } else {
                return Err(Error::InvalidMethodCall(
                    "translate_world requires three arguments: x, y, z distances".to_string(),
                ));
            }
        } else if args.len() == 3 {
            (args[0].clone(), args[1].clone(), args[2].clone())
        } else {
            return Err(Error::InvalidMethodCall(
                "translate_world requires three arguments: x, y, z distances".to_string(),
            ));
        };

        Ok(Expr::Formula(
            Arc::new(TranslateWorld),
            Box::new(Expr::Tuple(vec![recv, x_expr, y_expr, z_expr])),
        ))
    }

    fn to_qualified_field(&self, _expr: &Expr) -> Result<String, Error> {
        Err(Error::InvalidMethodCall(
            "translate_world is only supported in editor runtime, not in SQL queries".to_string(),
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Tuple(elements) = expr
            && elements.len() == 4
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("translate_world({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && component.name.ends_with(".world_pos")
        {
            return vec!["translate_world(".to_string()];
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
    fn test_translate_world_x_parse() {
        let context = create_test_context();
        let expr = context
            .parse_str("bdx.world_pos.translate_world_x(-8.0)")
            .unwrap();

        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "translate_world_x");
            if let Expr::Tuple(elements) = *inner {
                assert_eq!(elements.len(), 2);
                assert!(matches!(elements[0], Expr::ComponentPart(_)));
                assert!(matches!(elements[1], Expr::FloatLiteral(-8.0)));
            } else {
                panic!("Expected Tuple in Formula");
            }
        } else {
            panic!("Expected Formula expression");
        }
    }

    #[test]
    fn test_translate_world_parse_debug() {
        let context = create_test_context();
        let result = context.parse_str("bdx.world_pos.translate_world(-8.0, -8.0, 4.0)");
        if let Err(e) = &result {
            eprintln!("Parse error: {}", e);
        }
        result.unwrap();
    }

    #[test]
    fn test_translate_world_parse() {
        let context = create_test_context();
        let expr = context
            .parse_str("bdx.world_pos.translate_world(-8.0, -8.0, 4.0)")
            .unwrap();

        if let Expr::Formula(formula, inner) = expr {
            assert_eq!(formula.name(), "translate_world");
            if let Expr::Tuple(elements) = *inner {
                assert_eq!(elements.len(), 4);
                assert!(matches!(elements[0], Expr::ComponentPart(_)));
                assert!(matches!(elements[1], Expr::FloatLiteral(-8.0)));
                assert!(matches!(elements[2], Expr::FloatLiteral(-8.0)));
                assert!(matches!(elements[3], Expr::FloatLiteral(4.0)));
            }
        } else {
            panic!("Expected Formula expression");
        }
    }
}

