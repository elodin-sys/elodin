use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Clip;

impl super::Formula for Clip {
    fn name(&self) -> &'static str {
        "clip"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        // Syntax: value.clip(min, max) → GREATEST(min, LEAST(value, max))
        // Due to comma precedence, args might be a single Tuple instead of two separate args
        let (min_expr, max_expr) = if args.len() == 1 {
            // Parser treated comma-separated args as a tuple
            if let Expr::Tuple(tuple_elements) = &args[0] {
                if tuple_elements.len() == 2 {
                    (tuple_elements[0].clone(), tuple_elements[1].clone())
                } else {
                    return Err(Error::InvalidMethodCall(
                        "clip requires two arguments: value.clip(min, max)".to_string(),
                    ));
                }
            } else {
                return Err(Error::InvalidMethodCall(
                    "clip requires two arguments: value.clip(min, max)".to_string(),
                ));
            }
        } else if args.len() == 2 {
            (args[0].clone(), args[1].clone())
        } else {
            return Err(Error::InvalidMethodCall(
                "clip requires two arguments: value.clip(min, max)".to_string(),
            ));
        };

        Ok(Expr::Formula(
            Arc::new(Clip),
            Box::new(Expr::Tuple(vec![recv, min_expr, max_expr])),
        ))
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        if let Expr::Tuple(elements) = expr {
            if elements.len() == 3 {
                let value = elements[0].to_qualified_field()?;
                let min_val = elements[1].to_qualified_field()?;
                let max_val = elements[2].to_qualified_field()?;
                Ok(format!(
                    "GREATEST({}, LEAST({}, {}))",
                    min_val, value, max_val
                ))
            } else {
                Err(Error::InvalidMethodCall(
                    "clip requires 3 operands".to_string(),
                ))
            }
        } else {
            Err(Error::InvalidMethodCall(
                "clip requires tuple operands".to_string(),
            ))
        }
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        // For clip, we want a descriptive column name
        if let Expr::Tuple(elements) = expr
            && elements.len() == 3
        {
            let value_name = elements[0].to_column_name().unwrap_or_default();
            if !value_name.is_empty() {
                return Some(format!("clip({})", value_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        match expr {
            Expr::ArrayAccess(_, _) | Expr::FloatLiteral(_) | Expr::BinaryOp(_, _, _) => {
                vec!["clip(".to_string()]
            }
            _ => Vec::new(),
        }
    }

    fn to_sql(&self, expr: &Expr, _context: &Context) -> Result<String, Error> {
        // Handle tuple operands - need to determine table from first element
        if let Expr::Tuple(elements) = expr
            && elements.len() == 3
        {
            // Get table from first element (value)
            let table = elements[0].to_table()?;
            let qualified_field = self.to_qualified_field(expr)?;
            let column_name = self.to_column_name(expr);
            let select_part = if let Some(col_name) = column_name {
                format!("{} as '{}'", qualified_field, col_name)
            } else {
                qualified_field
            };
            return Ok(format!("select {} from {}", select_part, table));
        }
        // Fall back to default implementation
        super::Formula::to_sql(self, expr, _context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Component, ComponentPart, Context, Expr};
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType, Timestamp};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn create_test_component() -> Arc<Component> {
        Arc::new(Component::new(
            "a.value".to_string(),
            ComponentId::new("a.value"),
            Schema::new(PrimType::F64, Vec::<u64>::new()).unwrap(),
        ))
    }

    fn create_test_context() -> Context {
        Context::from_leaves([create_test_component()], Timestamp(0), Timestamp(1000))
    }

    #[test]
    fn test_clip_sql() {
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let context = create_test_context();
        let expr = Expr::Formula(
            Arc::new(Clip),
            Box::new(Expr::Tuple(vec![
                Expr::ComponentPart(part),
                Expr::FloatLiteral(0.000000000001),
                Expr::FloatLiteral(999999.0),
            ])),
        );
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("GREATEST("));
        assert!(sql.contains("LEAST("));
        assert!(sql.contains("0.000000000001"));
    }

    #[test]
    fn test_clip_parse() {
        let context = create_test_context();
        // Test parsing clip with two float arguments
        // The parser should handle: a.value.clip(min, max)
        let expr = context
            .parse_str("a.value.clip(0.000000000001, 999999)")
            .unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("GREATEST("));
        assert!(sql.contains("LEAST("));
    }

    #[test]
    fn test_clip_with_float_literals() {
        let context = create_test_context();
        let expr = context.parse_str("a.value.clip(0.0, 100.0)").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("GREATEST("));
        assert!(sql.contains("LEAST("));
    }
}
