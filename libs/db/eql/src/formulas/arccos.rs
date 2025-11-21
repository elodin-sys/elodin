use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Arccos;

impl super::Formula for Arccos {
    fn name(&self) -> &'static str {
        "arccos"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.is_empty() {
            Ok(Expr::Formula(Arc::new(Arccos), Box::new(recv)))
        } else {
            Err(Error::InvalidMethodCall(
                "arccos takes no arguments".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        // Clip input to [-1, 1] to match Python's arccos behavior which raises ValueError
        // for values outside this range. PostgreSQL's acos() may return NaN for out-of-range
        // values, so we ensure the input is always in the valid domain.
        let inner = expr.to_qualified_field()?;
        Ok(format!("acos(GREATEST(-1.0, LEAST(1.0, {})))", inner))
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        match expr {
            Expr::ComponentPart(_)
            | Expr::ArrayAccess(_, _)
            | Expr::FloatLiteral(_)
            | Expr::BinaryOp(_, _, _)
            | Expr::Formula(_, _) => {
                vec!["arccos()".to_string()]
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formulas::Formula;
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
    fn test_arccos_sql() {
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let context = create_test_context();
        let expr = Expr::Formula(Arc::new(Arccos), Box::new(Expr::ComponentPart(part)));
        let sql = expr.to_sql(&context).unwrap();
        // Should clip input to [-1, 1] to match Python's arccos behavior
        assert_eq!(
            sql,
            "select acos(GREATEST(-1.0, LEAST(1.0, a_value.a_value))) as 'arccos(a.value)' from a_value"
        );
    }

    #[test]
    fn test_arccos_parse() {
        let context = create_test_context();
        let expr = context.parse_str("a.value.arccos()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("acos("));
    }

    #[test]
    fn test_arccos_with_arithmetic() {
        let context = create_test_context();
        let expr = context.parse_str("(a.value * -1.0).arccos()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("acos("));
        assert!(sql.contains("GREATEST(-1.0, LEAST(1.0"));
        assert!(sql.contains(" * "));
    }

    #[test]
    fn test_arccos_clips_input() {
        // Test that arccos clips its input to [-1, 1] to match Python's behavior
        let context = create_test_context();

        // Test with a value that would be outside [-1, 1]
        let expr = context.parse_str("a.value.arccos()").unwrap();
        let sql = expr.to_sql(&context).unwrap();

        // Should contain the clipping logic
        assert!(sql.contains("GREATEST(-1.0, LEAST(1.0"));
        assert!(sql.contains("acos("));
    }

    #[test]
    fn test_arccos_suggestions() {
        let context = create_test_context();
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let expr = Expr::ComponentPart(part);
        let arccos_formula = Arccos;
        let suggestions = arccos_formula.suggestions(&expr, &context);
        assert!(suggestions.contains(&"arccos()".to_string()));
    }
}
