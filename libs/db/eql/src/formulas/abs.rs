use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Abs;

impl super::Formula for Abs {
    fn name(&self) -> &'static str {
        "abs"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.is_empty() {
            Ok(Expr::Formula(Arc::new(Abs), Box::new(recv)))
        } else {
            Err(Error::InvalidMethodCall(
                "abs takes no arguments".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        Ok(format!("abs({})", expr.to_qualified_field()?))
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        match expr {
            Expr::ComponentPart(_)
            | Expr::ArrayAccess(_, _)
            | Expr::FloatLiteral(_)
            | Expr::BinaryOp(_, _, _)
            | Expr::Formula(_, _) => {
                vec!["abs()".to_string()]
            }
            _ => Vec::new(),
        }
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
    fn test_abs_sql() {
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let context = create_test_context();
        let expr = Expr::Formula(Arc::new(Abs), Box::new(Expr::ComponentPart(part)));
        let sql = expr.to_sql(&context).unwrap();
        assert_eq!(
            sql,
            "select abs(a_value.a_value) as 'abs(a.value)' from a_value"
        );
    }

    #[test]
    fn test_abs_parse() {
        let context = create_test_context();
        let expr = context.parse_str("a.value.abs()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("abs("));
    }

    #[test]
    fn test_abs_with_arithmetic() {
        let context = create_test_context();
        let expr = context.parse_str("(a.value * -1.0).abs()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("abs("));
        assert!(sql.contains(" * "));
    }

    #[test]
    fn test_abs_suggestions() {
        use crate::formulas::Formula;
        let context = create_test_context();
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let expr = Expr::ComponentPart(part);
        let abs_formula = Abs;
        let suggestions = abs_formula.suggestions(&expr, &context);
        assert!(suggestions.contains(&"abs()".to_string()));
    }

    #[test]
    fn test_abs_with_negative_literal() {
        let context = create_test_context();
        let expr = context.parse_str("(a.value * -1.0).abs()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("abs("));
        assert!(sql.contains(" * -1"));
    }

    #[test]
    fn test_abs_chained() {
        // Test chaining abs with other operations
        let context = create_test_context();
        let expr = context.parse_str("a.value.abs().sqrt()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("abs("));
        assert!(sql.contains("sqrt("));
        // sqrt should wrap abs
        assert!(sql.contains("sqrt(abs("));
    }
}
