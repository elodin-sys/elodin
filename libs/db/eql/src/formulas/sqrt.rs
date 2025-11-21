use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Sqrt;

impl super::Formula for Sqrt {
    fn name(&self) -> &'static str {
        "sqrt"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.is_empty() {
            Ok(Expr::Formula(Arc::new(Sqrt), Box::new(recv)))
        } else {
            Err(Error::InvalidMethodCall(
                "sqrt takes no arguments".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        Ok(format!("sqrt({})", expr.to_qualified_field()?))
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        match expr {
            Expr::ComponentPart(_)
            | Expr::ArrayAccess(_, _)
            | Expr::FloatLiteral(_)
            | Expr::BinaryOp(_, _, _)
            | Expr::Formula(_, _) => {
                vec!["sqrt()".to_string()]
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
    fn test_sqrt_sql() {
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let context = create_test_context();
        let expr = Expr::Formula(Arc::new(Sqrt), Box::new(Expr::ComponentPart(part)));
        let sql = expr.to_sql(&context).unwrap();
        assert_eq!(
            sql,
            "select sqrt(a_value.a_value) as 'sqrt(a.value)' from a_value"
        );
    }

    #[test]
    fn test_sqrt_parse() {
        let context = create_test_context();
        let expr = context.parse_str("a.value.sqrt()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("sqrt("));
    }

    #[test]
    fn test_sqrt_with_arithmetic() {
        let context = create_test_context();
        let expr = context.parse_str("(a.value * a.value).sqrt()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("sqrt("));
        assert!(sql.contains(" * "));
    }

    #[test]
    fn test_sqrt_chained_with_abs() {
        // Test chaining sqrt with other formulas
        let context = create_test_context();
        let expr = context.parse_str("a.value.sqrt().abs()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("sqrt("));
        assert!(sql.contains("abs("));
        // abs should wrap sqrt
        assert!(sql.contains("abs(sqrt("));
    }

    #[test]
    fn test_sqrt_with_addition() {
        // Test sqrt with sum of squares (common pattern)
        let context = create_test_context();
        let expr = context
            .parse_str("(a.value * a.value + 1.0).sqrt()")
            .unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("sqrt("));
        assert!(sql.contains(" + "));
        assert!(sql.contains(" * "));
    }
}
