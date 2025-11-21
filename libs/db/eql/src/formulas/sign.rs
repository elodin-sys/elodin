use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Sign;

impl super::Formula for Sign {
    fn name(&self) -> &'static str {
        "sign"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.is_empty() {
            Ok(Expr::Formula(Arc::new(Sign), Box::new(recv)))
        } else {
            Err(Error::InvalidMethodCall(
                "sign takes no arguments".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        // DataFusion (the SQL backend used by elodin-db) doesn't have a native sign() function.
        // Implement it using a CASE statement that returns -1 for negative, 0 for zero, 1 for positive.
        // This matches the behavior of NumPy's sign() and PostgreSQL's sign() functions.
        let inner = expr.to_qualified_field()?;
        Ok(format!(
            "CASE WHEN {} > 0 THEN 1 WHEN {} < 0 THEN -1 ELSE 0 END",
            inner, inner
        ))
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        match expr {
            Expr::ComponentPart(_)
            | Expr::ArrayAccess(_, _)
            | Expr::FloatLiteral(_)
            | Expr::BinaryOp(_, _, _)
            | Expr::Formula(_, _) => {
                vec!["sign()".to_string()]
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
    fn test_sign_sql() {
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let context = create_test_context();
        let expr = Expr::Formula(Arc::new(Sign), Box::new(Expr::ComponentPart(part)));
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("CASE WHEN"));
        assert!(sql.contains("THEN 1"));
        assert!(sql.contains("THEN -1"));
        assert!(sql.contains("ELSE 0 END"));
    }

    #[test]
    fn test_sign_parse() {
        let context = create_test_context();
        let expr = context.parse_str("a.value.sign()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("CASE WHEN"));
        assert!(sql.contains("THEN 1"));
        assert!(sql.contains("THEN -1"));
        assert!(sql.contains("ELSE 0 END"));
    }

    #[test]
    fn test_sign_with_arithmetic() {
        let context = create_test_context();
        let expr = context.parse_str("(a.value * -1.0).sign()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("CASE WHEN"));
        assert!(sql.contains("THEN 1"));
        assert!(sql.contains("THEN -1"));
        assert!(sql.contains("ELSE 0 END"));
        assert!(sql.contains(" * "));
    }

    #[test]
    fn test_sign_suggestions() {
        let context = create_test_context();
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let expr = Expr::ComponentPart(part);
        let sign_formula = Sign;
        let suggestions = sign_formula.suggestions(&expr, &context);
        assert!(suggestions.contains(&"sign()".to_string()));
    }

    #[test]
    fn test_sign_with_negation() {
        // Test sign of negated value (common pattern in AoA calculation)
        let context = create_test_context();
        let expr = context.parse_str("(a.value * -1.0).sign()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("CASE WHEN"));
        assert!(sql.contains("THEN 1"));
        assert!(sql.contains("THEN -1"));
        assert!(sql.contains("ELSE 0 END"));
        assert!(sql.contains(" * -1"));
    }

    #[test]
    fn test_sign_case_statement_structure() {
        // Verify the CASE statement is correctly structured
        let context = create_test_context();
        let expr = context.parse_str("a.value.sign()").unwrap();
        let sql = expr.to_sql(&context).unwrap();

        // Should have positive case first, then negative, then zero
        let case_pos = sql.find("> 0 THEN 1").expect("Should have positive case");
        let case_neg = sql.find("< 0 THEN -1").expect("Should have negative case");
        let case_zero = sql.find("ELSE 0").expect("Should have zero case");

        // Verify order: positive < negative < zero
        assert!(
            case_pos < case_neg,
            "Positive case should come before negative"
        );
        assert!(
            case_neg < case_zero,
            "Negative case should come before zero"
        );
    }
}
