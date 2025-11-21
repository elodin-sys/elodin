use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Degrees;

impl super::Formula for Degrees {
    fn name(&self) -> &'static str {
        "degrees"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.is_empty() {
            Ok(Expr::Formula(Arc::new(Degrees), Box::new(recv)))
        } else {
            Err(Error::InvalidMethodCall(
                "degrees takes no arguments".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        Ok(format!("degrees({})", expr.to_qualified_field()?))
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        match expr {
            Expr::ComponentPart(_)
            | Expr::ArrayAccess(_, _)
            | Expr::FloatLiteral(_)
            | Expr::BinaryOp(_, _, _)
            | Expr::Formula(_, _) => {
                vec!["degrees()".to_string()]
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
            "a.angle".to_string(),
            ComponentId::new("a.angle"),
            Schema::new(PrimType::F64, Vec::<u64>::new()).unwrap(),
        ))
    }

    fn create_test_context() -> Context {
        Context::from_leaves([create_test_component()], Timestamp(0), Timestamp(1000))
    }

    #[test]
    fn test_degrees_sql() {
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.angle".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let context = create_test_context();
        let expr = Expr::Formula(Arc::new(Degrees), Box::new(Expr::ComponentPart(part)));
        let sql = expr.to_sql(&context).unwrap();
        assert_eq!(
            sql,
            "select degrees(a_angle.a_angle) as 'degrees(a.angle)' from a_angle"
        );
    }

    #[test]
    fn test_degrees_parse() {
        let context = create_test_context();
        let expr = context.parse_str("a.angle.degrees()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("degrees("));
    }

    #[test]
    fn test_degrees_suggestions() {
        let context = create_test_context();
        let part = context.component_parts.get("a").unwrap();
        let component = part.children.get("angle").unwrap();
        let expr = Expr::ComponentPart(Arc::new(component.clone()));
        let suggestions = context.get_suggestions(&expr);
        // degrees() should be suggested for component parts
        assert!(
            suggestions.contains(&"degrees()".to_string()),
            "Suggestions: {:?}",
            suggestions
        );
    }
}
