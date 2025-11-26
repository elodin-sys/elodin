use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Atan2;

impl super::Formula for Atan2 {
    fn name(&self) -> &'static str {
        "atan2"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        // Syntax: y.atan2(x) → atan2(y, x)
        // recv is y, args[0] is x
        // Note: Due to parser behavior, args should be a single element
        if args.len() == 1 {
            Ok(Expr::Formula(
                Arc::new(Atan2),
                Box::new(Expr::Tuple(vec![recv, args[0].clone()])),
            ))
        } else {
            Err(Error::InvalidMethodCall(
                "atan2 requires one argument: y.atan2(x)".to_string(),
            ))
        }
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        // Extract y and x from tuple
        if let Expr::Tuple(elements) = expr {
            if elements.len() == 2 {
                let y = elements[0].to_qualified_field()?;
                let x = elements[1].to_qualified_field()?;
                Ok(format!("atan2({}, {})", y, x))
            } else {
                Err(Error::InvalidMethodCall(
                    "atan2 requires 2 operands".to_string(),
                ))
            }
        } else {
            Err(Error::InvalidMethodCall(
                "atan2 requires tuple operands".to_string(),
            ))
        }
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        // For atan2, we want a descriptive column name
        if let Expr::Tuple(elements) = expr
            && elements.len() == 2
        {
            let y_name = elements[0].to_column_name().unwrap_or_default();
            let x_name = elements[1].to_column_name().unwrap_or_default();
            if !y_name.is_empty() && !x_name.is_empty() {
                return Some(format!("atan2({}, {})", y_name, x_name));
            }
        }
        None
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        // Suggest atan2 for numeric expressions
        match expr {
            Expr::ArrayAccess(_, _) | Expr::FloatLiteral(_) | Expr::BinaryOp(_, _, _) => {
                vec!["atan2(".to_string()]
            }
            _ => Vec::new(),
        }
    }

    fn to_sql(&self, expr: &Expr, _context: &Context) -> Result<String, Error> {
        // Handle tuple operands - need to collect all tables from elements
        if let Expr::Tuple(elements) = expr {
            if elements.len() == 2 {
                // Collect all tables from both elements (they might be BinaryOps or other expressions)
                use std::collections::BTreeSet;
                let mut table_names = BTreeSet::new();

                // Try to get table from first element (y)
                if let Ok(table) = elements[0].to_table() {
                    table_names.insert(table);
                }

                // Try to get table from second element (x)
                if let Ok(table) = elements[1].to_table() {
                    table_names.insert(table);
                }

                // If we have no tables (e.g., both are literals), that's an error
                if table_names.is_empty() {
                    return Err(Error::InvalidFieldAccess(
                        "atan2 requires at least one component reference".to_string(),
                    ));
                }

                let qualified_field = self.to_qualified_field(expr)?;
                let column_name = self.to_column_name(expr);
                let select_part = if let Some(col_name) = column_name {
                    format!("{} as '{}'", qualified_field, col_name)
                } else {
                    qualified_field
                };

                // Build FROM clause - if multiple tables, join them
                if table_names.len() == 1 {
                    Ok(format!(
                        "select {} from {}",
                        select_part,
                        table_names.first().unwrap()
                    ))
                } else {
                    let mut from_clause = table_names.first().unwrap().clone();
                    let first_table = from_clause.clone();
                    for table in table_names.iter().skip(1) {
                        from_clause.push_str(&format!(
                            " JOIN {} ON {}.time = {}.time",
                            table, first_table, table
                        ));
                    }
                    Ok(format!("select {} from {}", select_part, from_clause))
                }
            } else {
                Err(Error::InvalidMethodCall(
                    "atan2 requires 2 operands".to_string(),
                ))
            }
        } else {
            // Fall back to default implementation
            super::Formula::to_sql(self, expr, _context)
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

    fn create_test_components() -> (Arc<Component>, Arc<Component>) {
        let y_comp = Arc::new(Component::new(
            "a.y".to_string(),
            ComponentId::new("a.y"),
            Schema::new(PrimType::F64, Vec::<u64>::new()).unwrap(),
        ));
        let x_comp = Arc::new(Component::new(
            "a.x".to_string(),
            ComponentId::new("a.x"),
            Schema::new(PrimType::F64, Vec::<u64>::new()).unwrap(),
        ));
        (y_comp, x_comp)
    }

    fn create_test_context() -> Context {
        let (y_comp, x_comp) = create_test_components();
        Context::from_leaves([y_comp, x_comp], Timestamp(0), Timestamp(1000))
    }

    #[test]
    fn test_atan2_sql() {
        let (y_comp, x_comp) = create_test_components();
        let y_part = Arc::new(ComponentPart {
            name: "a.y".to_string(),
            id: y_comp.id,
            component: Some(y_comp),
            children: BTreeMap::new(),
        });
        let x_part = Arc::new(ComponentPart {
            name: "a.x".to_string(),
            id: x_comp.id,
            component: Some(x_comp),
            children: BTreeMap::new(),
        });
        let context = create_test_context();
        let expr = Expr::Formula(
            Arc::new(Atan2),
            Box::new(Expr::Tuple(vec![
                Expr::ComponentPart(y_part),
                Expr::ComponentPart(x_part),
            ])),
        );
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("atan2("));
        assert!(sql.contains("a_y"));
        assert!(sql.contains("a_x"));
    }

    #[test]
    fn test_atan2_parse() {
        let context = create_test_context();
        // Note: EQL parser may require parentheses around the receiver
        let expr = context.parse_str("(a.y).atan2(a.x)").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("atan2("));
    }

    #[test]
    fn test_atan2_with_arithmetic() {
        let context = create_test_context();
        // Test: (a.y * -1.0).atan2(a.x)
        let expr = context.parse_str("(a.y * -1.0).atan2(a.x)").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("atan2("));
    }

    #[test]
    fn test_atan2_with_clip_formula() {
        // Test the actual customer use case: (v_body[2] * -1.0).atan2(v_body[0].clip(...))
        let v_body_comp = Arc::new(Component::new(
            "rocket.v_body".to_string(),
            ComponentId::new("rocket.v_body"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));
        let context = Context::from_leaves([v_body_comp], Timestamp(0), Timestamp(1000));

        // This matches the actual expression from the customer
        let expr = context
            .parse_str(
                "(rocket.v_body[2] * -1.0).atan2(rocket.v_body[0].clip(0.000000000001, 999999))",
            )
            .unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("atan2("));
        assert!(sql.contains("rocket_v_body"));
    }

    #[test]
    fn test_atan2_with_float_literals() {
        // Test atan2 with literal values
        let context = create_test_context();

        // Test y.atan2(1.0) - using literal for x
        let expr = context.parse_str("(a.y).atan2(1.0)").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("atan2("));
        // Literal may be formatted as "1" or "1.0"
        assert!(sql.contains("1"));
    }

    #[test]
    fn test_atan2_chained_with_degrees() {
        // Test common pattern: atan2().degrees()
        let context = create_test_context();
        let expr = context.parse_str("(a.y).atan2(a.x).degrees()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("atan2("));
        assert!(sql.contains("degrees("));
        // degrees should wrap atan2
        assert!(sql.contains("degrees(atan2("));
    }

    #[test]
    fn test_atan2_argument_order() {
        // Verify atan2(y, x) argument order is correct
        let context = create_test_context();
        let expr = context.parse_str("(a.y).atan2(a.x)").unwrap();
        let sql = expr.to_sql(&context).unwrap();

        // Should be atan2(y, x) not atan2(x, y)
        assert!(sql.contains("atan2("));

        // Find positions of y and x in the atan2 call
        let atan2_start = sql.find("atan2(").expect("Should contain atan2");
        let y_pos = sql[atan2_start..]
            .find("a_y")
            .expect("Should contain y component");
        let x_pos = sql[atan2_start..]
            .find("a_x")
            .expect("Should contain x component");

        // y should come before x in the function call
        assert!(y_pos < x_pos, "atan2 should have y before x: atan2(y, x)");
    }
}
