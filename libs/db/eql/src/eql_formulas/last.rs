use crate::{Context, Error, Expr, parse_duration};

pub(crate) fn parse(recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
    if let [Expr::StringLiteral(duration)] = args {
        let duration = parse_duration(duration)?;
        Ok(Expr::Last(Box::new(recv), duration))
    } else {
        Err(Error::InvalidMethodCall("last".to_string()))
    }
}

pub(crate) fn to_sql(
    expr: &Expr,
    duration: &hifitime::Duration,
    context: &Context,
) -> Result<String, Error> {
    let sql = expr.to_sql(context)?;
    let duration_micros = (duration.total_nanoseconds() / 1000) as i64;
    let lower_bound = context.last_timestamp.0 - duration_micros;
    let lower_bound = lower_bound as f64 * 1e-6;
    let time_field = expr.to_sql_time_field()?;
    Ok(format!(
        "{sql} where {time_field} >= to_timestamp({lower_bound})",
    ))
}

pub(crate) fn component_suggestion() -> String {
    "last(".to_string()
}

pub(crate) fn keyword_suggestion() -> String {
    "last".to_string()
}

#[cfg(test)]
mod tests {
    use crate::{Component, ComponentPart, Context, Expr, parse_duration};
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType, Timestamp};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn last_generates_expected_sql() {
        let component = Arc::new(Component::new(
            "a.world_pos".to_string(),
            ComponentId::new("a.world_pos"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));
        let part = Arc::new(ComponentPart {
            name: "a.world_pos".to_string(),
            id: component.id,
            component: Some(component.clone()),
            children: BTreeMap::new(),
        });
        let context = Context::new(BTreeMap::new(), Timestamp(0), Timestamp(1_000_000));
        let duration = parse_duration("PT0.5S").unwrap();
        let expr = Expr::Last(Box::new(Expr::ComponentPart(part)), duration);

        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains(">= to_timestamp(0.5)"));
    }
}
