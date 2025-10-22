use crate::{Context, Error, Expr, parse_duration};

pub(crate) fn parse(recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
    if let [Expr::StringLiteral(duration)] = args {
        let duration = parse_duration(duration)?;
        Ok(Expr::First(Box::new(recv), duration))
    } else {
        Err(Error::InvalidMethodCall("first".to_string()))
    }
}

pub(crate) fn to_sql(
    expr: &Expr,
    duration: &hifitime::Duration,
    context: &Context,
) -> Result<String, Error> {
    let sql = expr.to_sql(context)?;
    let duration_micros = (duration.total_nanoseconds() / 1000) as i64;
    let upper_bound = context.earliest_timestamp.0 + duration_micros;
    let upper_bound = upper_bound as f64 * 1e-6;
    let time_field = expr.to_sql_time_field()?;
    Ok(format!(
        "{sql} where {time_field} <= to_timestamp({upper_bound})",
    ))
}

pub(crate) fn component_suggestion() -> String {
    "first(".to_string()
}

pub(crate) fn keyword_suggestion() -> String {
    "first".to_string()
}

#[derive(Debug, Clone)]
pub struct First;

impl super::EqlFormula for First {
    fn name(&self) -> &'static str {
        "first"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        parse(recv, args)
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        match expr {
            Expr::ComponentPart(_) | Expr::ArrayAccess(_, _) => {
                vec!["first(".to_string()]
            }
            Expr::Tuple(_) | Expr::Fft(_) | Expr::FftFreq(_) => vec!["first".to_string()],
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Component, ComponentPart, Context, Expr, parse_duration};
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType, Timestamp};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn first_generates_expected_sql() {
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
        let expr = Expr::First(Box::new(Expr::ComponentPart(part)), duration);

        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("<= to_timestamp(0.5)"));
    }
}
