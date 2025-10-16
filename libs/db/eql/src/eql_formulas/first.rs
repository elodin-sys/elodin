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
