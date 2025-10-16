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
