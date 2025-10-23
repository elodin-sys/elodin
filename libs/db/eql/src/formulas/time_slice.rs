use crate::{Context, Error, Expr, parse_duration};
use hifitime::Duration;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum TimeSlice {
    First(Option<Duration>),
    Last(Option<Duration>),
}

impl TimeSlice {
    fn duration(&self) -> Option<&Duration> {
        match self {
            TimeSlice::First(d) => d.as_ref(),
            TimeSlice::Last(d) => d.as_ref(),
        }
    }

    fn with_duration(&self, d: Duration) -> Self {
        match self {
            TimeSlice::First(_) => TimeSlice::First(Some(d)),
            TimeSlice::Last(_) => TimeSlice::Last(Some(d)),
        }
    }
}

impl super::Formula for TimeSlice {
    fn name(&self) -> &'static str {
        match self {
            TimeSlice::First(_) => "first",
            TimeSlice::Last(_) => "last",
        }
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if let [Expr::StringLiteral(duration)] = args {
            let duration = parse_duration(duration)?;
            // Ok(Expr::Last(Box::new(recv), duration))
            Ok(Expr::Formula(
                Arc::new(self.with_duration(duration)),
                Box::new(recv),
            ))
        } else {
            Err(Error::InvalidMethodCall("last".to_string()))
        }
    }
    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        match expr {
            Expr::ComponentPart(_) | Expr::ArrayAccess(_, _) => {
                vec![format!("{}(", self.name())]
            }
            Expr::Tuple(_) | Expr::Formula(_, _) => vec![self.name().to_string()],
            _ => Vec::new(),
        }
    }

    fn to_sql(&self, expr: &Expr, context: &Context) -> Result<String, Error> {
        if let Some(duration) = self.duration() {
            let sql = expr.to_sql(context)?;
            let duration_micros = (duration.total_nanoseconds() / 1000) as i64;
            let (bound, operator) = match self {
                TimeSlice::First(_) => {
                    let upper_bound = context.earliest_timestamp.0 + duration_micros;
                    let upper_bound = upper_bound as f64 * 1e-6;
                    (upper_bound, "<=")
                }
                TimeSlice::Last(_) => {
                    let lower_bound = context.last_timestamp.0 - duration_micros;
                    let lower_bound = lower_bound as f64 * 1e-6;
                    (lower_bound, ">=")
                }
            };
            let time_field = expr.to_sql_time_field()?;
            Ok(format!(
                "{sql} where {time_field} {operator} to_timestamp({bound})",
            ))
        } else {
            Err(Error::MissingDuration(self.name().to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Component, ComponentPart, Context, Expr, formulas::TimeSlice, parse_duration};
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
        let expr = Expr::Formula(
            Arc::new(TimeSlice::Last(Some(duration))),
            Box::new(Expr::ComponentPart(part)),
        );

        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains(">= to_timestamp(0.5)"), "WAS {}", sql);
    }

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
        let expr = Expr::Formula(
            Arc::new(TimeSlice::First(Some(duration))),
            Box::new(Expr::ComponentPart(part)),
        );

        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("<= to_timestamp(0.5)"));
    }
}
