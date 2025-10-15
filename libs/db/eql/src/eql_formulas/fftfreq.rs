use crate::{Error, Expr};

pub(crate) fn to_field(expr: &Expr) -> Result<String, Error> {
    Ok(format!("fftfreq({})", expr.to_field()?))
}

pub(crate) fn to_qualified_field(expr: &Expr) -> Result<String, Error> {
    Ok(format!("fftfreq({})", expr.to_qualified_field()?))
}

pub(crate) fn to_column_name(expr: &Expr) -> Option<String> {
    expr.to_column_name().map(|name| format!("fftfreq({name})"))
}

pub(crate) fn parse(recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
    if args.is_empty() && matches!(recv, Expr::Time(_)) {
        Ok(Expr::FftFreq(Box::new(recv)))
    } else {
        Err(Error::InvalidMethodCall("fftfreq".to_string()))
    }
}

pub(crate) fn suggestions_for_time() -> Vec<String> {
    vec!["fftfreq()".to_string()]
}

#[cfg(test)]
mod tests {
    use crate::{Component, Context, Expr};
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType};
    use std::sync::Arc;

    #[test]
    fn test_fftfreq_sql() {
        let component = Arc::new(Component::new(
            "a.world_pos".to_string(),
            ComponentId::new("a.world_pos"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));
        let context = Context::default();

        let time_expr = Expr::Time(component);
        let expr = Expr::FftFreq(Box::new(time_expr));
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select fftfreq(a_world_pos.time) from a_world_pos"
        );
    }
}
