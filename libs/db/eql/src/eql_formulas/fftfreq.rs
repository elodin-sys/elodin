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
