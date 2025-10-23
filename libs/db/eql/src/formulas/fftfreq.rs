use crate::{Context, Error, Expr};
use std::sync::Arc;

fn to_field(expr: &Expr) -> Result<String, Error> {
    Ok(format!("fftfreq({})", expr.to_field()?))
}

fn to_qualified_field(expr: &Expr) -> Result<String, Error> {
    Ok(format!("fftfreq({})", expr.to_qualified_field()?))
}

fn to_column_name(expr: &Expr) -> Option<String> {
    expr.to_column_name().map(|name| format!("fftfreq({name})"))
}

fn suggestions_for_time() -> Vec<String> {
    vec!["fftfreq()".to_string()]
}

#[derive(Debug, Clone)]
pub struct FftFreq;

impl super::Formula for FftFreq {
    fn name(&self) -> &'static str {
        "fftfreq"
    }

    fn parse(
        &self,
        formula: Arc<dyn super::Formula>,
        recv: Expr,
        args: &[Expr],
    ) -> Result<Expr, Error> {
        if args.is_empty() && matches!(recv, Expr::Time(_)) {
            Ok(Expr::Formula(formula, Box::new(recv)))
        } else {
            Err(Error::InvalidMethodCall("fftfreq".to_string()))
        }
    }

    fn to_field(&self, expr: &Expr) -> Result<String, Error> {
        to_field(expr)
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        to_qualified_field(expr)
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        to_column_name(expr)
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if matches!(expr, Expr::Time(_)) {
            suggestions_for_time()
        } else {
            Vec::new()
        }
    }
}
