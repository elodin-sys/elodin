use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Fft;

impl super::Formula for Fft {
    fn name(&self) -> &'static str {
        "fft"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.is_empty() && matches!(recv, Expr::ArrayAccess(_, _)) {
            Ok(Expr::Formula(Arc::new(Fft), Box::new(recv)))
        } else {
            Err(Error::InvalidMethodCall("fft".to_string()))
        }
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if matches!(expr, Expr::ArrayAccess(_, _)) {
            vec!["fft()".to_string()]
        } else {
            Vec::new()
        }
    }
}
