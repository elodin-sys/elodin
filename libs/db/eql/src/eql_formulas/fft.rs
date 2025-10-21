use crate::{Error, Expr};

// pub(crate) fn to_field(expr: &Expr) -> Result<String, Error> {
//     Ok(format!("fft({})", expr.to_field()?))
// }

// pub(crate) fn to_qualified_field(expr: &Expr) -> Result<String, Error> {
//     Ok(format!("fft({})", expr.to_qualified_field()?))
// }

// pub(crate) fn to_column_name(expr: &Expr) -> Option<String> {
//     expr.to_column_name().map(|name| format!("fft({name})"))
// }

// pub(crate) fn parse(recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
//     if args.is_empty() && matches!(recv, Expr::ArrayAccess(_, _)) {
//         Ok(Expr::Fft(Box::new(recv)))
//     } else {
//         Err(Error::InvalidMethodCall("fft".to_string()))
//     }
// }

// pub(crate) fn array_access_suggestion() -> String {
//     "fft()".to_string()
// }

#[derive(Debug, Clone)]
pub struct Fft;

impl super::EqlFormula for Fft {
    fn name(&self) -> &'static str {
        "fft"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        if args.is_empty() && matches!(recv, Expr::ArrayAccess(_, _)) {
            Ok(Expr::Fft(Box::new(recv)))
        } else {
            Err(Error::InvalidMethodCall("fft".to_string()))
        }
    }

}
