use crate::{Context, Error, Expr};
use std::sync::Arc;

fn to_qualified_field(expr: &Expr) -> Result<String, Error> {
    let part = match expr {
        Expr::ComponentPart(p) => p.clone(),
        _ => {
            return Err(Error::InvalidMethodCall(
                "norm() expects a vector component".to_string(),
            ));
        }
    };
    let comp = part
        .component
        .as_ref()
        .ok_or_else(|| Error::InvalidFieldAccess("norm() on non-leaf component".to_string()))?;
    let dims = comp.schema.dim();
    if dims.is_empty() {
        return Err(Error::InvalidMethodCall(
            "norm() on scalar component".to_string(),
        ));
    }
    let n_elems: usize = dims.iter().copied().map(|d| d as usize).product();
    let mut terms = Vec::with_capacity(n_elems);
    for i in 0..n_elems {
        let qi = Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), i)
            .to_qualified_field()?;
        terms.push(format!("{qi} * {qi}"));
    }
    Ok(format!("sqrt({})", terms.join(" + ")))
}

fn to_column_name(expr: &Expr) -> Option<String> {
    expr.to_column_name().map(|name| format!("norm({name})"))
}

fn parse(recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
    if args.is_empty() && matches!(recv, Expr::ComponentPart(_)) {
        return Ok(Expr::Formula(Arc::new(Norm), Box::new(recv)));
    }
    Err(Error::InvalidMethodCall("norm".to_string()))
}

#[derive(Debug, Clone)]
pub struct Norm;

impl super::Formula for Norm {
    fn name(&self) -> &'static str {
        "norm"
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        parse(recv, args)
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        to_qualified_field(expr)
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        to_column_name(expr)
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        if let Expr::ComponentPart(part) = expr
            && let Some(component) = &part.component
            && !component.schema.dim().is_empty()
        {
            return vec!["norm()".to_string()];
        }
        Vec::new()
    }

    // fn to_sql(&self, expr: &Expr, context: &Context) -> Result<String, Error> {
    //     Ok(format!(
    //         "select {} as '{}' from {}",
    //         // self.name(),
    //         self.to_qualified_field(expr)?,
    //         self.to_column_name(expr)?,
    //         expr.to_table()?
    //     ))
    // }
}
