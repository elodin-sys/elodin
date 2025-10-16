use crate::{Component, Error, Expr};

pub(crate) fn to_qualified_field(expr: &Expr) -> Result<String, Error> {
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
        terms.push(format!("{qi} * {qi}")); // <- sans parenthèses, pour matcher les tests
    }
    Ok(format!("sqrt({})", terms.join(" + ")))
}

pub(crate) fn to_column_name(expr: &Expr) -> Option<String> {
    expr.to_column_name().map(|name| format!("norm({name})"))
}

pub(crate) fn parse(recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
    if args.is_empty() {
        if matches!(recv, Expr::ComponentPart(_)) {
            return Ok(Expr::Norm(Box::new(recv)));
        }
    }
    Err(Error::InvalidMethodCall("norm".to_string()))
}

pub(crate) fn extend_component_suggestions(component: &Component, suggestions: &mut Vec<String>) {
    if !component.schema.dim().is_empty() {
        suggestions.push("norm()".to_string());
    }
}
