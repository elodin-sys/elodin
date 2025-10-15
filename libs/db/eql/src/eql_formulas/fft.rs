use crate::{Error, Expr};

pub(crate) fn to_field(expr: &Expr) -> Result<String, Error> {
    Ok(format!("fft({})", expr.to_field()?))
}

pub(crate) fn to_qualified_field(expr: &Expr) -> Result<String, Error> {
    Ok(format!("fft({})", expr.to_qualified_field()?))
}

pub(crate) fn to_column_name(expr: &Expr) -> Option<String> {
    expr.to_column_name().map(|name| format!("fft({name})"))
}

pub(crate) fn parse(recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
    if args.is_empty() && matches!(recv, Expr::ArrayAccess(_, _)) {
        Ok(Expr::Fft(Box::new(recv)))
    } else {
        Err(Error::InvalidMethodCall("fft".to_string()))
    }
}

pub(crate) fn array_access_suggestion() -> String {
    "fft()".to_string()
}

#[cfg(test)]
mod tests {
    use crate::{Component, ComponentPart, Context, Expr};
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn test_fft_to_sql() {
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

        let expr = Expr::Fft(Box::new(Expr::ArrayAccess(
            Box::new(Expr::ComponentPart(part.clone())),
            0,
        )));
        let context = Context::default();
        let sql = expr.to_sql(&context).unwrap();
        assert_eq!(
            sql,
            "select fft(a_world_pos.a_world_pos[1]) as 'fft(a.world_pos.x)' from a_world_pos"
        );
    }
}
