use crate::{Context, Error, Expr};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastTarget {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Bool,
    F32,
    F64,
}

impl CastTarget {
    fn parse(target: &str) -> Option<Self> {
        match target.trim().to_ascii_lowercase().as_str() {
            "u8" | "uint8" => Some(Self::U8),
            "u16" | "uint16" => Some(Self::U16),
            "u32" | "uint32" => Some(Self::U32),
            "u64" | "uint64" => Some(Self::U64),
            "i8" | "int8" => Some(Self::I8),
            "i16" | "int16" => Some(Self::I16),
            "i32" | "int32" => Some(Self::I32),
            "i64" | "int64" => Some(Self::I64),
            "bool" | "boolean" => Some(Self::Bool),
            "f32" | "float" | "float32" | "real" => Some(Self::F32),
            "f64" | "double" | "float64" => Some(Self::F64),
            _ => None,
        }
    }

    fn eql_name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Bool => "bool",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    fn sql_name(self) -> &'static str {
        match self {
            Self::U8 => "TINYINT UNSIGNED",
            Self::U16 => "SMALLINT UNSIGNED",
            Self::U32 => "INTEGER UNSIGNED",
            Self::U64 => "BIGINT UNSIGNED",
            Self::I8 => "TINYINT",
            Self::I16 => "SMALLINT",
            Self::I32 => "INTEGER",
            Self::I64 => "BIGINT",
            Self::Bool => "BOOLEAN",
            Self::F32 => "FLOAT",
            Self::F64 => "DOUBLE",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Cast {
    target: Option<CastTarget>,
}

impl Cast {
    fn target(&self) -> Result<CastTarget, Error> {
        self.target
            .ok_or_else(|| Error::InvalidMethodCall("cast is missing a target type".to_string()))
    }
}

impl super::Formula for Cast {
    fn name(&self) -> &'static str {
        "cast"
    }

    fn editor_cast_target(&self) -> Option<CastTarget> {
        self.target
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error> {
        let [Expr::StringLiteral(target)] = args else {
            return Err(Error::InvalidMethodCall(
                "cast requires exactly one target type, e.g. value.cast(f64)".to_string(),
            ));
        };

        let Some(target) = CastTarget::parse(target) else {
            return Err(Error::InvalidMethodCall(format!(
                "unsupported cast target '{target}'; expected one of u8, u16, u32, u64, i8, i16, i32, i64, bool, f32, f64"
            )));
        };

        Ok(Expr::Formula(
            Arc::new(Cast {
                target: Some(target),
            }),
            Box::new(recv),
        ))
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        let target = self.target()?;
        Ok(format!(
            "CAST({} AS {})",
            expr.to_qualified_field()?,
            target.sql_name()
        ))
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        let target = self.target?;
        expr.to_column_name()
            .map(|name| format!("cast({}, {})", name, target.eql_name()))
    }

    fn suggestions(&self, expr: &Expr, _context: &Context) -> Vec<String> {
        match expr {
            Expr::ComponentPart(_)
            | Expr::ArrayAccess(_, _)
            | Expr::FloatLiteral(_)
            | Expr::BinaryOp(_, _, _)
            | Expr::Formula(_, _) => vec!["cast(".to_string()],
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Component, ComponentPart, Context, Expr};
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType, Timestamp};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn create_test_component() -> Arc<Component> {
        Arc::new(Component::new(
            "a.value".to_string(),
            ComponentId::new("a.value"),
            Schema::new(PrimType::U16, Vec::<u64>::new()).unwrap(),
        ))
    }

    fn create_test_context() -> Context {
        Context::from_leaves([create_test_component()], Timestamp(0), Timestamp(1000))
    }

    #[test]
    fn test_cast_sql() {
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });
        let context = create_test_context();
        let expr = Expr::Formula(
            Arc::new(Cast {
                target: Some(CastTarget::F64),
            }),
            Box::new(Expr::ComponentPart(part)),
        );

        let sql = expr.to_sql(&context).unwrap();
        assert_eq!(
            sql,
            "select CAST(a_value.a_value AS DOUBLE) as 'cast(a.value, f64)' from a_value"
        );
    }

    #[test]
    fn test_cast_parse_identifier_arg() {
        let context = create_test_context();
        let expr = context.parse_str("a.value.cast(f64)").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("CAST(a_value.a_value AS DOUBLE)"));
    }

    #[test]
    fn test_cast_parse_string_arg() {
        let context = create_test_context();
        let expr = context.parse_str("a.value.cast(\"f32\")").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("CAST(a_value.a_value AS FLOAT)"));
    }

    #[test]
    fn test_cast_with_arithmetic() {
        let context = create_test_context();
        let expr = context.parse_str("a.value.cast(f64) + 1.0").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert!(sql.contains("CAST(a_value.a_value AS DOUBLE)"));
        assert!(sql.contains(" + 1"));
    }

    #[test]
    fn test_cast_invalid_target() {
        let context = create_test_context();
        let err = context.parse_str("a.value.cast(quaternion)").unwrap_err();
        assert!(
            err.to_string().contains("unsupported cast target"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_cast_suggestions() {
        use crate::formulas::Formula;

        let context = create_test_context();
        let component = create_test_component();
        let part = Arc::new(ComponentPart {
            name: "a.value".to_string(),
            id: component.id,
            component: Some(component),
            children: BTreeMap::new(),
        });

        let suggestions = Cast::default().suggestions(&Expr::ComponentPart(part), &context);
        assert!(suggestions.contains(&"cast(".to_string()));
    }
}
