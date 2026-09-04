use crate::{BinaryOp, ComponentPart, Expr};

#[derive(Clone, Debug, PartialEq)]
pub enum EvalValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl EvalValue {
    pub fn into_values(self) -> Vec<f64> {
        match self {
            Self::Scalar(value) => vec![value],
            Self::Vector(values) => values,
        }
    }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        match self {
            Self::Scalar(value) => Self::Scalar(f(value)),
            Self::Vector(values) => Self::Vector(values.into_iter().map(f).collect()),
        }
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum EvalError {
    #[error("component '{0}' has no sample")]
    MissingComponent(String),
    #[error("array index {index} out of bounds for length {length}")]
    OutOfBounds { index: usize, length: usize },
    #[error("cannot broadcast vectors of length {left} and {right}")]
    Broadcast { left: usize, right: usize },
    #[error("{0} requires a scalar operand")]
    RequiresScalar(&'static str),
    #[error("{0} requires a vector operand")]
    RequiresVector(&'static str),
    #[error("formula '{0}' is not supported by sample evaluation")]
    UnsupportedFormula(&'static str),
    #[error("expression is not supported by sample evaluation")]
    UnsupportedExpression,
}

pub fn supports(expr: &Expr) -> bool {
    match expr {
        Expr::ComponentPart(_) | Expr::FloatLiteral(_) => true,
        Expr::ArrayAccess(inner, _) => supports(inner),
        Expr::Tuple(elements) => elements.iter().all(supports),
        Expr::BinaryOp(left, right, _) => supports(left) && supports(right),
        Expr::Formula(formula, inner) => {
            matches!(
                formula.name(),
                "sqrt" | "abs" | "degrees" | "arccos" | "sign" | "cast" | "norm" | "atan2" | "clip"
            ) && supports(inner)
        }
        Expr::Time(_) | Expr::StringLiteral(_) | Expr::Last(_, _) | Expr::First(_, _) => false,
    }
}

pub fn evaluate(
    expr: &Expr,
    component: &impl Fn(&ComponentPart) -> Result<EvalValue, EvalError>,
) -> Result<EvalValue, EvalError> {
    match expr {
        Expr::ComponentPart(part) => component(part),
        Expr::ArrayAccess(inner, index) => {
            let values = evaluate(inner, component)?.into_values();
            values
                .get(*index)
                .copied()
                .map(EvalValue::Scalar)
                .ok_or(EvalError::OutOfBounds {
                    index: *index,
                    length: values.len(),
                })
        }
        Expr::Tuple(elements) => {
            let mut values = Vec::new();
            for element in elements {
                values.extend(evaluate(element, component)?.into_values());
            }
            Ok(EvalValue::Vector(values))
        }
        Expr::FloatLiteral(value) => Ok(EvalValue::Scalar(*value)),
        Expr::BinaryOp(left, right, op) => {
            let left = evaluate(left, component)?;
            let right = evaluate(right, component)?;
            binary(left, right, *op)
        }
        Expr::Formula(formula, inner) => formula_eval(formula.name(), inner, component),
        Expr::Time(_) | Expr::StringLiteral(_) | Expr::Last(_, _) | Expr::First(_, _) => {
            Err(EvalError::UnsupportedExpression)
        }
    }
}

fn binary(left: EvalValue, right: EvalValue, op: BinaryOp) -> Result<EvalValue, EvalError> {
    let apply = |left: f64, right: f64| match op {
        BinaryOp::Add => left + right,
        BinaryOp::Sub => left - right,
        BinaryOp::Mul => left * right,
        BinaryOp::Div => left / right,
    };
    match (left, right) {
        (EvalValue::Scalar(left), EvalValue::Scalar(right)) => {
            Ok(EvalValue::Scalar(apply(left, right)))
        }
        (EvalValue::Scalar(left), EvalValue::Vector(right)) => Ok(EvalValue::Vector(
            right.into_iter().map(|right| apply(left, right)).collect(),
        )),
        (EvalValue::Vector(left), EvalValue::Scalar(right)) => Ok(EvalValue::Vector(
            left.into_iter().map(|left| apply(left, right)).collect(),
        )),
        (EvalValue::Vector(left), EvalValue::Vector(right)) => {
            if left.len() != right.len() {
                return Err(EvalError::Broadcast {
                    left: left.len(),
                    right: right.len(),
                });
            }
            Ok(EvalValue::Vector(
                left.into_iter()
                    .zip(right)
                    .map(|(left, right)| apply(left, right))
                    .collect(),
            ))
        }
    }
}

fn scalar(value: EvalValue, formula: &'static str) -> Result<f64, EvalError> {
    match value {
        EvalValue::Scalar(value) => Ok(value),
        EvalValue::Vector(values) if values.len() == 1 => Ok(values[0]),
        EvalValue::Vector(_) => Err(EvalError::RequiresScalar(formula)),
    }
}

fn formula_eval(
    name: &'static str,
    inner: &Expr,
    component: &impl Fn(&ComponentPart) -> Result<EvalValue, EvalError>,
) -> Result<EvalValue, EvalError> {
    match name {
        "sqrt" => Ok(evaluate(inner, component)?.map(f64::sqrt)),
        "abs" => Ok(evaluate(inner, component)?.map(f64::abs)),
        "degrees" => Ok(evaluate(inner, component)?.map(f64::to_degrees)),
        "arccos" => Ok(evaluate(inner, component)?.map(f64::acos)),
        "sign" => Ok(evaluate(inner, component)?.map(f64::signum)),
        "cast" => evaluate(inner, component),
        "norm" => {
            let EvalValue::Vector(values) = evaluate(inner, component)? else {
                return Err(EvalError::RequiresVector("norm"));
            };
            Ok(EvalValue::Scalar(
                values.iter().map(|value| value * value).sum::<f64>().sqrt(),
            ))
        }
        "atan2" => {
            let Expr::Tuple(elements) = inner else {
                return Err(EvalError::RequiresVector("atan2"));
            };
            if elements.len() != 2 {
                return Err(EvalError::RequiresVector("atan2"));
            }
            let y = scalar(evaluate(&elements[0], component)?, "atan2")?;
            let x = scalar(evaluate(&elements[1], component)?, "atan2")?;
            Ok(EvalValue::Scalar(y.atan2(x)))
        }
        "clip" => {
            let Expr::Tuple(elements) = inner else {
                return Err(EvalError::RequiresVector("clip"));
            };
            if elements.len() != 3 {
                return Err(EvalError::RequiresVector("clip"));
            }
            let value = evaluate(&elements[0], component)?;
            let min = scalar(evaluate(&elements[1], component)?, "clip")?;
            let max = scalar(evaluate(&elements[2], component)?, "clip")?;
            Ok(value.map(|value| value.clamp(min, max)))
        }
        _ => Err(EvalError::UnsupportedFormula(name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Component, Context};
    use impeller2::{
        schema::Schema,
        types::{ComponentId, PrimType, Timestamp},
    };
    use std::{collections::HashMap, sync::Arc};

    fn context() -> Context {
        let vector = Arc::new(Component::new(
            "sample.vector".to_string(),
            ComponentId::new("sample.vector"),
            Schema::new(PrimType::F64, vec![3_u64]).unwrap(),
        ));
        Context::from_leaves([vector], Timestamp(0), Timestamp(1))
    }

    fn evaluate_with_vector(expr: &Expr) -> Result<EvalValue, EvalError> {
        let values = HashMap::from([(
            ComponentId::new("sample.vector"),
            EvalValue::Vector(vec![3.0, 4.0, 12.0]),
        )]);
        evaluate(expr, &|part| {
            values
                .get(&part.id)
                .cloned()
                .ok_or_else(|| EvalError::MissingComponent(part.name.clone()))
        })
    }

    #[test]
    fn evaluates_sqrt_over_vector_arithmetic() {
        let expr = context()
            .parse_str("(sample.vector * sample.vector).sqrt()")
            .unwrap();
        assert_eq!(
            evaluate_with_vector(&expr).unwrap(),
            EvalValue::Vector(vec![3.0, 4.0, 12.0])
        );
    }

    #[test]
    fn evaluates_vector_norm() {
        let expr = context().parse_str("sample.vector.norm()").unwrap();
        assert_eq!(
            evaluate_with_vector(&expr).unwrap(),
            EvalValue::Scalar(13.0)
        );
    }

    #[test]
    fn evaluates_norm_over_computed_vector() {
        let expr = context().parse_str("(sample.vector / 2.0).norm()").unwrap();
        assert_eq!(evaluate_with_vector(&expr).unwrap(), EvalValue::Scalar(6.5));
    }

    #[test]
    fn preserves_tuple_element_order() {
        let expr = context()
            .parse_str("(sample.vector[0], sample.vector[2], sample.vector[1])")
            .unwrap();
        assert_eq!(
            evaluate_with_vector(&expr).unwrap(),
            EvalValue::Vector(vec![3.0, 12.0, 4.0])
        );
    }

    #[test]
    fn rejects_mismatched_vector_broadcast() {
        let expr = Expr::BinaryOp(
            Box::new(Expr::Tuple(vec![
                Expr::FloatLiteral(1.0),
                Expr::FloatLiteral(2.0),
            ])),
            Box::new(Expr::Tuple(vec![
                Expr::FloatLiteral(1.0),
                Expr::FloatLiteral(2.0),
                Expr::FloatLiteral(3.0),
            ])),
            BinaryOp::Add,
        );
        assert_eq!(
            evaluate_with_vector(&expr),
            Err(EvalError::Broadcast { left: 2, right: 3 })
        );
    }
}
