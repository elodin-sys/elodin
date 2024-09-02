use nox_ecs::nox::{Op, ReprMonad, Vector};
use pyo3::prelude::*;

use crate::Error;

#[pyfunction]
pub fn skew(arr: Vector<f64, 3, Op>, py: Python<'_>) -> Result<PyObject, Error> {
    let arr = arr.skew().into_inner().to_jax()?;
    // TODO: fix noxpr -> jax tracing so that reshaping is not needed
    let arr = arr.call_method1(py, "reshape", (3, 3))?;
    Ok(arr)
}
