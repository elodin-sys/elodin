use pyo3::{IntoPyObjectExt, prelude::*};

use crate::*;

impl<T: TensorItem, D: Dim> FromPyObject<'_> for Tensor<T, D, Op> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let tensor = Tensor::from_inner(Noxpr::jax(ob.into_py_any(ob.py())?));
        Ok(tensor)
    }
}
