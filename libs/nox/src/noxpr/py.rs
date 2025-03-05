use pyo3::{IntoPyObjectExt, exceptions::PyValueError, prelude::*};

use crate::*;

impl<T: TensorItem, D: Dim> FromPyObject<'_> for Tensor<T, D, Op> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let tensor = Tensor::from_inner(Noxpr::jax(ob.into_py_any(ob.py())?));
        Ok(tensor)
    }
}

#[allow(deprecated)]
impl<T: TensorItem, D: Dim> ToPyObject for Tensor<T, D, Op> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.inner
            .to_jax()
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .unwrap_or_else(|err| err.into_py_any(py).unwrap())
    }
}
