use pyo3::{exceptions::PyValueError, prelude::*};

use crate::*;

impl<T: TensorItem, D: Dim> FromPyObject<'_> for Tensor<T, D, Op> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let tensor = Tensor::from_inner(Noxpr::jax(ToPyObject::to_object(&ob, ob.py())));
        Ok(tensor)
    }
}

impl<T: TensorItem, D: Dim> IntoPy<PyObject> for Tensor<T, D, Op> {
    fn into_py(self, py: Python) -> PyObject {
        self.inner
            .to_jax()
            .map_err(|err| PyValueError::new_err(err.to_string()))
            .unwrap_or_else(|err| err.into_py(py))
    }
}
