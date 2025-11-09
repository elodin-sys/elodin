use crate::*;

#[pyclass]
#[derive(Clone)]
pub struct PyBufBytes {
    pub bytes: bytes::Bytes,
}

pub struct PyAsset {
    pub object: Py<PyAny>,
}

impl PyAsset {
    pub fn try_new(py: Python<'_>, object: Py<PyAny>) -> Result<Self, Error> {
        let _ = object.getattr(py, "asset_name")?;
        let _ = object.getattr(py, "bytes")?;
        Ok(Self { object })
    }
}

impl PyAsset {
    pub fn name(&self) -> Result<String, Error> {
        Python::with_gil(|py| {
            let asset_name: String = self.object.call_method0(py, "asset_name")?.extract(py)?;
            Ok(asset_name)
        })
    }

    pub fn bytes(&self) -> Result<bytes::Bytes, Error> {
        Python::with_gil(|py| {
            let bytes: PyBufBytes = self.object.call_method0(py, "bytes")?.extract(py)?;
            Ok(bytes.bytes)
        })
    }
}

// Simple Handle type for asset management
#[derive(Clone)]
pub struct SimpleHandle<T> {
    pub id: u64,
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Clone)]
#[pyclass]
pub struct Handle {
    pub inner: SimpleHandle<()>,
}

#[pymethods]
impl Handle {
    pub fn asarray(&self) -> Result<PyObject, Error> {
        Ok(nox::NoxprScalarExt::constant(self.inner.id).to_jax()?)
    }

    pub fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = nox::NoxprScalarExt::constant(self.inner.id).to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(_aux: PyObject, jax: PyObject) -> Self {
        // Extract the u64 id from the JAX array
        Python::with_gil(|py| {
            // The jax object should be a scalar u64
            let id = jax.extract::<u64>(py).unwrap_or(0);
            Handle {
                inner: SimpleHandle {
                    id,
                    _phantom: std::marker::PhantomData,
                },
            }
        })
    }

    #[staticmethod]
    fn from_array(arr: PyObject) -> Self {
        // Extract the u64 id from a numpy/JAX array
        Python::with_gil(|py| {
            // The array should contain a single u64 value
            let id = arr.extract::<u64>(py).unwrap_or(0);
            Handle {
                inner: SimpleHandle {
                    id,
                    _phantom: std::marker::PhantomData,
                },
            }
        })
    }

    #[classattr]
    fn metadata() -> Component {
        Component {
            name: "handle".to_string(),
            ty: Some(ComponentType::u64()),
            metadata: Default::default(),
        }
    }

    #[classattr]
    fn __metadata__() -> (Component,) {
        (Self::metadata(),)
    }
}
