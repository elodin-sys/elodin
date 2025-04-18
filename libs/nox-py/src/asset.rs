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

#[derive(Clone)]
#[pyclass]
pub struct Handle {
    pub inner: nox_ecs::Handle<()>,
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
    fn unflatten(_aux: PyObject, _jax: PyObject) -> Self {
        todo!()
    }

    #[staticmethod]
    fn from_array(_arr: PyObject) -> Self {
        todo!()
    }

    #[classattr]
    fn metadata() -> Component {
        Component {
            name: "handle".to_string(),
            ty: Some(ComponentType::u64()),
            asset: true,
            metadata: Default::default(),
        }
    }

    #[classattr]
    fn __metadata__() -> (Component,) {
        (Self::metadata(),)
    }
}
