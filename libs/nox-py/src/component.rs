use crate::*;

use std::{collections::HashMap, ops::Deref, sync::Arc};

use nox_ecs::conduit;
use nox_ecs::conduit::TagValue;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::types::{PyBytes, PySequence};

use crate::Metadata;

#[derive(Clone)]
#[pyclass]
pub struct Component {
    #[pyo3(set)]
    pub id: ComponentId,
    #[pyo3(get, set)]
    pub ty: ComponentType,
    #[pyo3(get, set)]
    pub asset: bool,
    pub metadata: HashMap<String, TagValue>,
}

#[pymethods]
impl Component {
    #[new]
    pub fn new(
        py: Python<'_>,
        id: PyObject,
        ty: ComponentType,
        mut name: Option<String>,
        asset: Option<bool>,
    ) -> Result<Self, Error> {
        if name.is_none() {
            if let Ok(id) = id.extract::<String>(py) {
                name = Some(id)
            }
        }

        let id = if let Ok(id) = id.extract::<ComponentId>(py) {
            id
        } else {
            ComponentId::new(py, id)?
        };
        let metadata = name
            .into_iter()
            .map(|n| ("name".to_string(), TagValue::String(n)))
            .collect();
        Ok(Self {
            id,
            ty,
            metadata,
            asset: asset.unwrap_or_default(),
        })
    }

    #[staticmethod]
    pub fn id(py: Python<'_>, component: PyObject) -> Result<ComponentId, Error> {
        let metadata_attr = component.getattr(py, "__metadata__")?;
        let metadata = metadata_attr
            .downcast::<PySequence>(py)
            .map_err(PyErr::from)?;
        let component = metadata.get_item(0)?.extract::<Self>()?;
        Ok(component.id)
    }

    pub fn tag(&mut self, key: String, value: &PyAny) -> Result<(), Error> {
        let value = if let Ok(s) = value.extract::<String>() {
            TagValue::String(s)
        } else if let Ok(f) = value.extract::<bool>() {
            TagValue::Bool(f)
        } else if let Ok(u) = value.extract::<String>() {
            TagValue::String(u)
        } else if let Ok(b) = value
            .call_method0("bytes")
            .and_then(|x| x.extract::<&PyBytes>())
        {
            TagValue::Bytes(b.as_bytes().to_vec())
        } else {
            return Err(Error::UnexpectedInput);
        };
        self.metadata.insert(key, value);
        Ok(())
    }

    pub fn to_metadata(&self) -> Metadata {
        let inner = Arc::new(conduit::Metadata {
            component_id: self.id.inner,
            component_type: self.ty.clone().into(),
            asset: self.asset,
            tags: self.metadata.clone(),
        });
        Metadata { inner }
    }
}

#[derive(Clone, Copy, Debug)]
#[pyclass]
pub struct ComponentId {
    pub inner: conduit::ComponentId,
}

#[pymethods]
impl ComponentId {
    #[new]
    fn new(py: Python<'_>, inner: PyObject) -> Result<Self, Error> {
        if let Ok(s) = inner.extract::<String>(py) {
            Ok(Self {
                inner: conduit::ComponentId::new(&s),
            })
        } else if let Ok(s) = inner.extract::<u64>(py) {
            Ok(Self {
                inner: conduit::ComponentId(s),
            })
        } else {
            Err(Error::UnexpectedInput)
        }
    }

    fn __str__(&self) -> String {
        self.inner.0.to_string()
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ComponentType {
    #[pyo3(get, set)]
    pub ty: PrimitiveType,
    #[pyo3(get, set)]
    pub shape: Py<PyArray1<i64>>,
}

#[pymethods]
impl ComponentType {
    #[new]
    pub fn new(ty: PrimitiveType, shape: numpy::PyArrayLike1<i64>) -> Self {
        let py_readonly: &PyReadonlyArray1<i64> = shape.deref();
        let py_array: &PyArray1<i64> = py_readonly.deref();
        let shape = py_array.to_owned();
        Self { ty, shape }
    }

    #[classattr]
    #[pyo3(name = "SpatialPosF64")]
    pub fn spatial_pos_f64(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![7]).to_owned();
        Self {
            ty: PrimitiveType::F64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "SpatialMotionF64")]
    pub fn spatial_motion_f64(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![6]).to_owned();
        Self {
            ty: PrimitiveType::F64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "U64")]
    pub fn u64(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![]).to_owned();
        Self {
            ty: PrimitiveType::U64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "F32")]
    pub fn f32(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![]).to_owned();
        Self {
            ty: PrimitiveType::F32,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "F64")]
    pub fn f64(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![]).to_owned();
        Self {
            ty: PrimitiveType::F64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "Edge")]
    pub fn edge(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![2]).to_owned();
        Self {
            ty: PrimitiveType::U64,
            shape,
        }
    }

    #[classattr]
    #[pyo3(name = "Quaternion")]
    pub fn quaternion(py: Python<'_>) -> Self {
        let shape = numpy::PyArray1::from_vec(py, vec![4]).to_owned();
        Self {
            ty: PrimitiveType::F64,
            shape,
        }
    }
}

impl From<ComponentType> for conduit::ComponentType {
    fn from(val: ComponentType) -> Self {
        Python::with_gil(|py| {
            let shape = val.shape.as_ref(py);
            let shape = shape.to_vec().unwrap().into();
            conduit::ComponentType {
                primitive_ty: val.ty.into(),
                shape,
            }
        })
    }
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveType {
    F64,
    F32,
    U64,
    U32,
    U16,
    U8,
    I64,
    I32,
    I16,
    I8,
    Bool,
}

impl From<PrimitiveType> for conduit::PrimitiveTy {
    fn from(val: PrimitiveType) -> Self {
        match val {
            PrimitiveType::F64 => conduit::PrimitiveTy::F64,
            PrimitiveType::F32 => conduit::PrimitiveTy::F32,
            PrimitiveType::U64 => conduit::PrimitiveTy::U64,
            PrimitiveType::U32 => conduit::PrimitiveTy::U32,
            PrimitiveType::U16 => conduit::PrimitiveTy::U16,
            PrimitiveType::U8 => conduit::PrimitiveTy::U8,
            PrimitiveType::I64 => conduit::PrimitiveTy::I64,
            PrimitiveType::I32 => conduit::PrimitiveTy::I32,
            PrimitiveType::I16 => conduit::PrimitiveTy::I16,
            PrimitiveType::I8 => conduit::PrimitiveTy::I8,
            PrimitiveType::Bool => conduit::PrimitiveTy::Bool,
        }
    }
}
