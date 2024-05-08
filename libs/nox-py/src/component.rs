use crate::*;

use std::{collections::HashMap, ops::Deref};

use nox_ecs::conduit;
use nox_ecs::conduit::TagValue;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::types::PyList;

#[derive(Clone)]
#[pyclass]
pub struct Component {
    #[pyo3(set)]
    pub name: String,
    #[pyo3(get, set)]
    pub ty: Option<ComponentType>,
    #[pyo3(get, set)]
    pub asset: bool,
    pub metadata: HashMap<String, TagValue>,
}

#[pymethods]
impl Component {
    #[new]
    #[pyo3(signature = (name, ty = None, asset = false, metadata = HashMap::default()))]
    pub fn new(
        py: Python<'_>,
        name: String,
        ty: Option<ComponentType>,
        asset: bool,
        metadata: HashMap<String, PyObject>,
    ) -> Result<Self, Error> {
        let metadata = metadata
            .into_iter()
            .map(|(k, v)| {
                let value = if let Ok(s) = v.extract::<String>(py) {
                    TagValue::String(s)
                } else if let Ok(f) = v.extract::<bool>(py) {
                    TagValue::Bool(f)
                } else if let Ok(v) = v.extract::<i64>(py) {
                    TagValue::Int(v)
                } else {
                    TagValue::Unit
                };
                (k, value)
            })
            .collect();

        Ok(Self {
            name,
            ty,
            metadata,
            asset,
        })
    }

    #[staticmethod]
    pub fn id(py: Python<'_>, component: PyObject) -> Result<String, Error> {
        Self::name(py, component)
    }

    #[staticmethod]
    pub fn name(py: Python<'_>, component: PyObject) -> Result<String, Error> {
        Metadata::of(py, component).map(|metadata| metadata.inner.name)
    }

    #[staticmethod]
    pub fn index(py: Python<'_>, component: PyObject) -> Result<ShapeIndexer, Error> {
        let metadata = Metadata::of(py, component)?.inner;
        let ty = metadata.component_type;
        let strides: Vec<usize> = ty
            .shape
            .iter()
            .rev()
            .scan(1, |state, &x| {
                let result = *state;
                *state *= x as usize;
                Some(result)
            })
            .collect();
        let strides = strides.into_iter().rev().collect();
        let shape = ty.shape.iter().map(|x| *x as usize).collect();
        Ok(ShapeIndexer::new(metadata.name, shape, vec![], strides))
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

impl From<conduit::ComponentType> for ComponentType {
    fn from(val: conduit::ComponentType) -> Self {
        Python::with_gil(|py| {
            let shape = PyArray1::from_vec(py, val.shape.to_vec()).to_owned();
            ComponentType {
                ty: val.primitive_ty.into(),
                shape,
            }
        })
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

impl From<conduit::PrimitiveTy> for PrimitiveType {
    fn from(val: conduit::PrimitiveTy) -> Self {
        match val {
            conduit::PrimitiveTy::F64 => PrimitiveType::F64,
            conduit::PrimitiveTy::F32 => PrimitiveType::F32,
            conduit::PrimitiveTy::U64 => PrimitiveType::U64,
            conduit::PrimitiveTy::U32 => PrimitiveType::U32,
            conduit::PrimitiveTy::U16 => PrimitiveType::U16,
            conduit::PrimitiveTy::U8 => PrimitiveType::U8,
            conduit::PrimitiveTy::I64 => PrimitiveType::I64,
            conduit::PrimitiveTy::I32 => PrimitiveType::I32,
            conduit::PrimitiveTy::I16 => PrimitiveType::I16,
            conduit::PrimitiveTy::I8 => PrimitiveType::I8,
            conduit::PrimitiveTy::Bool => PrimitiveType::Bool,
        }
    }
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

#[pyclass]
#[derive(Clone, Debug)]
pub struct ShapeIndexer {
    pub component_name: String,
    strides: Vec<usize>,
    shape: Vec<usize>,
    index: Vec<usize>,
    py_list: Py<PyList>,
    items: Vec<ShapeIndexer>,
}

#[pymethods]
impl ShapeIndexer {
    #[new]
    fn new(
        component_name: String,
        shape: Vec<usize>,
        index: Vec<usize>,
        strides: Vec<usize>,
    ) -> Self {
        let items = if shape.is_empty() {
            vec![]
        } else {
            let mut shape = shape.clone();
            let count = shape.remove(0);
            (0..count)
                .map(|i| {
                    let mut index = index.clone();
                    index.insert(0, i);
                    ShapeIndexer::new(
                        component_name.clone(),
                        shape.clone(),
                        index,
                        strides.clone(),
                    )
                })
                .collect()
        };
        let py_list = Python::with_gil(|py| {
            PyList::new(py, items.iter().map(|x| Py::new(py, x.clone()).unwrap())).into()
        });
        ShapeIndexer {
            component_name,
            shape,
            index,
            strides,
            py_list,
            items,
        }
    }

    pub fn indexes(&self) -> Vec<usize> {
        if self.shape.is_empty() {
            vec![self
                .index
                .iter()
                .zip(self.strides.iter().rev())
                .map(|(i, stride)| i * stride)
                .sum()]
        } else {
            self.items.iter().flat_map(|item| item.indexes()).collect()
        }
    }

    fn __getitem__(&self, py: Python<'_>, index: PyObject) -> PyResult<PyObject> {
        self.py_list.call_method1(py, "__getitem__", (index,))
    }
}
