use crate::*;

use std::collections::HashMap;

use impeller2_wkt::MetadataValue;
use pyo3::types::PyList;
use pyo3::{intern, types::PySequence};

#[derive(Clone, Debug)]
#[pyclass]
pub struct Component {
    #[pyo3(set)]
    pub name: String,
    #[pyo3(get, set)]
    pub ty: Option<ComponentType>,
    #[pyo3(get, set)]
    pub asset: bool,
    pub metadata: HashMap<String, MetadataValue>,
}

impl Component {
    pub fn component_id(&self) -> ComponentId {
        ComponentId::new(&self.name)
    }

    pub fn from_component<C: impeller2::component::Component>() -> Self {
        let schema = C::schema();
        Component {
            name: C::NAME.to_string(),
            ty: Some(ComponentType {
                shape: schema.shape().iter().map(|&x| x as u64).collect(),
                ty: schema.prim_type().into(),
            }),
            asset: C::ASSET,
            metadata: Default::default(),
        }
    }
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
                    MetadataValue::String(s)
                } else if let Ok(f) = v.extract::<bool>(py) {
                    MetadataValue::Bool(f)
                } else if let Ok(v) = v.extract::<i64>(py) {
                    MetadataValue::I64(v)
                } else {
                    MetadataValue::Unit
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
        Component::of(py, component).map(|metadata| metadata.name.to_string())
    }

    #[staticmethod]
    pub fn index(py: Python<'_>, component: PyObject) -> Result<ShapeIndexer, Error> {
        let component = Component::of(py, component)?;
        //let metadata = Metadata::of(py, component)?.inner;
        let ty = component.ty.unwrap();
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
        Ok(ShapeIndexer::new(
            component.name.to_string(),
            shape,
            vec![],
            strides,
        ))
    }

    #[staticmethod]
    pub fn of(py: Python<'_>, component: PyObject) -> Result<Self, Error> {
        let mut component_data = component
            .getattr(py, intern!(py, "__metadata__"))
            .and_then(|metadata| {
                metadata
                    .downcast_bound::<PySequence>(py)
                    .map_err(PyErr::from)
                    .and_then(|seq| seq.get_item(0))
                    .and_then(|item| item.extract::<Component>())
            })?;

        if component_data.ty.is_none() {
            if let Some(base_ty) = component
                .getattr(py, intern!(py, "__origin__"))
                .and_then(|origin| origin.getattr(py, intern!(py, "__metadata__")))
                .and_then(|metadata| {
                    metadata
                        .downcast_bound::<PySequence>(py)
                        .map_err(PyErr::from)
                        .and_then(|seq| seq.get_item(0))
                        .and_then(|item| item.extract::<Component>())
                })
                .ok()
                .and_then(|component| component.ty)
            {
                component_data.ty = Some(base_ty);
            }
        }

        if component_data.ty.is_none() {
            println!("{:?}", component_data);
            return Err(PyValueError::new_err("component type not found").into());
        }
        Ok(component_data)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ComponentType {
    #[pyo3(get, set)]
    pub ty: PrimitiveType,
    #[pyo3(get, set)]
    pub shape: Vec<u64>,
}

#[pymethods]
impl ComponentType {
    #[new]
    pub fn new(ty: PrimitiveType, shape: Vec<u64>) -> Self {
        Self { ty, shape }
    }

    #[classattr]
    #[pyo3(name = "SpatialPosF64")]
    pub fn spatial_pos_f64() -> Self {
        Self {
            ty: PrimitiveType::F64,
            shape: vec![7],
        }
    }

    #[classattr]
    #[pyo3(name = "SpatialMotionF64")]
    pub fn spatial_motion_f64() -> Self {
        Self {
            ty: PrimitiveType::F64,
            shape: vec![6],
        }
    }

    #[classattr]
    #[pyo3(name = "U64")]
    pub fn u64() -> Self {
        Self {
            ty: PrimitiveType::U64,
            shape: vec![],
        }
    }

    #[classattr]
    #[pyo3(name = "F32")]
    pub fn f32() -> Self {
        Self {
            ty: PrimitiveType::F32,
            shape: vec![],
        }
    }

    #[classattr]
    #[pyo3(name = "F64")]
    pub fn f64() -> Self {
        Self {
            ty: PrimitiveType::F64,
            shape: vec![],
        }
    }

    #[classattr]
    #[pyo3(name = "Edge")]
    pub fn edge() -> Self {
        Self {
            ty: PrimitiveType::U64,
            shape: vec![2],
        }
    }

    #[classattr]
    #[pyo3(name = "Quaternion")]
    pub fn quaternion() -> Self {
        Self {
            ty: PrimitiveType::F64,
            shape: vec![4],
        }
    }
}

impl From<impeller_db::ComponentSchema> for ComponentType {
    fn from(val: impeller_db::ComponentSchema) -> Self {
        ComponentType {
            ty: val.prim_type.into(),
            shape: val.shape.to_vec(),
        }
    }
}

impl From<Component> for impeller_db::ComponentSchema {
    fn from(val: Component) -> Self {
        let ty = val.ty.unwrap();
        impeller_db::ComponentSchema {
            component_id: ComponentId::new(&val.name),
            prim_type: ty.ty.into(),
            shape: ty.shape.iter().copied().collect(),
            dim: ty.shape.iter().map(|x| *x as usize).collect(),
        }
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

impl From<impeller2::types::PrimType> for PrimitiveType {
    fn from(val: impeller2::types::PrimType) -> Self {
        match val {
            impeller2::types::PrimType::F64 => PrimitiveType::F64,
            impeller2::types::PrimType::F32 => PrimitiveType::F32,
            impeller2::types::PrimType::U64 => PrimitiveType::U64,
            impeller2::types::PrimType::U32 => PrimitiveType::U32,
            impeller2::types::PrimType::U16 => PrimitiveType::U16,
            impeller2::types::PrimType::U8 => PrimitiveType::U8,
            impeller2::types::PrimType::I64 => PrimitiveType::I64,
            impeller2::types::PrimType::I32 => PrimitiveType::I32,
            impeller2::types::PrimType::I16 => PrimitiveType::I16,
            impeller2::types::PrimType::I8 => PrimitiveType::I8,
            impeller2::types::PrimType::Bool => PrimitiveType::Bool,
        }
    }
}

impl From<PrimitiveType> for impeller2::types::PrimType {
    fn from(val: PrimitiveType) -> Self {
        match val {
            PrimitiveType::F64 => impeller2::types::PrimType::F64,
            PrimitiveType::F32 => impeller2::types::PrimType::F32,
            PrimitiveType::U64 => impeller2::types::PrimType::U64,
            PrimitiveType::U32 => impeller2::types::PrimType::U32,
            PrimitiveType::U16 => impeller2::types::PrimType::U16,
            PrimitiveType::U8 => impeller2::types::PrimType::U8,
            PrimitiveType::I64 => impeller2::types::PrimType::I64,
            PrimitiveType::I32 => impeller2::types::PrimType::I32,
            PrimitiveType::I16 => impeller2::types::PrimType::I16,
            PrimitiveType::I8 => impeller2::types::PrimType::I8,
            PrimitiveType::Bool => impeller2::types::PrimType::Bool,
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
            PyList::new_bound(py, items.iter().map(|x| Py::new(py, x.clone()).unwrap())).into()
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
