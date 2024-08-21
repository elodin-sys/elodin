use crate::*;

use nox_ecs::impeller;
use nox_ecs::impeller::Asset;
use pyo3::{intern, types::PySequence};

#[pyclass]
#[derive(Clone)]
pub struct EntityMetadata {
    pub inner: impeller::well_known::EntityMetadata,
}

#[pymethods]
impl EntityMetadata {
    #[new]
    pub fn new(name: String, color: Option<Color>) -> Self {
        let color = color.unwrap_or(Color::new(1.0, 1.0, 1.0));
        Self {
            inner: impeller::well_known::EntityMetadata {
                name,
                color: color.inner,
            },
        }
    }

    pub fn asset_name(&self) -> &'static str {
        impeller::well_known::EntityMetadata::ASSET_NAME
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Metadata {
    pub inner: impeller::Metadata,
}

impl From<Metadata> for Component {
    fn from(value: Metadata) -> Self {
        Component {
            name: value.inner.name.to_string(),
            asset: value.inner.asset,
            metadata: value.inner.tags.unwrap_or_default(),
            ty: Some(value.inner.component_type.into()),
        }
    }
}

impl std::ops::Deref for Metadata {
    type Target = impeller::Metadata;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[pymethods]
impl Metadata {
    #[getter]
    pub fn ty(&self) -> ComponentType {
        self.inner.component_type.clone().into()
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

        let component_type = component_data
            .ty
            .ok_or(PyValueError::new_err("component type not found"))?;
        let inner = impeller::Metadata {
            name: component_data.name.into(),
            component_type: component_type.into(),
            asset: component_data.asset,
            tags: Some(component_data.metadata),
        };
        Ok(Self { inner })
    }
}
