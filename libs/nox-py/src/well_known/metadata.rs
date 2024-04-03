use crate::*;

use std::sync::Arc;

use nox_ecs::conduit;
use nox_ecs::conduit::{Asset, TagValue};

use crate::{Color, PyBufBytes};

#[pyclass]
#[derive(Clone)]
pub struct EntityMetadata {
    pub inner: conduit::well_known::EntityMetadata,
}

#[pymethods]
impl EntityMetadata {
    #[new]
    pub fn new(name: String, color: Option<Color>) -> Self {
        let color = color.unwrap_or(Color::new(1.0, 1.0, 1.0));
        Self {
            inner: conduit::well_known::EntityMetadata {
                name,
                color: color.inner,
            },
        }
    }

    pub fn asset_id(&self) -> u64 {
        self.inner.asset_id().0
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Metadata {
    pub inner: Arc<conduit::Metadata>,
}

#[pymethods]
impl Metadata {
    #[new]
    pub fn new(component_id: ComponentId, ty: ComponentType, name: Option<String>) -> Self {
        let inner = Arc::new(conduit::Metadata {
            component_id: component_id.inner,
            component_type: ty.into(),
            tags: name
                .into_iter()
                .map(|n| ("name".to_string(), TagValue::String(n)))
                .collect(),
        });
        Metadata { inner }
    }
}
