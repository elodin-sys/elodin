use crate::EntityId;
use crate::{Color, Error, PyBufBytes};
use conduit::ComponentId;
use nox_ecs::conduit;
use nox_ecs::conduit::Asset;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct VectorArrow {
    inner: conduit::well_known::VectorArrow,
}

#[pymethods]
impl VectorArrow {
    #[new]
    #[pyo3(signature = (entity, name, offset=0, color=Color::new(1.0, 1.0, 1.0), attached=true, body_frame=true, scale=1.0))]
    fn new(
        entity: EntityId,
        name: String,
        offset: usize,
        color: Color,
        attached: bool,
        body_frame: bool,
        scale: f32,
    ) -> Self {
        Self {
            inner: conduit::well_known::VectorArrow {
                entity_id: entity.inner,
                id: ComponentId::new(&name),
                range: offset..offset + 2,
                color: color.inner,
                attached,
                body_frame,
                scale,
            },
        }
    }

    pub fn asset_name(&self) -> &'static str {
        conduit::well_known::VectorArrow::ASSET_NAME
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }
}
