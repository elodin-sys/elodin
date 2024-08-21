use crate::EntityId;
use crate::{Color, Error, PyBufBytes};
use impeller::ComponentId;
use nox_ecs::impeller;
use nox_ecs::impeller::Asset;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct VectorArrow {
    inner: impeller::well_known::VectorArrow,
}

#[pymethods]
impl VectorArrow {
    #[new]
    #[pyo3(signature = (entity, component_name, offset=0, color=Color::new(1.0, 1.0, 1.0), attached=true, body_frame=true, scale=1.0))]
    fn new(
        entity: EntityId,
        component_name: String,
        offset: usize,
        color: Color,
        attached: bool,
        body_frame: bool,
        scale: f32,
    ) -> Self {
        Self {
            inner: impeller::well_known::VectorArrow {
                entity_id: entity.inner,
                id: ComponentId::new(&component_name),
                range: offset..offset + 2,
                color: color.inner,
                attached,
                body_frame,
                scale,
            },
        }
    }

    pub fn asset_name(&self) -> &'static str {
        impeller::well_known::VectorArrow::ASSET_NAME
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BodyAxes {
    inner: impeller::well_known::BodyAxes,
}

#[pymethods]
impl BodyAxes {
    #[new]
    #[pyo3(signature = (entity, scale=1.0))]
    fn new(entity: EntityId, scale: f32) -> Self {
        Self {
            inner: impeller::well_known::BodyAxes {
                entity_id: entity.inner,
                scale,
            },
        }
    }

    pub fn asset_name(&self) -> &'static str {
        impeller::well_known::BodyAxes::ASSET_NAME
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }
}
