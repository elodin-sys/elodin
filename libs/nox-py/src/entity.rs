use crate::*;
use core::fmt;

use nox_ecs::conduit;

#[derive(Clone, Copy)]
#[pyclass]
pub struct EntityId {
    pub inner: conduit::EntityId,
}

#[pymethods]
impl EntityId {
    #[new]
    fn new(id: u64) -> Self {
        EntityId { inner: id.into() }
    }

    fn __str__(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.0.fmt(f)
    }
}
