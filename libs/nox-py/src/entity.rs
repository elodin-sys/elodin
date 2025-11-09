use crate::*;
use std::fmt;

#[derive(Clone, Copy)]
#[pyclass]
pub struct EntityId {
    pub inner: impeller2::types::EntityId,
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
    
    fn __int__(&self) -> u64 {
        self.inner.0
    }
    
    fn id(&self) -> u64 {
        self.inner.0
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.0.fmt(f)
    }
}
