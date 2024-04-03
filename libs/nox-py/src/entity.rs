use crate::*;
use core::fmt;

use nox_ecs::conduit;

#[derive(Clone)]
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

#[derive(Clone)]
#[pyclass]
pub struct Entity {
    pub id: EntityId,
    pub world: Py<WorldBuilder>,
}

#[pymethods]
impl Entity {
    pub fn id(&self) -> EntityId {
        self.id.clone()
    }

    pub fn insert(&mut self, py: Python<'_>, archetype: Spawnable<'_>) -> Result<Self, Error> {
        let mut world = self.world.borrow_mut(py);
        world.spawn_with_entity_id(archetype, self.id())?;
        Ok(self.clone())
    }

    pub fn metadata(&mut self, py: Python<'_>, metadata: EntityMetadata) -> Self {
        let mut world = self.world.borrow_mut(py);
        let metadata = world.world.insert_asset(metadata.inner);
        world.world.spawn_with_id(metadata, self.id.inner);
        self.clone()
    }

    pub fn name(&mut self, py: Python<'_>, name: String) -> Self {
        self.metadata(py, EntityMetadata::new(name, None))
    }
}
