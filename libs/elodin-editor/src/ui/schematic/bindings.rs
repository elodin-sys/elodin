use bevy::prelude::{Entity, Resource};
use impeller2_wkt::NodeId;
use std::collections::HashMap;

/// Maps schematic `NodeId`s to Bevy `Entity`s at runtime.
///
/// Ephemeral bindings are rebuilt every frame by `tiles_to_schematic`.
#[derive(Resource, Default)]
pub struct SchematicBindings {
    map: HashMap<NodeId, Entity>,
    ephemeral_keys: Vec<NodeId>,
}

impl SchematicBindings {
    pub fn clear_ephemeral(&mut self) {
        for id in self.ephemeral_keys.drain(..) {
            self.map.remove(&id);
        }
    }

    pub fn bind_ephemeral(&mut self, id: NodeId, entity: Entity) {
        self.map.insert(id, entity);
        self.ephemeral_keys.push(id);
    }

    pub fn bind(&mut self, id: NodeId, entity: Entity) {
        self.map.insert(id, entity);
    }

    pub fn get(&self, id: NodeId) -> Option<Entity> {
        self.map.get(&id).copied()
    }

    pub fn remove(&mut self, id: NodeId) {
        self.map.remove(&id);
    }
}
