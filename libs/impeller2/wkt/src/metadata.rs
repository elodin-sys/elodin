use impeller2::types::{ComponentId, EntityId, Msg, PacketId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ComponentMetadata {
    pub component_id: ComponentId,
    pub name: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    #[serde(default)]
    pub asset: bool,
}

impl ComponentMetadata {
    pub fn element_names(&self) -> &str {
        self.metadata
            .get("element_names")
            .map(|v| v.as_str())
            .unwrap_or_default()
    }
}

impl Msg for ComponentMetadata {
    const ID: PacketId = [224, 11];
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct EntityMetadata {
    pub entity_id: EntityId,
    pub name: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Msg for EntityMetadata {
    const ID: PacketId = [224, 30];
}

pub trait MetadataExt {
    fn metadata_mut(&mut self) -> &mut HashMap<String, String>;
    fn metadata(&self) -> &HashMap<String, String>;
    fn priority(&self) -> i64 {
        self.metadata()
            .get("priority")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10)
    }
    fn set_priority(&mut self, priority: i64) {
        self.set("priority", &priority.to_string());
    }
    fn set(&mut self, key: &str, value: &str) {
        self.metadata_mut()
            .insert(key.to_string(), value.to_string());
    }
    fn get(&self, key: &str) -> Option<&str> {
        self.metadata().get(key).map(|v| v.as_str())
    }
}

impl MetadataExt for ComponentMetadata {
    fn metadata_mut(&mut self) -> &mut HashMap<String, String> {
        &mut self.metadata
    }
    fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

impl MetadataExt for EntityMetadata {
    fn metadata_mut(&mut self) -> &mut HashMap<String, String> {
        &mut self.metadata
    }
    fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}
