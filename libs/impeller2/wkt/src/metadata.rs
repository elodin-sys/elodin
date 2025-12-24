use impeller2::types::{ComponentId, EntityId};
use postcard_schema::Schema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Schema)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct ComponentMetadata {
    pub component_id: ComponentId,
    pub name: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl ComponentMetadata {
    pub fn element_names(&self) -> &str {
        self.metadata
            .get("element_names")
            .map(|v| v.as_str())
            .unwrap_or_default()
    }

    /// Returns true if this component is a timestamp source field.
    /// Timestamp source components contain raw clock values used as timestamps
    /// for other components, and should be excluded from time range calculations.
    pub fn is_timestamp_source(&self) -> bool {
        self.metadata
            .get("_is_timestamp_source")
            .map(|v| v == "true")
            .unwrap_or(false)
    }

    /// Marks this component as a timestamp source field.
    pub fn set_timestamp_source(&mut self, is_source: bool) {
        if is_source {
            self.metadata
                .insert("_is_timestamp_source".to_string(), "true".to_string());
        } else {
            self.metadata.remove("_is_timestamp_source");
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Schema)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct EntityMetadata {
    pub entity_id: EntityId,
    pub name: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
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
