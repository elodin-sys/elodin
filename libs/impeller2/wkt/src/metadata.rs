use impeller2::types::{ComponentId, EntityId, Msg, PacketId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Color;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ComponentMetadata {
    pub component_id: ComponentId,
    pub name: String,
    #[serde(default)]
    pub metadata: HashMap<String, MetadataValue>,
    #[serde(default)]
    pub asset: bool,
}

impl ComponentMetadata {
    pub fn element_names(&self) -> &str {
        self.metadata
            .get("element_names")
            .and_then(MetadataValue::as_str)
            .unwrap_or_default()
    }
}

impl Msg for ComponentMetadata {
    const ID: PacketId = [224, 0, 11];
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct EntityMetadata {
    pub entity_id: EntityId,
    pub name: String,
    #[serde(default)]
    pub metadata: HashMap<String, MetadataValue>,
}

impl Msg for EntityMetadata {
    const ID: PacketId = [224, 0, 11];
}

pub trait MetadataExt {
    fn metadata_mut(&mut self) -> &mut HashMap<String, MetadataValue>;
    fn metadata(&self) -> &HashMap<String, MetadataValue>;
    fn priority(&self) -> i64 {
        self.metadata()
            .get("priority")
            .and_then(|v| match &v {
                MetadataValue::I64(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(10)
    }
    fn color(&self) -> Color {
        self.metadata()
            .get("color")
            .and_then(|v| match &v {
                MetadataValue::Bytes(b) => postcard::from_bytes(b).ok(),
                _ => None,
            })
            .unwrap_or(Color::YOLK)
    }
    fn set_priority(&mut self, priority: i64) {
        self.metadata_mut()
            .insert("priority".to_string(), MetadataValue::I64(priority));
    }
    fn set_color(&mut self, color: Color) {
        self.metadata_mut().insert(
            "color".to_string(),
            MetadataValue::Bytes(postcard::to_allocvec(&color).unwrap()),
        );
    }
}

impl MetadataExt for ComponentMetadata {
    fn metadata_mut(&mut self) -> &mut HashMap<String, MetadataValue> {
        &mut self.metadata
    }
    fn metadata(&self) -> &HashMap<String, MetadataValue> {
        &self.metadata
    }
}

impl MetadataExt for EntityMetadata {
    fn metadata_mut(&mut self) -> &mut HashMap<String, MetadataValue> {
        &mut self.metadata
    }
    fn metadata(&self) -> &HashMap<String, MetadataValue> {
        &self.metadata
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum MetadataValue {
    Unit,
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(v) = self {
            Some(v)
        } else {
            None
        }
    }
}
