use impeller2::types::{ComponentId, EntityId, Msg, PacketId};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::HashMap};

use crate::Color;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ComponentMetadata {
    pub component_id: ComponentId,
    pub name: Cow<'static, str>,
    pub metadata: Metadata,
    pub asset: bool,
}

impl ComponentMetadata {
    pub fn element_names(&self) -> &str {
        self.metadata
            .metadata
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
    pub metadata: Metadata,
}

impl Msg for EntityMetadata {
    const ID: PacketId = [224, 0, 11];
}

#[derive(Clone, Serialize, Deserialize, Default, Debug, PartialEq)]
pub struct Metadata {
    pub metadata: HashMap<String, MetadataValue>,
}

impl Metadata {
    pub fn priority(&self) -> i64 {
        self.metadata
            .get("priority")
            .and_then(|v| match &v {
                MetadataValue::I64(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(10)
    }

    pub fn color(&self) -> Color {
        self.metadata
            .get("priority")
            .and_then(|v| match &v {
                MetadataValue::Bytes(b) => postcard::from_bytes(b).ok(),
                _ => None,
            })
            .unwrap_or(Color::YOLK)
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
