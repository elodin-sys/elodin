use std::ops::Range;

use crate::{well_known::Color, Asset, ComponentId, EntityId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct EntityMetadata {
    pub name: String,
    pub color: Color,
}

impl Asset for EntityMetadata {
    const ASSET_NAME: &'static str = "entity_metadata";
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct VectorArrow {
    pub id: ComponentId,
    pub entity_id: EntityId,
    pub range: Range<usize>,
    pub color: Color,
    pub attached: bool,
    pub body_frame: bool,
    pub scale: f32,
}

impl Asset for VectorArrow {
    const ASSET_NAME: &'static str = "arrow";
}
