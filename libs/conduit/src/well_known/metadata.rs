use std::ops::Range;

use crate::{well_known::Color, Asset, ComponentId};
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
pub struct Gizmo {
    pub id: ComponentId,
    pub ty: GizmoType,
}

impl Asset for Gizmo {
    const ASSET_NAME: &'static str = "gizmo";
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum GizmoType {
    Vector { range: Range<usize>, color: Color },
}
