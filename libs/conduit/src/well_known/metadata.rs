use std::ops::Range;

use crate::{well_known::Color, Asset, AssetId, ComponentId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct EntityMetadata {
    pub name: String,
    pub color: Color,
}

impl Asset for EntityMetadata {
    const ASSET_ID: crate::AssetId = AssetId(2245);

    fn asset_id(&self) -> crate::AssetId {
        Self::ASSET_ID
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Gizmo {
    pub id: ComponentId,
    pub ty: GizmoType,
}

impl Asset for Gizmo {
    const ASSET_ID: crate::AssetId = AssetId(2243);

    fn asset_id(&self) -> crate::AssetId {
        Self::ASSET_ID
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum GizmoType {
    Vector { range: Range<usize>, color: Color },
}
