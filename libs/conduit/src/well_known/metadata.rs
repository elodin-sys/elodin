use crate::{well_known::Color, Asset, AssetId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct EntityMetadata {
    pub name: String,
    pub color: Color,
}

impl Asset for EntityMetadata {
    const ASSET_ID: crate::AssetId = AssetId(2242);

    fn asset_id(&self) -> crate::AssetId {
        Self::ASSET_ID
    }
}
