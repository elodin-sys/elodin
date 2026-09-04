//! Types for configuring terrains.
//!
#[cfg(feature = "high_precision")]
use big_space::prelude::{CellCoord, CellTransformOwned, Grid};

use crate::terrain::{
    math::TerrainModel,
    terrain_data::{tile_atlas::TileAtlas, AttachmentConfig},
};
use bevy::{camera::visibility::NoFrustumCulling, ecs::entity::EntityHashMap, prelude::*};

/// Resource that stores components that are associated to a terrain entity.
/// This is used to persist components in the render world.
#[derive(Deref, DerefMut, Resource)]
pub struct TerrainComponents<C>(EntityHashMap<C>);

impl<C> Default for TerrainComponents<C> {
    fn default() -> Self {
        Self(default())
    }
}

/// The configuration of a terrain.
///
/// Here you can define all fundamental parameters of the terrain.
#[derive(Clone)]
pub struct TerrainConfig {
    /// The count of level of detail layers.
    pub lod_count: u32,
    pub model: TerrainModel,
    /// The amount of tiles the can be loaded simultaneously in the tile atlas.
    pub atlas_size: u32,
    /// The path to the terrain folder inside the assets directory.
    pub path: String,
    /// The attachments of the terrain.
    pub attachments: Vec<AttachmentConfig>,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            lod_count: 1,
            model: TerrainModel::sphere(default(), 1.0, 0.0, 1.0),
            atlas_size: 1024,
            path: default(),
            attachments: default(),
        }
    }
}

impl TerrainConfig {
    pub fn add_attachment(mut self, attachment_config: AttachmentConfig) -> Self {
        self.attachments.push(attachment_config);
        self
    }
}

/// Smallest planar clipmap that stayed in-index on RC-jet (32 was too small).
pub const PLANAR_ATLAS_MIN: u32 = 64;
/// Working-set cap. LOD5 planar datasets are 341 tiles; 1024 layers is ~2 GiB.
pub const PLANAR_ATLAS_MAX: u32 = 256;

/// GPU atlas layers for a planar region: at least [`PLANAR_ATLAS_MIN`], at
/// most [`PLANAR_ATLAS_MAX`], and never larger than the on-disk tile count
/// when that count already exceeds the minimum.
pub fn planar_atlas_size(dataset_tiles: u32) -> u32 {
    if dataset_tiles == 0 {
        return PLANAR_ATLAS_MIN;
    }
    dataset_tiles.clamp(PLANAR_ATLAS_MIN, PLANAR_ATLAS_MAX)
}

/// Atlas layers for the preprocess binary: the whole planar tile pyramid,
/// `(4^lod_count - 1) / 3` tiles (341 at LOD 5), lives in the atlas at once.
pub fn planar_preprocess_atlas_size(dataset_tiles: u32, lod_count: u32) -> u32 {
    dataset_tiles.max((4u32.pow(lod_count) - 1) / 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_atlas_size_uses_min_when_dataset_is_unknown() {
        assert_eq!(planar_atlas_size(0), PLANAR_ATLAS_MIN);
    }

    #[test]
    fn planar_atlas_size_clamps_to_working_set() {
        assert_eq!(planar_atlas_size(32), PLANAR_ATLAS_MIN);
        assert_eq!(planar_atlas_size(80), 80);
        assert_eq!(planar_atlas_size(341), PLANAR_ATLAS_MAX);
        assert_eq!(planar_atlas_size(1024), PLANAR_ATLAS_MAX);
    }

    #[test]
    fn planar_preprocess_atlas_holds_the_full_dataset() {
        assert_eq!(planar_preprocess_atlas_size(0, 5), 341);
        assert_eq!(planar_preprocess_atlas_size(341, 5), 341);
        assert_eq!(planar_preprocess_atlas_size(2048, 5), 2048);
    }
}

/// The components of a terrain.
///
/// Does not include loader(s) and a material.
#[derive(Bundle)]
pub struct TerrainBundle {
    pub tile_atlas: TileAtlas,
    #[cfg(feature = "high_precision")]
    pub cell: CellCoord,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub visibility: Visibility,
    pub inherited_visibility: InheritedVisibility,
    pub view_visibility: ViewVisibility,
    pub no_frustum_culling: NoFrustumCulling,
}

impl TerrainBundle {
    /// Creates a new terrain bundle from the config.
    pub fn new(tile_atlas: TileAtlas, #[cfg(feature = "high_precision")] frame: &Grid) -> Self {
        #[cfg(feature = "high_precision")]
        let CellTransformOwned { transform, cell } = tile_atlas.model.grid_transform(frame);
        #[cfg(not(feature = "high_precision"))]
        let transform = tile_atlas.model.transform();

        Self {
            tile_atlas,
            transform,
            #[cfg(feature = "high_precision")]
            cell,
            global_transform: default(),
            visibility: Visibility::Visible,
            inherited_visibility: default(),
            view_visibility: default(),
            no_frustum_culling: NoFrustumCulling,
        }
    }
}
