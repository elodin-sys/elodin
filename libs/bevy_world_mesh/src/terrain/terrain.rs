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
