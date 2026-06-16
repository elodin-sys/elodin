//! Types for configuring terrain views.

use bevy::{platform::collections::HashMap, prelude::*};

/// Resource that stores components that are associated to a terrain entity and a view entity.
#[derive(Deref, DerefMut, Resource)]
pub struct TerrainViewComponents<C>(HashMap<(Entity, Entity), C>);

impl<C> Default for TerrainViewComponents<C> {
    fn default() -> Self {
        Self(default())
    }
}

/// The configuration of a terrain view.
///
/// A terrain view describes the quality settings the corresponding terrain will be rendered with.
#[derive(Clone)]
pub struct TerrainViewConfig {
    /// The count of tiles in x and y direction per tile tree layer.
    pub tree_size: u32,
    /// The size of the tile buffer.
    pub geometry_tile_count: u32,
    /// The amount of steps the tile list will be refined.
    pub refinement_count: u32,
    /// The number of rows and columns of the tile grid.
    pub grid_size: u32,
    /// The percentage tolerance added to the morph distance during tile subdivision.
    /// This is required to counteracted the distortion of the subdivision distance estimation near the corners of the cube sphere.
    /// For planar terrains this can be set to zero and for spherical / ellipsoidal terrains a value of around 0.1 is necessary.
    pub subdivision_tolerance: f64,
    pub precision_threshold_distance: f64,
    pub load_distance: f64,
    /// The distance measured in tile sizes between adjacent LOD layers.
    /// This currently has to be larger than about 6, since the tiles can only morph to the adjacent layer.
    /// Should the morph distance be too small, this will result in morph transitions suddenly being canceled, by the next LOD.
    /// This is dependent on the morph distance, the morph ratio and the subdivision tolerance. It can be debug with the show tiles debug view.
    pub morph_distance: f64,
    pub blend_distance: f64,
    /// The morph percentage of the mesh.
    pub morph_range: f32,
    /// The blend percentage in the vertex and fragment shader.
    pub blend_range: f32,
    pub origin_lod: u32,
}

impl Default for TerrainViewConfig {
    fn default() -> Self {
        Self {
            tree_size: 8,
            geometry_tile_count: 1000000,
            refinement_count: 30,
            grid_size: 16,
            subdivision_tolerance: 0.1,
            load_distance: 2.5,
            morph_distance: 16.0,
            blend_distance: 2.0,
            morph_range: 0.2,
            blend_range: 0.2,
            precision_threshold_distance: 0.001,
            origin_lod: 10,
        }
    }
}
