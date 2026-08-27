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
            geometry_tile_count: geometry_tile_capacity(8, 5, 6),
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

/// STORAGE slots for the refine/draw tile lists (two buffers of
/// `count * 16` bytes per view). `1_000_000` was ~15.26 MiB each.
///
/// The data clipmap is `tree_size² × lods × sides`. GPU `refine_tiles`
/// can 4× that working set each step, so the buffer also holds one
/// generation of `4^REFINE_STEPS` children per side.
pub fn geometry_tile_capacity(tree_size: u32, lod_count: u32, side_count: u32) -> u32 {
    const LEAF_HEADROOM: u32 = 16;
    const REFINE_STEPS: u32 = 8;
    const MIN: u32 = 4096;
    const MAX: u32 = 262_144;
    let sides = side_count.max(1);
    let clipmap = tree_size
        .saturating_mul(tree_size)
        .saturating_mul(lod_count.max(1))
        .saturating_mul(sides)
        .saturating_mul(LEAF_HEADROOM);
    let refine = sides.saturating_mul(1 << (2 * REFINE_STEPS));
    clipmap.max(refine).clamp(MIN, MAX)
}

/// Matches `try_reserve_children` in `refine_tiles.wgsl`: four ping-pong
/// slots, or none. A failed reserve must not move `child_index`.
pub fn try_reserve_refine_children(child_index: i32, counter: i32, tile_count: i32) -> Option<i32> {
    let last = child_index.checked_add(counter.checked_mul(3)?)?;
    if (0..tile_count).contains(&child_index) && (0..tile_count).contains(&last) {
        Some(child_index)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_tile_capacity_fits_planar_refine() {
        assert_eq!(geometry_tile_capacity(8, 5, 1), 65_536);
    }

    #[test]
    fn geometry_tile_capacity_fits_globe_refine() {
        assert_eq!(geometry_tile_capacity(8, 5, 6), 262_144);
    }

    #[test]
    fn geometry_tile_capacity_clamps() {
        assert_eq!(geometry_tile_capacity(1, 1, 1), 65_536);
        assert_eq!(geometry_tile_capacity(32, 16, 6), 262_144);
    }

    #[test]
    fn refine_reserve_rejects_a_partial_generation() {
        assert_eq!(try_reserve_refine_children(0, 1, 8), Some(0));
        assert_eq!(try_reserve_refine_children(5, 1, 8), None);
        assert_eq!(try_reserve_refine_children(7, -1, 8), Some(7));
        assert_eq!(try_reserve_refine_children(2, -1, 8), None);
    }
}
