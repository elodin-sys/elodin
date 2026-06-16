use crate::terrain::{
    terrain_data::tile_tree::{TileTree, TileTreeEntry},
    terrain_view::TerrainViewComponents,
    util::StaticBuffer,
};
use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        Extract,
    },
};
use bytemuck::cast_slice;
use ndarray::{Array2, Array4};
use std::mem;

/// Stores the GPU representation of the [`TileTree`] (array texture)
/// alongside the data to update it.
///
/// The data is synchronized each frame by copying it from the [`TileTree`] to the texture.
#[derive(Component)]
pub struct GpuTileTree {
    pub(crate) tile_tree_buffer: StaticBuffer<()>,
    pub(crate) origins_buffer: StaticBuffer<()>,
    /// The current cpu tile_tree data. This is synced each frame with the tile_tree data.
    data: Array4<TileTreeEntry>,
    origins: Array2<UVec2>,
}

impl GpuTileTree {
    fn new(device: &RenderDevice, tile_tree: &TileTree) -> Self {
        let tile_tree_buffer = StaticBuffer::empty_sized(
            None,
            device,
            (tile_tree.data.len() * mem::size_of::<TileTreeEntry>()) as BufferAddress,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        let origins_buffer = StaticBuffer::empty_sized(
            None,
            device,
            (tile_tree.origins.len() * mem::size_of::<UVec2>()) as BufferAddress,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        Self {
            tile_tree_buffer,
            origins_buffer,
            data: default(),
            origins: default(),
        }
    }

    /// Initializes the [`GpuTileTree`] of newly created terrains.
    pub(crate) fn initialize(
        device: Res<RenderDevice>,
        mut gpu_tile_trees: ResMut<TerrainViewComponents<GpuTileTree>>,
        tile_trees: Extract<Res<TerrainViewComponents<TileTree>>>,
    ) {
        for (&(terrain, view), tile_tree) in tile_trees.iter() {
            if gpu_tile_trees.contains_key(&(terrain, view)) {
                continue;
            }

            gpu_tile_trees.insert((terrain, view), GpuTileTree::new(&device, tile_tree));
        }
    }

    /// Extracts the current data from all [`TileTree`]s into the corresponding [`GpuTileTree`]s.
    pub(crate) fn extract(
        mut gpu_tile_trees: ResMut<TerrainViewComponents<GpuTileTree>>,
        tile_trees: Extract<Res<TerrainViewComponents<TileTree>>>,
    ) {
        for (&(terrain, view), tile_tree) in tile_trees.iter() {
            let Some(gpu_tile_tree) = gpu_tile_trees.get_mut(&(terrain, view)) else {
                continue;
            };

            gpu_tile_tree.data = tile_tree.data.clone();
            gpu_tile_tree.origins = tile_tree.origins.clone();
        }
    }

    /// Prepares the tile_tree data to be copied into the tile_tree texture.
    pub(crate) fn prepare(
        queue: Res<RenderQueue>,
        mut gpu_tile_trees: ResMut<TerrainViewComponents<GpuTileTree>>,
    ) {
        for gpu_tile_tree in gpu_tile_trees.values_mut() {
            let data = cast_slice(gpu_tile_tree.data.as_slice().unwrap());
            gpu_tile_tree.tile_tree_buffer.update_bytes(&queue, data);

            let origins = cast_slice(gpu_tile_tree.origins.as_slice().unwrap());
            gpu_tile_tree.origins_buffer.update_bytes(&queue, origins);
        }
    }
}
