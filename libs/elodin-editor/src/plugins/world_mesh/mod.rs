use bevy::prelude::*;
use bevy_world_mesh::prelude::WorldMeshPlugin as BevyWorldMeshRendererPlugin;
use bevy_world_mesh::terrain::{
    terrain_data::{tile_atlas::TileAtlas, tile_tree::TileTree},
    terrain_view::{TerrainViewComponents, TerrainViewConfig},
};

use crate::MainCamera;

/// Marker for terrain entities spawned from a schematic `world_mesh` element.
#[derive(Component)]
pub struct WorldMeshTerrain;

/// Editor integration layer for the real `bevy_world_mesh` terrain renderer.
///
/// The renderer/material plugin lives in `bevy_world_mesh`; this editor plugin
/// only adds Elodin-specific dynamic viewport wiring.
pub struct EditorWorldMeshPlugin;

impl Plugin for EditorWorldMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(BevyWorldMeshRendererPlugin)
            .add_systems(Update, sync_terrain_view_components);
    }
}

/// The terrain renderer needs one [`TileTree`] per `(terrain, camera)` pair.
/// Editor viewports are spawned dynamically from KDL, so wire the pairs after
/// both the terrain entity and viewport cameras exist.
fn sync_terrain_view_components(
    terrains: Query<(Entity, &TileAtlas), With<WorldMeshTerrain>>,
    cameras: Query<Entity, With<MainCamera>>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
) {
    tile_trees
        .retain(|(terrain, view), _| terrains.get(*terrain).is_ok() && cameras.get(*view).is_ok());

    let view_config = TerrainViewConfig::default();
    for (terrain, tile_atlas) in &terrains {
        for view in &cameras {
            tile_trees
                .entry((terrain, view))
                .or_insert_with(|| TileTree::new(tile_atlas, &view_config));
        }
    }
}
