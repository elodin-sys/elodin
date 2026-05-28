use bevy::prelude::*;

use bevy_world_mesh::prelude::{TerrainMaterialPlugin, TerrainPlugin};
use bevy_world_mesh::scenes::planar::WorldMeshMaterial;

/// Editor integration layer for `bevy_world_mesh`.
///
/// The editor always links the terrain renderer so a schematic can opt-in to
/// spawning a world_mesh element without requiring a cargo feature.
pub struct WorldMeshPlugin;

impl Plugin for WorldMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            TerrainPlugin,
            TerrainMaterialPlugin::<WorldMeshMaterial>::default(),
        ));
    }
}
