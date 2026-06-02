use bevy::{
    prelude::*, reflect::TypePath, render::render_resource::AsBindGroup, shader::ShaderRef,
};
use bevy_world_mesh::prelude::{TerrainMaterialPlugin, TerrainPlugin};
use bevy_world_mesh::terrain::{
    terrain_data::{tile_atlas::TileAtlas, tile_tree::TileTree},
    terrain_view::{TerrainViewComponents, TerrainViewConfig},
};

use crate::MainCamera;

/// Material used by Elodin's editor-hosted world mesh.
///
/// The renderer itself comes from the real `world_mesh` crate. This material is
/// intentionally tiny: it points the terrain renderer at the world-mesh fragment
/// shader, which samples height/albedo attachments from the atlas.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct WorldMeshMaterial {}

impl Material for WorldMeshMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/world_mesh.wgsl".into()
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }
}

/// Marker for terrain entities spawned from a schematic `world_mesh` element.
#[derive(Component)]
pub struct WorldMeshTerrain;

/// Editor integration layer for the real `world_mesh` terrain renderer.
///
/// The editor always links the terrain renderer so a schematic can opt in to
/// spawning a `world_mesh` element without requiring a cargo feature.
pub struct WorldMeshPlugin;

impl Plugin for WorldMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            TerrainPlugin,
            TerrainMaterialPlugin::<WorldMeshMaterial>::default(),
        ))
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
