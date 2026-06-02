//! Minimal example: flat planar terrain with the `DebugTerrainMaterial`
//! (gray height visualisation). Useful as a build-test of the renderer
//! + preprocess pipeline without pulling in any real-data scaffolding.

use bevy::math::DVec3;
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;
use bevy_world_mesh::prelude::*;

const PATH: &str = "terrains/planar";
const TERRAIN_SIZE: f64 = 1000.0;
const HEIGHT: f32 = 250.0;
const TEXTURE_SIZE: u32 = 512;
const LOD_COUNT: u32 = 4;

fn main() {
    App::new()
        .add_plugins((
            // big_space requires TransformPlugin disabled.
            DefaultPlugins.build().disable::<TransformPlugin>(),
            TerrainPlugin,
            TerrainMaterialPlugin::<DebugTerrainMaterial>::default(),
            TerrainDebugPlugin,
            EnvScreenshotPlugin,
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<DebugTerrainMaterial>>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let config = TerrainConfig {
        lod_count: LOD_COUNT,
        model: TerrainModel::planar(DVec3::new(0.0, -100.0, 0.0), TERRAIN_SIZE, 0.0, HEIGHT),
        path: PATH.to_string(),
        ..default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::R16,
    });

    let view_config = TerrainViewConfig::default();

    let tile_atlas = TileAtlas::new(&config);
    let tile_tree = TileTree::new(&tile_atlas, &view_config);

    commands.spawn_big_space(Grid::default(), |root| {
        let frame = root.grid().clone();

        let terrain = root
            .spawn_spatial((
                TerrainBundle::new(tile_atlas, &frame),
                MeshMaterial3d(materials.add(DebugTerrainMaterial::default())),
            ))
            .id();

        let view = root
            .spawn_spatial((
                Camera3d::default(),
                Projection::Perspective(PerspectiveProjection {
                    near: 0.1,
                    ..default()
                }),
                Transform::from_xyz(0.0, 250.0, 600.0).looking_at(Vec3::ZERO, Vec3::Y),
                FloatingOrigin,
                BigSpaceCameraController::default().with_slowing(true),
            ))
            .id();

        tile_trees.insert((terrain, view), tile_tree);

        root.spawn_spatial((
            Mesh3d(meshes.add(Cuboid::from_length(10.0))),
            Transform::from_xyz(TERRAIN_SIZE as f32 / 2.0, 100.0, TERRAIN_SIZE as f32 / 2.0),
        ));
    });
}
