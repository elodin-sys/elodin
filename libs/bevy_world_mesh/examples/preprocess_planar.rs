use bevy::prelude::*;
use bevy_world_mesh::prelude::*;

const PATH: &str = "terrains/planar";
const TEXTURE_SIZE: u32 = 512;
const LOD_COUNT: u32 = 4;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, TerrainPlugin, TerrainPreprocessPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let config = TerrainConfig {
        lod_count: LOD_COUNT,
        path: PATH.to_string(),
        ..default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        format: AttachmentFormat::R16,
        ..default()
    })
    .add_attachment(AttachmentConfig {
        name: "albedo".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        format: AttachmentFormat::Rgba8,
        ..default()
    });

    let mut tile_atlas = TileAtlas::new(&config);

    let preprocessor = Preprocessor::new()
        .clear_attachment(0, &mut tile_atlas)
        .clear_attachment(1, &mut tile_atlas)
        .preprocess_tile(
            PreprocessDataset {
                attachment_index: 0,
                path: format!("{PATH}/source/height.png"),
                lod_range: 0..LOD_COUNT,
                ..default()
            },
            &asset_server,
            &mut tile_atlas,
        )
        .preprocess_tile(
            PreprocessDataset {
                attachment_index: 1,
                path: format!("{PATH}/source/albedo.png"),
                lod_range: 0..LOD_COUNT,
                ..default()
            },
            &asset_server,
            &mut tile_atlas,
        );

    commands.spawn((tile_atlas, preprocessor));
}
