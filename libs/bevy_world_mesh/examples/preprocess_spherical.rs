use bevy::prelude::*;
use bevy_world_mesh::prelude::*;

const PATH: &str = "terrains/spherical";
const TEXTURE_SIZE: u32 = 512;
const LOD_COUNT: u32 = 5;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.build().disable::<TransformPlugin>(),
            TerrainPlugin,
            TerrainPreprocessPlugin,
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let config = TerrainConfig {
        lod_count: LOD_COUNT,
        path: PATH.to_string(),
        atlas_size: 2048,
        ..default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        format: AttachmentFormat::R16,
        ..default()
    });

    let mut tile_atlas = TileAtlas::new(&config);

    let preprocessor = Preprocessor::new()
        .clear_attachment(0, &mut tile_atlas)
        .preprocess_spherical(
            SphericalDataset {
                attachment_index: 0,
                paths: (0..6)
                    .map(|side| format!("{PATH}/source/height/face{side}.tif"))
                    .collect(),
                lod_range: 0..LOD_COUNT,
            },
            &asset_server,
            &mut tile_atlas,
        );

    commands.spawn((tile_atlas, preprocessor));
}
