// Preprocessor binary. Drives the `world_mesh` `Preprocessor` to convert
// the source height + albedo PNGs into a tile atlas under
// `assets/terrains/planar/<region>/`, where `<region>` is selected by the
// `WORLD_MESH_REGION` env var (default: `death_valley`).
//
// Auto-exits a few frames after the per-region `config.tc` is written
// (which the preprocessor does as its final step) so this can be scripted.

use bevy::prelude::*;
use bevy_world_mesh::prelude::*;
use bevy_world_mesh::scenes::planar::{planar_path, terrain_config, LOD_COUNT};
use std::path::Path;

fn main() {
    App::new()
        .add_plugins((
            // big_space panics at init if Bevy's default TransformPlugin is
            // running; `TerrainPlugin` pulls in `BigSpaceDefaultPlugins` when
            // the `high_precision` feature is enabled, so disable the stock
            // one here even though the preprocess pipeline doesn't actually
            // use big_space.
            DefaultPlugins.build().disable::<TransformPlugin>(),
            TerrainPlugin,
            TerrainPreprocessPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, exit_when_atlas_saved)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let config = terrain_config();
    let mut tile_atlas = TileAtlas::new(&config);
    let path = planar_path();

    let preprocessor = Preprocessor::new()
        .clear_attachment(0, &mut tile_atlas)
        .clear_attachment(1, &mut tile_atlas)
        .preprocess_tile(
            PreprocessDataset {
                attachment_index: 0,
                path: format!("{path}/source/height.png"),
                lod_range: 0..LOD_COUNT,
                ..default()
            },
            &asset_server,
            &mut tile_atlas,
        )
        .preprocess_tile(
            PreprocessDataset {
                attachment_index: 1,
                path: format!("{path}/source/albedo.png"),
                lod_range: 0..LOD_COUNT,
                ..default()
            },
            &asset_server,
            &mut tile_atlas,
        );

    commands.spawn((tile_atlas, preprocessor));
}

fn exit_when_atlas_saved(mut exit: MessageWriter<AppExit>, mut frames_after: Local<u32>) {
    let atlas_config = format!("assets/{}/config.tc", planar_path());
    if Path::new(&atlas_config).exists() {
        *frames_after += 1;
        // Give a healthy buffer for in-flight save tasks to flush to disk.
        if *frames_after > 120 {
            eprintln!("preprocess: atlas at {atlas_config}, exiting.");
            exit.write(AppExit::Success);
        }
    }
}
