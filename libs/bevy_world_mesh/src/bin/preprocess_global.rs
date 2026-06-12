// Spherical-atlas preprocessor for the top-level globe app. Reads the 6
// height + 6 albedo cube-face TIFFs that `fetch_global_spherical` produced
// (or the `synthesize_spherical_faces` synthetic fallback for height) and
// drives the `Preprocessor::preprocess_spherical` pipeline for both
// attachment indices, writing the runtime atlas under
// `assets/terrains/spherical/data/`.
//
// Modelled on `examples/preprocess_spherical.rs` but with a second
// attachment for the albedo and the same exit-when-atlas-saved guard the
// planar `preprocess` binary uses, so this is scriptable.

use bevy::{
    app::ScheduleRunnerPlugin,
    asset::AssetPlugin,
    prelude::*,
    window::{ExitCondition, WindowPlugin},
    winit::WinitPlugin,
};
use bevy_world_mesh::prelude::*;
use bevy_world_mesh::scenes::globe::{PATH, globe_terrain_config};
use bevy_world_mesh::terrain::util::{asset_path, assets_root};
use std::time::Duration;

fn main() {
    App::new()
        .add_plugins((
            // Run headless: preprocessing only needs Bevy's asset/render/wgpu
            // stack for compute shaders, not a visible window.
            //
            // big_space requires TransformPlugin disabled, matching the
            // upstream spherical example. Preprocess itself never touches
            // big_space, but `scenes` enables `high_precision` workspace-
            // wide so we keep the same DefaultPlugins shape everywhere.
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: assets_root().to_string_lossy().into_owned(),
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<TransformPlugin>(),
            ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(1.0 / 60.0)),
            TerrainPlugin,
            TerrainPreprocessPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, exit_when_atlas_saved)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let config = globe_terrain_config();
    let lod_count = config.lod_count;
    let mut tile_atlas = TileAtlas::new(&config);

    let preprocessor = Preprocessor::new()
        .clear_attachment(0, &mut tile_atlas)
        .clear_attachment(1, &mut tile_atlas)
        .preprocess_spherical(
            SphericalDataset {
                attachment_index: 0,
                paths: (0..6)
                    .map(|side| format!("{PATH}/source/height/face{side}.tif"))
                    .collect(),
                lod_range: 0..lod_count,
            },
            &asset_server,
            &mut tile_atlas,
        )
        .preprocess_spherical(
            SphericalDataset {
                attachment_index: 1,
                // PNG for albedo: bevy_terrain's TIFF loader is hardcoded to
                // `R16Unorm`, and the `Rgba8` attachment expects 4-channel
                // sRGB. Bevy's PNG loader auto-expands RGB→RGBA8UnormSrgb.
                paths: (0..6)
                    .map(|side| format!("{PATH}/source/albedo/face{side}.png"))
                    .collect(),
                lod_range: 0..lod_count,
            },
            &asset_server,
            &mut tile_atlas,
        );

    commands.spawn((tile_atlas, preprocessor));
}

fn exit_when_atlas_saved(mut exit: MessageWriter<AppExit>, mut frames_after: Local<u32>) {
    let atlas_config = asset_path(format!("{PATH}/config.tc"));
    if atlas_config.exists() {
        *frames_after += 1;
        if *frames_after > 120 {
            eprintln!(
                "preprocess_global: atlas at {}, exiting.",
                atlas_config.display()
            );
            exit.write(AppExit::Success);
        }
    }
}
