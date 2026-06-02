use bevy::{asset::embedded_asset, prelude::*};
use itertools::Itertools;

// `embedded_asset!(app, "X")` produces a URL of the form
// `embedded://<CARGO_PKG_NAME>/<path from src/ to the asset>`.
// This crate's src lays them out under `src/terrain/shaders/`, so the
// URLs are `embedded://bevy_world_mesh/terrain/shaders/...`.

pub const DEFAULT_VERTEX_SHADER: &str =
    "embedded://bevy_world_mesh/terrain/shaders/render/vertex.wgsl";
pub const DEFAULT_FRAGMENT_SHADER: &str =
    "embedded://bevy_world_mesh/terrain/shaders/render/fragment.wgsl";
pub const PREPARE_PREPASS_SHADER: &str =
    "embedded://bevy_world_mesh/terrain/shaders/tiling_prepass/prepare_prepass.wgsl";
pub const REFINE_TILES_SHADER: &str =
    "embedded://bevy_world_mesh/terrain/shaders/tiling_prepass/refine_tiles.wgsl";
pub(crate) const SPLIT_SHADER: &str =
    "embedded://bevy_world_mesh/terrain/shaders/preprocess/split.wgsl";
pub(crate) const STITCH_SHADER: &str =
    "embedded://bevy_world_mesh/terrain/shaders/preprocess/stitch.wgsl";
pub(crate) const DOWNSAMPLE_SHADER: &str =
    "embedded://bevy_world_mesh/terrain/shaders/preprocess/downsample.wgsl";

#[derive(Default, Resource)]
pub(crate) struct InternalShaders(Vec<Handle<Shader>>);

impl InternalShaders {
    pub(crate) fn load(app: &mut App, shaders: &[&'static str]) {
        let mut shaders = shaders
            .iter()
            .map(|&shader| app.world_mut().resource_mut::<AssetServer>().load(shader))
            .collect_vec();

        let mut internal_shaders = app.world_mut().resource_mut::<InternalShaders>();
        internal_shaders.0.append(&mut shaders);
    }
}

pub(crate) fn load_terrain_shaders(app: &mut App) {
    embedded_asset!(app, "types.wgsl");
    embedded_asset!(app, "attachments.wgsl");
    embedded_asset!(app, "bindings.wgsl");
    embedded_asset!(app, "functions.wgsl");
    embedded_asset!(app, "debug.wgsl");
    embedded_asset!(app, "render/vertex.wgsl");
    embedded_asset!(app, "render/fragment.wgsl");
    embedded_asset!(app, "tiling_prepass/prepare_prepass.wgsl");
    embedded_asset!(app, "tiling_prepass/refine_tiles.wgsl");

    InternalShaders::load(
        app,
        &[
            "embedded://bevy_world_mesh/terrain/shaders/types.wgsl",
            "embedded://bevy_world_mesh/terrain/shaders/attachments.wgsl",
            "embedded://bevy_world_mesh/terrain/shaders/bindings.wgsl",
            "embedded://bevy_world_mesh/terrain/shaders/functions.wgsl",
            "embedded://bevy_world_mesh/terrain/shaders/debug.wgsl",
            "embedded://bevy_world_mesh/terrain/shaders/render/vertex.wgsl",
            "embedded://bevy_world_mesh/terrain/shaders/render/fragment.wgsl",
        ],
    );
}

pub(crate) fn load_preprocess_shaders(app: &mut App) {
    embedded_asset!(app, "preprocess/preprocessing.wgsl");
    embedded_asset!(app, "preprocess/split.wgsl");
    embedded_asset!(app, "preprocess/stitch.wgsl");
    embedded_asset!(app, "preprocess/downsample.wgsl");

    InternalShaders::load(
        app,
        &["embedded://bevy_world_mesh/terrain/shaders/preprocess/preprocessing.wgsl"],
    );
}
