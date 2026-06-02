//! Planar scene plugin ([`PlanarScenePlugin`]) — km-scale real-world region
//! rendering driven by [`crate::regions`] presets. Wires the terrain
//! renderer, a `big_space` floating-origin camera, the debug overlay
//! plugin, and the screenshot harness into a single ready-to-fly app.

use crate::prelude::*;
use crate::regions::RegionManifest;
pub use crate::terrain::render::world_mesh_material::WorldMeshMaterial;
use bevy::math::DVec3;
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;

/// Per-region atlas path (relative to `assets/`). Multiple regions coexist
/// under `terrains/planar/<region>/` so swapping between them doesn't wipe
/// the previous fetch + preprocess. The active region is selected at
/// runtime via the `WORLD_MESH_REGION` env var; the default
/// (`death_valley`) keeps the original entry point working without any env
/// var.
pub fn planar_path() -> String {
    let region = std::env::var("WORLD_MESH_REGION").unwrap_or_else(|_| "death_valley".to_string());
    format!("terrains/planar/{region}")
}

pub const TEXTURE_SIZE: u32 = 512;
/// Atlas LOD depth for the planar path. LOD_COUNT=5 gives deepest LOD at
/// 2^4 * TEXTURE_SIZE = 8192 px per terrain dim; for a 20-30 km region that
/// resolves to ~2-4 m/px display, backed by Sentinel-2 z=14 source via
/// OUTPUT_SIZE=4096. Default `atlas_size=1024` comfortably fits
/// (4^5 - 1) / 3 = 341 total tiles.
pub const LOD_COUNT: u32 = 5;

/// End-to-end scene plugin for planar terrain. Reads the active region's
/// `region.toml` manifest, boots up the renderer with the right terrain
/// scale + elevation bracket, spawns a `big_space` fly-camera, and wires
/// in the debug overlays + screenshot harness.
pub struct PlanarScenePlugin;

impl Plugin for PlanarScenePlugin {
    fn build(&self, app: &mut App) {
        let manifest = RegionManifest::load_or_default();
        let title = format!(
            "World Mesh — {} ({} km, [{}, {}] m)",
            manifest.name, manifest.side_km, manifest.min_height_m, manifest.max_height_m
        );
        app.insert_resource(manifest)
            .add_plugins((
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title,
                            resolution: (1280, 800).into(),
                            ..default()
                        }),
                        ..default()
                    })
                    .build()
                    .disable::<TransformPlugin>(),
                WorldMeshPlugin,
                TerrainDebugPlugin,
                EnvScreenshotPlugin,
            ))
            .add_systems(Startup, setup);
    }
}

/// Tile-atlas configuration. Reads the same manifest the runtime renderer
/// uses, so the `preprocess` binary builds an atlas with matching scale +
/// height bounds.
pub fn terrain_config() -> TerrainConfig {
    let manifest = RegionManifest::load_or_default();
    let terrain_size = manifest.terrain_size_m();
    let height = manifest.height_m();

    TerrainConfig {
        lod_count: LOD_COUNT,
        // y=0 sits at min_height_m on the ground. Anchor the centre of the
        // terrain a fraction below origin so the camera math is symmetric.
        model: TerrainModel::planar(
            DVec3::new(0.0, -(height as f64) * 0.4, 0.0),
            terrain_size,
            0.0,
            height,
        ),
        path: planar_path(),
        ..default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::R16,
    })
    .add_attachment(AttachmentConfig {
        name: "albedo".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::Rgba8,
    })
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<WorldMeshMaterial>>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
    manifest: Res<RegionManifest>,
) {
    let config = terrain_config();
    let view_config = TerrainViewConfig::default();

    let tile_atlas = TileAtlas::new(&config);
    let tile_tree = TileTree::new(&tile_atlas, &view_config);

    let scale = manifest.terrain_size_m() as f32;
    let off = manifest.camera_offset_xyz_norm;
    let tgt = manifest.camera_target_xyz_norm;
    let camera_pos = Vec3::new(off[0] * scale, off[1] * scale, off[2] * scale);
    let camera_target = Vec3::new(tgt[0] * scale, tgt[1] * scale, tgt[2] * scale);

    // big_space is required because the workspace enables `high_precision` on
    // bevy_terrain (the globe path needs Earth-radius coordinates). For the
    // planar km-scale case this is a no-op — `translation_to_grid` returns
    // cell=(0,0,0) and translation=position — but the API shape is the same.
    commands.spawn_big_space(Grid::default(), |root| {
        let frame = root.grid().clone();

        let terrain = root
            .spawn_spatial((
                TerrainBundle::new(tile_atlas, &frame),
                MeshMaterial3d(materials.add(WorldMeshMaterial::default())),
            ))
            .id();

        let (cell, translation) = frame.translation_to_grid(camera_pos.as_dvec3());
        // big_space 0.12's `BigSpaceCameraController` replaces our old
        // `DebugCameraBundle` outright: WASD/Space/Ctrl/Q/E/mouse/Shift are
        // wired up for free by the controller's companion
        // `default_camera_inputs` system.
        //
        // `with_slowing(false)` here because the planar `TerrainBundle` has
        // no `Aabb` component — `slow_near_objects` iterates entities with
        // `Aabb` to find the nearest and scale the speed by that distance,
        // but with no candidates it silently falls back to
        // `effective_speed = c × (c + boost)`, which collapses to 1 m/s at
        // the default `c = 1.0`. Imperceptible in a 22 km scene. Fixed
        // `with_speed` is the simpler fit: with `c = sqrt(target_speed)`
        // the controller gives us a constant cruise speed. `c = 300` →
        // `≈ 90 km/s` cruise, sized for the 500 km Mojave Desert region
        // at aircraft altitude. The 22 km Death Valley scene works fine
        // at this speed too — just faster traversal. Shift boost adds a
        // marginal nudge (`c × (c+1)`); a region-adaptive speed tied to
        // `manifest.side_km` is a future enhancement if the small regions
        // start to feel twitchy.
        let view = root
            .spawn_spatial((
                Camera3d::default(),
                Projection::Perspective(PerspectiveProjection {
                    near: 0.1,
                    ..default()
                }),
                Transform::from_translation(translation).looking_at(camera_target, Vec3::Y),
                cell,
                FloatingOrigin,
                BigSpaceCameraController::default()
                    .with_speed(300.0)
                    .with_slowing(false),
            ))
            .id();

        tile_trees.insert((terrain, view), tile_tree);
    });
}

// Screenshot harness lives in
// `crate::terrain::debug::screenshot::EnvScreenshotPlugin` (re-exported as
// `EnvScreenshotPlugin` in the crate prelude under the `debug` feature). Set
// `WORLD_MESH_SCREENSHOT`, `WORLD_MESH_SCREENSHOT_DELAY`, and
// `WORLD_MESH_SCREENSHOT_EXIT` to drive it.
