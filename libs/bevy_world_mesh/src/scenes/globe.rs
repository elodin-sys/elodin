//! Globe scene plugin ([`GlobeScenePlugin`]): WGS84 ellipsoid Earth driven
//! by the same `WorldMeshMaterial` the planar scene uses, fed by 6
//! cube-face TIFFs (height + albedo) under
//! `assets/terrains/spherical/{source,data}/`.
//!
//! The structural difference from the planar scene is mostly the
//! [`crate::terrain::math::TerrainModel`]: we use
//! `TerrainModel::ellipsoid` (semi-major 6 378 137 m, semi-minor
//! 6 356 752.314 245 m) instead of `TerrainModel::planar`, and the camera
//! is parked at `RADIUS * camera_distance_radii` along -X for an orbital
//! marble shot. Both paths share `WorldMeshMaterial` and
//! `assets/shaders/world_mesh.wgsl` — `bevy_terrain` injects the
//! `SPHERICAL` shader_def automatically based on `gpu_tile_atlas.is_spherical`.

use crate::prelude::*;
use crate::scenes::planar::WorldMeshMaterial;
use bevy::math::DVec3;
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const PATH: &str = "terrains/spherical";
pub const TEXTURE_SIZE: u32 = 512;
/// Number of tile slots in the atlas. Spherical preprocess at LOD_COUNT=5
/// needs `6 * (4^5 - 1) / 3 = 2046` slots, so 2048 fits with 2 spare.
///
/// **Hard ceiling**: wgpu / Metal cap 2D texture array layers at 2048
/// (`MTLMaxTextureArrayLayers`) on Apple Silicon, which means `atlas_size`
/// can never exceed 2048 on this backend no matter how much unified memory
/// is available. That in turn caps `LOD_COUNT` at 5 for spherical (LOD 5
/// alone would need 6144 slots). Push past that by either sharding across
/// multiple atlas textures or switching to a 3D-texture layout — neither
/// is implemented upstream.
pub const ATLAS_SIZE: u32 = 2048;
/// Atlas LOD depth. LOD_COUNT=5 at `face_size=8192` gives the deepest LOD
/// (LOD 4 = 8192 px per face dim) a **1:1 mapping** with face TIFF source —
/// no upsampling at the deepest level, so every pixel visible is backed by
/// real z=8 data. Equator display resolution at LOD 4: 1.22 km/px.
pub const DEFAULT_LOD_COUNT: u32 = 5;
pub const DEFAULT_CAMERA_DISTANCE_RADII: f32 = 3.0;

/// WGS84 semi-major axis (equatorial radius) in metres.
pub const MAJOR_AXES: f64 = 6_378_137.0;
/// WGS84 semi-minor axis (polar radius) in metres.
pub const MINOR_AXES: f64 = 6_356_752.314_245;
/// Mean Earth radius. Used to scale the camera distance and the sun's
/// placeholder mesh (matching upstream `spherical.rs`).
pub const RADIUS: f64 = 6_371_000.0;
/// Lowest elevation we encode in the cube-face TIFFs (Mariana Trench bottom
/// is around -10 994 m; -12 000 leaves headroom).
pub const MIN_HEIGHT_M: f32 = -12_000.0;
/// Highest elevation we encode in the cube-face TIFFs (Everest is 8 848 m).
pub const MAX_HEIGHT_M: f32 = 9_000.0;

/// Runtime metadata for the spherical Earth atlas. Written by
/// `bevy_world_mesh::bin::fetch_global_spherical` into
/// `assets/terrains/spherical/globe.toml`, read here at startup.
///
/// If the manifest is missing (e.g. when the synthetic
/// `synthesize_spherical_faces` populated the height TIFFs without imagery),
/// we fall back to the synthetic-defaults `Default` impl below.
#[derive(Debug, Clone, Resource, Serialize, Deserialize)]
pub struct GlobeManifest {
    pub name: String,
    pub zoom: u8,
    pub face_size: u32,
    pub min_height_m: f32,
    pub max_height_m: f32,
    pub lod_count: u32,
    pub camera_distance_radii: f32,
}

impl Default for GlobeManifest {
    fn default() -> Self {
        Self {
            name: "earth".to_string(),
            zoom: 8,
            face_size: 8192,
            min_height_m: MIN_HEIGHT_M,
            max_height_m: MAX_HEIGHT_M,
            lod_count: DEFAULT_LOD_COUNT,
            camera_distance_radii: DEFAULT_CAMERA_DISTANCE_RADII,
        }
    }
}

impl GlobeManifest {
    pub fn try_load() -> Option<Self> {
        let path = Path::new("assets/terrains/spherical/globe.toml");
        let text = std::fs::read_to_string(path).ok()?;
        toml::from_str(&text).ok()
    }

    pub fn load_or_default() -> Self {
        Self::try_load().unwrap_or_default()
    }

    /// Cap the on-disk manifest's `lod_count` at the atlas budget so an old
    /// `globe.toml` that asked for too many LODs doesn't crash the
    /// preprocessor with "Atlas out of indices".
    pub fn clamped_lod_count(&self) -> u32 {
        self.lod_count.min(DEFAULT_LOD_COUNT)
    }
}

/// Tile-atlas configuration. Reads the manifest the runtime renderer uses,
/// so the `preprocess_global` binary builds an atlas with matching scale +
/// height bounds.
pub fn globe_terrain_config() -> TerrainConfig {
    let manifest = GlobeManifest::load_or_default();
    TerrainConfig {
        lod_count: manifest.clamped_lod_count(),
        model: TerrainModel::ellipsoid(
            DVec3::ZERO,
            MAJOR_AXES,
            MINOR_AXES,
            manifest.min_height_m,
            manifest.max_height_m,
        ),
        path: PATH.to_string(),
        atlas_size: ATLAS_SIZE,
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

/// End-to-end scene plugin for the spherical Earth. Mirrors
/// [`super::planar::PlanarScenePlugin`] but builds a WGS84 ellipsoid
/// terrain model, parks the camera at 3 Earth radii for the orbital
/// shot, and wires in the cube-face `globe.toml` manifest.
pub struct GlobeScenePlugin;

impl Plugin for GlobeScenePlugin {
    fn build(&self, app: &mut App) {
        let manifest = GlobeManifest::load_or_default();
        let title = format!(
            "World Mesh — {} (globe, z={}, {}\u{00B2} faces)",
            manifest.name, manifest.zoom, manifest.face_size
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
                TerrainPlugin,
                TerrainMaterialPlugin::<WorldMeshMaterial>::default(),
                TerrainDebugPlugin,
                EnvScreenshotPlugin,
            ))
            .add_systems(Startup, setup);
    }
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<WorldMeshMaterial>>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
    mut meshes: ResMut<Assets<Mesh>>,
    manifest: Res<GlobeManifest>,
) {
    let config = globe_terrain_config();
    let view_config = TerrainViewConfig::default();
    let tile_atlas = TileAtlas::new(&config);
    let tile_tree = TileTree::new(&tile_atlas, &view_config);

    let camera_dist = RADIUS * manifest.camera_distance_radii as f64;

    commands.spawn_big_space(Grid::default(), |root| {
        let frame = root.grid().clone();

        let terrain = root
            .spawn_spatial((
                TerrainBundle::new(tile_atlas, &frame),
                MeshMaterial3d(materials.add(WorldMeshMaterial::default())),
            ))
            .id();

        // big_space 0.12's `BigSpaceCameraController` owns camera movement
        // now (WASD/Space/Ctrl/Q/E/mouse/Shift, auto-registered via
        // `BigSpaceDefaultPlugins`). `with_slowing(true)` enables
        // `slow_near_objects`, which scales the effective speed by the
        // distance to the nearest AABB (the sun sphere, via its Mesh3d).
        // The controller formula is
        //   `effective_speed = nearest_distance * (controller.speed + boost)`,
        // so `with_speed(0.1)` multiplies that by 0.1 to give 10× slower
        // fly-cam movement (roughly 9 Mm/s at orbital view vs the previous
        // 90 Mm/s). Staying well below `with_speed(1e6)` also avoids the
        // numerical drift documented in `context/MIGRATION_NOTES_0.16_AND_BEYOND.md:2026-05-01d`.
        //
        // `Transform::looking_to(Vec3::X, Vec3::Y)` sets forward = +X
        // frame-independently; `looking_at(Vec3::ZERO, ...)` would be
        // wrong because `translation` is the *within-cell* offset, not the
        // absolute position.
        let (cell, translation) = frame.translation_to_grid(-DVec3::X * camera_dist);
        let view = root
            .spawn_spatial((
                Camera3d::default(),
                Projection::Perspective(PerspectiveProjection {
                    near: 0.1,
                    ..default()
                }),
                Transform::from_translation(translation).looking_to(Vec3::X, Vec3::Y),
                cell,
                FloatingOrigin,
                BigSpaceCameraController::default()
                    .with_speed(0.1)
                    .with_slowing(true),
            ))
            .id();

        tile_trees.insert((terrain, view), tile_tree);

        // Sun sphere placeholder (visual-only; lighting is supplied by the
        // bevy_terrain debug plugin's directional light).
        let sun_position = DVec3::new(-1.0, 1.0, -1.0) * RADIUS * 10.0;
        let (sun_cell, sun_translation) = frame.translation_to_grid(sun_position);
        root.spawn_spatial((
            Mesh3d(meshes.add(Sphere::new(RADIUS as f32 * 2.0).mesh().build())),
            Transform::from_translation(sun_translation),
            sun_cell,
        ));
    });
}
