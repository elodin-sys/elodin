//! # World Mesh
//!
//! A large-scale real-world terrain renderer for the [Bevy](https://bevy.org)
//! game engine. Renders planar regions (km-scale patches of real terrain) and
//! full spherical worlds (Earth-scale, WGS84 ellipsoid) from the same
//! pipeline, built on Uniform Distance-Dependent Level of Detail (UDLOD)
//! geometry + a Chunked Clipmap tile atlas. Incorporates a modernised fork
//! of [`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain)
//! — see [`NOTICE`](https://github.com/elodin-sys/world_mesh/blob/main/NOTICE)
//! for attribution.
//!
//! # Feature flags
//!
//! - `terrain` (default) — core renderer: `WorldMeshPlugin`, `TerrainPlugin`,
//!   `TerrainBundle`, `TerrainConfig`, `TerrainMaterialPlugin<M>`, `TileAtlas`, `TileTree`,
//!   `TerrainModel`, `Preprocessor`, `TerrainPreprocessPlugin`. Minimal
//!   dependency set (no HTTP, no TIFF, no toml/serde).
//! - `high_precision` — enables the [`big_space`] integration (double-
//!   precision camera + `FloatingOrigin` + `CellCoord`) required for
//!   Earth-radius scenes.
//! - `fetch` — HTTP slippy-tile fetcher ([`crate::fetch::TileFetcher`])
//!   with on-disk cache, plus the decoders for AWS Terrain Tiles
//!   (terrarium) and EOX Sentinel-2 Cloudless. Pulls in `ureq`.
//! - `regions` — region-preset catalogue ([`crate::regions::Region`],
//!   `BRIENZ`, `DEATH_VALLEY`, `lookup`). Pulls in `toml`, `serde`.
//! - `scenes` — end-to-end scene scaffolding: [`scenes::planar::PlanarScenePlugin`]
//!   and [`scenes::globe::GlobeScenePlugin`]. Depends on `regions` +
//!   `high_precision`.
//! - `synth` — procedural FBM noise generators used by the
//!   `synthesize_height` / `synthesize_spherical_faces` binaries; pulls in
//!   the `noise` crate. Not needed at runtime.
//!
//! # Minimal usage
//!
//! ```toml
//! # Cargo.toml
//! [dependencies]
//! bevy_world_mesh = { version = "0.1", default-features = false, features = ["terrain", "high_precision"] }
//! ```
//!
//! ```ignore
//! use bevy::prelude::*;
//! use bevy_world_mesh::prelude::*;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins((DefaultPlugins, WorldMeshPlugin))
//!         .run();
//! }
//! ```
//!
//! For a ready-to-fly demo, enable `scenes` + `regions` + `fetch` + `debug`
//! and use [`scenes::planar::PlanarScenePlugin`] /
//! [`scenes::globe::GlobeScenePlugin`] directly.

pub mod terrain;

#[cfg(feature = "fetch")]
pub mod fetch;

#[cfg(feature = "regions")]
pub mod regions;

#[cfg(feature = "scenes")]
pub mod scenes;

/// Common re-exports. `use bevy_world_mesh::prelude::*;` gets you the renderer
/// plugin + bundles, the material traits + `TerrainMaterialPlugin<M>`, and
/// (under the relevant features) the `big_space` integration and the scene
/// plugins.
pub mod prelude {
    #[cfg(feature = "high_precision")]
    pub use big_space::prelude::{
        BigSpaceCameraController, BigSpaceCommands, FloatingOrigin, Grid,
    };

    pub use crate::terrain::{
        debug::{
            screenshot::EnvScreenshotPlugin, DebugTerrainMaterial, LoadingImages,
            TerrainDebugPlugin,
        },
        math::TerrainModel,
        plugin::{TerrainPlugin, WorldMeshPlugin},
        preprocess::{
            preprocessor::{PreprocessDataset, Preprocessor, SphericalDataset},
            TerrainPreprocessPlugin,
        },
        render::{terrain_material::TerrainMaterialPlugin, world_mesh_material::WorldMeshMaterial},
        terrain::{TerrainBundle, TerrainConfig},
        terrain_data::{
            tile_atlas::TileAtlas,
            tile_tree::{TerrainViewPosition, TileTree},
            AttachmentConfig, AttachmentFormat,
        },
        terrain_view::{TerrainViewComponents, TerrainViewConfig},
    };

    #[cfg(feature = "scenes")]
    pub use crate::scenes::{
        globe::{GlobeManifest, GlobeScenePlugin},
        planar::PlanarScenePlugin,
    };

    #[cfg(feature = "regions")]
    pub use crate::regions::RegionManifest;
}
