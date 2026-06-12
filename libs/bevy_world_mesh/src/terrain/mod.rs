//! Terrain rendering internals: UDLOD geometry, chunked-clipmap tile atlas,
//! and the compute-shader preprocessing pipeline that builds atlases on disk.
//!
//! This module is the heart of the crate. Downstream users will most often
//! reach for the re-exports in [`crate::prelude`] instead of touching these
//! submodules directly.
//!
//! # Concepts
//!
//! - **`TileTree`** ([`terrain_data::tile_tree::TileTree`]) — a per-view
//!   wrapping quadtree that decides which tiles should be resident based on
//!   the camera's position and the configured LOD distances.
//! - **`TileAtlas`** ([`terrain_data::tile_atlas::TileAtlas`]) — a 2D
//!   texture array that stores the currently resident tile data and serves
//!   paging requests from `TileTree`.
//! - **`TerrainModel`** ([`math::TerrainModel`]) — the shape of the
//!   terrain: `Planar`, `Spherical`, or `Ellipsoidal` (WGS84).
//! - **`Preprocessor`** ([`preprocess::preprocessor::Preprocessor`]) —
//!   runs a GPU compute-shader pipeline (split / stitch / downsample) that
//!   turns source PNG/TIFF data into the atlas pyramid on disk.
//!
//! See the crate-level [`ARCHITECTURE.md`](https://github.com/elodin-sys/world_mesh/blob/main/ARCHITECTURE.md)
//! for a full walkthrough.

// The `ShaderType` derive (from `encase`, re-exported via bevy) generates a
// module-level `check()` fn that references every field of the decorated
// struct for compile-time layout assertions. rustc's `dead_code` lint cannot
// see that use (the fn is never called at runtime), so every GPU-buffer
// struct in this crate otherwise trips ~4-13 false "field never used"
// warnings. We own all of this code and still want `-Dwarnings` to stay
// green, so silence that one lint module-wide here. Any *real* unused-field
// mistakes in non-GPU structs will still show up under clippy's other lints.
#![allow(dead_code)]

pub mod debug;
pub mod formats;
pub mod math;
pub mod plugin;
pub mod preprocess;
pub mod render;
pub mod shaders;
// Historical module name from the `bevy_terrain` fork. The types here
// (`TerrainBundle`, `TerrainConfig`, `TerrainComponents`) are the
// terrain-entity building blocks rather than "terrain inside terrain";
// a rename would ripple through dozens of call sites for little gain.
#[allow(clippy::module_inception)]
pub mod terrain;
pub mod terrain_data;
pub mod terrain_view;
pub mod util;
