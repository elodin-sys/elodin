//! End-to-end scene plugins that wire the terrain renderer, data fetchers,
//! cameras, debug overlays, and screenshot harness into a ready-to-fly
//! demo app.
//!
//! - [`planar::PlanarScenePlugin`] — a km-scale real-world region driven by
//!   [`crate::regions`] presets + the [`crate::fetch`] pipeline.
//! - [`globe::GlobeScenePlugin`] — a WGS84-ellipsoid Earth renderer with
//!   cube-face source data.
//!
//! Enabled by the `scenes` feature, which transitively enables `regions`
//! and `high_precision`.

pub mod globe;
pub mod planar;
