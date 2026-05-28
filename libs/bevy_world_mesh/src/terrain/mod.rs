//! Minimal terrain types.

use bevy::prelude::*;

#[cfg(not(target_family = "wasm"))]
pub mod atlas;

/// Marker component for the root entity of a spawned terrain.
#[derive(Component, Reflect, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[reflect(Component)]
pub struct TerrainRoot;

/// High-level configuration for a terrain instance.
///
/// The full upstream renderer supports region-based atlases and multiple terrain
/// models. For the first-pass vendoring we only keep the fields the editor needs
/// to spawn a simple planar mesh.
#[derive(Debug, Clone)]
pub struct TerrainConfig {
    /// User-facing region identifier (e.g. "death_valley").
    pub region: String,
    /// Width of the planar backdrop in meters.
    pub width_m: f32,
    /// Depth of the planar backdrop in meters.
    pub depth_m: f32,
}

impl TerrainConfig {
    pub fn new(region: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            width_m: 2000.0,
            depth_m: 2000.0,
        }
    }
}
