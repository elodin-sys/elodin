//! Bevy world-mesh (first-pass vendored integration).
//!
//! This is a minimal, in-tree placeholder for the original `elodin-sys/world_mesh`
//! terrain renderer. The initial goal is to provide a small API surface that the
//! editor can link against and spawn a basic terrain backdrop.

#![forbid(unsafe_code)]

use bevy::prelude::*;

pub mod prelude;
pub mod scenes;
pub mod terrain;

/// Core plugin for world-mesh terrain rendering.
///
/// In this first-pass integration this is intentionally lightweight; it exists
/// so downstream crates can unconditionally add the plugin and we can iterate
/// towards the full renderer.
#[derive(Default, Debug, Clone, Copy)]
pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<terrain::TerrainRoot>();
    }
}

/// Registers the render pipeline / asset types for a terrain material `M`.
///
/// This mirrors the upstream `world_mesh` public surface. For now it is a thin
/// wrapper over Bevy's [`bevy::pbr::MaterialPlugin`].
#[derive(Debug, Clone, Copy)]
pub struct TerrainMaterialPlugin<M: bevy::pbr::Material>(std::marker::PhantomData<M>);

impl<M: bevy::pbr::Material> Default for TerrainMaterialPlugin<M> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<M> Plugin for TerrainMaterialPlugin<M>
where
    M: bevy::pbr::Material,
    M::Data: PartialEq + Eq + std::hash::Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.add_plugins(bevy::pbr::MaterialPlugin::<M>::default());
    }
}
