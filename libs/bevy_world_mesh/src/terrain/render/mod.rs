//! This module contains the implementation of the Uniform Distance-Dependent Level of Detail (UDLOD).
//!
//! This algorithm is responsible for approximating the terrain geometry.
//! Therefore tiny mesh tiles are refined in a tile_tree-like manner in a compute shader prepass for
//! each view. Then they are drawn using a single draw indirect call and morphed together to form
//! one continuous surface.

pub mod culling_bind_group;
pub mod terrain_bind_group;
pub mod terrain_material;
pub mod terrain_view_bind_group;
pub mod tiling_prepass;
pub mod world_mesh_material;

use bevy::render::{
    render_resource::{BindGroupLayout, BindGroupLayoutDescriptor},
    renderer::RenderDevice,
};

/// Materialize a concrete [`BindGroupLayout`] from a [`BindGroupLayoutDescriptor`].
///
/// Bevy 0.18 (PR#21205) reshaped `RenderPipelineDescriptor` / `ComputePipelineDescriptor`
/// to hold `Vec<BindGroupLayoutDescriptor>` instead of `Vec<BindGroupLayout>`; the
/// `PipelineCache` materializes the concrete layouts lazily on first pipeline use.
/// Our own `create_bind_group` call sites still need a concrete [`BindGroupLayout`]
/// to pass to `device.create_bind_group(...)`, so this helper bridges the two:
/// it's called in each bind-group-creation path to instantiate a layout from the
/// same descriptor the pipeline stores. A `PipelineCache`-backed alternative
/// (`pipeline_cache.get_bind_group_layout(&desc)`) exists but would force us to
/// plumb `Res<PipelineCache>` through every prepare system -- this is simpler
/// and incurs the same one-layout-per-bind-group cost we had in 0.17.
pub(crate) fn instantiate_layout(
    device: &RenderDevice,
    desc: &BindGroupLayoutDescriptor,
) -> BindGroupLayout {
    device.create_bind_group_layout(Some(desc.label.as_ref()), &desc.entries)
}
