use bevy::{
    prelude::*, reflect::TypePath, render::render_resource::AsBindGroup, shader::ShaderRef,
};

use crate::terrain::shaders::WORLD_MESH_FRAGMENT_SHADER;

/// Default material for world-mesh terrain.
///
/// The material carries no per-material bindings of its own. Height, albedo,
/// and normals come from the terrain renderer's tile-atlas attachments; this
/// shader just routes the albedo attachment into Bevy's PBR terrain fragment
/// path.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct WorldMeshMaterial {}

impl Material for WorldMeshMaterial {
    fn fragment_shader() -> ShaderRef {
        WORLD_MESH_FRAGMENT_SHADER.into()
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }
}
