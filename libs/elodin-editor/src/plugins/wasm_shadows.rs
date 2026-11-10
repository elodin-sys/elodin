//! Compatibility for Bevy 0.19's WebGL shaders on the WebGPU backend.
use bevy::{
    asset::AssetEventSystems,
    prelude::*,
    render::renderer::RenderAdapterInfo,
    shader::{ShaderImport, Source},
};

pub(crate) fn plugin(app: &mut App) {
    // Patch newly loaded shaders before their asset events reach the renderer.
    app.add_systems(PostUpdate, patch_shadow_sampling.before(AssetEventSystems));
}

fn patch_shadow_sampling(
    adapter: Res<RenderAdapterInfo>,
    mut shaders: ResMut<Assets<Shader>>,
    mut shadow_id: Local<Option<AssetId<Shader>>>,
) {
    if adapter.backend != wgpu::Backend::BrowserWebGpu {
        return;
    }
    let Some(id) = (*shadow_id).or_else(|| {
        shaders
            .iter()
            .find_map(|(id, shader)| {
                matches!(&shader.import_path, ShaderImport::Custom(path) if path == "bevy_pbr::shadow_sampling")
                    .then_some(id)
            })
    }) else {
        return;
    };
    *shadow_id = Some(id);
    let Some(shader) = shaders.get(id) else {
        *shadow_id = None;
        return;
    };
    let Source::Wgsl(source) = &shader.source else {
        return;
    };
    if !source.contains("textureSampleCompare(") {
        return;
    }
    // The WebGL variants use implicit derivatives in non-uniform control flow.
    // WebGPU rejects that. Shadow maps have one mip, so explicit level-zero
    // comparisons preserve the result and satisfy WebGPU's uniformity rules.
    // Keep WebGL unchanged: its GLSL backend needs the implicit variant.
    let source = source.replace("textureSampleCompare(", "textureSampleCompareLevel(");
    if let Some(mut shader) = shaders.get_mut(id) {
        shader.source = Source::Wgsl(source.into());
    }
}
