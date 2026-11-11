//! Compatibility for Bevy 0.19's WebGL shaders on the WebGPU backend.
use super::gpu_info::GpuInfo;
use bevy::{
    asset::AssetEventSystems,
    prelude::*,
    shader::{ShaderImport, Source},
};

pub(crate) fn plugin(app: &mut App) {
    // Patch newly loaded shaders before their asset events reach the renderer.
    app.add_systems(PostUpdate, patch_shadow_sampling.before(AssetEventSystems));
}

fn patch_shadow_sampling(
    gpu: Option<Res<GpuInfo>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut shadow_id: Local<Option<AssetId<Shader>>>,
) {
    if !gpu.is_some_and(|gpu| gpu.0.backend == wgpu::Backend::BrowserWebGpu) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::gpu_info::tests::backend;

    fn shadow_shader() -> Shader {
        Shader::from_wgsl(
            "#define_import_path bevy_pbr::shadow_sampling\n\
             fn shadow() { textureSampleCompare(map, sampler, uv, depth); }",
            "shadow_sampling.wgsl",
        )
    }

    #[test]
    fn webgpu_patch_waits_for_adapter_and_preserves_unrelated_shaders() {
        let mut app = App::new();
        app.init_resource::<Assets<Shader>>().add_plugins(plugin);
        let shadow = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(shadow_shader());
        let other = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                "fn unrelated() { textureSampleCompare(map, sampler, uv, depth); }",
                "other.wgsl",
            ));
        app.update();
        assert!(matches!(
            &app.world().resource::<Assets<Shader>>().get(&shadow).unwrap().source,
            Source::Wgsl(source) if source.contains("textureSampleCompare(")
        ));
        // Browser adapters may report Other: the backend, not the device
        // classification, decides whether the shader needs adaptation.
        app.insert_resource(backend(
            wgpu::Backend::BrowserWebGpu,
            wgpu::DeviceType::Other,
        ));
        app.update();
        let shaders = app.world().resource::<Assets<Shader>>();
        assert!(matches!(
            &shaders.get(&shadow).unwrap().source,
            Source::Wgsl(source) if source.contains("textureSampleCompareLevel(")
        ));
        assert!(matches!(
            &shaders.get(&other).unwrap().source,
            Source::Wgsl(source) if source.contains("textureSampleCompare(")
        ));
    }

    #[test]
    fn webgl_and_native_shadows_remain_unchanged() {
        for api in [
            wgpu::Backend::Gl,
            wgpu::Backend::Metal,
            wgpu::Backend::Vulkan,
        ] {
            let mut app = App::new();
            app.init_resource::<Assets<Shader>>()
                .insert_resource(backend(api, wgpu::DeviceType::IntegratedGpu))
                .add_plugins(plugin);
            let shadow = app
                .world_mut()
                .resource_mut::<Assets<Shader>>()
                .add(shadow_shader());
            app.update();
            assert!(matches!(
                &app.world().resource::<Assets<Shader>>().get(&shadow).unwrap().source,
                Source::Wgsl(source) if source.contains("textureSampleCompare(")
            ));
        }
    }
}
