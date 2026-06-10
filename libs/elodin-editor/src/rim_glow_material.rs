use bevy::{
    asset::embedded_asset,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

pub struct RimGlowMaterialPlugin;

impl Plugin for RimGlowMaterialPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "rim_glow.wgsl");
        app.add_plugins(MaterialPlugin::<RimGlowMaterial>::default());
    }
}

#[derive(ShaderType, Copy, Clone, Debug)]
pub struct RimGlowParams {
    pub color: Vec4,
    pub strength: f32,
    pub power: f32,
}

impl Default for RimGlowParams {
    fn default() -> Self {
        Self {
            color: Vec4::ONE,
            strength: 0.0,
            power: 3.0,
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct RimGlowExt {
    #[uniform(100)]
    pub params: RimGlowParams,
}

impl MaterialExtension for RimGlowExt {
    fn fragment_shader() -> ShaderRef {
        "embedded://elodin_editor/rim_glow.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }
}

pub type RimGlowMaterial = ExtendedMaterial<StandardMaterial, RimGlowExt>;
