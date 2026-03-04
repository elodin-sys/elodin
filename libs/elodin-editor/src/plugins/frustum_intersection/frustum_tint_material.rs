//! Per-fragment frustum tint: ellipsoid inside vs outside frustum get different colors.

use bevy::{
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

/// GPU uniforms for frustum tint.
#[derive(ShaderType, Clone, Debug)]
pub struct FrustumTintParams {
    pub planes: [Vec4; 6],
    pub inside_color: Vec4,
    pub outside_color: Vec4,
    pub enabled: u32,
}

/// Extension uniforms for frustum-based tinting.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct FrustumTintExt {
    #[uniform(100)]
    pub params: FrustumTintParams,
}

impl Default for FrustumTintExt {
    fn default() -> Self {
        Self {
            params: FrustumTintParams {
                planes: [Vec4::ZERO; 6],
                inside_color: Vec4::new(0.18, 0.92, 0.34, 0.5),
                outside_color: Vec4::new(1.0, 0.58, 0.12, 0.5),
                enabled: 0,
            },
        }
    }
}

impl MaterialExtension for FrustumTintExt {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            "embedded://elodin_editor/plugins/frustum_intersection/frustum_tint.wgsl".into(),
        )
    }

    /// Forward rendering only; deferred pipeline falls back to base material.
    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }
}

/// Material that tints ellipsoid fragments inside vs outside a frustum.
pub type FrustumTintMaterial = ExtendedMaterial<StandardMaterial, FrustumTintExt>;
