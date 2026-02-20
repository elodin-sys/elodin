use bevy::{
    pbr::MaterialExtension,
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

/// GPU-side parameters for the lower-triangular transform.
///
/// - `lower_tri`: linear transform in a Mat3 (the lower-triangular 3x3).
/// - `normal_matrix`: inverse-transpose of the top-left 3x3, for correct normal transformation under shear.
#[derive(ShaderType, Copy, Clone, Debug, Reflect)]
pub struct LowerTriParams {
    pub lower_tri: Mat3,
    pub normal_matrix: Mat3,
    // WebGL2 (and some std140-like layouts) can be picky; keeping this struct aligned is helpful.
    // Mat3 is 48 bytes in WGSL layout rules, so padding is often required if you extend further.
}

impl Default for LowerTriParams {
    fn default() -> Self {
        Self {
            lower_tri: Mat3::IDENTITY,
            normal_matrix: Mat3::IDENTITY,
        }
    }
}

/// Material extension that overrides the vertex shader and provides the uniforms used there.
///
/// We use binding slot 100 to avoid colliding with StandardMaterial's bindings, following Bevy's example convention. citeturn1view1
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct LowerTriTransformExt {
    #[uniform(100)]
    pub params: LowerTriParams,
}

impl MaterialExtension for LowerTriTransformExt {
    fn vertex_shader() -> ShaderRef {
        "shaders/lower_tri_transform.wgsl".into()
    }

    /// Use the same vertex deformation in the prepass (depth / shadow) pass so shadows match the ellipsoid.
    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/lower_tri_prepass.wgsl".into()
    }
}

/// Convenience alias: Standard PBR material + our vertex-shader extension.
pub type LowerTriMaterial = bevy::pbr::ExtendedMaterial<StandardMaterial, LowerTriTransformExt>;

/// Helper to build a `LowerTriParams` from a 3x3 linear transform.
///
/// This computes the correct normal matrix (`inverse().transpose()`).
pub fn params_from_linear(linear: Mat3) -> LowerTriParams {
    let lower_tri = linear;
    let normal_matrix = linear.inverse().transpose();
    LowerTriParams {
        lower_tri,
        normal_matrix,
    }
}
