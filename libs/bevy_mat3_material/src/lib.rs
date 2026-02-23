use bevy::{
    pbr::MaterialExtension,
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
    asset::embedded_asset,
};

pub struct Mat3MaterialPlugin;

impl Plugin for Mat3MaterialPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "mat3_prepass.wgsl");
        embedded_asset!(app, "mat3_transform.wgsl");
        app.add_plugins((
            MaterialPlugin::<Mat3Material>::default(),
        ))
        .register_type::<Mat3ParamsComponent>();
    }
}

/// GPU-side parameters for a 3×3 linear transform.
///
/// - `linear`: linear transform in a Mat3.
/// - `normal_matrix`: inverse-transpose of the top-left 3x3, for correct normal transformation under shear.
#[derive(ShaderType, Copy, Clone, Debug, Reflect)]
pub struct Mat3Params {
    pub linear: Mat3,
    pub normal_matrix: Mat3,
    // Mat3 is 48 bytes in WGSL layout rules, so padding is often required if you extend further.
    // #[cfg(target_arch = "wasm32")]
    // _webgl2_padding: Vec3,
}

impl Default for Mat3Params {
    fn default() -> Self {
        Self {
            linear: Mat3::IDENTITY,
            normal_matrix: Mat3::IDENTITY,
        }
    }
}

/// This computes the correct normal matrix (`inverse().transpose()`).
impl From<Mat3> for Mat3Params {
    fn from(linear: Mat3) -> Mat3Params {
        let normal_matrix = linear.inverse().transpose();
        Mat3Params {
            linear,
            normal_matrix,
        }
    }
}

/// Material extension that overrides the vertex shader and provides the uniforms used there.
///
/// We use binding slot 100 to avoid colliding with StandardMaterial's bindings, following Bevy's example convention.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct Mat3TransformExt {
    #[uniform(100)]
    pub params: Mat3Params,
}

impl MaterialExtension for Mat3TransformExt {
    fn vertex_shader() -> ShaderRef {
        "embedded://bevy_mat3_material/mat3_transform.wgsl".into()
    }

    /// Use the same vertex deformation in the prepass (depth / shadow) pass so shadows match the ellipsoid.
    fn prepass_vertex_shader() -> ShaderRef {
        "embedded://bevy_mat3_material/mat3_prepass.wgsl".into()
    }
}

/// Convenience alias: Standard PBR material + our vertex-shader extension.
pub type Mat3Material = bevy::pbr::ExtendedMaterial<StandardMaterial, Mat3TransformExt>;

/// Component that drives [`Mat3Material`] from the inspector.
///
/// Edit `linear` in the inspector; it is synced to the material each frame and the normal matrix
/// is derived automatically. Attach this to any entity with `MeshMaterial3d<Mat3Material>`
/// that uses a material handle unique to that entity (or shared only with its grid child).
#[derive(Default, Component, Debug, Reflect)]
#[reflect(Component)]
pub struct Mat3ParamsComponent {
    /// The 3×3 linear transform applied in the vertex shader (lower-triangular convention).
    /// The normal matrix is computed as `linear.inverse().transpose()` when syncing to the material.
    pub linear: Mat3,
}
