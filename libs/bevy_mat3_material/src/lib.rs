// #![doc(html_root_url = "https://docs.rs/bevy_mat3_material/0.1.0")]
#![doc = include_str!("../README.md")]
#![forbid(missing_docs)]
use bevy::{
    asset::{embedded_asset, RenderAssetUsages},
    mesh::{Indices, Mesh, PrimitiveTopology},
    pbr::MaterialExtension,
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

/// Plugin for `Mat3Material`.
pub struct Mat3MaterialPlugin;

impl Plugin for Mat3MaterialPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "mat3_prepass.wgsl");
        embedded_asset!(app, "mat3_transform.wgsl");
        app.add_plugins(MaterialPlugin::<Mat3Material>::default())
            .add_systems(Update, sync_mat3_params_from_component)
            .register_type::<Mat3Params>();
    }
}

/// Syncs [`Mat3Params`] into the entity's [`Mat3Material`] so inspector edits apply.
#[allow(clippy::type_complexity)]
fn sync_mat3_params_from_component(
    mut materials: ResMut<Assets<Mat3Material>>,
    query: Query<
        (
            Entity,
            &Mat3Params,
            Option<&MeshMaterial3d<Mat3Material>>,
            Option<&Children>,
        ),
        Changed<Mat3Params>,
    >,
    child_mesh_materials: Query<&MeshMaterial3d<Mat3Material>>,
    children_query: Query<&Children>,
) {
    for (entity, comp, maybe_mesh_material, maybe_children) in &query {
        let params: Mat3Uniforms = comp.linear.into();

        if let Some(mesh_material) = maybe_mesh_material {
            if let Some(material) = materials.get_mut(&mesh_material.0) {
                material.extension.params = params;
            }
        }

        if maybe_children.is_some() {
            for descendant in children_query.iter_descendants(entity) {
                if let Ok(mesh_material) = child_mesh_materials.get(descendant) {
                    if let Some(material) = materials.get_mut(&mesh_material.0) {
                        material.extension.params = params;
                    }
                }
            }
        }
    }
}

/// GPU-side parameters for a 3×3 linear transform.
///
/// - `linear`: linear transform of points in the mesh.
/// - `normal_matrix`: inverse-transpose of the `linear` Mat3, to correctly
///   transform mesh normals.
#[derive(ShaderType, Copy, Clone, Debug, Reflect)]
pub struct Mat3Uniforms {
    /// The linear transformation
    pub linear: Mat3, // 48 bytes, 3 * 16 bytes
    /// Its inverse transpose used to transform normals
    pub normal_matrix: Mat3, // 48 bytes, 3 * 16 bytes
}
// WebGL2/WASM structs must be 16 byte aligned.
// #[cfg(target_arch = "wasm32")]
// _webgl2_padding: Vec3,

impl Default for Mat3Uniforms {
    fn default() -> Self {
        Self {
            linear: Mat3::IDENTITY,
            normal_matrix: Mat3::IDENTITY,
        }
    }
}

/// This computes the correct normal matrix (`inverse().transpose()`).
impl From<Mat3> for Mat3Uniforms {
    fn from(linear: Mat3) -> Mat3Uniforms {
        let normal_matrix = linear.inverse().transpose();
        Mat3Uniforms {
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
    /// The linear transformations
    #[uniform(100)]
    pub params: Mat3Uniforms,
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
/// Edit `linear` in the inspector; it is synced to the material when changed and the normal matrix
/// is derived automatically. Attach this to any entity with `MeshMaterial3d<Mat3Material>`
/// that uses a material handle unique to that entity (or shared only with its grid child).
#[derive(Default, Component, Debug, Reflect)]
#[reflect(Component)]
pub struct Mat3Params {
    /// The 3×3 linear transform applied in the vertex shader (lower-triangular convention).
    /// The normal matrix is computed as `linear.inverse().transpose()` when syncing to the material.
    pub linear: Mat3,
}

/// Builds a line-list mesh for the UV sphere grid (quad-like edges). Uses the same vertex layout
/// as Bevy's `SphereMeshBuilder::uv`. Pass `Mat4::IDENTITY` for a unit sphere grid, or a deform
/// matrix to match a deformed ellipsoid.
/// Builds a line-list mesh for the UV sphere grid in **unit-sphere** space (radius, no deform).
/// Deformation is applied at runtime by using a material with a vertex shader (e.g. `Mat3Material`).
pub fn uv_sphere_grid_line_mesh(radius: f32, sectors: u32, stacks: u32) -> Mesh {
    use std::f32::consts::PI;

    let sector_step = 2.0 * PI / sectors as f32;
    let stack_step = PI / stacks as f32;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(((stacks + 1) * (sectors + 1)) as usize);

    for i in 0..=stacks {
        let stack_angle = PI / 2.0 - (i as f32) * stack_step;
        let xy = radius * stack_angle.cos();
        let z = radius * stack_angle.sin();
        for j in 0..=sectors {
            let sector_angle = (j as f32) * sector_step;
            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();
            positions.push([x, y, z]);
        }
    }

    let mut line_indices: Vec<u32> = Vec::new();
    // Ring edges (constant stack i): (i*(sectors+1)+j) -> (i*(sectors+1)+j+1)
    for i in 0..=stacks {
        for j in 0..sectors {
            let a = i * (sectors + 1) + j;
            let b = a + 1;
            line_indices.extend_from_slice(&[a, b]);
        }
    }
    // Meridian edges (constant sector j): (i*(sectors+1)+j) -> ((i+1)*(sectors+1)+j)
    for j in 0..=sectors {
        for i in 0..stacks {
            let a = i * (sectors + 1) + j;
            let b = a + (sectors + 1);
            line_indices.extend_from_slice(&[a, b]);
        }
    }

    Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_indices(Indices::U32(line_indices))
}
