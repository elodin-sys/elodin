#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip
}

// This matches `LowerTriParams` in Rust (`LowerTriTransformExt` uniform at binding 100).
struct LowerTriParams {
    lower_tri: mat3x3<f32>,
    normal_matrix: mat3x3<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> params: LowerTriParams;

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    // Apply the lower-triangular transform in LOCAL space.
    let local_pos = params.lower_tri * in.position;

    // Convert local -> world using Bevy helpers (preserves instancing, etc.)
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let world_pos = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(local_pos, 1.0)
    );

    // Fill Bevy's expected vertex output.
    var out: VertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);
#ifdef VERTEX_NORMALS
    // Correct normal transformation for shear: n' = inverse-transpose(M3) * n
    let local_normal = normalize(params.normal_matrix * in.normal);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(local_normal, in.instance_index);
#else
    out.world_normal = vec3<f32>(0.0);
#endif
#ifdef VERTEX_UVS_A
    out.uv = in.uv;
#endif
#ifdef VERTEX_TANGENTS
    // Tangents are directions: transform by M (not inverse-transpose), then re-orthonormalize vs normal when available.
    var local_tangent_dir = params.lower_tri * in.tangent.xyz;
#ifdef VERTEX_NORMALS
    local_tangent_dir = local_tangent_dir - local_normal * dot(local_normal, local_tangent_dir);
#endif
    local_tangent_dir = normalize(local_tangent_dir);

    // If the linear transform flips handedness (negative determinant), flip tangent.w so bitangent matches.
    let det_sign = select(-1.0, 1.0, determinant(params.lower_tri) >= 0.0);
    let local_tangent = vec4<f32>(local_tangent_dir, in.tangent.w * det_sign);
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(world_from_local, local_tangent, in.instance_index);
#endif

    return out;
}
