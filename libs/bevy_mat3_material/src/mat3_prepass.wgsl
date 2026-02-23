// Prepass (depth / normal / shadow) vertex shader: applies the same 3×3 linear transform
// so shadow maps and prepass use the deformed geometry, not the original mesh.
#import bevy_pbr::{
    mesh_functions,
    prepass_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}

struct Mat3Params {
    linear: mat3x3<f32>,
    normal_matrix: mat3x3<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> params: Mat3Params;

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    // Apply the same local-space transform as the forward pass.
    let local_pos = params.linear * in.position;

    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);

    var out: VertexOutput;
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(local_pos, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);

#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif

#ifdef VERTEX_UVS_A
    out.uv = in.uv;
#endif

#ifdef VERTEX_UVS_B
    out.uv_b = in.uv_b;
#endif

#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
#ifdef VERTEX_NORMALS
    let local_normal = params.normal_matrix * in.normal;
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        local_normal,
        in.instance_index,
    );
#endif

#ifdef VERTEX_TANGENTS
    var local_tangent_dir = params.linear * in.tangent.xyz;
#ifdef VERTEX_NORMALS
    local_tangent_dir = local_tangent_dir - local_normal * dot(local_normal, local_tangent_dir);
#endif
    local_tangent_dir = normalize(local_tangent_dir);
    let det_sign = select(-1.0, 1.0, determinant(params.linear) >= 0.0);
    let local_tangent = vec4<f32>(local_tangent_dir, in.tangent.w * det_sign);
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        local_tangent,
        in.instance_index,
    );
#endif
#endif

#ifdef VERTEX_COLORS
    out.color = in.color;
#endif

#ifdef MOTION_VECTOR_PREPASS
    // Static deformation: previous frame position uses same transformed local position.
    let prev_model = mesh_functions::get_previous_world_from_local(in.instance_index);
    out.previous_world_position = mesh_functions::mesh_position_local_to_world(
        prev_model,
        vec4<f32>(local_pos, 1.0),
    );
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = in.instance_index;
#endif

#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        in.instance_index,
        world_from_local[3],
    );
#endif

    return out;
}
