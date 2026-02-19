#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip
}

// This matches `LowerTriParams` in Rust (`LowerTriTransformExt` uniform at binding 100).
struct LowerTriParams {
    lower_tri: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> params: LowerTriParams;

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    // Apply the lower-triangular transform in LOCAL space.
    let local_pos = (params.lower_tri * vec4<f32>(in.position, 1.0)).xyz;

    // Convert local -> world using Bevy helpers (preserves instancing, etc.)
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let world_pos = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(local_pos, 1.0)
    );

    // Correct normal transformation for shear: n' = inverse-transpose(M3) * n
    let local_normal = normalize(params.normal_matrix * in.normal);
    let world_normal = mesh_functions::mesh_normal_local_to_world(local_normal, in.instance_index);

    // Fill Bevy's expected vertex output.
    var out: VertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);
    out.world_normal = world_normal;
    out.uv = in.uv;

    return out;
}
