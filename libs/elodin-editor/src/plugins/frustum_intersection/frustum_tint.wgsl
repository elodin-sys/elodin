// Fragment shader extension: tints ellipsoid inside vs outside frustum.
// Replaces base PBR fragment and overrides base_color before lighting.
#import bevy_pbr::{
    pbr_types,
    pbr_functions::alpha_discard,
    pbr_fragment::pbr_input_from_standard_material,
    decal::clustered::apply_decal_base_color,
}
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
}

#ifdef FORWARD_DECAL
#import bevy_pbr::decal::forward::get_forward_decal_info
#endif

struct FrustumTintParams {
    planes: array<vec4<f32>, 6>,
    inside_color: vec4<f32>,
    outside_color: vec4<f32>,
    enabled: u32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> frustum_tint: FrustumTintParams;

fn is_inside_frustum(world_pos: vec3<f32>) -> bool {
    for (var i = 0; i < 6; i++) {
        let plane = frustum_tint.planes[i];
        if dot(plane.xyz, world_pos) + plane.w > 0.0 {
            return false;
        }
    }
    return true;
}

@fragment
fn fragment(
    vertex_output: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var in = vertex_output;

#ifdef VISIBILITY_RANGE_DITHER
    pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);
#endif

#ifdef FORWARD_DECAL
    let forward_decal_info = get_forward_decal_info(in);
    in.world_position = forward_decal_info.world_position;
    in.uv = forward_decal_info.uv;
#endif

    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    pbr_input.material.base_color = apply_decal_base_color(
        in.world_position.xyz,
        in.position.xy,
        pbr_input.material.base_color
    );

    if frustum_tint.enabled != 0u {
        if is_inside_frustum(in.world_position.xyz) {
            pbr_input.material.base_color = frustum_tint.inside_color;
        } else {
            pbr_input.material.base_color = frustum_tint.outside_color;
        }
    }

    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

#ifdef FORWARD_DECAL
    out.color.a = min(forward_decal_info.alpha, out.color.a);
#endif

    return out;
}
