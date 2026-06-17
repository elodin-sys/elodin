// Fragment shader extension: view-dependent rim emissive for GLB glow.
#import bevy_pbr::{
    pbr_types,
    pbr_functions::alpha_discard,
    pbr_fragment::pbr_input_from_standard_material,
}
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
}

struct RimGlowParams {
    color: vec4<f32>,
    strength: f32,
    power: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> rim_glow: RimGlowParams;

@fragment
fn fragment(
    vertex_output: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var in = vertex_output;

#ifdef VISIBILITY_RANGE_DITHER
    pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);
#endif

    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    let n_dot_v = clamp(dot(normalize(pbr_input.N), normalize(pbr_input.V)), 0.0, 1.0);
    let rim = pow(1.0 - n_dot_v, max(rim_glow.power, 0.001)) * rim_glow.strength;
    pbr_input.material.emissive += vec4<f32>(rim_glow.color.rgb * rim, 0.0);

    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color + pbr_input.material.emissive;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    return out;
}
