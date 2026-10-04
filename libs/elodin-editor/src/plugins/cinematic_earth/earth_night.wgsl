// Per-pixel city-night mask on the globe emissive sheet.
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

struct EarthNightParams {
    to_sun_world: vec4<f32>,
    band: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> earth_night: EarthNightParams;

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

    let s = dot(normalize(in.world_normal), normalize(earth_night.to_sun_world.xyz));
    let start = earth_night.band.x;
    let full = earth_night.band.y;
    let mask = saturate((start - s) / (start - full));
    pbr_input.material.emissive *= mask;

    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color + pbr_input.material.emissive;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    return out;
}
