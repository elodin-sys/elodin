#import bevy_terrain::types::AtlasTile
#import bevy_terrain::attachments::{sample_attachment0 as sample_height, sample_normal, sample_attachment1 as sample_albedo}
#import bevy_terrain::fragment::{FragmentInput, FragmentOutput, fragment_info, fragment_output, fragment_debug}
#import bevy_terrain::functions::lookup_tile
#import bevy_pbr::pbr_types::{PbrInput, pbr_input_new}

@group(3) @binding(0)
var gradient: texture_1d<f32>;
@group(3) @binding(1)
var gradient_sampler: sampler;

fn sample_color(tile: AtlasTile) -> vec4<f32> {
#ifdef ALBEDO
    return sample_albedo(tile);
#else
    let height = sample_height(tile).x;

    return textureSampleLevel(gradient, gradient_sampler, pow(height, 0.9), 0.0);
#endif
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var info = fragment_info(input);

    let tile   = lookup_tile(info.coordinate, info.blend, 0u);
    var color  = sample_color(tile);
    var normal = sample_normal(tile, info.world_normal);

    if (info.blend.ratio > 0.0) {
        let tile2 = lookup_tile(info.coordinate, info.blend, 1u);
        color     = mix(color,  sample_color(tile2),                     info.blend.ratio);
        normal    = mix(normal, sample_normal(tile2, info.world_normal), info.blend.ratio);
    }

    var output: FragmentOutput;
    fragment_output(&info, &output, color, normal);
    fragment_debug(&info, &output, tile, normal);
    return output;
}
