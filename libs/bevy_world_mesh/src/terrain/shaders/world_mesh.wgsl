// World Mesh fragment shader — the default surface render for bevy_world_mesh.
// Pulls colour out of attachment 1 (albedo) instead
// of sampling a 1D gradient LUT, then routes through bevy_terrain's
// fragment_output (PBR lighting) and fragment_debug (debug-overlay support).
//
// Imports the same shader-module surface upstream's planar.wgsl uses, so this
// stays compatible with bevy_terrain's `LIGHTING` / `MORPH` / `BLEND` /
// `SAMPLE_GRAD` shader_defs.

#import bevy_terrain::types::AtlasTile
#import bevy_terrain::attachments::{
    sample_attachment0 as sample_height_attachment,
    sample_attachment1 as sample_albedo,
    sample_normal,
}
#import bevy_terrain::fragment::{
    FragmentInput, FragmentOutput,
    fragment_info, fragment_output, fragment_debug,
}
#import bevy_terrain::functions::lookup_tile

fn sample_color(tile: AtlasTile) -> vec4<f32> {
    return sample_albedo(tile);
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var info = fragment_info(input);

    let tile = lookup_tile(info.coordinate, info.blend, 0u);
    var color = sample_color(tile);
    var normal = sample_normal(tile, info.world_normal);

    if (info.blend.ratio > 0.0) {
        let tile2 = lookup_tile(info.coordinate, info.blend, 1u);
        color = mix(color, sample_color(tile2), info.blend.ratio);
        normal = mix(normal, sample_normal(tile2, info.world_normal), info.blend.ratio);
    }

    var output: FragmentOutput;
    fragment_output(&info, &output, color, normal);
    fragment_debug(&info, &output, tile, normal);
    return output;
}
