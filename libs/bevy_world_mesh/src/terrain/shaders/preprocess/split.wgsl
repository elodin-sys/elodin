#import bevy_terrain::preprocessing::{AtlasTile, atlas, attachment, pixel_coords, pixel_value, process_entry, is_border, inverse_mix}
#import bevy_terrain::functions::{inside_square, tile_count};

struct SplitData {
    tile: AtlasTile,
    top_left: vec2<f32>,
    bottom_right: vec2<f32>,
    tile_index: u32,
}

@group(1) @binding(0)
var<uniform> split_data: SplitData;
@group(1) @binding(1)
var source_tile: texture_2d<f32>;
@group(1) @binding(2)
var source_tile_sampler: sampler;

override fn pixel_value(coords: vec2<u32>) -> vec4<f32> {
    if (is_border(coords)) {
        return vec4<f32>(0.0);
    }

    let tile_coordinate = split_data.tile.coordinate;
    let tile_offset =  vec2<f32>(f32(tile_coordinate.x), f32(tile_coordinate.y));
    let tile_coords = vec2<f32>(coords - vec2<u32>(attachment.border_size)) / f32(attachment.center_size);
    let tile_scale = tile_count(tile_coordinate.lod);

    var source_coords = (tile_offset + tile_coords) / tile_scale;

    source_coords = inverse_mix(split_data.top_left, split_data.bottom_right, source_coords);

    let value = textureSampleLevel(source_tile, source_tile_sampler, source_coords, 0.0);

    let is_valid  = all(textureGather(0u, source_tile, source_tile_sampler, source_coords) != vec4<f32>(0.0));
    let is_inside = inside_square(tile_coords, vec2<f32>(0.0), 1.0) == 1.0;

    if (is_valid && is_inside) {
        return value;
    }
    else {
        return textureLoad(atlas, coords, split_data.tile.atlas_index, 0);
    }
}

// Todo: respect memory coalescing
@compute @workgroup_size(8, 8, 1)
fn split(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    process_entry(vec3<u32>(invocation_id.xy, split_data.tile_index));
}