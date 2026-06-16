#define_import_path bevy_terrain::preprocessing

const FORMAT_R8: u32 = 2u;
const FORMAT_RGBA8: u32 = 0u;
const FORMAT_R16: u32 = 1u;

const INVALID_ATLAS_INDEX: u32 = 4294967295u;

struct TileCoordinate {
    side: u32,
    lod: u32,
    x: u32,
    y: u32,
}

struct AtlasTile {
    coordinate: TileCoordinate,
    atlas_index: u32,
    _padding_a: u32,
    _padding_b: u32,
    _padding_c: u32,
}

struct AttachmentMeta {
    format_id: u32,
    lod_count: u32,
    texture_size: u32,
    border_size: u32,
    center_size: u32,
    pixels_per_entry: u32,
    entries_per_side: u32,
    entries_per_tile: u32,
}

@group(0) @binding(0)
var<storage, read_write> atlas_write_section: array<u32>;
@group(0) @binding(1)
var atlas: texture_2d_array<f32>;
@group(0) @binding(2)
var atlas_sampler: sampler;
@group(0) @binding(3)
var<uniform> attachment: AttachmentMeta;

fn inverse_mix(lower: vec2<f32>, upper: vec2<f32>, value: vec2<f32>) -> vec2<f32> {
    return (value - lower) / (upper - lower);
}

fn inside(coords: vec2<u32>, bounds: vec4<u32>) -> bool {
    return coords.x >= bounds.x &&
           coords.x <  bounds.x + bounds.z &&
           coords.y >= bounds.y &&
           coords.y <  bounds.y + bounds.w;
}

fn is_border(coords: vec2<u32>) -> bool {
    return !inside(coords, vec4<u32>(attachment.border_size, attachment.border_size, attachment.center_size, attachment.center_size));
}

fn pixel_coords(entry_coords: vec3<u32>, pixel_offset: u32) -> vec2<u32> {
    return vec2<u32>(entry_coords.x * attachment.pixels_per_entry + pixel_offset, entry_coords.y);
}

virtual fn pixel_value(coords: vec2<u32>) -> vec4<f32> { return vec4<f32>(0.0); }

fn store_entry(entry_coords: vec3<u32>, entry_value: u32) {
    let entry_index = entry_coords.z * attachment.entries_per_tile +
                      entry_coords.y * attachment.entries_per_side +
                      entry_coords.x;

    atlas_write_section[entry_index] = entry_value;
}

fn process_entry(entry_coords: vec3<u32>) {
    if (attachment.format_id == FORMAT_R8) {
        let entry_value = pack4x8unorm(vec4<f32>(pixel_value(pixel_coords(entry_coords, 0u)).x,
                                                 pixel_value(pixel_coords(entry_coords, 1u)).x,
                                                 pixel_value(pixel_coords(entry_coords, 2u)).x,
                                                 pixel_value(pixel_coords(entry_coords, 3u)).x));
        store_entry(entry_coords, entry_value);
    }
    if (attachment.format_id == FORMAT_RGBA8) {
        let entry_value = pack4x8unorm(pixel_value(pixel_coords(entry_coords, 0u)));
        store_entry(entry_coords, entry_value);
    }
    if (attachment.format_id == FORMAT_R16) {
        let entry_value = pack2x16unorm(vec2<f32>(pixel_value(pixel_coords(entry_coords, 0u)).x,
                                              pixel_value(pixel_coords(entry_coords, 1u)).x));
        store_entry(entry_coords, entry_value);
    }
}
