#define_import_path bevy_terrain::debug

#import bevy_terrain::types::{Coordinate, AtlasTile, Blend}
#import bevy_terrain::bindings::{config, tile_tree, view_config, geometry_tiles, attachments, origins, terrain_model_approximation}
#import bevy_terrain::functions::{inverse_mix, compute_coordinate, lookup_best, approximate_view_distance, compute_blend, tree_lod, inside_square, tile_coordinate, coordinate_from_local_position, compute_subdivision_coordinate}
#import bevy_pbr::mesh_view_bindings::view

fn index_color(index: u32) -> vec4<f32> {
    var COLOR_ARRAY = array(
        vec4(1.0, 0.0, 0.0, 1.0),
        vec4(0.0, 1.0, 0.0, 1.0),
        vec4(0.0, 0.0, 1.0, 1.0),
        vec4(1.0, 1.0, 0.0, 1.0),
        vec4(1.0, 0.0, 1.0, 1.0),
        vec4(0.0, 1.0, 1.0, 1.0),
    );

    return mix(COLOR_ARRAY[index % 6u], vec4<f32>(0.6), 0.2);
}

fn tile_tree_outlines(uv: vec2<f32>) -> f32 {
    let thickness = 0.015;

    return 1.0 - inside_square(uv, vec2<f32>(thickness), 1.0 - 2.0 * thickness);
}

fn checker_color(coordinate: Coordinate, ratio: f32) -> vec4<f32> {
    var color        = index_color(coordinate.lod);
    var parent_color = index_color(coordinate.lod - 1);
    color            = select(color,        mix(color,        vec4(0.0), 0.5), (coordinate.xy.x + coordinate.xy.y) % 2u == 0u);
    parent_color     = select(parent_color, mix(parent_color, vec4(0.0), 0.5), ((coordinate.xy.x >> 1) + (coordinate.xy.y >> 1)) % 2u == 0u);

    return mix(color, parent_color, ratio);
}

fn show_data_lod(blend: Blend, tile: AtlasTile) -> vec4<f32> {
#ifdef TILE_TREE_LOD
    let ratio = 0.0;
#else
    let ratio = select(0.0, blend.ratio, blend.lod == tile.coordinate.lod);
#endif

    var color = checker_color(tile.coordinate, ratio);

    if (ratio > 0.95 && blend.lod == tile.coordinate.lod) {
        color = mix(color, vec4<f32>(0.0), 0.8);
    }

#ifdef SPHERICAL
    color = mix(color, index_color(tile.coordinate.side), 0.3);
#endif

    return color;
}

fn show_geometry_lod(coordinate: Coordinate) -> vec4<f32> {
    let view_distance  = approximate_view_distance(coordinate, view.world_position);
    let target_lod     = log2(2.0 * view_config.morph_distance / view_distance);

#ifdef MORPH
    let ratio = select(inverse_mix(f32(coordinate.lod) + view_config.morph_range, f32(coordinate.lod), target_lod), 0.0, coordinate.lod == 0);
#else
    let ratio = 0.0;
#endif

    var color = checker_color(coordinate, ratio);

    if (distance(coordinate.uv, compute_subdivision_coordinate(coordinate).uv) < 0.1) {
        color = mix(index_color(coordinate.lod + 1), vec4(0.0), 0.7);
    }

    if (fract(target_lod) < 0.01 && target_lod >= 1.0) {
        color = mix(color, vec4<f32>(0.0), 0.8);
    }

#ifdef SPHERICAL
    color = mix(color, index_color(coordinate.side), 0.3);
#endif

    if (max(0.0, target_lod) < f32(coordinate.lod) - 1.0 + view_config.morph_range) {
        // The view_distance and morph range are not sufficient.
        // The same tile overlapps two morph zones.
        // -> increase morph distance
        color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    if (floor(target_lod) > f32(coordinate.lod)) {
        // The view_distance and morph range are not sufficient.
        // The tile does have an insuffient LOD.
        // -> increase morph tolerance
        color = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    }

    return color;
}
fn show_tile_tree(coordinate: Coordinate) -> vec4<f32> {
    let view_distance  = approximate_view_distance(coordinate, view.world_position);
    let target_lod     = log2(view_config.load_distance / view_distance);

    let best_lookup = lookup_best(coordinate);

    var color = checker_color(best_lookup.tile.coordinate, 0.0);
    color     = mix(color, vec4<f32>(0.1), tile_tree_outlines(best_lookup.tile_tree_uv));

    if (fract(target_lod) < 0.01 && target_lod >= 1.0) {
        color = mix(index_color(u32(target_lod)), vec4<f32>(0.0), 0.8);
    }

    return color;
}

fn show_pixels(tile: AtlasTile) -> vec4<f32> {
    let pixel_size = 4.0;
    let pixel_coordinate = tile.coordinate.uv * f32(attachments[0].size) / pixel_size;

    let is_even = (u32(pixel_coordinate.x) + u32(pixel_coordinate.y)) % 2u == 0u;

    if (is_even) { return vec4<f32>(0.5, 0.5, 0.5, 1.0); }
    else {         return vec4<f32>(0.1, 0.1, 0.1, 1.0); }
}
