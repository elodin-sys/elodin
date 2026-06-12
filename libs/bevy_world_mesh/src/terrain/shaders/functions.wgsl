#define_import_path bevy_terrain::functions

#import bevy_terrain::bindings::{mesh, config, origins, view_config, geometry_tiles, tile_tree, terrain_model_approximation}
#import bevy_terrain::types::{TileCoordinate, TileTree, TileTreeEntry, AtlasTile, Blend, BestLookup, Coordinate, Morph}
#import bevy_pbr::mesh_view_bindings::view
#import bevy_render::maths::{affine3_to_square, mat2x4_f32_to_mat3x3_unpack}

const F0 = 0u;
const F1 = 1u;
const PS = 2u;
const PT = 3u;
const C_SQR = 0.87 * 0.87;

fn normal_local_to_world(local_position: vec3<f32>) -> vec3<f32> {
#ifdef SPHERICAL
    let local_normal = local_position;
#else
    let local_normal = vec3<f32>(0.0, 1.0, 0.0);
#endif

    let world_from_local = mat2x4_f32_to_mat3x3_unpack(mesh[0].local_from_world_transpose_a,
                                                       mesh[0].local_from_world_transpose_b);
    return normalize(world_from_local * local_normal);
}

fn position_local_to_world(local_position: vec3<f32>) -> vec3<f32> {
    let world_from_local = affine3_to_square(mesh[0].world_from_local);
    return (world_from_local * vec4<f32>(local_position, 1.0)).xyz;
}

fn inverse_mix(a: f32, b: f32, value: f32) -> f32 {
    return saturate((value - a) / (b - a));
}

fn compute_morph(coordinate: Coordinate, view_distance: f32) -> Coordinate {
#ifdef MORPH
    // Morphing more than one layer at once is not possible, since the approximate view distance for vertices that
    // should be placed on the same position will be slightly different, so the target lod and thus the ratio will be
    // slightly off as well, which results in a pop.
    let even_uv = vec2<f32>(vec2<u32>(coordinate.uv * view_config.grid_size) & vec2<u32>(~1u)) / view_config.grid_size;

    let target_lod  = log2(2.0 * view_config.morph_distance / view_distance);
    let ratio       = select(inverse_mix(f32(coordinate.lod) + view_config.morph_range, f32(coordinate.lod), target_lod), 0.0, coordinate.lod == 0);

    return Coordinate(coordinate.side, coordinate.lod, coordinate.xy, mix(coordinate.uv, even_uv, ratio));
#else
    return coordinate;
#endif
}

fn compute_blend(view_distance: f32) -> Blend {
    let target_lod = min(log2(view_config.blend_distance / view_distance), f32(config.lod_count) - 0.00001);
    let lod        = u32(target_lod);

#ifdef BLEND
    let ratio = select(inverse_mix(f32(lod) + view_config.blend_range, f32(lod), target_lod), 0.0, lod == 0u);

    return Blend(lod, ratio);
#else
    return Blend(lod, 0.0);
#endif
}

fn compute_tile_uv(vertex_index: u32) -> vec2<f32>{
    // use first and last indices of the rows twice, to form degenerate triangles
    let grid_index   = vertex_index % view_config.vertices_per_tile;
    let row_index    = clamp(grid_index % view_config.vertices_per_row, 1u, view_config.vertices_per_row - 2u) - 1u;
    let column_index = grid_index / view_config.vertices_per_row;

    return vec2<f32>(f32(column_index + (row_index & 1u)), f32(row_index >> 1u)) / view_config.grid_size;
}

fn compute_local_position(coordinate: Coordinate) -> vec3<f32> {
    var uv = (vec2<f32>(coordinate.xy) + coordinate.uv) / tile_count(coordinate.lod);

#ifdef SPHERICAL
    uv = (uv - 0.5) / 0.5;
    uv = uv / sqrt(1.0 + C_SQR - C_SQR * uv * uv);

    var local_position: vec3<f32>;

    switch (coordinate.side) {
        case 0u:      { local_position = vec3( -1.0, -uv.y,  uv.x); }
        case 1u:      { local_position = vec3( uv.x, -uv.y,   1.0); }
        case 2u:      { local_position = vec3( uv.x,   1.0,  uv.y); }
        case 3u:      { local_position = vec3(  1.0, -uv.x,  uv.y); }
        case 4u:      { local_position = vec3( uv.y, -uv.x,  -1.0); }
        case 5u:      { local_position = vec3( uv.y,  -1.0,  uv.x); }
        case default: {}
    }

    return normalize(local_position);
#else
    return vec3<f32>(uv.x - 0.5, 0.0, uv.y - 0.5);
#endif
}

fn compute_relative_position(coord: Coordinate) -> vec3<f32> {
    var coordinate = coord;
    coordinate_change_lod(&coordinate, terrain_model_approximation.origin_lod);

    let params = terrain_model_approximation.sides[coordinate.side];
    let relative_st = (vec2<f32>(vec2<i32>(coordinate.xy) - params.view_xy) + coordinate.uv - params.view_uv) / tile_count(terrain_model_approximation.origin_lod);

    let s = relative_st.x;
    let t = relative_st.y;
    let c = params.c;
    let c_s = params.c_s;
    let c_t = params.c_t;
    let c_ss = params.c_ss;
    let c_st = params.c_st;
    let c_tt = params.c_tt;

    return c + c_s * s + c_t * t + c_ss * s * s + c_st * s * t + c_tt * t * t;
}

fn approximate_view_distance(coordinate: Coordinate, view_world_position: vec3<f32>) -> f32 {
    let local_position = compute_local_position(coordinate);
    var world_position = position_local_to_world(local_position);
    let world_normal   = normal_local_to_world(local_position);
    var view_distance  = distance(world_position + terrain_model_approximation.approximate_height * world_normal, view_world_position);

#ifdef HIGH_PRECISION
    if (view_distance < view_config.precision_threshold_distance) {
        let relative_position = compute_relative_position(coordinate);
        view_distance         = length(relative_position + terrain_model_approximation.approximate_height * world_normal);
    }
#endif

    return view_distance;
}

fn compute_subdivision_coordinate(coordinate: Coordinate) -> Coordinate {
    let params  = terrain_model_approximation.sides[coordinate.side];

#ifdef FRAGMENT
    var view_coordinate = Coordinate(coordinate.side, terrain_model_approximation.origin_lod, vec2<u32>(params.view_xy), params.view_uv, vec2<f32>(0.0), vec2<f32>(0.0));
#else
    var view_coordinate = Coordinate(coordinate.side, terrain_model_approximation.origin_lod, vec2<u32>(params.view_xy), params.view_uv);
#endif

    coordinate_change_lod(&view_coordinate, coordinate.lod);
    var offset = vec2<i32>(view_coordinate.xy) - vec2<i32>(coordinate.xy);
    var uv = view_coordinate.uv;

    if      (offset.x < 0) { uv.x = 0.0; }
    else if (offset.x > 0) { uv.x = 1.0; }
    if      (offset.y < 0) { uv.y = 0.0; }
    else if (offset.y > 0) { uv.y = 1.0; }

    var subdivision_coordinate = coordinate;
    subdivision_coordinate.uv = uv;
    return subdivision_coordinate;
}

fn tile_count(lod: u32) -> f32 { return f32(1u << lod); }

fn inside_square(position: vec2<f32>, origin: vec2<f32>, size: f32) -> f32 {
    let inside = step(origin, position) * step(position, origin + size);

    return inside.x * inside.y;
}

fn coordinate_change_lod(coordinate: ptr<function, Coordinate>, new_lod: u32) {
    let lod_difference = i32(new_lod) - i32((*coordinate).lod);

    if (lod_difference == 0) { return; }

    let delta_count = 1u << u32(abs(lod_difference));
    let delta_size  = pow(2.0, f32(lod_difference));

    (*coordinate).lod = new_lod;

    if (lod_difference > 0) {
        let scaled_uv    = (*coordinate).uv * delta_size;
        (*coordinate).xy = (*coordinate).xy * delta_count + vec2<u32>(scaled_uv);
        (*coordinate).uv = scaled_uv % 1.0;
    } else {
        let xy = (*coordinate).xy;
        (*coordinate).xy = xy / delta_count;
        (*coordinate).uv = (vec2<f32>(xy % delta_count) + (*coordinate).uv) * delta_size;
    }

#ifdef FRAGMENT
    (*coordinate).uv_dx *= delta_size;
    (*coordinate).uv_dy *= delta_size;
#endif
}

fn compute_tile_tree_uv(coordinate: Coordinate) -> vec2<f32> {
    let origin_xy = vec2<i32>(origins[coordinate.side * config.lod_count + coordinate.lod]);
    let tree_size = min(f32(view_config.tree_size), tile_count(coordinate.lod));

    return (vec2<f32>(vec2<i32>(coordinate.xy) - origin_xy) + coordinate.uv) / tree_size;
}


fn lookup_tile_tree_entry(coordinate: Coordinate) -> TileTreeEntry {
    let tree_xy    = vec2<u32>(coordinate.xy) % view_config.tree_size;
    let tree_index = ((coordinate.side * config.lod_count +
                       coordinate.lod) * view_config.tree_size +
                       tree_xy.x)      * view_config.tree_size +
                       tree_xy.y;

    return tile_tree[tree_index];
}

// Todo: implement this more efficiently
fn lookup_best(lookup_coordinate: Coordinate) -> BestLookup {
    var coordinate: Coordinate; var tile_tree_uv: vec2<f32>;

    var new_coordinate   = lookup_coordinate;
    coordinate_change_lod(&new_coordinate , 0u);
    var new_tile_tree_uv = new_coordinate.uv;

    while (new_coordinate.lod < config.lod_count && !any(new_tile_tree_uv <= vec2<f32>(0.0)) && !any(new_tile_tree_uv >= vec2<f32>(1.0))) {
        coordinate  = new_coordinate;
        tile_tree_uv = new_tile_tree_uv;

        new_coordinate = lookup_coordinate;
        coordinate_change_lod(&new_coordinate, coordinate.lod + 1u);
        new_tile_tree_uv = compute_tile_tree_uv(new_coordinate);
    }

    let tile_tree_entry = lookup_tile_tree_entry(coordinate);

    coordinate_change_lod(&coordinate, tile_tree_entry.atlas_lod);

    return BestLookup(AtlasTile(tile_tree_entry.atlas_index, coordinate), tile_tree_uv);
}

fn lookup_tile(lookup_coordinate: Coordinate, blend: Blend, lod_offset: u32) -> AtlasTile {
#ifdef TILE_TREE_LOD
    return lookup_best(lookup_coordinate).tile;
#else
    var coordinate = lookup_coordinate;

    coordinate_change_lod(&coordinate, blend.lod - lod_offset);

    let tile_tree_entry = lookup_tile_tree_entry(coordinate);

    coordinate_change_lod(&coordinate, tile_tree_entry.atlas_lod);

    return AtlasTile(tile_tree_entry.atlas_index, coordinate);
#endif
}
