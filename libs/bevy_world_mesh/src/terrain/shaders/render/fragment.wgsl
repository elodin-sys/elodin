#define_import_path bevy_terrain::fragment

// Bevy 0.16 note: we used to call `bevy_pbr::pbr_functions::apply_pbr_lighting`
// here, which depends on `bevy_pbr::pbr_bindings`. That module declares
// `@group(2) @binding(0)` for its `StandardMaterial` bindings, which conflicts
// with our terrain's `@group(2)` (terrain_view). naga_oil's composed module
// flags that as a bind-group mismatch and the shader fails validation, but
// because bevy routes shader creation through `ShaderSource::Naga`, the source
// string is empty by the time wgpu reports the error -- you just see a
// `byte index X out of bounds of ``` panic inside `naga::span::Span::location`.
// To avoid importing pbr_bindings at all, we keep the View uniform import
// (which is `@group(0) @binding(0)`) but implement our own minimal Lambertian
// + ambient lighting locally. This is intentional for our heightmap use case:
// terrain surfaces are pure matte, we don't need PBR, and the heavy bevy_pbr
// lighting pipeline (clustered forward, shadows, light probes, irradiance
// volumes, transmission, etc.) is inapplicable.

#import bevy_terrain::types::{Blend, AtlasTile, Coordinate}
#import bevy_terrain::bindings::{config, view_config, geometry_tiles}
#import bevy_terrain::functions::{compute_blend, lookup_tile}
#import bevy_terrain::attachments::{sample_normal, sample_color}
#import bevy_terrain::debug::{show_data_lod, show_geometry_lod, show_tile_tree, show_pixels}
#import bevy_pbr::mesh_view_bindings::view

struct FragmentInput {
    @builtin(position)     clip_position: vec4<f32>,
    @location(0)           tile_index: u32,
    @location(1)           coordinate_uv: vec2<f32>,
    @location(2)           world_position: vec4<f32>,
    @location(3)           world_normal: vec3<f32>,
}

struct FragmentOutput {
    @location(0)             color: vec4<f32>
}

struct FragmentInfo {
    coordinate: Coordinate,
    view_distance: f32,
    blend: Blend,
    clip_position: vec4<f32>,
    world_normal: vec3<f32>,
    world_position: vec4<f32>,
    color: vec4<f32>,
    normal: vec3<f32>,
}

fn fragment_info(input: FragmentInput) -> FragmentInfo{
    let tile          = geometry_tiles[input.tile_index];
    let uv            = input.coordinate_uv;
    let view_distance = distance(input.world_position.xyz, view.world_position);

    var info: FragmentInfo;
    info.coordinate     = Coordinate(tile.side, tile.lod, tile.xy, uv, dpdx(uv), dpdy(uv));
    info.view_distance  = view_distance;
    info.blend          = compute_blend(view_distance);
    info.clip_position  = input.clip_position;
    info.world_normal   = input.world_normal;
    info.world_position = input.world_position;

    return info;
}

/// Minimal Lambertian + ambient lighting. Matches the direction of the
/// DirectionalLight that `TerrainDebugPlugin::debug_lighting` spawns (pointing
/// roughly down-and-forward from the sky), and uses an ambient floor so that
/// shadowed faces aren't pitch black. This replaces bevy_pbr's `apply_pbr_lighting`
/// and its `@group(2)` StandardMaterial bindings; see the module-level note.
fn terrain_lighting(base_color: vec4<f32>, world_normal: vec3<f32>) -> vec4<f32> {
    let N = normalize(world_normal);
    // Sun from upper-right, matching `debug_lighting`'s transform
    // (`Vec3::new(0.5, -1.0, -0.5).normalize()` as forward light direction,
    // i.e. the light direction toward the scene). We point the normal the
    // opposite way to dot against the incoming light.
    let L = normalize(vec3<f32>(-0.5, 1.0, 0.5));
    let diffuse = max(dot(N, L), 0.0);
    let ambient = 0.25;
    let intensity = clamp(ambient + (1.0 - ambient) * diffuse, 0.0, 1.0);
    return vec4<f32>(base_color.rgb * intensity, base_color.a);
}

fn fragment_output(info: ptr<function, FragmentInfo>, output: ptr<function, FragmentOutput>, color: vec4<f32>, normal: vec3<f32>) {
#ifdef LIGHTING
    (*output).color = terrain_lighting(color, normal);
#else
    (*output).color = color;
#endif
}

fn fragment_debug(info: ptr<function, FragmentInfo>, output: ptr<function, FragmentOutput>, tile: AtlasTile, normal: vec3<f32>) {
#ifdef SHOW_DATA_LOD
    (*output).color = show_data_lod((*info).blend, tile);
#endif
#ifdef SHOW_GEOMETRY_LOD
    (*output).color = show_geometry_lod((*info).coordinate);
#endif
#ifdef SHOW_TILE_TREE
    (*output).color = show_tile_tree((*info).coordinate);
#endif
#ifdef SHOW_PIXELS
    (*output).color = mix((*output).color, show_pixels(tile), 0.5);
#endif
#ifdef SHOW_UV
    (*output).color = vec4<f32>(tile.coordinate.uv, 0.0, 1.0);
#endif
#ifdef SHOW_NORMALS
    (*output).color = vec4<f32>(normal, 1.0);
#endif

    // Todo: move this somewhere else
    if ((*info).view_distance < view_config.precision_threshold_distance) {
        (*output).color = mix((*output).color, vec4<f32>(0.1), 0.7);
    }
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
