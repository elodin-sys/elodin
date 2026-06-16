#define_import_path bevy_terrain::attachments

#import bevy_terrain::types::AtlasTile
#import bevy_terrain::bindings::{config, atlas_sampler, attachments, attachment0_atlas, attachment1_atlas, attachment2_atlas}
#import bevy_terrain::functions::tile_count

fn attachment_uv(uv: vec2<f32>, attachment_index: u32) -> vec2<f32> {
    let attachment = attachments[attachment_index];
    return uv * attachment.scale + attachment.offset;
}

fn sample_attachment0(tile: AtlasTile) -> vec4<f32> {
    let uv = attachment_uv(tile.coordinate.uv, 0u);

#ifdef FRAGMENT
#ifdef SAMPLE_GRAD
    return textureSampleGrad(attachment0_atlas, atlas_sampler, uv, tile.index, tile.coordinate.uv_dx, tile.coordinate.uv_dy);
#else
    return textureSampleLevel(attachment0_atlas, atlas_sampler, uv, tile.index, 0.0);
#endif
#else
    return textureSampleLevel(attachment0_atlas, atlas_sampler, uv, tile.index, 0.0);
#endif
}

fn sample_attachment1(tile: AtlasTile) -> vec4<f32> {
    let uv = attachment_uv(tile.coordinate.uv, 1u);

#ifdef FRAGMENT
#ifdef SAMPLE_GRAD
    return textureSampleGrad(attachment1_atlas, atlas_sampler, uv, tile.index, tile.coordinate.uv_dx, tile.coordinate.uv_dy);
#else
    return textureSampleLevel(attachment1_atlas, atlas_sampler, uv, tile.index, 0.0);
#endif
#else
    return textureSampleLevel(attachment1_atlas, atlas_sampler, uv, tile.index, 0.0);
#endif
}

fn sample_attachment1_gather0(tile: AtlasTile) -> vec4<f32> {
    let uv = attachment_uv(tile.coordinate.uv, 1u);
    return textureGather(0, attachment1_atlas, atlas_sampler, uv, tile.index);
}

fn sample_height(tile: AtlasTile) -> f32 {
    let height = sample_attachment0(tile).x;

    return mix(config.min_height, config.max_height, height);
}

fn sample_normal(tile: AtlasTile, vertex_normal: vec3<f32>) -> vec3<f32> {
    let uv = attachment_uv(tile.coordinate.uv, 0u);

#ifdef SPHERICAL
    var FACE_UP = array(
        vec3( 0.0, 1.0,  0.0),
        vec3( 0.0, 1.0,  0.0),
        vec3( 0.0, 0.0, -1.0),
        vec3( 0.0, 0.0, -1.0),
        vec3(-1.0, 0.0,  0.0),
        vec3(-1.0, 0.0,  0.0),
    );

    let face_up = FACE_UP[tile.coordinate.side];

    let normal    = normalize(vertex_normal);
    let tangent   = cross(face_up, normal);
    let bitangent = cross(normal, tangent);
    let TBN       = mat3x3(tangent, bitangent, normal);

    let side_length = 3.14159265359 / 4.0 * config.scale;
#else
    let TBN = mat3x3(1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0,
                     0.0, 1.0, 0.0);

    let side_length = config.scale;
#endif

    // Todo: this is only an approximation of the S2 distance (pixels are not spaced evenly and they are not perpendicular)
    let pixels_per_side = attachments[0u].size * tile_count(tile.coordinate.lod);
    let distance_between_samples = side_length / pixels_per_side;
    let offset = 0.5 / attachments[0u].size;

#ifdef FRAGMENT
#ifdef SAMPLE_GRAD
    let left  = mix(config.min_height, config.max_height, textureSampleGrad(attachment0_atlas, atlas_sampler, uv + vec2<f32>(-offset,     0.0), tile.index, tile.coordinate.uv_dx, tile.coordinate.uv_dy).x);
    let up    = mix(config.min_height, config.max_height, textureSampleGrad(attachment0_atlas, atlas_sampler, uv + vec2<f32>(    0.0, -offset), tile.index, tile.coordinate.uv_dx, tile.coordinate.uv_dy).x);
    let right = mix(config.min_height, config.max_height, textureSampleGrad(attachment0_atlas, atlas_sampler, uv + vec2<f32>( offset,     0.0), tile.index, tile.coordinate.uv_dx, tile.coordinate.uv_dy).x);
    let down  = mix(config.min_height, config.max_height, textureSampleGrad(attachment0_atlas, atlas_sampler, uv + vec2<f32>(    0.0,  offset), tile.index, tile.coordinate.uv_dx, tile.coordinate.uv_dy).x);
#else
    let left  = mix(config.min_height, config.max_height, textureSampleLevel(attachment0_atlas, atlas_sampler, uv + vec2<f32>(-offset,     0.0), tile.index, 0.0).x);
    let up    = mix(config.min_height, config.max_height, textureSampleLevel(attachment0_atlas, atlas_sampler, uv + vec2<f32>(    0.0, -offset), tile.index, 0.0).x);
    let right = mix(config.min_height, config.max_height, textureSampleLevel(attachment0_atlas, atlas_sampler, uv + vec2<f32>( offset,     0.0), tile.index, 0.0).x);
    let down  = mix(config.min_height, config.max_height, textureSampleLevel(attachment0_atlas, atlas_sampler, uv + vec2<f32>(    0.0,  offset), tile.index, 0.0).x);
#endif
#else
    let left  = mix(config.min_height, config.max_height, textureSampleLevel(attachment0_atlas, atlas_sampler, uv + vec2<f32>(-offset,     0.0), tile.index, 0.0).x);
    let up    = mix(config.min_height, config.max_height, textureSampleLevel(attachment0_atlas, atlas_sampler, uv + vec2<f32>(    0.0, -offset), tile.index, 0.0).x);
    let right = mix(config.min_height, config.max_height, textureSampleLevel(attachment0_atlas, atlas_sampler, uv + vec2<f32>( offset,     0.0), tile.index, 0.0).x);
    let down  = mix(config.min_height, config.max_height, textureSampleLevel(attachment0_atlas, atlas_sampler, uv + vec2<f32>(    0.0,  offset), tile.index, 0.0).x);
#endif

    let surface_normal = normalize(vec3<f32>(left - right, down - up, distance_between_samples));

    return normalize(TBN * surface_normal);
}

fn sample_color(tile: AtlasTile) -> vec4<f32> {
    let height = sample_attachment0(tile).x;

    return vec4<f32>(height * 0.5);
}
