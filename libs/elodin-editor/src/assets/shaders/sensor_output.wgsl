#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var texture_sampler: sampler;

struct SensorOutputSettings {
    // x: effect, y: palette, z: flags, w: deterministic frame seed
    mode: vec4<u32>,
    // width, height, near, far
    viewport: vec4<f32>,
    // legacy effect param_a, param_b, reserved, reserved
    legacy: vec4<f32>,
    // air °C, sky zenith °C, terrain base °C, solar/luminance gain °C
    thermal: vec4<f32>,
    // AGC min °C, max °C, smoothing, DDE strength
    agc: vec4<f32>,
    // MTF blur px, temporal noise DN, column FPN DN, vignette strength
    sensor: vec4<f32>,
    // transmission km, dead-pixel ppm, low percentile, high percentile
    range: vec4<f32>,
}
@group(0) @binding(3) var<uniform> settings: SensorOutputSettings;
@group(0) @binding(4) var palette_lut: texture_2d<f32>;
@group(0) @binding(5) var thermal_mask: texture_2d<f32>;

const EFFECT_NORMAL: u32 = 0u;
const EFFECT_THERMAL: u32 = 1u;
const EFFECT_NIGHT_VISION: u32 = 2u;
const EFFECT_DEPTH: u32 = 3u;
const EFFECT_LWIR: u32 = 4u;
const FLAG_THERMAL_MASK: u32 = 1u;
const PI: f32 = 3.141592653589793;

fn hash(value: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(value.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn frame_noise(pixel: vec2<f32>) -> f32 {
    let seed = f32(settings.mode.w);
    let first = max(hash(pixel + vec2(seed * 0.013, seed * 0.037)), 1e-7);
    let second = hash(pixel.yx + vec2(seed * 0.071, seed * 0.019));
    return sqrt(-2.0 * log(first)) * cos(2.0 * PI * second);
}

fn texture_coord(uv: vec2<f32>) -> vec2<i32> {
    let dimensions = vec2<i32>(textureDimensions(depth_texture));
    return clamp(
        vec2<i32>(uv * vec2<f32>(dimensions)),
        vec2<i32>(0),
        dimensions - vec2<i32>(1),
    );
}

fn raw_depth_at(uv: vec2<f32>) -> f32 {
    return textureLoad(depth_texture, texture_coord(uv), 0);
}

fn view_distance(raw_depth: f32) -> f32 {
    if raw_depth <= 1e-7 {
        return settings.viewport.w;
    }
    return clamp(settings.viewport.z / raw_depth, settings.viewport.z, settings.viewport.w);
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

fn palette(value: f32) -> vec3<f32> {
    return textureSampleLevel(
        palette_lut,
        texture_sampler,
        vec2(clamp(value, 0.0, 1.0), 0.5),
        0.0,
    ).rgb;
}

fn apply_legacy_thermal(color: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let contrasted = clamp(
        (luminance(color.rgb) - 0.5) * settings.legacy.x + 0.5,
        0.0,
        1.0,
    );
    let noise = frame_noise(uv * settings.viewport.xy) * settings.legacy.y;
    return vec4(palette(contrasted + noise), 1.0);
}

fn apply_night_vision(color: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let amplified = clamp(luminance(color.rgb) * settings.legacy.x, 0.0, 1.0);
    let grain = frame_noise(uv * settings.viewport.xy) * settings.legacy.y;
    let value = clamp(amplified + grain, 0.0, 1.0);
    return vec4(0.1 * value, value, 0.05 * value, 1.0);
}

fn apply_depth(uv: vec2<f32>) -> vec4<f32> {
    let raw_depth = raw_depth_at(uv);
    if raw_depth <= 1e-7 {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    let distance = view_distance(raw_depth);
    let log_range = max(log2(settings.viewport.w / settings.viewport.z), 1e-6);
    let normalized = clamp(log2(distance / settings.viewport.z) / log_range, 0.0, 1.0);
    return vec4(vec3(1.0 - normalized), 1.0);
}

fn mask_temperature(uv: vec2<f32>) -> f32 {
    let dimensions = vec2<i32>(textureDimensions(thermal_mask));
    let coord = clamp(
        vec2<i32>(uv * vec2<f32>(dimensions)),
        vec2<i32>(0),
        dimensions - vec2<i32>(1),
    );
    let encoded = textureLoad(thermal_mask, coord, 0).r;
    if encoded <= 0.0 {
        return -10000.0;
    }
    return encoded * 500.0 - 100.0;
}

fn pseudo_temperature(uv: vec2<f32>) -> f32 {
    let raw_depth = raw_depth_at(uv);
    if raw_depth <= 1e-7 {
        let zenith = clamp(abs(uv.y - 0.5) * 2.0, 0.0, 1.0);
        return mix(settings.thermal.x, settings.thermal.y, zenith);
    }

    if (settings.mode.z & FLAG_THERMAL_MASK) != 0u {
        let tagged_temperature = mask_temperature(uv);
        if tagged_temperature > -1000.0 {
            return tagged_temperature;
        }
    }

    let color = textureSampleLevel(screen_texture, texture_sampler, uv, 0.0);
    let lit_luminance = luminance(color.rgb);
    let surface_temperature =
        settings.thermal.z + settings.thermal.w * (lit_luminance - 0.5) * 2.0;
    let distance = view_distance(raw_depth);
    let transmission_distance = max(settings.range.x * 1000.0, 1.0);
    let transmission = exp(-distance / transmission_distance);
    return mix(settings.thermal.x, surface_temperature, transmission);
}

fn apply_lwir(uv: vec2<f32>) -> vec4<f32> {
    let pixel = 1.0 / settings.viewport.xy;
    let center = pseudo_temperature(uv);
    let left = pseudo_temperature(clamp(uv - vec2(pixel.x, 0.0), vec2(0.0), vec2(1.0)));
    let right = pseudo_temperature(clamp(uv + vec2(pixel.x, 0.0), vec2(0.0), vec2(1.0)));
    let up = pseudo_temperature(clamp(uv - vec2(0.0, pixel.y), vec2(0.0), vec2(1.0)));
    let down = pseudo_temperature(clamp(uv + vec2(0.0, pixel.y), vec2(0.0), vec2(1.0)));
    let cross_average = (left + right + up + down) * 0.25;

    let optically_softened = mix(center, cross_average, clamp(settings.sensor.x, 0.0, 1.0));
    let detailed = optically_softened + settings.agc.w * (center - cross_average);
    let span = max(settings.agc.y - settings.agc.x, 1e-3);
    var signal = clamp((detailed - settings.agc.x) / span, 0.0, 1.0);
    signal = pow(signal, max(settings.legacy.z, 0.1));

    let pixel_coord = uv * settings.viewport.xy;
    let temporal = frame_noise(pixel_coord) * settings.sensor.y / 255.0;
    let column = (hash(vec2(floor(pixel_coord.x), 17.0)) - 0.5)
        * 3.4641016
        * settings.sensor.z
        / 255.0;
    signal += temporal + column;

    let centered = uv * 2.0 - 1.0;
    signal *= 1.0 - settings.sensor.w * dot(centered, centered) * 0.5;

    let dead_threshold = max(settings.range.y, 0.0) / 1000000.0;
    let dead_hash = hash(floor(pixel_coord) + vec2(91.0, 37.0));
    if dead_hash < dead_threshold {
        signal = select(0.0, 1.0, dead_hash < dead_threshold * 0.25);
    }

    return vec4(palette(signal), 1.0);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(screen_texture, texture_sampler, in.uv);
    switch settings.mode.x {
        case EFFECT_NORMAL: {
            return color;
        }
        case EFFECT_THERMAL: {
            return apply_legacy_thermal(color, in.uv);
        }
        case EFFECT_NIGHT_VISION: {
            return apply_night_vision(color, in.uv);
        }
        case EFFECT_DEPTH: {
            return apply_depth(in.uv);
        }
        case EFFECT_LWIR: {
            return apply_lwir(in.uv);
        }
        default: {
            return color;
        }
    }
}
