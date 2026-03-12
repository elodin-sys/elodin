#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

struct SensorEffectSettings {
    // 0 = normal (passthrough), 1 = thermal, 2 = night vision, 3 = depth
    effect_type: u32,
    // For thermal: contrast. For night vision: gain.
    param_a: f32,
    // For thermal: noise_sigma. For night vision: noise_sigma.
    param_b: f32,
    // Elapsed time in seconds (for animated noise).
    time: f32,
}
@group(0) @binding(2) var<uniform> settings: SensorEffectSettings;

// Simple hash for noise generation
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise_at(uv: vec2<f32>, t: f32) -> f32 {
    return hash(uv * 1000.0 + vec2<f32>(t * 7.13, t * 3.71)) - 0.5;
}

// Iron-bow thermal colormap: black → purple → red → orange → yellow → white
fn thermal_colormap(t: f32) -> vec3<f32> {
    let c = clamp(t, 0.0, 1.0);
    let r = clamp(1.5 * c - 0.25, 0.0, 1.0);
    let g = clamp(2.0 * c - 0.75, 0.0, 1.0);
    var b = clamp(3.0 * (c - 0.1) * (0.6 - c), 0.0, 1.0) + clamp(c - 0.85, 0.0, 1.0) * 3.0;
    b = clamp(b, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn apply_thermal(color: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let contrast = settings.param_a;
    let noise_sigma = settings.param_b;

    let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let contrasted = clamp((luminance - 0.5) * contrast + 0.5, 0.0, 1.0);
    let noise = noise_at(uv, settings.time) * noise_sigma;
    let t = clamp(contrasted + noise, 0.0, 1.0);

    return vec4<f32>(thermal_colormap(t), 1.0);
}

fn apply_night_vision(color: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let gain = settings.param_a;
    let noise_sigma = settings.param_b;

    let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let amplified = clamp(luminance * gain, 0.0, 1.0);
    let grain = noise_at(uv, settings.time) * noise_sigma;
    let noisy = clamp(amplified + grain, 0.0, 1.0);

    // Phosphor green tint
    return vec4<f32>(0.1 * noisy, noisy, 0.05 * noisy, 1.0);
}

fn apply_depth(color: vec4<f32>) -> vec4<f32> {
    let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(luminance, luminance, luminance, 1.0);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(screen_texture, texture_sampler, in.uv);

    switch settings.effect_type {
        // Normal: passthrough
        case 0u: {
            return color;
        }
        // Thermal
        case 1u: {
            return apply_thermal(color, in.uv);
        }
        // Night vision
        case 2u: {
            return apply_night_vision(color, in.uv);
        }
        // Depth (grayscale)
        case 3u: {
            return apply_depth(color);
        }
        default: {
            return color;
        }
    }
}
