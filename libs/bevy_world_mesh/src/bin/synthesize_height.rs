// Generate the synthetic input assets that the bevy_terrain examples and the
// top-level world_mesh app feed through bevy_terrain's preprocess pipeline.
// Idempotent — files only written if missing.
//
// Outputs:
//   assets/terrains/planar/source/height.png
//   assets/terrains/planar/source/albedo.png
//   assets/textures/gradient.png
//   assets/textures/gradient2.png

use image::{ImageBuffer, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use std::fs;
use std::path::Path;

/// Heightmap resolution. Larger = more detail at every LOD; 2048×2048 produces
/// a noticeably crisper render than the 1024 we used originally.
const TERRAIN_SIZE: u32 = 2048;
/// World-space scale of the noise field. Lower number → wider features.
const NOISE_SCALE: f64 = 1.6;
/// Snowline cutoff in normalised height [0..1].
const SNOWLINE: f32 = 0.78;
/// Treeline cutoff (above this, terrain becomes bare rock).
const TREELINE: f32 = 0.55;

fn main() {
    let height_field = build_height_field(TERRAIN_SIZE, 0xC0DE_C0DE);
    write_planar_height(
        "assets/terrains/planar/source/height.png",
        TERRAIN_SIZE,
        &height_field,
    );
    write_planar_albedo(
        "assets/terrains/planar/source/albedo.png",
        TERRAIN_SIZE,
        &height_field,
    );
    write_gradient("assets/textures/gradient.png", earth_gradient_stops());
    write_gradient("assets/textures/gradient2.png", planar_gradient_stops());
}

/// Sample a fractal Brownian motion field driven by Perlin noise on a regular
/// grid. Returns a `size × size` row-major buffer of values in [0..1] with
/// elevation-like statistics (a few prominent ridges, lots of mid-band rolling
/// terrain, occasional valleys / lakebeds).
fn build_height_field(size: u32, seed: u32) -> Vec<f32> {
    // Two-octave-flavoured generator: a wide, low-amplitude continent layer
    // adds large-scale variation; a high-frequency `Fbm` overlay carves the
    // ridges, valleys, and erosion-like detail you'd see in a real DEM.
    let continent = Fbm::<Perlin>::new(seed)
        .set_octaves(4)
        .set_frequency(0.4)
        .set_persistence(0.5)
        .set_lacunarity(2.0);
    let detail = Fbm::<Perlin>::new(seed.wrapping_add(7))
        .set_octaves(7)
        .set_frequency(2.5)
        .set_persistence(0.55)
        .set_lacunarity(2.05);

    let mut out = vec![0.0_f32; (size * size) as usize];
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for y in 0..size {
        for x in 0..size {
            let xn = x as f64 / (size - 1) as f64;
            let yn = y as f64 / (size - 1) as f64;
            let p = [xn * NOISE_SCALE, yn * NOISE_SCALE];
            let pd = [xn * NOISE_SCALE * 1.0, yn * NOISE_SCALE * 1.0];

            let c = continent.get(p) as f32; // ~[-1, 1]
            let d = detail.get(pd) as f32; // ~[-1, 1]

            // Re-shape the field: continents bias the mean, detail adds the
            // high-frequency carving, and a soft floor squashes deep valleys
            // so we don't end up with totally flat lakebeds.
            let raw = 0.55 * c + 0.45 * d;
            let h = 0.5 + 0.5 * raw.tanh();

            // Sharpen the ridges: pow > 1 deepens valleys without blowing
            // out the highs (visually similar to thermal-erosion shaping).
            let h = h.powf(1.25);

            out[(y * size + x) as usize] = h;
            if h < min {
                min = h;
            }
            if h > max {
                max = h;
            }
        }
    }

    // Renormalise so the dynamic range fills [0, 1] regardless of seed.
    let span = (max - min).max(1e-6);
    for v in &mut out {
        *v = ((*v - min) / span).clamp(0.0, 1.0);
    }
    out
}

fn ensure_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create asset parent dir");
    }
}

fn write_planar_height(path: &str, size: u32, heights: &[f32]) {
    let path_ref = Path::new(path);
    if path_ref.exists() {
        println!("skip {path} (already exists)");
        return;
    }
    ensure_parent(path_ref);

    let mut img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let h = heights[(y * size + x) as usize];
            img.put_pixel(x, y, Luma([(h * u16::MAX as f32) as u16]));
        }
    }
    img.save(path).expect("save height png");
    println!("wrote {path} ({size}x{size} u16 grayscale)");
}

/// Synthesise an "ortho-photo style" albedo with biome classification driven
/// by elevation and finite-difference slope of the heightfield. Low + flat →
/// grass, mid + slope → forest with rocky crags, high → bare rock, very high
/// → snow. The result reads as a real terrain photo from cruising altitude
/// rather than a flat gradient stripe.
fn write_planar_albedo(path: &str, size: u32, heights: &[f32]) {
    let path_ref = Path::new(path);
    if path_ref.exists() {
        println!("skip {path} (already exists)");
        return;
    }
    ensure_parent(path_ref);

    // Independent FBM for biome dithering — keeps the albedo from looking
    // perfectly correlated with the heightmap (real ortho-photos have
    // texture variation that doesn't track elevation 1:1).
    let veg = Fbm::<Perlin>::new(0x1F1F)
        .set_octaves(4)
        .set_frequency(8.0)
        .set_persistence(0.55);

    let pixel = |x: u32, y: u32| heights[(y * size + x) as usize];
    let slope = |x: u32, y: u32| -> f32 {
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(size - 1);
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(size - 1);
        let dx = pixel(xp, y) - pixel(xm, y);
        let dy = pixel(x, yp) - pixel(x, ym);
        // Convert normalised heightfield differences into a 0..1 slope-ish
        // metric. The constant scales it so slopes of around 30° read as ~0.4.
        ((dx * dx + dy * dy).sqrt() * 30.0).clamp(0.0, 1.0)
    };

    let mut img: RgbImage = RgbImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let h = pixel(x, y);
            let s = slope(x, y);
            let xn = x as f64 / (size - 1) as f64;
            let yn = y as f64 / (size - 1) as f64;
            let dither = (veg.get([xn * 4.0, yn * 4.0]) as f32) * 0.5 + 0.5;

            // Anchor colours.
            let grass = vec3(0.32, 0.50, 0.22) * (0.85 + 0.30 * dither);
            let forest = vec3(0.18, 0.34, 0.16) * (0.80 + 0.25 * dither);
            let rock = vec3(0.45, 0.42, 0.38);
            let snow = vec3(0.94, 0.95, 0.97);
            let scrub = vec3(0.42, 0.41, 0.28);

            // Blend lowland: grass ↔ forest based on dither (mottling).
            let lowland = lerp_v3(grass, forest, smoothstep(0.45, 0.85, dither));

            // Treeline → rock transition.
            let with_rock = if h < TREELINE {
                lowland
            } else {
                let t = smoothstep(TREELINE, TREELINE + 0.15, h);
                lerp_v3(lowland, scrub, t)
            };
            let with_rock = lerp_v3(with_rock, rock, smoothstep(0.20, 0.65, s));

            // Snow on top.
            let mut col = if h < SNOWLINE {
                with_rock
            } else {
                let t = smoothstep(SNOWLINE, SNOWLINE + 0.1, h);
                lerp_v3(with_rock, snow, t)
            };

            // Subtle elevation darkening at very low altitudes (river beds).
            if h < 0.10 {
                let t = smoothstep(0.0, 0.10, h);
                col = lerp_v3(vec3(0.18, 0.22, 0.20), col, t);
            }

            img.put_pixel(
                x,
                y,
                Rgb([
                    (col.0 * 255.0).clamp(0.0, 255.0) as u8,
                    (col.1 * 255.0).clamp(0.0, 255.0) as u8,
                    (col.2 * 255.0).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    img.save(path).expect("save albedo png");
    println!("wrote {path} ({size}x{size} 8-bit RGB)");
}

fn write_gradient(path: &str, stops: &[(f32, [u8; 4])]) {
    let path_ref = Path::new(path);
    if path_ref.exists() {
        println!("skip {path} (already exists)");
        return;
    }
    ensure_parent(path_ref);

    const WIDTH: u32 = 256;
    let mut img = RgbaImage::new(WIDTH, 1);
    for x in 0..WIDTH {
        let t = x as f32 / (WIDTH - 1) as f32;
        let c = sample_stops(stops, t);
        img.put_pixel(x, 0, Rgba(c));
    }
    img.save(path).expect("save gradient png");
    println!("wrote {path} ({WIDTH}x1 8-bit RGBA gradient)");
}

fn earth_gradient_stops() -> &'static [(f32, [u8; 4])] {
    &[
        (0.00, [10, 30, 90, 255]),
        (0.45, [40, 90, 160, 255]),
        (0.50, [220, 200, 140, 255]),
        (0.55, [60, 110, 50, 255]),
        (0.75, [120, 100, 70, 255]),
        (0.90, [180, 180, 180, 255]),
        (1.00, [255, 255, 255, 255]),
    ]
}

fn planar_gradient_stops() -> &'static [(f32, [u8; 4])] {
    &[
        (0.00, [60, 90, 50, 255]),
        (0.40, [110, 130, 70, 255]),
        (0.70, [140, 130, 110, 255]),
        (0.90, [200, 200, 200, 255]),
        (1.00, [255, 255, 255, 255]),
    ]
}

fn sample_stops(stops: &[(f32, [u8; 4])], t: f32) -> [u8; 4] {
    if stops.is_empty() {
        return [255, 255, 255, 255];
    }
    if t <= stops[0].0 {
        return stops[0].1;
    }
    if t >= stops[stops.len() - 1].0 {
        return stops[stops.len() - 1].1;
    }
    for window in stops.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if t >= t0 && t <= t1 {
            let mix = (t - t0) / (t1 - t0).max(1e-6);
            return [
                lerp_u8(c0[0], c1[0], mix),
                lerp_u8(c0[1], c1[1], mix),
                lerp_u8(c0[2], c1[2], mix),
                lerp_u8(c0[3], c1[3], mix),
            ];
        }
    }
    stops[stops.len() - 1].1
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let a = a as f32;
    let b = b as f32;
    (a + (b - a) * t).round().clamp(0.0, 255.0) as u8
}

#[derive(Copy, Clone)]
struct Vec3(f32, f32, f32);

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3(self.0 * s, self.1 * s, self.2 * s)
    }
}

fn vec3(r: f32, g: f32, b: f32) -> Vec3 {
    Vec3(r, g, b)
}

fn lerp_v3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3(
        a.0 + (b.0 - a.0) * t,
        a.1 + (b.1 - a.1) * t,
        a.2 + (b.2 - a.2) * t,
    )
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
