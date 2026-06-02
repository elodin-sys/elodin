// Generate six 16-bit grayscale TIFFs that the upstream
// `preprocess_spherical` example consumes as the cube faces of a synthetic
// Earth-like sphere. Each face is 1024×1024 by default; pixels project to a
// 3D direction on the unit sphere, where we sample a multi-frequency
// trig-noise field. Continuity at face edges is approximate (no proper
// great-circle continuation) but visually clean enough for a demo.
//
// Output: assets/terrains/spherical/source/height/face{0..5}.tif

use image::{ImageBuffer, Luma};
use std::fs;
use std::path::Path;

const FACE_SIZE: u32 = 1024;
const OUT_DIR: &str = "assets/terrains/spherical/source/height";

fn main() {
    fs::create_dir_all(OUT_DIR).expect("create spherical source dir");

    for face in 0..6u32 {
        let out_path = format!("{OUT_DIR}/face{face}.tif");
        if Path::new(&out_path).exists() {
            println!("skip {out_path} (already exists)");
            continue;
        }

        let mut img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(FACE_SIZE, FACE_SIZE);

        for y in 0..FACE_SIZE {
            for x in 0..FACE_SIZE {
                let u = (x as f32 + 0.5) / FACE_SIZE as f32; // 0..1
                let v = (y as f32 + 0.5) / FACE_SIZE as f32;

                // Map (u, v) on this face to a direction on the unit sphere.
                // Each face spans [-1, 1] x [-1, 1] in its tangent plane.
                let s = 2.0 * u - 1.0;
                let t = 2.0 * v - 1.0;

                let dir = match face {
                    0 => normalize([1.0, -t, -s]),  // +X
                    1 => normalize([-1.0, -t, s]),  // -X
                    2 => normalize([s, 1.0, t]),    // +Y
                    3 => normalize([s, -1.0, -t]),  // -Y
                    4 => normalize([s, -t, 1.0]),   // +Z
                    5 => normalize([-s, -t, -1.0]), // -Z
                    _ => unreachable!(),
                };

                let h = noise3d(dir[0], dir[1], dir[2]);
                let h = h.clamp(0.0, 1.0);
                img.put_pixel(x, y, Luma([(h * u16::MAX as f32) as u16]));
            }
        }

        img.save(&out_path).expect("save face tif");
        println!("wrote {out_path} ({FACE_SIZE}x{FACE_SIZE} u16 grayscale)");
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Multi-frequency trig "noise" sampled at a point on the unit sphere.
/// Composed of low-frequency continent-scale variation plus higher-frequency
/// wrinkles. Output ~ [0.2, 0.85] — most of the dynamic range without ever
/// hitting hard min/max.
fn noise3d(x: f32, y: f32, z: f32) -> f32 {
    use std::f32::consts::TAU;

    let f = |a: f32, b: f32, c: f32, freq: f32| {
        ((a * freq).sin() * (b * freq * 0.83).cos() + (c * freq * 1.17).sin()) * 0.333
    };

    let continents = f(x, y, z, TAU * 0.5);
    let regions = f(x + 1.7, y - 0.9, z + 2.3, TAU * 1.3);
    let ranges = f(x * 1.1, y * 1.3, z * 0.9, TAU * 2.7);
    let detail = f(x * 1.7, y * 1.9, z * 1.3, TAU * 5.3);

    0.5 + 0.30 * continents + 0.20 * regions + 0.10 * ranges + 0.05 * detail
}
