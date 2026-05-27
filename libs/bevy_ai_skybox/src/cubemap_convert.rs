use bevy::prelude::Vec3;
use image::{Rgba, RgbaImage};

pub const BUNDLED_CUBEMAP_FACE_SIZE: u32 = 2048;

pub fn equirect_to_stacked_cubemap(source: &RgbaImage, face_size: u32) -> RgbaImage {
    let mut output = RgbaImage::new(face_size, face_size * 6);
    for face in 0..6 {
        for y in 0..face_size {
            for x in 0..face_size {
                let direction = cube_face_uv_to_direction(face, x, y, face_size);
                let (u, v) = direction_to_equirect_uv(direction);
                output.put_pixel(x, y + face * face_size, sample_equirect(source, u, v));
            }
        }
    }
    output
}

pub fn stacked_face(stacked: &RgbaImage, face: u32, face_size: u32) -> RgbaImage {
    let mut face_img = RgbaImage::new(face_size, face_size);
    for y in 0..face_size {
        for x in 0..face_size {
            face_img.put_pixel(x, y, *stacked.get_pixel(x, y + face * face_size));
        }
    }
    face_img
}

fn cube_face_uv_to_direction(face: u32, x: u32, y: u32, face_size: u32) -> Vec3 {
    let s = 2.0 * ((x as f32 + 0.5) / face_size as f32) - 1.0;
    let t = 2.0 * ((y as f32 + 0.5) / face_size as f32) - 1.0;
    match face {
        0 => Vec3::new(1.0, -t, -s).normalize(),
        1 => Vec3::new(-1.0, -t, s).normalize(),
        2 => Vec3::new(s, 1.0, t).normalize(),
        3 => Vec3::new(s, -1.0, -t).normalize(),
        4 => Vec3::new(s, -t, 1.0).normalize(),
        _ => Vec3::new(-s, -t, -1.0).normalize(),
    }
}

fn direction_to_equirect_uv(direction: Vec3) -> (f32, f32) {
    let direction = direction.normalize();
    let u = (0.5 + direction.z.atan2(direction.x) / std::f32::consts::TAU).fract();
    let v = (direction.y.acos() / std::f32::consts::PI).clamp(0.0, 1.0);
    (u, v)
}

fn sample_equirect(source: &RgbaImage, u: f32, v: f32) -> Rgba<u8> {
    let width = source.width();
    let height = source.height();
    let x = u * width as f32 - 0.5;
    let y = (v * (height.saturating_sub(1)) as f32).clamp(0.0, height.saturating_sub(1) as f32);
    let x_floor = x.floor();
    let y_floor = y.floor();
    let x0 = wrap_pixel_x(x_floor as i32, width);
    let x1 = wrap_pixel_x(x_floor as i32 + 1, width);
    let y0 = y_floor as u32;
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = x - x_floor;
    let ty = y - y_floor;
    let p00 = source.get_pixel(x0, y0);
    let p10 = source.get_pixel(x1, y0);
    let p01 = source.get_pixel(x0, y1);
    let p11 = source.get_pixel(x1, y1);
    let mut rgba = [0u8; 4];
    for i in 0..4 {
        let top = lerp(p00[i] as f32, p10[i] as f32, tx);
        let bottom = lerp(p01[i] as f32, p11[i] as f32, tx);
        rgba[i] = lerp(top, bottom, ty).round().clamp(0.0, 255.0) as u8;
    }
    Rgba(rgba)
}

fn wrap_pixel_x(x: i32, width: u32) -> u32 {
    let width = width as i32;
    x.rem_euclid(width) as u32
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
