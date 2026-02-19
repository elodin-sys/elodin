use ab_glyph::{Font, FontRef, GlyphId, ScaleFont, point};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use std::collections::HashMap;

#[derive(Resource, Default)]
pub struct IconTextureCache {
    cache: HashMap<(String, u32), Handle<Image>>,
    codepoint_cache: HashMap<(char, u32), Handle<Image>>,
}

impl IconTextureCache {
    pub fn get_or_insert(
        &mut self,
        icon_name: &str,
        px_size: u32,
        images: &mut Assets<Image>,
    ) -> Handle<Image> {
        let key = (icon_name.to_string(), px_size);
        if let Some(handle) = self.cache.get(&key) {
            return handle.clone();
        }

        let Some(codepoint) = impeller2_wkt::builtin_icon_char(icon_name) else {
            warn!(
                "Unknown built-in icon '{}', using fallback circle",
                icon_name
            );
            let handle = self.get_or_insert_codepoint('\u{ef4a}', px_size, images);
            self.cache.insert(key, handle.clone());
            return handle;
        };

        let handle = self.get_or_insert_codepoint(codepoint, px_size, images);
        self.cache.insert(key, handle.clone());
        handle
    }

    fn get_or_insert_codepoint(
        &mut self,
        codepoint: char,
        px_size: u32,
        images: &mut Assets<Image>,
    ) -> Handle<Image> {
        let key = (codepoint, px_size);
        if let Some(handle) = self.codepoint_cache.get(&key) {
            return handle.clone();
        }
        let image = rasterize_material_icon(codepoint, px_size);
        let handle = images.add(image);
        self.codepoint_cache.insert(key, handle.clone());
        handle
    }
}

pub fn rasterize_material_icon(codepoint: char, px_size: u32) -> Image {
    let font = FontRef::try_from_slice(egui_material_icons::FONT_DATA)
        .expect("Material Icons font data should be valid");

    let glyph_id = font.glyph_id(codepoint);
    if glyph_id == GlyphId(0) {
        warn!(
            "Material Icons font has no glyph for codepoint U+{:04X}, generating fallback",
            codepoint as u32
        );
        return generate_fallback_icon(px_size);
    }

    let scale = px_size as f32;
    let scaled_font = font.as_scaled(scale);
    let glyph = glyph_id.with_scale_and_position(scale, point(0.0, 0.0));

    let Some(outlined) = font.outline_glyph(glyph) else {
        warn!(
            "Could not outline glyph for codepoint U+{:04X}, generating fallback",
            codepoint as u32
        );
        return generate_fallback_icon(px_size);
    };

    let bounds = outlined.px_bounds();
    let glyph_w = bounds.width().ceil() as u32;
    let glyph_h = bounds.height().ceil() as u32;

    if glyph_w == 0 || glyph_h == 0 {
        return generate_fallback_icon(px_size);
    }

    let canvas_size = px_size.max(glyph_w.max(glyph_h));

    let h_advance = scaled_font.h_advance(glyph_id);
    let ascent = scaled_font.ascent();
    let descent = scaled_font.descent();

    let baseline_x = (canvas_size as f32 - h_advance) / 2.0;
    let baseline_y = (canvas_size as f32 + ascent + descent) / 2.0;

    let glyph = glyph_id.with_scale_and_position(scale, point(baseline_x, baseline_y));
    let Some(outlined) = font.outline_glyph(glyph) else {
        return generate_fallback_icon(px_size);
    };
    let bounds = outlined.px_bounds();

    let mut pixels = vec![0u8; (canvas_size * canvas_size * 4) as usize];

    outlined.draw(|x, y, coverage| {
        let px = x as i32 + bounds.min.x as i32;
        let py = y as i32 + bounds.min.y as i32;
        if px >= 0 && py >= 0 && (px as u32) < canvas_size && (py as u32) < canvas_size {
            let idx = ((py as u32 * canvas_size + px as u32) * 4) as usize;
            let alpha = (coverage * 255.0).round() as u8;
            pixels[idx] = 255;
            pixels[idx + 1] = 255;
            pixels[idx + 2] = 255;
            pixels[idx + 3] = alpha;
        }
    });

    Image::new(
        Extent3d {
            width: canvas_size,
            height: canvas_size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        default(),
    )
}

fn generate_fallback_icon(px_size: u32) -> Image {
    let size = px_size.max(4);
    let mut pixels = vec![0u8; (size * size * 4) as usize];
    let center = size as f32 / 2.0;
    let radius = center * 0.8;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center + 0.5;
            let dy = y as f32 - center + 0.5;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist <= radius {
                let alpha = if dist > radius - 1.5 {
                    ((radius - dist) / 1.5 * 255.0).round() as u8
                } else {
                    255
                };
                let idx = ((y * size + x) * 4) as usize;
                pixels[idx] = 255;
                pixels[idx + 1] = 255;
                pixels[idx + 2] = 255;
                pixels[idx + 3] = alpha;
            }
        }
    }

    Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        default(),
    )
}
