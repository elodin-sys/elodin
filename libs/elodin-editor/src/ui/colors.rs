use egui::Color32;
use impeller2_wkt::Color;

pub const WHITE: Color32 = Color32::WHITE;
pub const TRANSPARENT: Color32 = Color32::TRANSPARENT;

pub const PRIMARY_CREAME: Color32 = Color32::from_rgb(0xFF, 0xFB, 0xF0);
pub const PRIMARY_CREAME_6: Color32 = Color32::from_rgb(0x99, 0x97, 0x90);
pub const PRIMARY_CREAME_8: Color32 = Color32::from_rgb(0xCC, 0xC9, 0xC0);
pub const PRIMARY_CREAME_9: Color32 = Color32::from_rgb(0xE6, 0xE2, 0xD8);

pub const PRIMARY_SMOKE: Color32 = Color32::from_rgb(0x0D, 0x0D, 0x0D);

pub const BLACK_BLACK_600: Color32 = Color32::from_rgb(0x1F, 0x1F, 0x1F);

pub const BORDER_GREY: Color32 = Color32::from_rgb(0x20, 0x20, 0x20); // white * 0.05

pub const PRIMARY_ONYX: Color32 = Color32::from_rgb(0x17, 0x16, 0x15);
pub const PRIMARY_ONYX_5: Color32 = Color32::from_rgb(0x97, 0x96, 0x96);
pub const PRIMARY_ONYX_6: Color32 = Color32::from_rgb(0x74, 0x73, 0x73);
pub const PRIMARY_ONYX_8: Color32 = Color32::from_rgb(0x45, 0x45, 0x44);
pub const PRIMARY_ONYX_9: Color32 = Color32::from_rgb(0x2E, 0x2D, 0x2C);

pub const TURQUOISE_DEFAULT: Color32 = Color32::from_rgb(0x69, 0xB3, 0xBF);
pub const SLATE_DEFAULT: Color32 = Color32::from_rgb(0x7F, 0x70, 0xFF);
pub const PUMPKIN_DEFAULT: Color32 = Color32::from_rgb(0xFF, 0x6F, 0x1E);
pub const YOLK_DEFAULT: Color32 = Color32::from_rgb(0xFE, 0xC5, 0x04);
pub const PEACH_DEFAULT: Color32 = Color32::from_rgb(0xFF, 0xD7, 0xB3);
pub const REDDISH_DEFAULT: Color32 = Color32::from_rgb(0xE9, 0x4B, 0x14);
pub const HYPERBLUE_DEFAULT: Color32 = Color32::from_rgb(0x14, 0x5F, 0xCF);
pub const MINT_DEFAULT: Color32 = Color32::from_rgb(0x88, 0xDE, 0x9F);
pub const BONE_DEFAULT: Color32 = Color32::from_rgb(0xE4, 0xD9, 0xC3);

pub const TURQUOISE_40: Color32 = Color32::from_rgb(0x38, 0x55, 0x59);
pub const SLATE_40: Color32 = Color32::from_rgb(0x41, 0x3A, 0x73);
pub const PUMPKIN_40: Color32 = Color32::from_rgb(0x74, 0x3A, 0x1A);
pub const YOLK_40: Color32 = Color32::from_rgb(0x73, 0x5C, 0x0D);
pub const PEACH_40: Color32 = Color32::from_rgb(0x74, 0x63, 0x54);
pub const REDDISH_40: Color32 = Color32::from_rgb(0x6B, 0x2B, 0x15);
pub const HYPERBLUE_40: Color32 = Color32::from_rgb(0x16, 0x33, 0x60);
pub const MINT_40: Color32 = Color32::from_rgb(0x43, 0x66, 0x4C);

pub const SURFACE_SECONDARY: Color32 = Color32::from_rgb(0x16, 0x16, 0x16);

pub fn get_color_by_index_solid(index: usize) -> Color32 {
    let colors = [
        TURQUOISE_DEFAULT,
        SLATE_DEFAULT,
        PUMPKIN_DEFAULT,
        YOLK_DEFAULT,
        PEACH_DEFAULT,
        REDDISH_DEFAULT,
        HYPERBLUE_DEFAULT,
        MINT_DEFAULT,
    ];
    colors[index % colors.len()]
}

pub fn get_color_by_index_all(index: usize) -> Color32 {
    let colors_dark = [
        Color32::from_rgb(0x6A, 0x9B, 0xA5),
        Color32::from_rgb(0x72, 0x57, 0xB3),
        Color32::from_rgb(0xC6, 0x6B, 0x42),
        Color32::from_rgb(0xD6, 0xA4, 0x36),
        Color32::from_rgb(0xB6, 0x52, 0x2D),
        Color32::from_rgb(0x42, 0x69, 0xA8),
        Color32::from_rgb(0x7E, 0xBF, 0x7F),
        Color32::from_rgb(0x5A, 0x82, 0x90),
        Color32::from_rgb(0x64, 0x48, 0xA8),
        Color32::from_rgb(0xB0, 0x62, 0x3E),
        Color32::from_rgb(0xC9, 0x93, 0x36),
        Color32::from_rgb(0x9D, 0x4B, 0x2A),
        Color32::from_rgb(0x34, 0x5C, 0x91),
        Color32::from_rgb(0x6A, 0x9C, 0x6A),
        Color32::from_rgb(0x50, 0x6A, 0x7A),
        Color32::from_rgb(0x5C, 0x3C, 0xA5),
        Color32::from_rgb(0x9C, 0x58, 0x3A),
        Color32::from_rgb(0xB6, 0x88, 0x30),
        Color32::from_rgb(0x82, 0x36, 0x21),
        Color32::from_rgb(0x2C, 0x4B, 0x7B),
        Color32::from_rgb(0x5F, 0x8A, 0x5F),
        Color32::from_rgb(0x43, 0x5B, 0x70),
        Color32::from_rgb(0x4A, 0x31, 0x92),
        Color32::from_rgb(0x85, 0x48, 0x2F),
        Color32::from_rgb(0x9B, 0x71, 0x30),
        Color32::from_rgb(0x6B, 0x27, 0x17),
        Color32::from_rgb(0x24, 0x3C, 0x66),
        Color32::from_rgb(0x48, 0x6D, 0x48),
        Color32::from_rgb(0x37, 0x48, 0x5A),
        Color32::from_rgb(0x3A, 0x25, 0x86),
        Color32::from_rgb(0x6C, 0x3C, 0x28),
        Color32::from_rgb(0x84, 0x5D, 0x28),
        Color32::from_rgb(0x52, 0x1F, 0x14),
        Color32::from_rgb(0x1C, 0x2A, 0x58),
        Color32::from_rgb(0x3D, 0x68, 0x3D),
        Color32::from_rgb(0x2C, 0x3A, 0x48),
        Color32::from_rgb(0x33, 0x23, 0x7A),
        Color32::from_rgb(0x53, 0x2D, 0x1C),
        Color32::from_rgb(0x63, 0x46, 0x22),
        Color32::from_rgb(0x39, 0x19, 0x0E),
        Color32::from_rgb(0x14, 0x1A, 0x3A),
        Color32::from_rgb(0x2A, 0x4B, 0x2A),
        Color32::from_rgb(0x1E, 0x29, 0x36),
        Color32::from_rgb(0x25, 0x18, 0x64),
        Color32::from_rgb(0x45, 0x24, 0x1A),
        Color32::from_rgb(0x57, 0x35, 0x1E),
        Color32::from_rgb(0x2C, 0x11, 0x0A),
        Color32::from_rgb(0x10, 0x16, 0x2E),
        Color32::from_rgb(0x22, 0x3C, 0x22),
        Color32::from_rgb(0x16, 0x20, 0x2C),
    ];
    colors_dark[index % colors_dark.len()]
}

pub fn with_opacity(color: Color32, opacity: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), (255.0 * opacity) as u8)
}

pub trait EColor {
    fn into_color32(self) -> Color32;
}

impl EColor for Color {
    fn into_color32(self) -> Color32 {
        MINT_DEFAULT
        // TODO: Enable when colors will be used
        // Color32::from_rgb(
        //     (255.0 * self.r) as u8,
        //     (255.0 * self.g) as u8,
        //     (255.0 * self.b) as u8,
        // )
    }
}

pub trait ColorExt {
    fn into_bevy(self) -> ::bevy::prelude::Color;
    fn opacity(self, opacity: f32) -> Self;
}

impl ColorExt for Color32 {
    fn into_bevy(self) -> ::bevy::prelude::Color {
        let [r, g, b, a] = self.to_srgba_unmultiplied().map(|c| c as f32 / 255.0);
        ::bevy::prelude::Color::srgba(r, g, b, a)
    }

    fn opacity(self, opacity: f32) -> Self {
        with_opacity(self, opacity)
    }
}

pub mod bevy {
    use bevy::prelude::Color;
    pub const RED: Color = Color::srgb(0.91, 0.29, 0.08);
    pub const GREEN: Color = Color::srgb(0.53, 0.87, 0.62);
    pub const BLUE: Color = Color::srgb(0.08, 0.38, 0.82);
    pub const GREY_900: Color = Color::srgb(0.2, 0.2, 0.2);
}
