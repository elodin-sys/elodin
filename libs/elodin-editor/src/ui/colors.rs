use conduit::well_known::Color;
use egui::Color32;

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
pub const REDDISH_40: Color32 = Color32::from_rgb(0x74, 0x63, 0x54);
pub const HYPERBLUE_40: Color32 = Color32::from_rgb(0x6B, 0x2B, 0x15);
pub const MINT_40: Color32 = Color32::from_rgb(0x43, 0x66, 0x4C);

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
    let colors = [
        TURQUOISE_DEFAULT,
        SLATE_DEFAULT,
        PUMPKIN_DEFAULT,
        YOLK_DEFAULT,
        PEACH_DEFAULT,
        REDDISH_DEFAULT,
        HYPERBLUE_DEFAULT,
        MINT_DEFAULT,
        TURQUOISE_40,
        SLATE_40,
        PUMPKIN_40,
        YOLK_40,
        PEACH_40,
        REDDISH_40,
        HYPERBLUE_40,
        MINT_40,
    ];
    colors[index % colors.len()]
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
}

impl ColorExt for Color32 {
    fn into_bevy(self) -> ::bevy::prelude::Color {
        let [r, g, b, a] = self.to_srgba_unmultiplied().map(|c| c as f32 / 255.0);
        ::bevy::prelude::Color::srgba(r, g, b, a)
    }
}

pub mod bevy {
    use bevy::prelude::Color;
    pub const RED: Color = Color::srgb(0.91, 0.29, 0.08);
    pub const GREEN: Color = Color::srgb(0.53, 0.87, 0.62);
    pub const BLUE: Color = Color::srgb(0.08, 0.38, 0.82);
    pub const GREY_900: Color = Color::srgb(0.2, 0.2, 0.2);
}
