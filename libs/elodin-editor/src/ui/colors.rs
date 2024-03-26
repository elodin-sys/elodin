use bevy_egui::egui::Color32;
use conduit::well_known::Color;

//pub const HYPER_RED: Color32 = Color32::from_rgb(0xEE, 0x3A, 0x43);
pub const BLACK: Color32 = Color32::from_rgb(0x1F, 0x1F, 0x1F);
pub const STONE_950: Color32 = Color32::from_rgb(0x0D, 0x0D, 0x0D);
#[allow(dead_code)]
pub const INTERFACE_BACKGROUND_BLACK: Color32 = Color32::from_rgb(0x17, 0x16, 0x15);

pub const WHITE: Color32 = Color32::WHITE;
pub const CREMA: Color32 = Color32::from_rgb(255, 251, 240);

pub const GREY_OPACITY_500: Color32 = Color32::from_rgb(0x99, 0x99, 0x99);

// pub const NEUTRAL_900: Color32 = Color32::from_rgb(0x17, 0x16, 0x15);
pub const GREEN_300: Color32 = Color32::from_rgb(0x88, 0xDE, 0x9F);
pub const ORANGE_50: Color32 = Color32::from_rgb(0xFF, 0xFB, 0xF0);

pub const BORDER_GREY: Color32 = Color32::from_rgb(0x20, 0x20, 0x20); // white * 0.05

pub const ONYX: Color32 = Color32::from_rgb(0x3d, 0x3d, 0x3d); // NOTE: this color does not have a name in figma
pub const ONYX_8: Color32 = Color32::from_rgb(0x45, 0x45, 0x44);

// TODO: Colors used by EPlot, needs to be replaced and removed
pub const EPLOT_STONE_950: Color32 = Color32::from_rgb(0x0D, 0x0D, 0x0D);
pub const EPLOT_ZINC_800: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
pub const EPLOT_NEUTRAL_900: Color32 = Color32::from_rgb(0x17, 0x16, 0x15);
pub const EPLOT_GREEN_300: Color32 = Color32::from_rgb(0x88, 0xDE, 0x9F);
pub const EPLOT_ORANGE_50: Color32 = Color32::from_rgb(0xFF, 0xFB, 0xF0);

pub const TURQUOISE_DEFAULT: Color32 = Color32::from_rgb(0x69, 0xB3, 0xBF);
pub const SLATE_DEFAULT: Color32 = Color32::from_rgb(0x7F, 0x70, 0xFF);
pub const PUMPKIN_DEFAULT: Color32 = Color32::from_rgb(0xFF, 0x6F, 0x1E);
pub const YOLK_DEFAULT: Color32 = Color32::from_rgb(0xFE, 0xC5, 0x04);
pub const PEACH_DEFAULT: Color32 = Color32::from_rgb(0xFF, 0xD7, 0xB3);
pub const REDDISH_DEFAULT: Color32 = Color32::from_rgb(0xE9, 0x4B, 0x14);
pub const HYPERBOLE_DEFAULT: Color32 = Color32::from_rgb(0x14, 0x5F, 0xCF);
pub const MINT_DEFAULT: Color32 = Color32::from_rgb(0x88, 0xDE, 0x9F);

pub fn get_color_by_index(index: usize) -> Color32 {
    let colors = [
        TURQUOISE_DEFAULT,
        SLATE_DEFAULT,
        PUMPKIN_DEFAULT,
        YOLK_DEFAULT,
        PEACH_DEFAULT,
        REDDISH_DEFAULT,
        HYPERBOLE_DEFAULT,
        MINT_DEFAULT,
    ];

    *colors.get(index).unwrap_or(&WHITE)
}

pub fn with_opacity(color: Color32, opacity: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), (255.0 * opacity) as u8)
}

pub trait EColor {
    fn into_color32(self) -> Color32;
}

impl EColor for Color {
    fn into_color32(self) -> Color32 {
        GREEN_300
        // TODO: Enable when colors will be used
        // Color32::from_rgb(
        //     (255.0 * self.r) as u8,
        //     (255.0 * self.g) as u8,
        //     (255.0 * self.b) as u8,
        // )
    }
}

pub mod bevy {
    use bevy::prelude::Color;
    pub const RED: Color = Color::rgb(0.91, 0.29, 0.08);
    pub const GREEN: Color = Color::rgb(0.53, 0.87, 0.62);
    pub const BLUE: Color = Color::rgb(0.08, 0.38, 0.82);
    pub const GREY_900: Color = Color::rgb(0.2, 0.2, 0.2);
}
