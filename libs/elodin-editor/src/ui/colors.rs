use bevy_egui::egui::Color32;
use conduit::well_known::Color;

//pub const HYPER_RED: Color32 = Color32::from_rgb(0xEE, 0x3A, 0x43);
// pub const STONE_900: Color32 = Color32::from_rgb(0x1F, 0x1F, 0x1F);
pub const STONE_950: Color32 = Color32::from_rgb(0x0D, 0x0D, 0x0D);
#[allow(dead_code)]
pub const INTERFACE_BACKGROUND_BLACK: Color32 = Color32::from_rgb(0x17, 0x16, 0x15);

pub const WHITE: Color32 = Color32::WHITE;

pub const GREY_OPACITY_500: Color32 = Color32::from_rgb(0x99, 0x99, 0x99);

// pub const NEUTRAL_900: Color32 = Color32::from_rgb(0x17, 0x16, 0x15);
pub const GREEN_300: Color32 = Color32::from_rgb(0x88, 0xDE, 0x9F);
pub const ORANGE_50: Color32 = Color32::from_rgb(0xFF, 0xFB, 0xF0);

pub const BORDER_GREY: Color32 = Color32::from_rgb(0x20, 0x20, 0x20); // white * 0.05

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
    pub const BLUE: Color = Color::rgb(0.0, 0.78, 1.0);
}
