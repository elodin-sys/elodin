//! Theme colors for the ViewCube widget
//!
//! Automatically adapts to the editor's active color scheme (dark/light mode).

use bevy::prelude::*;
use bevy_egui::egui;

use super::components::CubeElement;
use crate::ui::colors;

/// Colors for ViewCube elements
pub struct ViewCubeColors {
    pub face_normal: Color,
    pub face_hover: Color,
    pub edge_normal: Color,
    pub edge_hover: Color,
    pub corner_normal: Color,
    pub corner_hover: Color,
    pub arrow_normal: Color,
    pub arrow_hover: Color,
    pub highlight_emissive: LinearRgba,
}

impl Default for ViewCubeColors {
    /// Automatically selects dark or light theme based on the editor's active color scheme.
    fn default() -> Self {
        let selection = colors::current_selection();
        if selection.mode == "light" {
            Self::light()
        } else {
            Self::dark()
        }
    }
}

impl ViewCubeColors {
    fn from_color32(color: egui::Color32) -> Color {
        let [r, g, b, a] = color.to_srgba_unmultiplied();
        Color::srgba(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        )
    }

    fn arrow_palette() -> (Color, Color) {
        let scheme = colors::get_scheme();
        (
            Self::from_color32(scheme.icon_primary),
            Self::from_color32(scheme.highlight),
        )
    }

    /// Dark theme colors - light ivory cube on dark background, CAD-style
    pub fn dark() -> Self {
        let (arrow_normal, arrow_hover) = Self::arrow_palette();

        Self {
            // Very bright neutral fill for maximum readability in dark mode.
            face_normal: Color::srgba(0.98, 0.98, 0.98, 1.0),
            face_hover: Color::srgb(1.0, 0.85, 0.3),
            edge_normal: Color::srgba(0.9, 0.9, 0.9, 1.0),
            edge_hover: Color::srgb(0.3, 0.76, 1.0),
            corner_normal: Color::srgba(0.94, 0.94, 0.94, 1.0),
            corner_hover: Color::srgb(1.0, 0.55, 0.18),
            // Arrows
            arrow_normal,
            arrow_hover,
            highlight_emissive: LinearRgba::new(0.5, 0.4, 0.15, 1.0),
        }
    }

    /// Light theme colors - soft cream cube with defined structure
    pub fn light() -> Self {
        let (arrow_normal, arrow_hover) = Self::arrow_palette();

        Self {
            // Keep a clear, light gray in light mode while preserving contrast with the bg.
            face_normal: Color::srgba(0.9, 0.9, 0.9, 1.0),
            face_hover: Color::srgb(0.2, 0.55, 1.0),
            edge_normal: Color::srgba(0.78, 0.78, 0.78, 1.0),
            edge_hover: Color::srgb(0.0, 0.7, 0.54),
            corner_normal: Color::srgba(0.84, 0.84, 0.84, 1.0),
            corner_hover: Color::srgb(0.92, 0.36, 0.22),
            // Arrows
            arrow_normal,
            arrow_hover,
            highlight_emissive: LinearRgba::new(0.1, 0.25, 0.5, 1.0),
        }
    }

    /// Get the normal color for a cube element
    pub fn get_element_color(&self, element: &CubeElement) -> Color {
        match element {
            CubeElement::Face(_) => self.face_normal,
            CubeElement::Edge(_) => self.edge_normal,
            CubeElement::Corner(_) => self.corner_normal,
        }
    }

    /// Get the hover color for a cube element
    pub fn get_element_hover(&self, element: &CubeElement) -> Color {
        match element {
            CubeElement::Face(_) => self.face_hover,
            CubeElement::Edge(_) => self.edge_hover,
            CubeElement::Corner(_) => self.corner_hover,
        }
    }
}
