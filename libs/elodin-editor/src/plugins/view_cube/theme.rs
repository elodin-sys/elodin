//! Theme colors for the ViewCube widget
//!
//! Automatically adapts to the editor's active color scheme (dark/light mode).

use bevy::prelude::*;

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
    /// Dark theme colors - light ivory cube on dark background, CAD-style
    pub fn dark() -> Self {
        Self {
            // Light ivory faces - bright and readable on dark bg
            face_normal: Color::srgba(0.92, 0.89, 0.82, 0.85),
            face_hover: Color::srgb(1.0, 0.85, 0.3),
            // Warm bronze edges - clearly distinct from faces
            edge_normal: Color::srgba(0.72, 0.62, 0.45, 0.9),
            edge_hover: Color::srgb(0.3, 0.76, 1.0),
            // Bright gold corners - accent points
            corner_normal: Color::srgba(0.85, 0.72, 0.4, 0.95),
            corner_hover: Color::srgb(1.0, 0.55, 0.18),
            // Arrows
            arrow_normal: Color::srgba(0.85, 0.8, 0.7, 0.6),
            arrow_hover: Color::srgba(1.0, 0.9, 0.4, 0.95),
            highlight_emissive: LinearRgba::new(0.5, 0.4, 0.15, 1.0),
        }
    }

    /// Light theme colors - soft cream cube with defined structure
    pub fn light() -> Self {
        Self {
            // Very bright ivory faces. Keep high alpha to avoid scene bleed-through.
            face_normal: Color::srgba(1.0, 0.99, 0.965, 0.96),
            face_hover: Color::srgb(0.2, 0.55, 1.0),
            // Light warm edges
            edge_normal: Color::srgba(0.9, 0.82, 0.68, 0.94),
            edge_hover: Color::srgb(0.0, 0.7, 0.54),
            // Bright corner accents
            corner_normal: Color::srgba(0.98, 0.89, 0.72, 0.97),
            corner_hover: Color::srgb(0.92, 0.36, 0.22),
            // Arrows
            arrow_normal: Color::srgba(0.74, 0.66, 0.52, 0.72),
            arrow_hover: Color::srgba(0.15, 0.45, 0.95, 0.95),
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
