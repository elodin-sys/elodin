//! Theme colors for the ViewCube widget

use bevy::prelude::*;

use super::components::CubeElement;

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
    fn default() -> Self {
        Self::dark()
    }
}

impl ViewCubeColors {
    /// Dark theme colors
    pub fn dark() -> Self {
        Self {
            face_normal: Color::srgba(0.6, 0.6, 0.65, 0.4),
            face_hover: Color::srgb(1.0, 0.9, 0.2),
            edge_normal: Color::srgba(0.4, 0.4, 0.45, 0.85),
            edge_hover: Color::srgb(1.0, 0.9, 0.2),
            corner_normal: Color::srgba(0.5, 0.5, 0.55, 0.9),
            corner_hover: Color::srgb(1.0, 0.9, 0.2),
            arrow_normal: Color::srgba(1.0, 1.0, 1.0, 0.4),
            arrow_hover: Color::srgba(1.0, 1.0, 0.3, 0.9),
            highlight_emissive: LinearRgba::new(0.5, 0.45, 0.1, 1.0),
        }
    }

    /// Light theme colors
    pub fn light() -> Self {
        Self {
            face_normal: Color::srgba(0.7, 0.7, 0.75, 0.5),
            face_hover: Color::srgb(0.2, 0.6, 1.0),
            edge_normal: Color::srgba(0.5, 0.5, 0.55, 0.7),
            edge_hover: Color::srgb(0.2, 0.6, 1.0),
            corner_normal: Color::srgba(0.4, 0.4, 0.45, 0.8),
            corner_hover: Color::srgb(0.2, 0.6, 1.0),
            arrow_normal: Color::srgba(0.3, 0.3, 0.3, 0.5),
            arrow_hover: Color::srgba(0.1, 0.5, 0.9, 0.9),
            highlight_emissive: LinearRgba::new(0.1, 0.3, 0.5, 1.0),
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
