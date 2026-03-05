//! Configuration for the ViewCube widget

use bevy::prelude::*;
use std::f32::consts::{FRAC_PI_2, PI};

use super::components::FaceDirection;

// ============================================================================
// Coordinate Systems
// ============================================================================

/// Supported coordinate systems
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoordinateSystem {
    /// East-North-Up: X=East, Y=Up, Z=North (aviation/robotics)
    #[default]
    ENU,
}

/// Axis definition with label, direction, and color
#[derive(Clone, Debug)]
pub struct AxisDefinition {
    pub positive_label: &'static str,
    pub negative_label: &'static str,
    pub direction: Vec3,
    pub color: Color,
    pub color_dim: Color,
}

impl CoordinateSystem {
    /// Get the three axis definitions for this coordinate system
    pub fn get_axes(&self) -> [AxisDefinition; 3] {
        match self {
            // ENU mapped to Bevy's Y-up coordinate system:
            // Bevy +X (right) → East (red)
            // Bevy +Y (up)    → Up (blue)
            // Bevy +Z (fwd)   → North (green)
            // See: https://docs.elodin.systems/reference/coords/
            CoordinateSystem::ENU => [
                AxisDefinition {
                    positive_label: "E",
                    negative_label: "W",
                    direction: Vec3::X,
                    color: Color::srgb(0.9, 0.2, 0.2), // Red
                    color_dim: Color::srgb(0.6, 0.15, 0.15),
                },
                AxisDefinition {
                    positive_label: "U",
                    negative_label: "D",
                    direction: Vec3::Y,
                    color: Color::srgb(0.2, 0.4, 0.9), // Blue (Up)
                    color_dim: Color::srgb(0.15, 0.3, 0.6),
                },
                AxisDefinition {
                    positive_label: "N",
                    negative_label: "S",
                    direction: Vec3::Z,
                    color: Color::srgb(0.2, 0.8, 0.2), // Green (North)
                    color_dim: Color::srgb(0.15, 0.5, 0.15),
                },
            ],
        }
    }

    /// Get face labels for all 6 faces.
    /// Positive directions use bright axis colors; opposite directions use dim axis colors.
    pub fn get_face_labels(&self, face_offset: f32) -> Vec<FaceLabelConfig> {
        let axes = self.get_axes();
        let mut labels = Vec::new();

        for axis in &axes {
            // The synced cube applies a Y-PI correction to match camera conventions.
            // Mirror X-only for label placement so E/W appear on expected visual faces.
            let visual_direction = Self::face_label_visual_direction(axis.direction);

            // Positive face label.
            labels.push(FaceLabelConfig {
                text: axis.positive_label,
                position: visual_direction * face_offset,
                rotation: Self::get_rotation_for_direction(visual_direction),
                color: axis.color,
                direction: Self::direction_to_face(axis.direction),
            });

            // Opposite face label.
            labels.push(FaceLabelConfig {
                text: axis.negative_label,
                position: -visual_direction * face_offset,
                rotation: Self::get_rotation_for_direction(-visual_direction),
                color: axis.color_dim,
                direction: Self::direction_to_face(-axis.direction),
            });
        }
        labels
    }

    fn face_label_visual_direction(dir: Vec3) -> Vec3 {
        if dir.x.abs() > 0.9 { -dir } else { dir }
    }

    fn get_rotation_for_direction(dir: Vec3) -> Quat {
        if dir.x.abs() > 0.9 {
            if dir.x > 0.0 {
                Quat::from_rotation_y(FRAC_PI_2)
            } else {
                Quat::from_rotation_y(-FRAC_PI_2)
            }
        } else if dir.y.abs() > 0.9 {
            if dir.y > 0.0 {
                Quat::from_rotation_x(-FRAC_PI_2)
            } else {
                Quat::from_rotation_x(FRAC_PI_2)
            }
        } else if dir.z > 0.0 {
            Quat::IDENTITY
        } else {
            Quat::from_rotation_y(PI)
        }
    }

    fn direction_to_face(dir: Vec3) -> FaceDirection {
        if dir.x > 0.5 {
            FaceDirection::East
        } else if dir.x < -0.5 {
            FaceDirection::West
        } else if dir.y > 0.5 {
            FaceDirection::Up
        } else if dir.y < -0.5 {
            FaceDirection::Down
        } else if dir.z > 0.5 {
            FaceDirection::North
        } else {
            FaceDirection::South
        }
    }
}

/// Configuration for a face label
#[derive(Clone, Debug)]
pub struct FaceLabelConfig {
    pub text: &'static str,
    pub position: Vec3,
    pub rotation: Quat,
    pub color: Color,
    pub direction: FaceDirection,
}

// ============================================================================
// Plugin Configuration
// ============================================================================

/// Main configuration resource for ViewCube
#[derive(Resource, Clone)]
pub struct ViewCubeConfig {
    pub system: CoordinateSystem,
    pub scale: f32,
    pub rotation_increment: f32,
    pub camera_distance: f32,
    /// When true, the cube mirrors the main camera orientation.
    pub sync_with_camera: bool,
    /// Optional extra rotation applied when syncing the cube to the camera.
    /// This is applied after the system-specific correction.
    pub axis_correction: Quat,
    /// Render layer used by the dedicated ViewCube overlay camera.
    pub render_layer: u8,
}

impl Default for ViewCubeConfig {
    fn default() -> Self {
        Self {
            system: CoordinateSystem::ENU,
            scale: 0.6,
            rotation_increment: 15.0 * PI / 180.0,
            camera_distance: 2.5,
            sync_with_camera: true,
            axis_correction: Quat::IDENTITY,
            render_layer: 31,
        }
    }
}

impl ViewCubeConfig {
    /// Single supported mode: editor overlay integration.
    pub fn editor_mode() -> Self {
        Self::default()
    }

    /// Base correction for Bevy's camera forward (-Z) vs cube face orientation (+Z).
    pub fn system_axis_correction(system: CoordinateSystem) -> Quat {
        match system {
            CoordinateSystem::ENU => Quat::from_rotation_y(PI),
        }
    }

    /// Full correction applied when syncing the cube to the camera.
    pub fn effective_axis_correction(&self) -> Quat {
        Self::system_axis_correction(self.system) * self.axis_correction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label_by_text<'a>(labels: &'a [FaceLabelConfig], text: &str) -> &'a FaceLabelConfig {
        labels
            .iter()
            .find(|label| label.text == text)
            .expect("label should exist")
    }

    #[test]
    fn enu_face_labels_keep_semantic_direction_and_flip_east_west_visual_side() {
        let labels = CoordinateSystem::ENU.get_face_labels(1.0);

        let east = label_by_text(&labels, "E");
        let west = label_by_text(&labels, "W");
        let up = label_by_text(&labels, "U");
        let south = label_by_text(&labels, "S");

        assert_eq!(east.direction, FaceDirection::East);
        assert_eq!(west.direction, FaceDirection::West);
        assert_eq!(east.position, Vec3::NEG_X);
        assert_eq!(west.position, Vec3::X);
        assert_eq!(up.position, Vec3::Y);
        assert_eq!(south.position, Vec3::NEG_Z);
    }
}
