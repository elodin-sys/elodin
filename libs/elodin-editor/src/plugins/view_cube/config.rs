//! Configuration for the ViewCube widget

use bevy::prelude::*;
use bevy_geo_frames::GeoFrame;
use std::f32::consts::{FRAC_PI_2, PI};

use super::components::FaceDirection;

// ============================================================================
// Coordinate Systems
// ============================================================================

/// Supported coordinate systems
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoordinateSystem(pub GeoFrame);

impl Default for CoordinateSystem {
    fn default() -> Self {
        CoordinateSystem(GeoFrame::ENU)
    }
}

/// Axis definition with label, direction, and color
#[derive(Clone, Debug)]
pub struct AxisDefinition {
    pub positive_label: &'static str,
    pub negative_label: &'static str,
    /// Positive direction in Bevy coordinate system.
    pub direction: Vec3,
    pub color: Color,
    pub color_dim: Color,
}

impl CoordinateSystem {
    /// Get the three axis definitions for XYZ in this coordinate system.
    pub fn get_axes(&self) -> [AxisDefinition; 3] {
        // Directions are frame-local cube axes. `GeoRotation::absolute`
        // on the mesh root is what places them in Bevy.
        match self.0 {
            GeoFrame::ENU =>
            // East / North / Up on cube +X / +Y / +Z.
            {
                [
                    AxisDefinition {
                        positive_label: "E",
                        negative_label: "W",
                        direction: Vec3::X,
                        color: Color::srgb(0.9, 0.2, 0.2), // Red
                        color_dim: Color::srgb(0.6, 0.15, 0.15),
                    },
                    AxisDefinition {
                        positive_label: "N",
                        negative_label: "S",
                        direction: Vec3::Y,
                        color: Color::srgb(0.2, 0.8, 0.2), // Green
                        color_dim: Color::srgb(0.15, 0.5, 0.15),
                    },
                    AxisDefinition {
                        positive_label: "U",
                        negative_label: "D",
                        direction: Vec3::Z,
                        color: Color::srgb(0.2, 0.4, 0.9), // Blue
                        color_dim: Color::srgb(0.15, 0.3, 0.6),
                    },
                ]
            }
            GeoFrame::NED =>
            // North / East / Down on cube +X / +Y / +Z.
            {
                [
                    AxisDefinition {
                        positive_label: "N",
                        negative_label: "S",
                        direction: Vec3::X,
                        color: Color::srgb(0.9, 0.2, 0.2), // Red
                        color_dim: Color::srgb(0.6, 0.15, 0.15),
                    },
                    AxisDefinition {
                        positive_label: "E",
                        negative_label: "W",
                        direction: Vec3::Y,
                        color: Color::srgb(0.2, 0.8, 0.2), // Green
                        color_dim: Color::srgb(0.15, 0.5, 0.15),
                    },
                    AxisDefinition {
                        positive_label: "D",
                        negative_label: "U",
                        direction: Vec3::Z,
                        color: Color::srgb(0.2, 0.4, 0.9), // Blue
                        color_dim: Color::srgb(0.15, 0.3, 0.6),
                    },
                ]
            }
            GeoFrame::ECEF =>
            // ECEF +X / +Y / +Z on cube +X / +Y / +Z.
            {
                [
                    AxisDefinition {
                        positive_label: "+X",
                        negative_label: "-X",
                        direction: Vec3::X,
                        color: Color::srgb(0.9, 0.2, 0.2), // Red
                        color_dim: Color::srgb(0.6, 0.15, 0.15),
                    },
                    AxisDefinition {
                        positive_label: "+Y",
                        negative_label: "-Y",
                        direction: Vec3::Y,
                        color: Color::srgb(0.2, 0.8, 0.2), // Green
                        color_dim: Color::srgb(0.15, 0.5, 0.15),
                    },
                    AxisDefinition {
                        positive_label: "+Z",
                        negative_label: "-Z",
                        direction: Vec3::Z,
                        color: Color::srgb(0.2, 0.4, 0.9), // Blue
                        color_dim: Color::srgb(0.15, 0.3, 0.6),
                    },
                ]
            }
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

            // Positive face label.
            labels.push(FaceLabelConfig {
                text: axis.positive_label,
                position: axis.direction * face_offset,
                rotation: Self::get_rotation_for_direction(axis.direction),
                color: axis.color,
                direction: Self::direction_to_face(axis.direction),
            });

            // Opposite face label.
            labels.push(FaceLabelConfig {
                text: axis.negative_label,
                position: -axis.direction * face_offset,
                rotation: Self::get_rotation_for_direction(-axis.direction),
                color: axis.color_dim,
                direction: Self::direction_to_face(-axis.direction),
            });
        }
        labels
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
}

impl Default for ViewCubeConfig {
    fn default() -> Self {
        Self {
            system: CoordinateSystem(GeoFrame::ENU),
            scale: 0.6,
            rotation_increment: 15.0 * PI / 180.0,
            camera_distance: 2.5,
            sync_with_camera: true,
            axis_correction: Quat::IDENTITY,
        }
    }
}

impl ViewCubeConfig {
    /// Single supported mode: editor overlay integration.
    pub fn editor_mode() -> Self {
        Self {
            rotation_increment: 5.0 * PI / 180.0,
            ..Self::default()
        }
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
    fn enu_face_labels_sit_on_frame_local_axes() {
        let labels = CoordinateSystem(GeoFrame::ENU).get_face_labels(1.0);

        let east = label_by_text(&labels, "E");
        let west = label_by_text(&labels, "W");
        let north = label_by_text(&labels, "N");
        let up = label_by_text(&labels, "U");

        assert_eq!(east.direction, FaceDirection::East);
        assert_eq!(west.direction, FaceDirection::West);
        assert_eq!(east.position, Vec3::X);
        assert_eq!(west.position, Vec3::NEG_X);
        assert_eq!(north.position, Vec3::Y);
        assert_eq!(up.position, Vec3::Z);
    }

    #[test]
    fn editor_mode_uses_fine_rotation_increment() {
        assert!(
            (ViewCubeConfig::editor_mode()
                .rotation_increment
                .to_degrees()
                - 5.0)
                .abs()
                < 1e-5
        );
        assert!((ViewCubeConfig::default().rotation_increment.to_degrees() - 15.0).abs() < 1e-5);
    }
}
