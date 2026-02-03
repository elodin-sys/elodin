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
    /// North-East-Down: X=North, Y=East, Z=Down (aerospace/navigation)
    NED,
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
            CoordinateSystem::ENU => [
                AxisDefinition {
                    positive_label: "East",
                    negative_label: "West",
                    direction: Vec3::X,
                    color: Color::srgb(0.9, 0.2, 0.2),
                    color_dim: Color::srgb(0.6, 0.15, 0.15),
                },
                AxisDefinition {
                    positive_label: "Up",
                    negative_label: "Down",
                    direction: Vec3::Y,
                    color: Color::srgb(0.2, 0.8, 0.2),
                    color_dim: Color::srgb(0.15, 0.5, 0.15),
                },
                AxisDefinition {
                    positive_label: "North",
                    negative_label: "South",
                    direction: Vec3::Z,
                    color: Color::srgb(0.2, 0.4, 0.9),
                    color_dim: Color::srgb(0.15, 0.3, 0.6),
                },
            ],
            CoordinateSystem::NED => [
                AxisDefinition {
                    positive_label: "North",
                    negative_label: "South",
                    direction: Vec3::X,
                    color: Color::srgb(0.9, 0.2, 0.2),
                    color_dim: Color::srgb(0.6, 0.15, 0.15),
                },
                AxisDefinition {
                    positive_label: "East",
                    negative_label: "West",
                    direction: Vec3::Y,
                    color: Color::srgb(0.2, 0.8, 0.2),
                    color_dim: Color::srgb(0.15, 0.5, 0.15),
                },
                AxisDefinition {
                    positive_label: "Down",
                    negative_label: "Up",
                    direction: Vec3::Z,
                    color: Color::srgb(0.2, 0.4, 0.9),
                    color_dim: Color::srgb(0.15, 0.3, 0.6),
                },
            ],
        }
    }

    /// Get all 6 face labels with their properties
    pub fn get_face_labels(&self, face_offset: f32) -> Vec<FaceLabelConfig> {
        let axes = self.get_axes();
        let mut labels = Vec::new();

        for axis in &axes {
            labels.push(FaceLabelConfig {
                text: axis.positive_label,
                position: axis.direction * face_offset,
                rotation: Self::get_rotation_for_direction(axis.direction),
                color: axis.color,
                direction: Self::direction_to_face(axis.direction),
            });
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
    /// When true, the plugin automatically handles camera rotation.
    /// The target camera must have the `ViewCubeTargetCamera` component.
    /// When false, only events are emitted for manual handling.
    pub auto_rotate: bool,
}

impl Default for ViewCubeConfig {
    fn default() -> Self {
        Self {
            system: CoordinateSystem::ENU,
            scale: 0.95,
            rotation_increment: 15.0 * PI / 180.0,
            camera_distance: 4.5,
            auto_rotate: true,
        }
    }
}
