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
            // NED mapped to Bevy's Y-up coordinate system:
            // Bevy +Z (fwd)   → North (red, 1st axis)
            // Bevy +X (right) → East (green, 2nd axis)
            // Bevy -Y (down)  → Down (blue, 3rd axis)
            CoordinateSystem::NED => [
                AxisDefinition {
                    positive_label: "N",
                    negative_label: "S",
                    direction: Vec3::Z,
                    color: Color::srgb(0.9, 0.2, 0.2), // Red (North)
                    color_dim: Color::srgb(0.6, 0.15, 0.15),
                },
                AxisDefinition {
                    positive_label: "E",
                    negative_label: "W",
                    direction: Vec3::X,
                    color: Color::srgb(0.2, 0.8, 0.2), // Green (East)
                    color_dim: Color::srgb(0.15, 0.5, 0.15),
                },
                AxisDefinition {
                    positive_label: "D",
                    negative_label: "U",
                    direction: Vec3::NEG_Y,
                    color: Color::srgb(0.2, 0.4, 0.9), // Blue (Down)
                    color_dim: Color::srgb(0.15, 0.3, 0.6),
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
    /// When true, the plugin automatically handles camera rotation.
    /// The target camera must have the `ViewCubeTargetCamera` component.
    /// When false, only events are emitted for manual handling.
    pub auto_rotate: bool,
    /// When true, the cube syncs its rotation with the main camera.
    /// Use this for overlay/gizmo mode where the cube shows world orientation.
    /// When false, the cube stays fixed (for standalone demo mode).
    pub sync_with_camera: bool,
    /// When true, renders the ViewCube as an overlay with its own camera.
    /// The ViewCube appears fixed in the top-right corner.
    /// When false, the ViewCube is part of the main scene.
    pub use_overlay: bool,
    /// Size of the overlay viewport in pixels (width and height).
    pub overlay_size: u32,
    /// Margin from the edge of the window in pixels.
    pub overlay_margin: f32,
    /// Render layer for overlay mode (should not conflict with other layers).
    pub render_layer: u8,
    /// Editor mode: use LookToTrigger from bevy_editor_cam instead of direct animation.
    /// This integrates better with the editor's camera system.
    pub use_look_to_trigger: bool,
    /// Editor mode: adapt viewport position based on main camera's viewport.
    /// Required for split views where gizmo should stay in each viewport's corner.
    pub follow_main_viewport: bool,
    /// Skip registering viewport positioning system.
    /// Use when integrating with existing viewport system (like navigation_gizmo).
    pub skip_viewport_system: bool,
}

impl Default for ViewCubeConfig {
    fn default() -> Self {
        Self {
            system: CoordinateSystem::ENU,
            scale: 0.95,
            rotation_increment: 15.0 * PI / 180.0,
            camera_distance: 3.5,
            auto_rotate: true,
            sync_with_camera: false,
            use_overlay: false,
            overlay_size: 160,
            overlay_margin: 8.0,
            render_layer: 31,
            use_look_to_trigger: false,
            follow_main_viewport: false,
            skip_viewport_system: false,
        }
    }
}

impl ViewCubeConfig {
    /// Configuration preset for editor integration.
    /// Uses LookToTrigger from bevy_editor_cam to properly rotate the camera
    /// around its anchor point, respecting EditorCam internal state.
    pub fn editor_mode() -> Self {
        Self {
            use_overlay: true,
            sync_with_camera: true,
            auto_rotate: false,
            use_look_to_trigger: true,
            follow_main_viewport: false, // Use existing set_camera_viewport
            skip_viewport_system: true,  // Don't add our viewport system
            overlay_size: 128,           // Match navigation_gizmo's side_length
            camera_distance: 2.5,        // Overlay camera distance from cube model
            scale: 0.6,                  // Cube model scale in overlay
            ..Default::default()
        }
    }
}
