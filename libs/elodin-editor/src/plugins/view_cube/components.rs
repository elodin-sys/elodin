//! Components for the ViewCube widget

use bevy::prelude::*;
use std::collections::HashMap;

// ============================================================================
// Marker Components
// ============================================================================

/// Marker for the ViewCube root entity
#[derive(Component)]
pub struct ViewCubeRoot;

/// Marker for the GLB cube mesh root
#[derive(Component)]
pub struct ViewCubeMeshRoot;

/// Marker for entities that have been set up by the plugin
#[derive(Component)]
pub struct ViewCubeSetup;

/// Marker for an active right-button drag interaction on a ViewCube root.
#[derive(Component)]
pub struct ViewCubeDragging;

/// Links a ViewCube to the main camera it should follow/control
#[derive(Component)]
pub struct ViewCubeLink {
    pub main_camera: Entity,
}

/// Marker for the ViewCube's dedicated camera (used in overlay mode)
#[derive(Component)]
pub struct ViewCubeCamera;

/// Stores the render layer for this ViewCube instance
#[derive(Component)]
pub struct ViewCubeRenderLayer(pub usize);

// ============================================================================
// Cube Elements
// ============================================================================

/// Identifies what type of cube element this is
#[derive(Component, Clone, Debug)]
pub enum CubeElement {
    Face(FaceDirection),
    Edge(EdgeDirection),
    Corner(CornerPosition),
}

/// Face directions based on Bevy coordinates (X=right, Y=up, Z=forward)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceDirection {
    East,  // +X (Right)
    West,  // -X (Left)
    North, // +Z (Front)
    South, // -Z (Back)
    Up,    // +Y (Top)
    Down,  // -Y (Bottom)
}

impl FaceDirection {
    /// Convert to camera look direction (where camera should be to see this face)
    pub fn to_look_direction(self) -> Vec3 {
        match self {
            FaceDirection::East => Vec3::X,
            FaceDirection::West => Vec3::NEG_X,
            FaceDirection::North => Vec3::Z,
            FaceDirection::South => Vec3::NEG_Z,
            FaceDirection::Up => Vec3::Y,
            FaceDirection::Down => Vec3::NEG_Y,
        }
    }

    /// Opposite cube face.
    pub fn opposite(self) -> Self {
        match self {
            FaceDirection::East => FaceDirection::West,
            FaceDirection::West => FaceDirection::East,
            FaceDirection::North => FaceDirection::South,
            FaceDirection::South => FaceDirection::North,
            FaceDirection::Up => FaceDirection::Down,
            FaceDirection::Down => FaceDirection::Up,
        }
    }
}

/// Edge directions (between two faces)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeDirection {
    // X-axis edges (horizontal, front-back)
    XTopFront,
    XTopBack,
    XBottomFront,
    XBottomBack,
    // Y-axis edges (vertical)
    YFrontLeft,
    YFrontRight,
    YBackLeft,
    YBackRight,
    // Z-axis edges (horizontal, left-right)
    ZTopLeft,
    ZTopRight,
    ZBottomLeft,
    ZBottomRight,
}

impl EdgeDirection {
    /// The two faces touching this edge.
    pub fn adjacent_faces(self) -> (FaceDirection, FaceDirection) {
        match self {
            EdgeDirection::XTopFront => (FaceDirection::Up, FaceDirection::North),
            EdgeDirection::XTopBack => (FaceDirection::Up, FaceDirection::South),
            EdgeDirection::XBottomFront => (FaceDirection::Down, FaceDirection::North),
            EdgeDirection::XBottomBack => (FaceDirection::Down, FaceDirection::South),
            EdgeDirection::YFrontLeft => (FaceDirection::North, FaceDirection::West),
            EdgeDirection::YFrontRight => (FaceDirection::North, FaceDirection::East),
            EdgeDirection::YBackLeft => (FaceDirection::South, FaceDirection::West),
            EdgeDirection::YBackRight => (FaceDirection::South, FaceDirection::East),
            EdgeDirection::ZTopLeft => (FaceDirection::Up, FaceDirection::West),
            EdgeDirection::ZTopRight => (FaceDirection::Up, FaceDirection::East),
            EdgeDirection::ZBottomLeft => (FaceDirection::Down, FaceDirection::West),
            EdgeDirection::ZBottomRight => (FaceDirection::Down, FaceDirection::East),
        }
    }
}

/// Corner positions
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CornerPosition {
    TopFrontLeft,
    TopFrontRight,
    TopBackLeft,
    TopBackRight,
    BottomFrontLeft,
    BottomFrontRight,
    BottomBackLeft,
    BottomBackRight,
}

impl CornerPosition {
    /// Convert to camera look direction (isometric view)
    pub fn to_look_direction(self) -> Vec3 {
        match self {
            CornerPosition::TopFrontLeft => Vec3::new(-1.0, 1.0, 1.0).normalize(),
            CornerPosition::TopFrontRight => Vec3::new(1.0, 1.0, 1.0).normalize(),
            CornerPosition::TopBackLeft => Vec3::new(-1.0, 1.0, -1.0).normalize(),
            CornerPosition::TopBackRight => Vec3::new(1.0, 1.0, -1.0).normalize(),
            CornerPosition::BottomFrontLeft => Vec3::new(-1.0, -1.0, 1.0).normalize(),
            CornerPosition::BottomFrontRight => Vec3::new(1.0, -1.0, 1.0).normalize(),
            CornerPosition::BottomBackLeft => Vec3::new(-1.0, -1.0, -1.0).normalize(),
            CornerPosition::BottomBackRight => Vec3::new(1.0, -1.0, -1.0).normalize(),
        }
    }
}

// ============================================================================
// Rotation Arrows
// ============================================================================

/// Direction for rotation arrows
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationArrow {
    Left,
    Right,
    Up,
    Down,
    RollLeft,
    RollRight,
}

/// Action buttons rendered in the ViewCube overlay.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewportActionButton {
    Reset,
    ZoomOut,
    ZoomIn,
}

/// Axis label rendered as a billboard, with metadata for screen-space spacing from the axis.
#[derive(Component, Clone, Copy)]
pub struct AxisLabelBillboard {
    pub axis_direction: Vec3,
    pub base_position: Vec3,
}

// ============================================================================
// Resources
// ============================================================================

/// Tracks the currently hovered element
#[derive(Resource, Default)]
pub struct HoveredElement {
    pub entity: Option<Entity>,
    pub entities: Vec<Entity>,
}

/// Stores original material colors for restoration after hover
#[derive(Resource, Default)]
pub struct OriginalMaterials {
    pub colors: HashMap<Entity, Color>,
}
