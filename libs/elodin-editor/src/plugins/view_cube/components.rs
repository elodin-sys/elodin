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
    /// Convert to camera look direction
    pub fn to_look_direction(self) -> Vec3 {
        match self {
            EdgeDirection::XTopFront => Vec3::new(0.0, 1.0, 1.0).normalize(),
            EdgeDirection::XTopBack => Vec3::new(0.0, 1.0, -1.0).normalize(),
            EdgeDirection::XBottomFront => Vec3::new(0.0, -1.0, 1.0).normalize(),
            EdgeDirection::XBottomBack => Vec3::new(0.0, -1.0, -1.0).normalize(),
            EdgeDirection::YFrontLeft => Vec3::new(-1.0, 0.0, 1.0).normalize(),
            EdgeDirection::YFrontRight => Vec3::new(1.0, 0.0, 1.0).normalize(),
            EdgeDirection::YBackLeft => Vec3::new(-1.0, 0.0, -1.0).normalize(),
            EdgeDirection::YBackRight => Vec3::new(1.0, 0.0, -1.0).normalize(),
            EdgeDirection::ZTopLeft => Vec3::new(-1.0, 1.0, 0.0).normalize(),
            EdgeDirection::ZTopRight => Vec3::new(1.0, 1.0, 0.0).normalize(),
            EdgeDirection::ZBottomLeft => Vec3::new(-1.0, -1.0, 0.0).normalize(),
            EdgeDirection::ZBottomRight => Vec3::new(1.0, -1.0, 0.0).normalize(),
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

// ============================================================================
// Resources
// ============================================================================

/// Tracks the currently hovered element
#[derive(Resource, Default)]
pub struct HoveredElement {
    pub entity: Option<Entity>,
}

/// Stores original material colors for restoration after hover
#[derive(Resource, Default)]
pub struct OriginalMaterials {
    pub colors: HashMap<Entity, Color>,
}
