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

    /// Select the face (among the two adjacent faces) that is most facing the camera.
    /// `camera_dir_world` must be the direction from subject -> camera.
    pub fn active_frame_face(
        self,
        camera_dir_world: Vec3,
    ) -> (FaceDirection, FaceDirection, f32, f32) {
        let (face_a, face_b) = self.adjacent_faces();
        let dot_a = face_a.to_look_direction().dot(camera_dir_world);
        let dot_b = face_b.to_look_direction().dot(camera_dir_world);
        if dot_a >= dot_b {
            (face_a, face_b, dot_a, dot_b)
        } else {
            (face_b, face_a, dot_b, dot_a)
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
    pub entities: Vec<Entity>,
}

/// Stores original material colors for restoration after hover
#[derive(Resource, Default)]
pub struct OriginalMaterials {
    pub colors: HashMap<Entity, Color>,
}
