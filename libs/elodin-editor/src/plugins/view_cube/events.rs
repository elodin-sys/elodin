//! Events emitted by the ViewCube widget

use bevy::prelude::*;

use super::components::{CornerPosition, EdgeDirection, FaceDirection, RotationArrow};

/// Events emitted when ViewCube elements are clicked
#[derive(Message, Clone, Debug)]
pub enum ViewCubeEvent {
    /// A face was clicked - rotate camera to face this direction
    FaceClicked(FaceDirection),
    /// An edge was clicked - rotate camera to face this edge
    EdgeClicked(EdgeDirection),
    /// A corner was clicked - rotate camera to face this corner
    CornerClicked(CornerPosition),
    /// A rotation arrow was clicked - apply incremental rotation
    ArrowClicked(RotationArrow),
}
