//! Events emitted by the ViewCube widget

use bevy::prelude::*;

use super::components::{CornerPosition, EdgeDirection, FaceDirection, RotationArrow};

/// Events emitted when ViewCube elements are clicked.
/// Each event carries the `source` entity (the ViewCubeRoot) so that handlers
/// can identify which ViewCube instance generated the event in multi-viewport setups.
#[derive(Message, Clone, Debug)]
pub enum ViewCubeEvent {
    /// A face was clicked - rotate camera to face this direction
    FaceClicked {
        direction: FaceDirection,
        source: Entity,
    },
    /// An edge frame segment was clicked.
    /// `target_face` is pre-resolved from interaction rules and represents the
    /// face to snap to (or opposite face when clicking the active front frame).
    EdgeClicked {
        direction: EdgeDirection,
        target_face: FaceDirection,
        source: Entity,
    },
    /// A corner was clicked - rotate camera to face this corner
    CornerClicked {
        position: CornerPosition,
        source: Entity,
    },
    /// A rotation arrow was clicked - apply incremental rotation
    ArrowClicked {
        arrow: RotationArrow,
        source: Entity,
    },
}
