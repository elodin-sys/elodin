//! ViewCube click events.

use bevy::prelude::*;

use super::components::{
    CornerPosition, EdgeDirection, FaceDirection, RotationArrow, ViewportActionButton,
};

#[derive(Message, Clone, Debug)]
pub enum ViewCubeEvent {
    FaceClicked {
        direction: FaceDirection,
        source: Entity,
    },
    EdgeClicked {
        direction: EdgeDirection,
        target_face: FaceDirection,
        source: Entity,
    },
    CornerClicked {
        position: CornerPosition,
        local_direction: Vec3,
        source: Entity,
    },
    ArrowClicked {
        arrow: RotationArrow,
        source: Entity,
    },
    ViewportActionClicked {
        action: ViewportActionButton,
        source: Entity,
    },
}
