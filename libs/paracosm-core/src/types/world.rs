use bevy::prelude::*;

use crate::spatial::{SpatialMotion, SpatialPos, SpatialTransform};

#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref)]
pub struct WorldPos(pub SpatialPos);
#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref)]
pub struct WorldAnchorPos(pub SpatialTransform);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component)]
pub struct WorldVel(pub SpatialMotion);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct WorldAccel(pub SpatialMotion);
