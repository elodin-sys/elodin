use crate::spatial::*;
use bevy::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct BodyPos(pub SpatialPos);
