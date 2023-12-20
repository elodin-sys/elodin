use crate::spatial::{SpatialMotion, SpatialPos, SpatialTransform};
use bevy::prelude::*;
use elodin_macros::Component as Comp;

#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref, Comp)]
pub struct WorldPos(pub SpatialPos);
#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref, Comp)]
pub struct WorldAnchorPos(pub SpatialTransform);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Comp)]
pub struct WorldVel(pub SpatialMotion);
#[derive(Debug, Clone, Copy, PartialEq, Component, Comp)]
pub struct WorldAccel(pub SpatialMotion);
