use crate::spatial::{SpatialMotion, SpatialPos, SpatialTransform};
use bevy::prelude::*;
use paracosm_macros::Component as Comp;

#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref, Comp)]
#[conduit(prefix = "31")]
pub struct WorldPos(pub SpatialPos);
#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref, Comp)]
#[conduit(prefix = "31")]
pub struct WorldAnchorPos(pub SpatialTransform);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Comp)]
#[conduit(prefix = "31")]
pub struct WorldVel(pub SpatialMotion);
#[derive(Debug, Clone, Copy, PartialEq, Component, Comp)]
#[conduit(prefix = "31")]
pub struct WorldAccel(pub SpatialMotion);
