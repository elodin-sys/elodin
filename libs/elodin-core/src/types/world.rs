use crate::spatial::{SpatialMotion, SpatialPos, SpatialTransform};
use bevy::prelude::*;

pub use elodin_conduit::well_known::WorldPos;
use nalgebra::UnitQuaternion;
#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref)]
pub struct WorldAnchorPos(pub SpatialTransform);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component)]
pub struct WorldVel(pub SpatialMotion);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct WorldAccel(pub SpatialMotion);

pub trait WorldPosExt {
    fn to_spatial(&self) -> SpatialPos;
}

impl WorldPosExt for WorldPos {
    fn to_spatial(&self) -> SpatialPos {
        SpatialPos::new(self.pos, UnitQuaternion::new_normalize(self.att))
    }
}
