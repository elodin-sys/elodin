use crate::spatial::*;
use bevy::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct JointPos(pub GeneralizedPos);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct JointVel(pub GeneralizedMotion);
#[derive(Debug, Clone, Copy, PartialEq, Component, Default)]
pub struct JointForce(pub GeneralizedForce);
#[derive(Debug, Clone, PartialEq, Component, Default)]
pub struct JointAccel(pub GeneralizedMotion);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct BiasForce(pub SpatialForce);
