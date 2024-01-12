use bevy::prelude::*;
use elodin_macros::Component as Comp;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct FixedBody(pub bool);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Picked(pub bool);
#[derive(Component, Clone, Comp, Serialize, Deserialize, Debug)]
#[conduit(postcard)]
pub struct TraceAnchor {
    pub anchor: Vector3<f64>,
}
