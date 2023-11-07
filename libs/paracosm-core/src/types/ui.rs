use bevy::prelude::*;
use nalgebra::Vector3;

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct FixedBody(pub bool);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Picked(pub bool);
#[derive(Component)]
pub struct TraceEntity(Entity);
#[derive(Component, Clone)]
pub struct TraceAnchor {
    pub anchor: Vector3<f64>,
}
