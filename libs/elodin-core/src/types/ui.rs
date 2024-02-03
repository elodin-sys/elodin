use bevy::prelude::*;
pub use elodin_conduit::well_known::TraceAnchor;

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct FixedBody(pub bool);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Picked(pub bool);
