use bevy::prelude::{Deref, DerefMut};
use bevy_ecs::{component::Component, event::Event};

#[derive(Event)]
pub struct SyncModels;
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Component, Clone, Copy, DerefMut, Deref, Debug)]
pub struct SyncedModel(pub bool);
