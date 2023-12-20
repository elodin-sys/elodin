use std::sync::{Arc, Mutex};

use crate::builder::{XpbdEffector, XpbdSensor};
use bevy::prelude::Component;
use bevy_ecs::{entity::Entity, system::Resource, world::World};

#[derive(Component, Default, Clone)]
pub struct Effectors(pub Vec<Arc<dyn XpbdEffector + Send + Sync>>);

#[cfg(feature = "nox")]
pub type XlaEffector = dyn for<'a> Fn(&'a mut World, Entity, &'a nox::Client) + Send + Sync;

#[cfg(feature = "nox")]
#[derive(Component, Default, Clone)]
pub struct XlaEffectors(pub Vec<Arc<XlaEffector>>);

#[cfg(feature = "nox")]
#[derive(Resource, Clone)]
pub struct XlaClient(pub Arc<Mutex<nox::Client>>);

#[derive(Component, Default, Clone)]
pub struct Sensors(pub Vec<Arc<dyn XpbdSensor + Send + Sync>>);
