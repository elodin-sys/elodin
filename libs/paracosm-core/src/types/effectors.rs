use std::sync::Arc;

use crate::builder::{XpbdEffector, XpbdSensor};
use bevy::prelude::Component;

#[derive(Component, Default, Clone)]
pub struct Effectors(pub Vec<Arc<dyn XpbdEffector + Send + Sync>>);

#[derive(Component, Default, Clone)]
pub struct Sensors(pub Vec<Arc<dyn XpbdSensor + Send + Sync>>);
