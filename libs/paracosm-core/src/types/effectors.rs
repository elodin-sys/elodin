use crate::builder::{XpbdEffector, XpbdSensor};
use bevy::prelude::Component;

#[derive(Component, Default)]
pub struct Effectors(pub Vec<Box<dyn XpbdEffector + Send + Sync>>);

#[derive(Component, Default)]
pub struct Sensors(pub Vec<Box<dyn XpbdSensor + Send + Sync>>);
