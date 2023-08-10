use bevy_ecs::{prelude::Component, system::Resource};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Default)]
pub struct Force(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Default)]
pub struct Torque(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component)]
pub struct Mass(pub f64);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component)]
pub struct Pos(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Resource)]
pub struct Time(pub f64);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component)]
pub struct Vel(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Att(pub UnitQuaternion<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct AngVel(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Inertia(pub Matrix3<f64>);
