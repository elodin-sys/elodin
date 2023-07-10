use nalgebra::Vector3;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Force(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Torque(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Mass(pub f64);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Pos(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Time(pub f64);
