use bevy::prelude::*;
use elodin_macros::Component as Comp;
use nalgebra::{matrix, Matrix3};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Comp)]
#[conduit(id = "31;mass")]
pub struct Mass(pub f64);

#[derive(Debug, Clone, Copy, PartialEq, Component, Comp)]
#[conduit(id = "31;intertia")]
pub struct Inertia(pub Matrix3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component, Comp)]
#[conduit(id = "31;inverse_inertia")]
pub struct InverseInertia(pub Matrix3<f64>);

impl Inertia {
    pub fn solid_box(width: f64, height: f64, depth: f64, mass: f64) -> Inertia {
        let h = height.powi(2);
        let w = width.powi(2);
        let d = depth.powi(2);
        let k = mass / 12.0;
        Inertia(matrix![
            k * (h + d), 0.0, 0.0;
            0.0, k * ( w + d ), 0.0;
            0.0, 0.0, k * (w + h)
        ])
    }

    pub fn sphere(radius: f64, mass: f64) -> Inertia {
        Inertia(Matrix3::from_diagonal_element(
            2.0 / 5.0 * mass * radius.powi(2),
        ))
    }
}
