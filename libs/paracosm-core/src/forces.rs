use nalgebra::Vector3;

use crate::{BodyPos, Force, Mass};

pub fn earth_gravity(Mass(m): Mass) -> Force {
    Force(Vector3::new(0.0, m * -9.81, 0.0))
}

pub fn gravity(body_mass: f64, body_pos: Vector3<f64>) -> impl Fn(Mass, BodyPos) -> Force {
    move |Mass(m), BodyPos(pos)| {
        const G: f64 = 6.649e-11;
        let r = body_pos - pos.pos;
        let mu = G * body_mass;
        Force(r * (mu * m / r.norm().powi(3)))
    }
}
