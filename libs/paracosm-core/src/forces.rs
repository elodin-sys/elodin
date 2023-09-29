use nalgebra::Vector3;

use crate::{Force, Mass, Pos};

pub fn earth_gravity(Mass(m): Mass) -> Force {
    Force(Vector3::new(0.0, m * -9.81, 0.0))
}

pub fn gravity(body_mass: f64, body_pos: Vector3<f64>) -> impl Fn(Mass, Pos) -> Force {
    move |Mass(m), Pos(pos)| {
        const G: f64 = 6.649e-11;
        let r = body_pos - pos;
        let mu = G * body_mass;
        Force(r * (mu * m / r.norm().powi(3)))
    }
}

#[cfg(test)]
mod tests {
    use crate::six_dof::SixDof;

    use super::*;
    use approx::assert_relative_eq;
    use plotters::prelude::*;

    #[test]
    fn test_perfect_simple_orbit() {
        let area = BitMapBackend::new("out.png", (4096, 4096)).into_drawing_area();
        area.fill(&WHITE).unwrap();
        let x_axis = (-1.5..1.5).step(0.1);
        let y_axis = (-1.5..1.5).step(0.1);
        let mut chart = ChartBuilder::on(&area)
            .caption("Grav Test", ("sans", 20))
            .build_cartesian_2d(x_axis, y_axis)
            .unwrap();

        let grav = gravity(1.0 / 6.649e-11, Vector3::zeros());
        let mut six_dof = SixDof::default()
            .pos(Vector3::new(1.0, 0., 0.))
            .vel(Vector3::new(0.0, 1.0, 0.0))
            .mass(1.0)
            .sim()
            .effector(grav);
        let mut time = 0.0;
        let mut points = vec![];
        let dt = 0.01;
        while time <= 2.0 * std::f64::consts::PI {
            six_dof.tick(dt);
            points.push((six_dof.state.pos.x, six_dof.state.pos.y));
            time += dt;
        }
        chart.draw_series(LineSeries::new(points, &BLACK)).unwrap();
        area.present().unwrap();
        assert_relative_eq!(six_dof.state.pos, Vector3::new(1.0, 0., 0.), epsilon = 0.01)
    }
}
