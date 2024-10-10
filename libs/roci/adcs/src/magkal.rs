//! A UKF based magnotmeter calibration estimator
//!
//! This filter is based on the paper
//! Real-Time Attitude-Independent !Three-Axis Magnetometer Calibration by Crassidis, Lai, and Harman
//! Located here https://ntrs.nasa.gov/api/citations/20040031762/downloads/20040031762.pdf and for a
//! slightly different version here https://www.acsu.buffalo.edu/~johnc/mag_cal05.pdf
//!
//! It estimates scale factors, bias, and nonorthogonality corrections

use nox::{tensor, ArrayRepr, Matrix, Matrix3, OwnedRepr, Vector};

use crate::ukf::{self, MerweConfig};

pub fn measure(
    state: Vector<f64, 9, ArrayRepr>,
    z: Vector<f64, 3, ArrayRepr>,
) -> Vector<f64, 1, ArrayRepr> {
    let b: Vector<f64, 3, _> = state.fixed_slice(&[0]);
    let d: Vector<f64, 6, _> = state.fixed_slice(&[3]);
    let d = d.into_buf();
    let d = tensor![[d[0], d[1], d[2]], [d[1], d[3], d[4]], [d[2], d[4], d[5]],];
    let z_t: Matrix<f64, 1, 3, ArrayRepr> = z.reshape();
    let d_eye = Matrix3::eye() + d;
    let c = d_eye.dot(&b);
    let e = 2.0 * d + d.dot(&d);
    (-1.0 * z_t).dot(&e).dot(&z) + 2.0 * z_t.dot(&c) - b.norm_squared()
}

pub struct State<R: OwnedRepr>(ukf::State<9, 1, 19, R>);

impl State<ArrayRepr> {
    pub fn new() -> Self {
        let q = Matrix::from_diag(tensor![
            50.0, 50.0, 50.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
        ]);
        let state = ukf::State {
            x_hat: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            covar: q,
            prop_covar: Matrix::zeros(),
            noise_covar: tensor![[1.0e-3]],
            config: MerweConfig::new(0.1, 2.0, -3.0),
        };
        Self(state)
    }
}

impl State<ArrayRepr> {
    pub fn update(
        self,
        z: Vector<f64, 3, ArrayRepr>,
        b: Vector<f64, 3, ArrayRepr>,
    ) -> Result<Self, nox::Error> {
        let Self(state) = self;
        let state = state.update(
            (z.norm_squared() - b.norm_squared()).reshape(),
            |x| x,
            move |x, _| measure(x, z),
        )?;
        Ok(Self(state))
    }

    pub fn h_hat(&self) -> Vector<f64, 3, ArrayRepr> {
        self.0.x_hat.fixed_slice(&[0])
    }

    pub fn d_hat(&self) -> Matrix3<f64, ArrayRepr> {
        let d: Vector<f64, 6, _> = self.0.x_hat.fixed_slice(&[3]);
        let d = d.into_buf();
        tensor![[d[0], d[1], d[2]], [d[1], d[3], d[4]], [d[2], d[4], d[5]],]
    }
}

impl Default for State<ArrayRepr> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nox::Quaternion;

    use crate::tests::test_mag_readings;

    use super::*;

    #[test]
    fn test_calibrate() {
        let readings = test_mag_readings();
        let mut state = State::default();
        for reading in readings.clone() {
            state = state
                .update(reading, tensor![1.0, 0.0, 0.0] * 31.99)
                .unwrap();
        }

        let expected = crate::tests::normalized_cal_mag_readings();

        let out = readings.map(|reading| {
            ((Matrix3::eye() + state.d_hat()).dot(&reading) - state.h_hat()).normalize()
        });

        for (o, e) in out.iter().zip(expected.iter()) {
            let cos = o.dot(e).into_buf();
            assert_relative_eq!(cos, 1.0, epsilon = 6e-3);
        }
    }

    #[test]
    fn test_offset_only() {
        let mut readings = vec![];
        let offset = tensor![5.0, 1.0, 3.0];
        for alpha in 0..8 {
            let alpha = alpha as f64 * std::f64::consts::PI * 2.0 / 8.0;
            for beta in 0..8 {
                let beta = beta as f64 * std::f64::consts::PI * 2.0 / 8.0;
                for theta in 0..8 {
                    let theta = theta as f64 * std::f64::consts::PI * 2.0 / 8.0;
                    let rot = Quaternion::from_euler(tensor![alpha, beta, theta]);
                    let z = rot * tensor![20.0, 0.0, 0.0] + offset;
                    readings.push(z);
                }
            }
        }

        let mut state = State::default();
        for reading in readings.clone() {
            state = state
                .update(reading, tensor![1.0, 0.0, 0.0] * 20.0)
                .unwrap();
        }
        assert_relative_eq!(state.h_hat(), offset, epsilon = 1e-5);
        assert_relative_eq!(state.d_hat(), Matrix::zeros(), epsilon = 1e-6);
    }
}
