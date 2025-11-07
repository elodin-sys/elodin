use nalgebra::{Matrix3, Matrix6, SMatrix, SVector, UnitQuaternion, Vector3, Vector6, vector};

pub fn calculate_covariance(sigma_g: Vector3<f64>, sigma_b: Vector3<f64>, dt: f64) -> Matrix6<f64> {
    let variance_g: Matrix3<f64> = Matrix3::from_diagonal(&(sigma_g.component_mul(&sigma_g) * dt));
    let variance_b = Matrix3::from_diagonal(&(sigma_b.component_mul(&sigma_b) * dt));
    let q_00 = variance_g + variance_b * dt * (dt / 3.0);
    let q_01 = variance_b * (dt / 2.0);
    let q_10 = q_01;
    let q_11 = variance_b;

    let mut q = Matrix6::zeros();
    q.fixed_view_mut::<3, 3>(0, 0).copy_from(&q_00);
    q.fixed_view_mut::<3, 3>(0, 3).copy_from(&q_01);
    q.fixed_view_mut::<3, 3>(3, 0).copy_from(&q_10);
    q.fixed_view_mut::<3, 3>(3, 3).copy_from(&q_11);
    q
}

fn propagate_quaternion(
    q: UnitQuaternion<f64>,
    omega: Vector3<f64>,
    dt: f64,
) -> UnitQuaternion<f64> {
    let omega_norm = omega.norm();
    let c = (0.5 * omega_norm * dt).cos();
    let s = (0.5 * omega_norm * dt).sin() / omega_norm;
    let omega_s = s * omega;
    let x = omega_s[0];
    let y = omega_s[1];
    let z = omega_s[2];
    let big_omega = SMatrix::<f64, 4, 4>::from_row_slice(&[
        c, z, -y, x, -z, c, x, y, y, -x, c, z, -x, -y, -z, c,
    ]);

    if omega_norm > 1e-5 {
        let q_vec = SVector::<f64, 4>::from_row_slice(&[q.i, q.j, q.k, q.w]);
        let new_q_vec = big_omega * q_vec;
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            new_q_vec[3],
            new_q_vec[0],
            new_q_vec[1],
            new_q_vec[2],
        ))
    } else {
        q
    }
}

fn propagate_state_covariance(
    big_p: Matrix6<f64>,
    omega: Vector3<f64>,
    yqy: Matrix6<f64>,
    dt: f64,
) -> Matrix6<f64> {
    let omega_norm_squared = omega.norm_squared();
    let omega_norm = omega.norm();
    let s = (omega_norm * dt).sin();
    let c = (omega_norm * dt).cos();
    let p = s / omega_norm;
    let q = (1.0 - c) / omega_norm_squared;
    let r = (omega_norm * dt - s) / (omega_norm_squared * omega_norm);

    let omega_cross = Matrix3::new(
        0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0,
    );
    let omega_cross_square = omega_cross * omega_cross;
    let eye = Matrix3::identity();

    let phi_00: Matrix3<f64> = if omega_norm > 1e-5 {
        eye - omega_cross * p + omega_cross_square * q
    } else {
        eye
    };
    let phi_01: Matrix3<f64> = if omega_norm > 1e-5 {
        omega_cross * q - eye * dt - omega_cross_square * r
    } else {
        Matrix3::identity() * -dt
    };
    let phi_10: Matrix3<f64> = Matrix3::zeros();
    let phi_11: Matrix3<f64> = Matrix3::identity();

    let mut phi = Matrix6::zeros();
    phi.fixed_view_mut::<3, 3>(0, 0).copy_from(&phi_00);
    phi.fixed_view_mut::<3, 3>(0, 3).copy_from(&phi_01);
    phi.fixed_view_mut::<3, 3>(3, 0).copy_from(&phi_10);
    phi.fixed_view_mut::<3, 3>(3, 3).copy_from(&phi_11);

    phi * big_p * phi.transpose() + yqy
}

#[derive(Debug)]
pub struct State {
    pub q_hat: UnitQuaternion<f64>,
    pub b_hat: Vector3<f64>,
    pub p: Matrix6<f64>,
    pub omega: Vector3<f64>,
    pub yqy: Matrix6<f64>,
    pub dt: f64,
}

impl State {
    pub fn new(sigma_g: Vector3<f64>, sigma_b: Vector3<f64>, dt: f64) -> Self {
        let y: Matrix6<f64> = Matrix6::from_diagonal(&vector![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        let q = calculate_covariance(sigma_g, sigma_b, dt);
        let yqy = y * q * y.transpose();
        Self {
            q_hat: UnitQuaternion::identity(),
            b_hat: Vector3::zeros(),
            p: Matrix6::identity(),
            omega: Vector3::zeros(),
            yqy,
            dt,
        }
    }

    pub fn estimate_attitude<const N: usize>(
        self,
        measured_bodys: [Vector3<f64>; N],
        references: [Vector3<f64>; N],
        sigma_r: [f64; N],
    ) -> Self {
        let Self {
            q_hat,
            b_hat,
            p,
            omega,
            yqy,
            dt,
        } = self;
        let omega = omega - b_hat;
        let q_hat = propagate_quaternion(q_hat, omega, dt);
        let mut p = propagate_state_covariance(p, omega, yqy, dt);
        let mut delta_x_hat: Vector6<f64> = Vector6::zeros();

        for ((reference, measured_body), sigma) in references
            .into_iter()
            .zip(measured_bodys.into_iter())
            .zip(sigma_r.into_iter())
        {
            let var_r = Matrix3::<f64>::identity() * sigma.powi(2);
            let body_r = q_hat.inverse() * reference;
            let e = measured_body - body_r;
            let skew_sym = Matrix3::new(
                0.0, -body_r[2], body_r[1], body_r[2], 0.0, -body_r[0], -body_r[1], body_r[0], 0.0,
            );

            let mut h = SMatrix::<f64, 3, 6>::zeros();
            h.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew_sym);
            // h.fixed_view_mut::<3, 3>(0, 3) is already zeros

            let h_trans = h.transpose();
            let s = (h * p * h_trans + var_r)
                .try_inverse()
                .unwrap_or_else(Matrix3::identity);
            let k = p * h_trans * s;
            p = (Matrix6::<f64>::identity() - k * h) * p;
            let d: Vector3<f64> = h * delta_x_hat;
            delta_x_hat += k * (e - d);
        }

        let delta_alpha: Vector3<f64> = delta_x_hat.fixed_rows::<3>(0).into_owned();
        let delta_beta: Vector3<f64> = delta_x_hat.fixed_rows::<3>(3).into_owned();

        // Integrate the quaternion update
        let delta_q = UnitQuaternion::from_scaled_axis(delta_alpha);
        let q_hat = q_hat * delta_q;
        let b_hat = b_hat + delta_beta;

        Self {
            q_hat,
            b_hat,
            p,
            omega,
            yqy,
            dt,
        }
    }
}

impl State {
    /// Checks if the estimation parameters are all finite (i.e not NaN or infinite)
    pub fn is_non_finite(&self) -> bool {
        !self.q_hat.coords.iter().all(|x| x.is_finite())
            || !self.b_hat.iter().all(|x| x.is_finite())
            || !self.p.iter().all(|x: &f64| x.is_finite())
            || !self.omega.iter().all(|x: &f64| x.is_finite())
    }

    /// Resets the state if any parameter is non-finite
    pub fn reset_if_invalid(&mut self) {
        if self.is_non_finite() {
            self.q_hat = UnitQuaternion::identity();
            self.b_hat = Vector3::zeros();
            self.p = Matrix6::identity();
            self.omega = Vector3::zeros();
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{matrix, vector};

    use super::*;

    #[test]
    fn test_calculate_covariance() {
        #[rustfmt::skip]
        let expected_q = matrix![
            8.33352623e-07, 0.00000000e+00, 0.00000000e+00, 3.47222222e-09, 0.00000000e+00, 0.00000000e+00;
            0.00000000e+00, 8.33352623e-07, 0.00000000e+00, 0.00000000e+00, 3.47222222e-09, 0.00000000e+00;
            0.00000000e+00, 0.00000000e+00, 8.33352623e-07, 0.00000000e+00, 0.00000000e+00, 3.47222222e-09;
            3.47222222e-09, 0.00000000e+00, 0.00000000e+00, 8.33333333e-07, 0.00000000e+00, 0.00000000e+00;
            0.00000000e+00, 3.47222222e-09, 0.00000000e+00, 0.00000000e+00, 8.33333333e-07, 0.00000000e+00;
            0.00000000e+00, 0.00000000e+00, 3.47222222e-09, 0.00000000e+00, 0.00000000e+00, 8.33333333e-07;
        ];
        let q = calculate_covariance(
            vector![0.01, 0.01, 0.01],
            vector![0.01, 0.01, 0.01],
            1.0 / 120.0,
        );
        assert_relative_eq!(q, expected_q, epsilon = 1e-5);
    }

    #[test]
    fn test_propagate_state_covariance() {
        // from: cube-sat.py
        let yqy = matrix![
            8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0, 0.0;
            0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0;
            0.0, 0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09;
            -3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0, 0.0;
            0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0;
            0.0, 0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07;
        ];
        let out = propagate_state_covariance(
            Matrix6::identity(),
            vector![1.0, 0.0, 0.0],
            yqy,
            1.0 / 120.0,
        );
        #[rustfmt::skip]
        let expected = matrix![
            1.00007028e+00,  0.00000000e+00,  0.00000000e+00, -8.33333681e-03,  0.00000000e+00,  0.00000000e+00;
            0.00000000e+00,  1.00007028e+00,  5.55768107e-19,  0.00000000e+00, -8.33324036e-03, -3.47220213e-05;
            0.00000000e+00,  5.55768107e-19,  1.00007028e+00,  0.00000000e+00,  3.47220213e-05, -8.33324036e-03;
           -8.33333681e-03,  0.00000000e+00,  0.00000000e+00,  1.00000083e+00,  0.00000000e+00,  0.00000000e+00;
            0.00000000e+00, -8.33324036e-03,  3.47220213e-05,  0.00000000e+00,  1.00000083e+00,  0.00000000e+00;
            0.00000000e+00, -3.47220213e-05, -8.33324036e-03,  0.00000000e+00,  0.00000000e+00,  1.00000083e+00;
        ];
        assert_relative_eq!(out, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_propagate_quaternion() {
        let q = propagate_quaternion(
            UnitQuaternion::identity(),
            vector![1.0, 0.0, 0.0],
            1.0 / 60.0,
        );
        let expected = nalgebra::Quaternion::new(0.99996528, 0.00833324, 0., 0.);
        assert_relative_eq!(q.coords, expected.coords, epsilon = 1e-5);
    }

    #[test]
    fn test_mekf() {
        #[rustfmt::skip]
        let expected_yqy = matrix![
            8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0, 0.0;
            0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0;
            0.0, 0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09;
            -3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0, 0.0;
            0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0;
            0.0, 0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07;
        ];
        let ref_a = vector![0.0, 1.0, 0.0].normalize();
        let ref_b = vector![1.0, 0.0, 0.0].normalize();
        let mut q: UnitQuaternion<f64> =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f64::consts::PI / 4.0);
        // MEKF uses body-to-inertial convention, so use q.inverse() to get inertial-to-body
        let body_a = (q.inverse() * ref_a).normalize();
        let body_b = (q.inverse() * ref_b).normalize();
        let dt = 1.0 / 120.0;
        let mut state = State::new(vector![0.01, 0.01, 0.01], vector![0.01, 0.01, 0.01], dt);
        assert_relative_eq!(state.yqy, expected_yqy, epsilon = 1e-4);
        for _ in 0..180 {
            state.omega = Vector3::zeros();
            state = state.estimate_attitude([body_a, body_b], [ref_a, ref_b], [0.03, 0.03]);
        }
        // Use angle_to for quaternion comparison to handle q and -q representing the same rotation
        assert!(
            state.q_hat.angle_to(&q) < 1e-3,
            "Quaternion angle difference: {}",
            state.q_hat.angle_to(&q)
        );
        for _ in 0..120 {
            q = q * UnitQuaternion::from_scaled_axis(vector![1.0 / 120.0, 0.0, 0.0]);
            let body_a = (q.inverse() * ref_a).normalize();
            let body_b = (q.inverse() * ref_b).normalize();
            state.omega = vector![1.0, 0.0, 0.0];
            state = state.estimate_attitude([body_a, body_b], [ref_a, ref_b], [0.03, 0.03]);
        }
        assert_relative_eq!(state.b_hat, vector![0.0, 0.0, 0.0], epsilon = 1e-2);
        // Use angle_to for quaternion comparison
        assert!(
            state.q_hat.angle_to(&q) < 1e-3,
            "Quaternion angle difference: {}",
            state.q_hat.angle_to(&q)
        );
    }
}
