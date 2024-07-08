use nox::{
    tensor, ArrayBuf, ArrayRepr, Field, Matrix, Matrix3, Matrix3x6, Matrix6, Quaternion, Vector,
};

pub fn calculate_covariance(
    sigma_g: Vector<f64, 3, ArrayRepr>,
    sigma_b: Vector<f64, 3, ArrayRepr>,
    dt: f64,
) -> Matrix<f64, 6, 6, ArrayRepr> {
    let variance_g: Matrix3<f64, ArrayRepr> = Matrix::from_diag(sigma_g * sigma_g * dt);
    let variance_b = Matrix::from_diag(sigma_b * sigma_b * dt);
    let q_00 = variance_g + variance_b * dt * (dt / 3.0);
    let q_01 = variance_b * (dt / 2.0);
    let q_10 = q_01;
    let q_11 = variance_b;
    let q_0: Matrix3x6<f64, ArrayRepr> = Matrix::concat_in_dim([q_00, q_01], 1);
    let q_1: Matrix3x6<f64, ArrayRepr> = Matrix::concat_in_dim([q_10, q_11], 1);
    Matrix::concat_in_dim([q_0, q_1], 0)
}

fn propogate_quaternion(
    q: Quaternion<f64, ArrayRepr>,
    omega: Vector<f64, 3, ArrayRepr>,
    dt: f64,
) -> Quaternion<f64, ArrayRepr> {
    let omega_norm = omega.norm();
    let c = (0.5 * omega_norm * dt).cos();
    let s = (0.5 * omega_norm * dt).sin() / omega_norm;
    let omega_s = s * omega;
    let [x, y, z] = omega_s.parts();
    let big_omega = Matrix::from_scalars([c, z, -y, x, -z, c, x, y, y, -x, c, z, -x, -y, -z, c]);
    if omega_norm.inner().buf > 1e-5 {
        Quaternion(big_omega.dot(&q.0))
    } else {
        q
    }
}

fn propogate_state_covariance(
    big_p: Matrix6<f64, ArrayRepr>,
    omega: Vector<f64, 3, ArrayRepr>,
    yqy: Matrix6<f64, ArrayRepr>,
    dt: f64,
) -> Matrix6<f64, ArrayRepr> {
    let omega_norm_squared = omega.norm_squared();
    let omega_norm = omega.norm();
    let s = (omega_norm * dt).sin();
    let c = (omega_norm * dt).cos();
    let p = s / omega_norm;
    let one = f64::one();
    let q = (one - c) / omega_norm_squared;
    let r = (omega_norm * dt - s) / (omega_norm_squared * omega_norm);
    let omega_cross = omega.skew();
    let omega_cross_square = omega_cross.dot(&omega_cross);
    let eye = Matrix3::eye();
    let phi_00: Matrix3<f64, ArrayRepr> = if omega_norm.into_buf() > 1e-5 {
        eye - omega_cross * p + omega_cross_square * q
    } else {
        eye
    };
    let phi_01: Matrix3<f64, ArrayRepr> = if omega_norm.into_buf() > 1e-5 {
        omega_cross * q - eye * dt - omega_cross_square * r
    } else {
        Matrix::eye() * -dt
    };
    let phi_10: Matrix3<f64, ArrayRepr> = Matrix::zeros();
    let phi_11: Matrix3<f64, ArrayRepr> = Matrix::eye();

    let phi_0: Matrix3x6<f64, ArrayRepr> = Matrix::concat_in_dim([phi_00, phi_01], 1);
    let phi_1: Matrix3x6<f64, ArrayRepr> = Matrix::concat_in_dim([phi_10, phi_11], 1);
    let phi: Matrix6<f64, ArrayRepr> = Matrix::concat_in_dim([phi_0, phi_1], 0);

    phi.dot(&big_p).dot(&phi.transpose()) + yqy
}

#[derive(Debug)]
pub struct State {
    pub q_hat: Quaternion<f64, ArrayRepr>,
    pub b_hat: Vector<f64, 3, ArrayRepr>,
    pub p: Matrix<f64, 6, 6, ArrayRepr>,
    pub omega: Vector<f64, 3, ArrayRepr>,
    pub yqy: Matrix6<f64, ArrayRepr>,
    pub dt: f64,
}

impl State {
    pub fn new(
        sigma_g: Vector<f64, 3, ArrayRepr>,
        sigma_b: Vector<f64, 3, ArrayRepr>,
        dt: f64,
    ) -> Self {
        let y: Matrix6<f64, ArrayRepr> =
            Matrix::from_diag(tensor![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        let q = calculate_covariance(sigma_g, sigma_b, dt);
        let yqy = y.dot(&q).dot(&y.transpose());
        Self {
            q_hat: Default::default(),
            b_hat: Default::default(),
            p: Matrix::eye(),
            omega: Default::default(),
            yqy,
            dt,
        }
    }

    pub fn estimate_attitude<const N: usize>(
        self,
        measured_bodys: [Vector<f64, 3, ArrayRepr>; N],
        references: [Vector<f64, 3, ArrayRepr>; N],
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
        let q_hat = propogate_quaternion(q_hat, omega, dt);
        let mut p = propogate_state_covariance(p, omega, yqy, dt);
        let mut delta_x_hat: Vector<f64, 6, ArrayRepr> = Vector::zeros();
        for ((reference, measured_body), sigma) in references
            .into_iter()
            .zip(measured_bodys.into_iter())
            .zip(sigma_r.into_iter())
        {
            let var_r = Matrix::<f64, 3, 3, ArrayRepr>::eye() * sigma.powi(2);
            let body_r = q_hat.inverse() * reference;
            let e = measured_body - body_r;
            let skew_sym = body_r.skew();
            let h: Matrix<f64, 3, 6, ArrayRepr> =
                Matrix::concat_in_dim([skew_sym, Matrix::<f64, 3, 3, ArrayRepr>::zeros()], 1);
            let h_trans = h.transpose();
            let s = (h.dot(&p).dot(&h_trans) + var_r)
                .try_inverse()
                .unwrap_or_else(|_| Matrix::eye());
            let k = p.dot(&h_trans.dot(&s));
            p = (Matrix::<f64, 6, 6, ArrayRepr>::eye() - k.dot(&h)).dot(&p);
            let d: Vector<f64, 3, ArrayRepr> = h.dot(&delta_x_hat);
            delta_x_hat = delta_x_hat + k.dot(&(e - d));
        }
        let delta_alpha: Vector<f64, 3, ArrayRepr> = delta_x_hat.fixed_slice(&[0]);
        let delta_beta: Vector<f64, 3, ArrayRepr> = delta_x_hat.fixed_slice(&[3]);
        let q_hat = q_hat.integrate_body(delta_alpha);
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
    /// Checks if the estimation paremters are all finite (i.e not NaN or infinite)
    pub fn is_non_finite(&self) -> bool {
        self.q_hat.0.into_buf().iter().any(|x| !x.is_finite())
            || self.b_hat.into_buf().iter().any(|x| !x.is_finite())
            || self
                .p
                .into_buf()
                .as_buf()
                .iter()
                .any(|&x: &f64| !x.is_finite())
            || self
                .omega
                .into_buf()
                .as_buf()
                .iter()
                .any(|&x: &f64| !x.is_finite())
    }

    /// Resets the state if any parameter is non-finite
    pub fn reset_if_invalid(&mut self) {
        if self.is_non_finite() {
            println!("reset");
            self.q_hat = Quaternion::identity();
            self.b_hat = Vector::zeros();
            self.p = Matrix::eye();
            self.omega = Vector::zeros();
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nox::{array, tensor};

    use super::*;

    #[test]
    fn test_calculate_covariance() {
        #[rustfmt::skip]
        let expected_q = tensor![
            [8.33352623e-07, 0.00000000e+00, 0.00000000e+00, 3.47222222e-09, 0.00000000e+00, 0.00000000e+00,],
            [0.00000000e+00, 8.33352623e-07, 0.00000000e+00, 0.00000000e+00, 3.47222222e-09, 0.00000000e+00,],
            [0.00000000e+00, 0.00000000e+00, 8.33352623e-07, 0.00000000e+00, 0.00000000e+00, 3.47222222e-09,],
            [3.47222222e-09, 0.00000000e+00, 0.00000000e+00, 8.33333333e-07, 0.00000000e+00, 0.00000000e+00,],
            [0.00000000e+00, 3.47222222e-09, 0.00000000e+00, 0.00000000e+00, 8.33333333e-07, 0.00000000e+00,],
            [0.00000000e+00, 0.00000000e+00, 3.47222222e-09, 0.00000000e+00, 0.00000000e+00, 8.33333333e-07,],
        ];
        let q = calculate_covariance(
            tensor![0.01, 0.01, 0.01],
            tensor![0.01, 0.01, 0.01],
            1.0 / 120.0,
        );
        assert_relative_eq!(q, expected_q, epsilon = 1e-5);
    }

    #[test]
    fn test_propogate_state_covariance() {
        // from: cube-sat.py
        // TODO(sphw): bring this function into rust
        let yqy = tensor![
            [8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0, 0.0],
            [0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0],
            [0.0, 0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09],
            [-3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0, 0.0],
            [0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0],
            [0.0, 0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07]
        ];
        let out =
            propogate_state_covariance(Matrix::eye(), tensor![1.0, 0.0, 0.0], yqy, 1.0 / 120.0);
        #[rustfmt::skip]
        let expected = tensor![[ 1.00007028e+00,  0.00000000e+00,  0.00000000e+00,
            -8.33333681e-03,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  1.00007028e+00,  5.55768107e-19,
            0.00000000e+00, -8.33324036e-03, -3.47220213e-05],
        [ 0.00000000e+00,  5.55768107e-19,  1.00007028e+00,
            0.00000000e+00,  3.47220213e-05, -8.33324036e-03],
        [-8.33333681e-03,  0.00000000e+00,  0.00000000e+00,
            1.00000083e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00, -8.33324036e-03,  3.47220213e-05,
            0.00000000e+00,  1.00000083e+00,  0.00000000e+00],
        [ 0.00000000e+00, -3.47220213e-05, -8.33324036e-03,
            0.00000000e+00,  0.00000000e+00,  1.00000083e+00]];
        assert_relative_eq!(out.inner(), expected.inner(), epsilon = 1e-6);
    }

    #[test]
    fn test_propogate_quaternion() {
        let q = propogate_quaternion(Quaternion::identity(), tensor![1.0, 0.0, 0.0], 1.0 / 60.0);
        assert_relative_eq!(
            q.0.inner(),
            &array![0.00833324, 0., 0., 0.99996528],
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_mekf() {
        #[rustfmt::skip]
        let expected_yqy = tensor![
            [8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0, 0.0],
            [0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0],
            [0.0, 0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09],
            [-3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0, 0.0],
            [0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0],
            [0.0, 0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07]
        ];
        let ref_a = tensor![0.0, 1.0, 0.0].normalize();
        let ref_b = tensor![1.0, 0.0, 0.0].normalize();
        let mut q: Quaternion<f64, ArrayRepr> =
            Quaternion::from_axis_angle(Vector::z_axis(), 3.14 / 4.0);
        let body_a = (q.inverse() * ref_a).normalize();
        let body_b = (q.inverse() * ref_b).normalize();
        let dt = 1.0 / 120.0;
        let mut state = State::new(tensor![0.01, 0.01, 0.01], tensor![0.01, 0.01, 0.01], dt);
        assert_relative_eq!(state.yqy, expected_yqy, epsilon = 1e-4);
        for _ in 0..180 {
            state.omega = Default::default();
            state = state.estimate_attitude([body_a, body_b], [ref_a, ref_b], [0.03, 0.03]);
        }
        assert_relative_eq!(state.q_hat.0, q.0, epsilon = 1e-3);
        for _ in 0..120 {
            q = q.integrate_body(tensor![1.0 / 120.0, 0.0, 0.0]);
            let body_a = (q.inverse() * ref_a).normalize();
            let body_b = (q.inverse() * ref_b).normalize();
            state.omega = tensor![1.0, 0.0, 0.0];
            state = state.estimate_attitude([body_a, body_b], [ref_a, ref_b], [0.03, 0.03]);
        }
        assert_relative_eq!(
            state.b_hat.inner(),
            tensor![0.0, 0.0, 0.0].inner(),
            epsilon = 1e-2
        );
        assert_relative_eq!(state.q_hat.0, q.0, epsilon = 1e-3);
    }
}
