use nox::{Error, Matrix, Repr, Scalar, Vector};

pub fn unscented_transform<const N: usize, const S: usize, R: Repr>(
    points: &Matrix<f64, S, N, R>,
    mean_weights: &Vector<f64, S, R>,
    covar_weights: &Vector<f64, S, R>,
) -> (Vector<f64, N, R>, Matrix<f64, N, N, R>) {
    let x_hat = points.transpose().dot(mean_weights);
    let y = points - &x_hat;
    let covar = y
        .transpose()
        .dot(&Matrix::from_diag(covar_weights.clone()).dot(&y));
    (x_hat, covar)
}

pub fn cross_covar<const N: usize, const Z: usize, const S: usize, R: Repr>(
    x_hat: &Vector<f64, N, R>,
    z_hat: &Vector<f64, Z, R>,
    points_x: Matrix<f64, S, N, R>,
    points_z: Matrix<f64, S, Z, R>,
    covar_weights: Vector<f64, S, R>,
) -> Matrix<f64, N, Z, R> {
    let mut covar = Matrix::zeros();
    for i in 0..S {
        let x = points_x.row(i);
        let z = points_z.row(i);
        let weight = covar_weights.get(i);
        let delta_x = x - x_hat;
        let delta_z = z - z_hat;
        covar = covar + weight * delta_x.outer(&delta_z);
    }
    covar
}

pub fn predict<const S: usize, const N: usize, R: Repr>(
    sigma_points: Matrix<f64, S, N, R>,
    prop_fn: impl Fn(Vector<f64, N, R>) -> Vector<f64, N, R>,
    mean_weights: &Vector<f64, S, R>,
    covar_weights: &Vector<f64, S, R>,
    prop_covar: &Matrix<f64, N, N, R>,
) -> (
    Matrix<f64, S, N, R>,
    Vector<f64, N, R>,
    Matrix<f64, N, N, R>,
) {
    let mut i = 0;
    let points = Matrix::from_rows([0; S].map(|_| {
        let point = sigma_points.row(i);
        let point = prop_fn(point);
        i += 1;
        point
    }));
    let (x_hat, covar) = unscented_transform(&points, mean_weights, covar_weights);
    let covar = covar + prop_covar;
    (points, x_hat, covar)
}

pub fn innovate<const S: usize, const N: usize, const Z: usize, R: Repr>(
    x_points: &Matrix<f64, S, N, R>,
    z: &Vector<f64, Z, R>,
    measure_fn: impl for<'a> Fn(Vector<f64, N, R>, Vector<f64, Z, R>) -> Vector<f64, Z, R>,
    mean_weights: &Vector<f64, S, R>,
    covar_weights: &Vector<f64, S, R>,
    noise_covar: &Matrix<f64, Z, Z, R>,
) -> (
    Matrix<f64, S, Z, R>,
    Vector<f64, Z, R>,
    Matrix<f64, Z, Z, R>,
) {
    let mut i = 0;
    let points = Matrix::from_rows([0; S].map(|_| {
        let point = x_points.row(i);
        let point = measure_fn(point, z.clone());
        i += 1;
        point
    }));
    let (z_hat, measure_covar) = unscented_transform(&points, mean_weights, covar_weights);
    let measure_covar = measure_covar + noise_covar;
    (points, z_hat, measure_covar)
}

pub struct MerweConfig<const N: usize> {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
    pub lambda: f64,
}

impl<const N: usize> MerweConfig<N> {
    pub fn new(alpha: f64, beta: f64, kappa: f64) -> Self {
        let n = N as f64;
        let lambda = alpha.powi(2) * (n + kappa) - n;

        Self {
            alpha,
            beta,
            kappa,
            lambda,
        }
    }

    pub fn sigma_points<const S: usize, R: Repr>(
        &self,
        x: Vector<f64, N, R>,
        sigma: Matrix<f64, N, N, R>,
    ) -> Result<Matrix<f64, S, N, R>, nox::Error> {
        let lambda = self.lambda;
        debug_assert_eq!(S, 2 * N + 1, "S must be 2 * N + 1");
        let mut i = 0;
        let u = ((N as f64 + lambda) * sigma).try_cholesky(true)?;
        let points = [0u8; S].map(|_| {
            let point = match i {
                0 => x.clone(),
                i if (1..=N).contains(&i) => &x + u.row(i - 1),
                _ => &x - u.row(i - N - 1),
            };
            i += 1;
            point
        });
        Ok(Matrix::from_rows(points))
    }

    pub fn mean_weights<const S: usize, R: Repr>(&self) -> Vector<f64, S, R> {
        let n = N as f64;
        let lambda = self.lambda;
        assert_eq!(S, N * 2 + 1, "S must be 2 * n + 1");
        let w_i = self.shared_weight();
        let w_0: Scalar<f64, R> = (lambda / (n + lambda)).into();
        Vector::from_scalars((0..S).map(|i| if i == 0 { w_0.clone() } else { w_i.clone() }))
    }

    pub fn covariance_weights<const S: usize, R: Repr>(&self) -> Vector<f64, S, R> {
        let Self {
            lambda,
            alpha,
            beta,
            ..
        } = self;
        assert_eq!(S, N * 2 + 1, "S must be 2 * n + 1");
        let n = N as f64;
        let w_i = self.shared_weight();
        let w_0: Scalar<f64, R> = (lambda / (n + lambda) + (1. - alpha.powi(2) + beta)).into();
        Vector::from_scalars((0..S).map(|i| if i == 0 { w_0.clone() } else { w_i.clone() }))
    }

    /// Calculated the shared weight used by both the covariance and mean weights
    pub fn shared_weight<R: Repr>(&self) -> Scalar<f64, R> {
        let lambda = self.lambda;
        let n = N as f64;
        let w = 1.0 / (2.0 * (n + lambda));
        w.into()
    }
}

pub struct State<const N: usize, const Z: usize, const S: usize, R: Repr> {
    pub x_hat: Vector<f64, N, R>,
    pub covar: Matrix<f64, N, N, R>,
    pub prop_covar: Matrix<f64, N, N, R>,
    pub noise_covar: Matrix<f64, Z, Z, R>,
    pub config: MerweConfig<N>,
}

impl<const N: usize, const Z: usize, const S: usize, R: Repr> State<N, Z, S, R> {
    pub fn update(
        mut self,
        z: Vector<f64, Z, R>,
        prop_fn: impl Fn(Vector<f64, N, R>) -> Vector<f64, N, R>,
        measure_fn: impl Fn(Vector<f64, N, R>, Vector<f64, Z, R>) -> Vector<f64, Z, R>,
    ) -> Result<Self, Error> {
        let sigma_points = self.config.sigma_points::<S, _>(self.x_hat, self.covar)?;
        let mean_weights = self.config.mean_weights::<S, _>();
        let covar_weights = self.config.covariance_weights::<S, _>();
        let (points_x, x_hat, covar) = predict(
            sigma_points,
            prop_fn,
            &mean_weights,
            &covar_weights,
            &self.prop_covar,
        );
        let (points_z, z_hat, z_covar) = innovate::<S, N, Z, _>(
            &points_x,
            &z,
            measure_fn,
            &mean_weights,
            &covar_weights,
            &self.noise_covar,
        );
        let cross_covar = cross_covar(&x_hat, &z_hat, points_x, points_z, covar_weights);
        let z_covar_inv = z_covar.try_inverse()?;
        let kalman_gain = cross_covar.dot(&z_covar_inv);
        let y = z - z_hat;
        self.x_hat = x_hat + kalman_gain.dot(&y);
        self.covar = covar - kalman_gain.dot(&z_covar.dot(&kalman_gain.transpose()));
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nox::{tensor, ArrayRepr, Matrix4};

    use super::*;

    #[test]
    fn test_sigma_points() {
        let config = MerweConfig::new(1., 2., 2.);
        let p = Matrix::eye();
        let points = config.sigma_points(tensor![0., 1., 0.], p).unwrap();
        let expected = tensor![
            [0., 1., 0.],
            [2.23606798, 1., 0.],
            [0., 3.23606798, 0.],
            [0., 1., 2.23606798],
            [-2.23606798, 1., 0.],
            [0., -1.23606798, 0.],
            [0., 1., -2.23606798],
        ];
        assert_relative_eq!(points, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_mean_weights() {
        let config = MerweConfig::<3>::new(1., 1., 2.);
        assert_eq!(config.lambda, 2.0);
        let weights = config.mean_weights();
        assert_eq!(weights, tensor![0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
    }

    #[test]
    fn test_mean_covars() {
        let config = MerweConfig::<3>::new(1., 2., 2.);
        assert_eq!(config.lambda, 2.0);
        let weights = config.covariance_weights::<7, ArrayRepr>();
        assert_eq!(weights, tensor![2.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,]);
    }

    #[test]
    fn test_unscented_transform() {
        let covar_weights = tensor![0.4, 0.1, 0.1];
        let mean_weights = tensor![0.4, 0.1, 0.1];
        let (x, p) = unscented_transform(
            &tensor![[1., 2.], [2., 4.], [5., 4.]],
            &mean_weights,
            &covar_weights,
        );
        assert_relative_eq!(x, tensor![1.1, 1.6], epsilon = 1e-7);
        assert_relative_eq!(p, tensor![[1.606, 1.136], [1.136, 1.216]], epsilon = 1e-7);
    }

    #[test]
    fn test_cross_covar() {
        let covar_weights = tensor![0.4, 0.1, 0.1];
        let x_hat = tensor![1., 2.];
        let z_hat = tensor![2., 3.];
        let points_x = tensor![[1., 2.], [2., 4.], [5., 4.]];
        let points_z = tensor![[2., 3.], [3., 5.], [6., 5.]];
        let p = cross_covar(&x_hat, &z_hat, points_x, points_z, covar_weights);
        assert_relative_eq!(p, tensor![[1.7, 1.], [1., 0.8]], epsilon = 1e-7);
    }

    #[test]
    fn test_simple_linear() {
        let z_std: f64 = 0.1;
        let mut state: State<4, 2, 9, ArrayRepr> = State {
            x_hat: tensor![-1.0, 1.0, -1.0, 1.0],
            covar: Matrix4::eye() * 0.02,
            prop_covar: tensor![
                [2.5e-09, 5.0e-08, 0.0e+00, 0.0e+00],
                [5.0e-08, 1.0e-06, 0.0e+00, 0.0e+00],
                [0.0e+00, 0.0e+00, 2.5e-09, 5.0e-08],
                [0.0e+00, 0.0e+00, 5.0e-08, 1.0e-06]
            ],
            noise_covar: Matrix::from_diag(tensor![z_std.powi(2), z_std.powi(2)]),
            config: MerweConfig::new(0.1, 2.0, -1.0),
        };
        let zs = [
            tensor![5.17725854e-02, -6.26320583e-03],
            tensor![7.76510642e-01, 9.41957633e-01],
            tensor![2.00841988e+00, 2.03743906e+00],
            tensor![2.99844618e+00, 3.11297309e+00],
            tensor![4.02493837e+00, 4.18458527e+00],
            tensor![5.15192102e+00, 4.94829940e+00],
            tensor![5.97820965e+00, 6.19408601e+00],
            tensor![6.90700638e+00, 7.09993104e+00],
            tensor![8.16174727e+00, 7.84922020e+00],
            tensor![9.07054319e+00, 9.02949139e+00],
            tensor![1.00943927e+01, 1.02153963e+01],
            tensor![1.10528857e+01, 1.09655620e+01],
            tensor![1.20433517e+01, 1.19606005e+01],
            tensor![1.30130301e+01, 1.30389662e+01],
            tensor![1.39492112e+01, 1.38780037e+01],
            tensor![1.50232252e+01, 1.50704141e+01],
            tensor![1.59498538e+01, 1.60893516e+01],
            tensor![1.70097415e+01, 1.70585561e+01],
            tensor![1.81292609e+01, 1.80656253e+01],
            tensor![1.90783022e+01, 1.89139529e+01],
            tensor![1.99490761e+01, 1.99682328e+01],
            tensor![2.10253265e+01, 2.09926241e+01],
            tensor![2.18166124e+01, 2.19475433e+01],
            tensor![2.29619247e+01, 2.29313189e+01],
            tensor![2.40366414e+01, 2.40207406e+01],
            tensor![2.50164997e+01, 2.50594340e+01],
            tensor![2.60602065e+01, 2.59104916e+01],
            tensor![2.68926856e+01, 2.68682419e+01],
            tensor![2.81448564e+01, 2.81699908e+01],
            tensor![2.89114209e+01, 2.90161936e+01],
            tensor![2.99632302e+01, 3.01334351e+01],
            tensor![3.09547757e+01, 3.09778803e+01],
            tensor![3.20168683e+01, 3.19300419e+01],
            tensor![3.28656686e+01, 3.30364708e+01],
            tensor![3.39697008e+01, 3.40282794e+01],
            tensor![3.50369500e+01, 3.51215329e+01],
            tensor![3.59710004e+01, 3.61372108e+01],
            tensor![3.70591558e+01, 3.69247502e+01],
            tensor![3.80522440e+01, 3.78498751e+01],
            tensor![3.90272805e+01, 3.90100329e+01],
            tensor![4.00377740e+01, 3.98368033e+01],
            tensor![4.08131455e+01, 4.09728212e+01],
            tensor![4.18214855e+01, 4.19909894e+01],
            tensor![4.31312654e+01, 4.29949226e+01],
            tensor![4.40398607e+01, 4.38911788e+01],
            tensor![4.50978163e+01, 4.49942054e+01],
            tensor![4.61407950e+01, 4.59709971e+01],
            tensor![4.69322204e+01, 4.71080633e+01],
            tensor![4.80521531e+01, 4.81292422e+01],
            tensor![4.91282949e+01, 4.90346729e+01],
        ];
        for z in zs {
            state = state
                .update(
                    z,
                    |x| {
                        let dt = 0.1;
                        let transition = tensor![
                            [1., dt, 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., dt],
                            [0., 0., 0., 1.]
                        ];
                        transition.dot(&x)
                    },
                    |x, _| Vector::from_scalars([x.get(0), x.get(2)]),
                )
                .unwrap();
        }
        assert_relative_eq!(
            state.x_hat,
            tensor![48.9118168, 9.96293597, 48.89106226, 9.95283274],
            epsilon = 1e-8
        );
    }
}
