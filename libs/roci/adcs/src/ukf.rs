#![allow(clippy::type_complexity)]

use nox::{
    BaseBroadcastDim, BroadcastDim, Const, Dim, Error, Matrix, NonScalarDim, NonTupleDim, Repr,
    Scalar, ShapeConstraint, SquareDim, Tensor, Vector,
};

pub fn unscented_transform<N, S, R>(
    points: &Tensor<f64, (S, N), R>,
    mean_weights: &Tensor<f64, S, R>,
    covar_weights: &Tensor<f64, S, R>,
) -> (Tensor<f64, N, R>, Tensor<f64, (N, N), R>)
where
    (N, S): Dim,
    (S, N): Dim,
    (N, N): Dim,
    N: Dim + NonScalarDim + NonTupleDim,
    S: Dim + NonScalarDim + NonTupleDim,
    (S, S): SquareDim<SideDim = S>,
    R: Repr,
{
    let points_t = points.transpose();
    let x_hat = points_t.dot(mean_weights);
    let y = points - &x_hat;
    let covar_weights_diag: Tensor<f64, (S, S), R> = Tensor::from_diag(covar_weights.clone());
    let covar = y.transpose().dot(&covar_weights_diag.dot(&y));
    (x_hat, covar)
}

pub fn cross_covar<N, Z, S, R>(
    x_hat: &Tensor<f64, N, R>,
    z_hat: &Tensor<f64, Z, R>,
    points_x: Tensor<f64, (S, N), R>,
    points_z: Tensor<f64, (S, Z), R>,
    covar_weights: Tensor<f64, S, R>,
) -> Tensor<f64, (N, Z), R>
where
    N: Dim + NonTupleDim + NonScalarDim,
    Z: Dim + NonTupleDim + NonScalarDim,
    S: Dim + NonTupleDim + NonScalarDim,
    ShapeConstraint: BaseBroadcastDim<N, N, Output = N>,
    ShapeConstraint: BaseBroadcastDim<Z, Z, Output = Z>,
    (N, Z): Dim,
    (S, N): Dim,
    (S, Z): Dim,
    (Z, N): Dim,
    R: Repr,
{
    let delta_x = points_x - x_hat;
    let delta_z = points_z - z_hat;
    delta_x
        .rows_iter()
        .zip(delta_z.rows_iter())
        .zip(covar_weights.rows_iter())
        .map(|((delta_x, delta_z), weight)| weight * delta_x.outer(&delta_z))
        .sum()
}

pub fn predict<S, N, R>(
    sigma_points: Tensor<f64, (S, N), R>,
    prop_fn: impl Fn(Tensor<f64, N, R>) -> Tensor<f64, N, R>,
    mean_weights: &Tensor<f64, S, R>,
    covar_weights: &Tensor<f64, S, R>,
    prop_covar: &Tensor<f64, (N, N), R>,
) -> (
    Tensor<f64, (S, N), R>,
    Tensor<f64, N, R>,
    Tensor<f64, (N, N), R>,
)
where
    S: Dim + NonTupleDim + NonScalarDim,
    N: Dim + NonTupleDim + NonScalarDim,
    ShapeConstraint: BaseBroadcastDim<N, N, Output = N>,
    (S, N): Dim,
    (N, N): Dim + SquareDim<SideDim = N>,
    (S, S): Dim + SquareDim<SideDim = S>,
    (N, S): Dim,
    R: Repr,
{
    let points = Tensor::stack(sigma_points.rows_iter().map(prop_fn), 0);
    let (x_hat, covar) = unscented_transform::<N, S, R>(&points, mean_weights, covar_weights);
    let covar = covar + prop_covar;
    (points, x_hat, covar)
}

pub fn innovate<S, N, Z, R>(
    x_points: &Tensor<f64, (S, N), R>,
    z: &Tensor<f64, Z, R>,
    measure_fn: impl for<'a> Fn(Tensor<f64, N, R>, Tensor<f64, Z, R>) -> Tensor<f64, Z, R>,
    mean_weights: &Tensor<f64, S, R>,
    covar_weights: &Tensor<f64, S, R>,
    noise_covar: &Tensor<f64, (Z, Z), R>,
) -> (
    Tensor<f64, (S, Z), R>,
    Tensor<f64, Z, R>,
    Tensor<f64, (Z, Z), R>,
)
where
    S: Dim + NonTupleDim + NonScalarDim,
    N: Dim + NonTupleDim + NonScalarDim,
    Z: Dim + NonTupleDim + NonScalarDim,
    R: Repr,
    ShapeConstraint: BaseBroadcastDim<N, N, Output = N>,
    ShapeConstraint: BaseBroadcastDim<Z, Z, Output = Z>,
    (S, N): Dim,
    (Z, Z): Dim + SquareDim<SideDim = Z>,
    (N, N): Dim + SquareDim<SideDim = N>,
    (S, S): Dim + SquareDim<SideDim = S>,
    (N, S): Dim,
    (S, Z): Dim,
    (Z, S): Dim,
{
    let points = Tensor::stack(
        x_points
            .rows_iter()
            .map(|point| measure_fn(point, z.clone())),
        0,
    );
    let (z_hat, measure_covar) = unscented_transform(&points, mean_weights, covar_weights);
    let measure_covar = measure_covar + noise_covar;
    (points, z_hat, measure_covar)
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct UncheckedMerweConfig {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
    pub lambda: f64,
    pub n: usize,
}

pub struct MerweConfig<const N: usize> {
    pub unchecked_config: UncheckedMerweConfig,
}

impl UncheckedMerweConfig {
    pub fn new(n: usize, alpha: f64, beta: f64, kappa: f64) -> Self {
        let lambda = alpha.powi(2) * (n as f64 + kappa) - n as f64;

        Self {
            alpha,
            beta,
            kappa,
            lambda,
            n,
        }
    }

    pub fn sigma_points<N, S, R>(
        &self,
        x: Tensor<f64, N, R>,
        sigma: Tensor<f64, (N, N), R>,
    ) -> Result<Tensor<f64, (S, N), R>, nox::Error>
    where
        (N, N): Dim + SquareDim<SideDim = N>,
        ShapeConstraint: BaseBroadcastDim<N, N, Output = N>,
        ShapeConstraint: BroadcastDim<N, N, Output = N>,
        R: Repr + 'static,
        N: Dim + NonScalarDim + 'static,
        (S, N): Dim,
    {
        let s = 2 * self.n + 1;
        let lambda = self.lambda;
        let u = ((self.n as f64 + lambda) * sigma)
            .try_cholesky()?
            .transpose();
        Ok(Tensor::stack(
            (0..s).map(move |i| match i {
                0 => x.clone(),
                i if (1..=self.n).contains(&i) => &x + u.row(i - 1),
                _ => &x - u.row(i - self.n - 1),
            }),
            0,
        ))
    }

    pub fn mean_weights<S: Dim, R: Repr>(&self) -> Tensor<f64, S, R> {
        let s = 2 * self.n + 1;
        let n = self.n as f64;
        let lambda = self.lambda;
        let w_i = self.shared_weight();
        let w_0: Scalar<f64, R> = (lambda / (n + lambda)).into();
        Tensor::stack(
            (0..s).map(|i| if i == 0 { w_0.clone() } else { w_i.clone() }),
            0,
        )
    }

    pub fn covariance_weights<S: Dim, R: Repr>(&self) -> Tensor<f64, S, R> {
        let Self {
            lambda,
            alpha,
            beta,
            n,
            ..
        } = self;
        let s = 2 * *n + 1;
        let n = *n as f64;
        let w_i = self.shared_weight();
        let w_0: Scalar<f64, R> = (lambda / (n + lambda) + (1. - alpha.powi(2) + beta)).into();
        Tensor::stack(
            (0..s).map(|i| if i == 0 { w_0.clone() } else { w_i.clone() }),
            0,
        )
    }

    /// Calculated the shared weight used by both the covariance and mean weights
    pub fn shared_weight<R: Repr>(&self) -> Scalar<f64, R> {
        let lambda = self.lambda;
        let n = self.n as f64;
        let w = 1.0 / (2.0 * (n + lambda));
        w.into()
    }
}

impl<const N: usize> MerweConfig<N> {
    pub fn new(alpha: f64, beta: f64, kappa: f64) -> Self {
        Self {
            unchecked_config: UncheckedMerweConfig::new(N, alpha, beta, kappa),
        }
    }

    pub fn sigma_points<const S: usize, R: Repr + 'static>(
        &self,
        x: Vector<f64, N, R>,
        sigma: Matrix<f64, N, N, R>,
    ) -> Result<Matrix<f64, S, N, R>, nox::Error> {
        debug_assert_eq!(S, 2 * N + 1, "S must be 2 * N + 1");
        self.unchecked_config.sigma_points(x, sigma)
    }

    pub fn mean_weights<const S: usize, R: Repr>(&self) -> Vector<f64, S, R> {
        self.unchecked_config.mean_weights()
    }

    pub fn covariance_weights<const S: usize, R: Repr>(&self) -> Vector<f64, S, R> {
        self.unchecked_config.covariance_weights()
    }

    /// Calculated the shared weight used by both the covariance and mean weights
    pub fn shared_weight<R: Repr>(&self) -> Scalar<f64, R> {
        self.unchecked_config.shared_weight()
    }
}

#[doc(hidden)]
pub struct UncheckedState<N: Dim, Z: Dim, R: Repr>
where
    (N, N): Dim,
    (Z, Z): Dim,
{
    pub x_hat: Tensor<f64, N, R>,
    pub covar: Tensor<f64, (N, N), R>,
    pub prop_covar: Tensor<f64, (N, N), R>,
    pub noise_covar: Tensor<f64, (Z, Z), R>,
}

impl<N: Dim, Z: Dim, R: Repr + 'static> UncheckedState<N, Z, R>
where
    N: NonScalarDim + NonTupleDim + 'static,
    Z: NonScalarDim + NonTupleDim,
    (N, N): Dim + SquareDim<SideDim = N>,
    (Z, Z): Dim + SquareDim<SideDim = Z>,
    ShapeConstraint: BaseBroadcastDim<N, N, Output = N>,
    ShapeConstraint: BroadcastDim<N, N, Output = N>,
    ShapeConstraint: BaseBroadcastDim<Z, Z, Output = Z>,
    ShapeConstraint: BroadcastDim<Z, Z, Output = Z>,
{
    pub fn update<S>(
        mut self,
        config: UncheckedMerweConfig,
        z: Tensor<f64, Z, R>,
        prop_fn: impl Fn(Tensor<f64, N, R>) -> Tensor<f64, N, R>,
        measure_fn: impl Fn(Tensor<f64, N, R>, Tensor<f64, Z, R>) -> Tensor<f64, Z, R>,
    ) -> Result<Self, Error>
    where
        S: Dim + NonTupleDim + NonScalarDim,
        (S, S): SquareDim<SideDim = S>,
        (S, N): Dim,
        (N, S): Dim,
        (S, Z): Dim,
        (Z, S): Dim,
        (N, Z): Dim,
        (Z, N): Dim,
    {
        let sigma_points =
            config.sigma_points::<N, S, R>(self.x_hat.clone(), self.covar.clone())?;
        let mean_weights = config.mean_weights::<S, R>();
        let covar_weights = config.covariance_weights::<S, R>();
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

pub struct State<const N: usize, const Z: usize, const S: usize, R: Repr> {
    pub x_hat: Vector<f64, N, R>,
    pub covar: Matrix<f64, N, N, R>,
    pub prop_covar: Matrix<f64, N, N, R>,
    pub noise_covar: Matrix<f64, Z, Z, R>,
    pub config: MerweConfig<N>,
}

impl<const N: usize, const Z: usize, const S: usize, R: Repr + 'static> State<N, Z, S, R> {
    pub fn update(
        self,
        z: Vector<f64, Z, R>,
        prop_fn: impl Fn(Vector<f64, N, R>) -> Vector<f64, N, R>,
        measure_fn: impl Fn(Vector<f64, N, R>, Vector<f64, Z, R>) -> Vector<f64, Z, R>,
    ) -> Result<Self, Error> {
        let Self {
            x_hat,
            covar,
            prop_covar,
            noise_covar,
            config,
        } = self;
        let unchecked_state = UncheckedState {
            x_hat,
            covar,
            prop_covar,
            noise_covar,
        };
        let UncheckedState {
            x_hat,
            covar,
            prop_covar,
            noise_covar,
        } = unchecked_state.update::<Const<S>>(config.unchecked_config, z, prop_fn, measure_fn)?;
        Ok(State {
            x_hat,
            covar,
            prop_covar,
            noise_covar,
            config,
        })
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
        assert_eq!(config.unchecked_config.lambda, 2.0);
        let weights = config.mean_weights();
        assert_eq!(weights, tensor![0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
    }

    #[test]
    fn test_mean_covars() {
        let config = MerweConfig::<3>::new(1., 2., 2.);
        assert_eq!(config.unchecked_config.lambda, 2.0);
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
