#![allow(clippy::type_complexity)]

use nalgebra::{SMatrix, SVector};

pub fn unscented_transform<const S: usize, const N: usize>(
    points: &SMatrix<f64, S, N>,
    mean_weights: &SVector<f64, S>,
    covar_weights: &SVector<f64, S>,
) -> (SVector<f64, N>, SMatrix<f64, N, N>) {
    let points_t = points.transpose();
    let x_hat = points_t * mean_weights;
    let y = points - SMatrix::from_fn(|_i, j| x_hat[j]);
    let covar_weights_diag = SMatrix::<f64, S, S>::from_diagonal(covar_weights);
    let covar = y.transpose() * covar_weights_diag * y;
    (x_hat, covar)
}

pub fn cross_covar<const N: usize, const Z: usize, const S: usize>(
    x_hat: &SVector<f64, N>,
    z_hat: &SVector<f64, Z>,
    points_x: SMatrix<f64, S, N>,
    points_z: SMatrix<f64, S, Z>,
    covar_weights: SVector<f64, S>,
) -> SMatrix<f64, N, Z> {
    let delta_x = points_x - SMatrix::from_fn(|_i, j| x_hat[j]);
    let delta_z = points_z - SMatrix::from_fn(|_i, j| z_hat[j]);
    
    let mut result = SMatrix::<f64, N, Z>::zeros();
    for i in 0..S {
        let dx = delta_x.row(i).transpose();
        let dz = delta_z.row(i).transpose();
        result += covar_weights[i] * (dx * dz.transpose());
    }
    result
}

pub fn predict<const S: usize, const N: usize>(
    sigma_points: SMatrix<f64, S, N>,
    prop_fn: impl Fn(SVector<f64, N>) -> SVector<f64, N>,
    mean_weights: &SVector<f64, S>,
    covar_weights: &SVector<f64, S>,
    prop_covar: &SMatrix<f64, N, N>,
) -> (
    SMatrix<f64, S, N>,
    SVector<f64, N>,
    SMatrix<f64, N, N>,
) {
    let points = SMatrix::<f64, S, N>::from_fn(|i, j| {
        let row = sigma_points.row(i);
        let vec = SVector::<f64, N>::from_fn(|k, _| row[k]);
        let result = prop_fn(vec);
        result[j]
    });
    let (x_hat, covar) = unscented_transform::<S, N>(&points, mean_weights, covar_weights);
    let covar = covar + prop_covar;
    (points, x_hat, covar)
}

pub fn innovate<const S: usize, const N: usize, const Z: usize>(
    x_points: &SMatrix<f64, S, N>,
    z: &SVector<f64, Z>,
    measure_fn: impl Fn(SVector<f64, N>, SVector<f64, Z>) -> SVector<f64, Z>,
    mean_weights: &SVector<f64, S>,
    covar_weights: &SVector<f64, S>,
    noise_covar: &SMatrix<f64, Z, Z>,
) -> (
    SMatrix<f64, S, Z>,
    SVector<f64, Z>,
    SMatrix<f64, Z, Z>,
) {
    let points = SMatrix::<f64, S, Z>::from_fn(|i, j| {
        let row = x_points.row(i);
        let vec = SVector::<f64, N>::from_fn(|k, _| row[k]);
        let result = measure_fn(vec, *z);
        result[j]
    });
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

    pub fn sigma_points<const S: usize, const N: usize>(
        &self,
        x: SVector<f64, N>,
        sigma: SMatrix<f64, N, N>,
    ) -> Result<SMatrix<f64, S, N>, String> {
        let s = 2 * self.n + 1;
        debug_assert_eq!(S, s, "S must be 2*N+1");
        let lambda = self.lambda;
        let u = ((self.n as f64 + lambda) * sigma)
            .cholesky()
            .ok_or("Cholesky decomposition failed")?
            .l()
            .transpose();
        
        Ok(SMatrix::<f64, S, N>::from_fn(|i, j| {
            match i {
                0 => x[j],
                i if (1..=self.n).contains(&i) => x[j] + u[(i - 1, j)],
                _ => x[j] - u[(i - self.n - 1, j)],
            }
        }))
    }

    pub fn mean_weights<const S: usize>(&self) -> SVector<f64, S> {
        let s = 2 * self.n + 1;
        debug_assert_eq!(S, s, "S must be 2*N+1");
        let n = self.n as f64;
        let lambda = self.lambda;
        let w_i = self.shared_weight();
        let w_0 = lambda / (n + lambda);
        SVector::<f64, S>::from_fn(|i, _| if i == 0 { w_0 } else { w_i })
    }

    pub fn covariance_weights<const S: usize>(&self) -> SVector<f64, S> {
        let Self {
            lambda,
            alpha,
            beta,
            n,
            ..
        } = self;
        let s = 2 * *n + 1;
        debug_assert_eq!(S, s, "S must be 2*N+1");
        let n = *n as f64;
        let w_i = self.shared_weight();
        let w_0 = lambda / (n + lambda) + (1. - alpha.powi(2) + beta);
        SVector::<f64, S>::from_fn(|i, _| if i == 0 { w_0 } else { w_i })
    }

    /// Calculated the shared weight used by both the covariance and mean weights
    pub fn shared_weight(&self) -> f64 {
        let lambda = self.lambda;
        let n = self.n as f64;
        1.0 / (2.0 * (n + lambda))
    }
}

impl<const N: usize> MerweConfig<N> {
    pub fn new(alpha: f64, beta: f64, kappa: f64) -> Self {
        Self {
            unchecked_config: UncheckedMerweConfig::new(N, alpha, beta, kappa),
        }
    }

    pub fn sigma_points<const S: usize>(
        &self,
        x: SVector<f64, N>,
        sigma: SMatrix<f64, N, N>,
    ) -> Result<SMatrix<f64, S, N>, String> {
        debug_assert_eq!(S, 2 * N + 1, "S must be 2 * N + 1");
        self.unchecked_config.sigma_points(x, sigma)
    }

    pub fn mean_weights<const S: usize>(&self) -> SVector<f64, S> {
        self.unchecked_config.mean_weights()
    }

    pub fn covariance_weights<const S: usize>(&self) -> SVector<f64, S> {
        self.unchecked_config.covariance_weights()
    }

    /// Calculated the shared weight used by both the covariance and mean weights
    pub fn shared_weight(&self) -> f64 {
        self.unchecked_config.shared_weight()
    }
}

#[doc(hidden)]
pub struct UncheckedState<const N: usize, const Z: usize> {
    pub x_hat: SVector<f64, N>,
    pub covar: SMatrix<f64, N, N>,
    pub prop_covar: SMatrix<f64, N, N>,
    pub noise_covar: SMatrix<f64, Z, Z>,
}

impl<const N: usize, const Z: usize> UncheckedState<N, Z> {
    pub fn update<const S: usize>(
        mut self,
        config: UncheckedMerweConfig,
        z: SVector<f64, Z>,
        prop_fn: impl Fn(SVector<f64, N>) -> SVector<f64, N>,
        measure_fn: impl Fn(SVector<f64, N>, SVector<f64, Z>) -> SVector<f64, Z>,
    ) -> Result<Self, String> {
        let sigma_points =
            config.sigma_points::<S, N>(self.x_hat, self.covar)?;
        let mean_weights = config.mean_weights::<S>();
        let covar_weights = config.covariance_weights::<S>();
        let (points_x, x_hat, covar) = predict(
            sigma_points,
            prop_fn,
            &mean_weights,
            &covar_weights,
            &self.prop_covar,
        );
        let (points_z, z_hat, z_covar) = innovate::<S, N, Z>(
            &points_x,
            &z,
            measure_fn,
            &mean_weights,
            &covar_weights,
            &self.noise_covar,
        );
        let cross_covar = cross_covar(&x_hat, &z_hat, points_x, points_z, covar_weights);
        let z_covar_inv = z_covar.try_inverse().ok_or("Failed to invert covariance")?;
        let kalman_gain = cross_covar * z_covar_inv;
        let y = z - z_hat;
        self.x_hat = x_hat + kalman_gain * y;
        self.covar = covar - kalman_gain * z_covar * kalman_gain.transpose();

        Ok(self)
    }
}

pub struct State<const N: usize, const Z: usize, const S: usize> {
    pub x_hat: SVector<f64, N>,
    pub covar: SMatrix<f64, N, N>,
    pub prop_covar: SMatrix<f64, N, N>,
    pub noise_covar: SMatrix<f64, Z, Z>,
    pub config: MerweConfig<N>,
}

impl<const N: usize, const Z: usize, const S: usize> State<N, Z, S> {
    pub fn update(
        self,
        z: SVector<f64, Z>,
        prop_fn: impl Fn(SVector<f64, N>) -> SVector<f64, N>,
        measure_fn: impl Fn(SVector<f64, N>, SVector<f64, Z>) -> SVector<f64, Z>,
    ) -> Result<Self, String> {
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
        } = unchecked_state.update::<S>(config.unchecked_config, z, prop_fn, measure_fn)?;
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
    use nalgebra::{matrix, vector, Matrix4, SMatrix};

    use super::*;

    #[test]
    fn test_sigma_points() {
        let config = MerweConfig::<3>::new(1., 2., 2.);
        let p = SMatrix::<f64, 3, 3>::identity();
        let points = config.sigma_points(vector![0., 1., 0.], p).unwrap();
        let expected = matrix![
            0., 1., 0.;
            2.23606798, 1., 0.;
            0., 3.23606798, 0.;
            0., 1., 2.23606798;
            -2.23606798, 1., 0.;
            0., -1.23606798, 0.;
            0., 1., -2.23606798;
        ];
        assert_relative_eq!(points, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_mean_weights() {
        let config = MerweConfig::<3>::new(1., 1., 2.);
        assert_eq!(config.unchecked_config.lambda, 2.0);
        let weights = config.mean_weights::<7>();
        assert_eq!(weights, vector![0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
    }

    #[test]
    fn test_mean_covars() {
        let config = MerweConfig::<3>::new(1., 2., 2.);
        assert_eq!(config.unchecked_config.lambda, 2.0);
        let weights = config.covariance_weights::<7>();
        assert_eq!(weights, vector![2.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
    }

    #[test]
    fn test_unscented_transform() {
        let covar_weights = vector![0.4, 0.1, 0.1];
        let mean_weights = vector![0.4, 0.1, 0.1];
        let (x, p) = unscented_transform(
            &matrix![1., 2.; 2., 4.; 5., 4.],
            &mean_weights,
            &covar_weights,
        );
        assert_relative_eq!(x, vector![1.1, 1.6], epsilon = 1e-7);
        assert_relative_eq!(p, matrix![1.606, 1.136; 1.136, 1.216], epsilon = 1e-7);
    }

    #[test]
    fn test_cross_covar() {
        let covar_weights = vector![0.4, 0.1, 0.1];
        let x_hat = vector![1., 2.];
        let z_hat = vector![2., 3.];
        let points_x = matrix![1., 2.; 2., 4.; 5., 4.];
        let points_z = matrix![2., 3.; 3., 5.; 6., 5.];
        let p = cross_covar(&x_hat, &z_hat, points_x, points_z, covar_weights);
        assert_relative_eq!(p, matrix![1.7, 1.; 1., 0.8], epsilon = 1e-7);
    }

    #[test]
    fn test_simple_linear() {
        let z_std: f64 = 0.1;
        let mut state: State<4, 2, 9> = State {
            x_hat: vector![-1.0, 1.0, -1.0, 1.0],
            covar: Matrix4::identity() * 0.02,
            prop_covar: matrix![
                2.5e-09, 5.0e-08, 0.0e+00, 0.0e+00;
                5.0e-08, 1.0e-06, 0.0e+00, 0.0e+00;
                0.0e+00, 0.0e+00, 2.5e-09, 5.0e-08;
                0.0e+00, 0.0e+00, 5.0e-08, 1.0e-06;
            ],
            noise_covar: SMatrix::<f64, 2, 2>::from_diagonal(&vector![z_std.powi(2), z_std.powi(2)]),
            config: MerweConfig::new(0.1, 2.0, -1.0),
        };
        let zs = [
            vector![5.17725854e-02, -6.26320583e-03],
            vector![7.76510642e-01, 9.41957633e-01],
            vector![2.00841988e+00, 2.03743906e+00],
            vector![2.99844618e+00, 3.11297309e+00],
            vector![4.02493837e+00, 4.18458527e+00],
            vector![5.15192102e+00, 4.94829940e+00],
            vector![5.97820965e+00, 6.19408601e+00],
            vector![6.90700638e+00, 7.09993104e+00],
            vector![8.16174727e+00, 7.84922020e+00],
            vector![9.07054319e+00, 9.02949139e+00],
            vector![1.00943927e+01, 1.02153963e+01],
            vector![1.10528857e+01, 1.09655620e+01],
            vector![1.20433517e+01, 1.19606005e+01],
            vector![1.30130301e+01, 1.30389662e+01],
            vector![1.39492112e+01, 1.38780037e+01],
            vector![1.50232252e+01, 1.50704141e+01],
            vector![1.59498538e+01, 1.60893516e+01],
            vector![1.70097415e+01, 1.70585561e+01],
            vector![1.81292609e+01, 1.80656253e+01],
            vector![1.90783022e+01, 1.89139529e+01],
            vector![1.99490761e+01, 1.99682328e+01],
            vector![2.10253265e+01, 2.09926241e+01],
            vector![2.18166124e+01, 2.19475433e+01],
            vector![2.29619247e+01, 2.29313189e+01],
            vector![2.40366414e+01, 2.40207406e+01],
            vector![2.50164997e+01, 2.50594340e+01],
            vector![2.60602065e+01, 2.59104916e+01],
            vector![2.68926856e+01, 2.68682419e+01],
            vector![2.81448564e+01, 2.81699908e+01],
            vector![2.89114209e+01, 2.90161936e+01],
            vector![2.99632302e+01, 3.01334351e+01],
            vector![3.09547757e+01, 3.09778803e+01],
            vector![3.20168683e+01, 3.19300419e+01],
            vector![3.28656686e+01, 3.30364708e+01],
            vector![3.39697008e+01, 3.40282794e+01],
            vector![3.50369500e+01, 3.51215329e+01],
            vector![3.59710004e+01, 3.61372108e+01],
            vector![3.70591558e+01, 3.69247502e+01],
            vector![3.80522440e+01, 3.78498751e+01],
            vector![3.90272805e+01, 3.90100329e+01],
            vector![4.00377740e+01, 3.98368033e+01],
            vector![4.08131455e+01, 4.09728212e+01],
            vector![4.18214855e+01, 4.19909894e+01],
            vector![4.31312654e+01, 4.29949226e+01],
            vector![4.40398607e+01, 4.38911788e+01],
            vector![4.50978163e+01, 4.49942054e+01],
            vector![4.61407950e+01, 4.59709971e+01],
            vector![4.69322204e+01, 4.71080633e+01],
            vector![4.80521531e+01, 4.81292422e+01],
            vector![4.91282949e+01, 4.90346729e+01],
        ];
        for z in zs {
            state = state
                .update(
                    z,
                    |x| {
                        let dt = 0.1;
                        let transition = matrix![
                            1., dt, 0., 0.;
                            0., 1., 0., 0.;
                            0., 0., 1., dt;
                            0., 0., 0., 1.;
                        ];
                        transition * x
                    },
                    |x, _| vector![x[0], x[2]],
                )
                .unwrap();
        }
        assert_relative_eq!(
            state.x_hat,
            vector![48.9118168, 9.96293597, 48.89106226, 9.95283274],
            epsilon = 1e-8
        );
    }
}
