"""
Pure Python/JAX implementation of Unscented Kalman Filter utilities.

This module provides UKF (Unscented Kalman Filter) functionality using pure
JAX/numpy operations, without any dependencies on roci or nalgebra.
"""

import jax.numpy as jnp


def unscented_transform(points, mean_weights, covar_weights):
    """
    Compute the mean and covariance from sigma points.

    Args:
        points: Sigma points matrix of shape (S, N) where S is number of points, N is dimension
        mean_weights: Weights for computing mean, shape (S,)
        covar_weights: Weights for computing covariance, shape (S,)

    Returns:
        Tuple of (mean, covariance) where mean is shape (N,) and covariance is (N, N)
    """
    points = jnp.asarray(points)
    mean_weights = jnp.asarray(mean_weights)
    covar_weights = jnp.asarray(covar_weights)

    # Compute weighted mean: x_hat = points.T @ mean_weights
    x_hat = points.T @ mean_weights

    # Compute deviations from mean
    y = points - x_hat[None, :]

    # Compute covariance: y.T @ diag(covar_weights) @ y
    weighted_y = y * covar_weights[:, None]
    covar = y.T @ weighted_y

    return x_hat, covar


def cross_covar(x_hat, z_hat, points_x, points_z, covar_weights):
    """
    Compute cross-covariance between state and measurement sigma points.

    Args:
        x_hat: State mean, shape (N,)
        z_hat: Measurement mean, shape (Z,)
        points_x: State sigma points, shape (S, N)
        points_z: Measurement sigma points, shape (S, Z)
        covar_weights: Weights for covariance, shape (S,)

    Returns:
        Cross-covariance matrix of shape (N, Z)
    """
    x_hat = jnp.asarray(x_hat)
    z_hat = jnp.asarray(z_hat)
    points_x = jnp.asarray(points_x)
    points_z = jnp.asarray(points_z)
    covar_weights = jnp.asarray(covar_weights)

    # Compute deviations
    delta_x = points_x - x_hat[None, :]
    delta_z = points_z - z_hat[None, :]

    # Compute cross-covariance
    weighted_delta_x = delta_x * covar_weights[:, None]
    cross_cov = weighted_delta_x.T @ delta_z

    return cross_cov


class UKFState:
    """
    Unscented Kalman Filter state using Merwe's scaled sigma point algorithm.

    This is a pure Python/JAX implementation that replaces the Rust/nalgebra version.
    """

    def __init__(self, x_hat, covar, prop_covar, noise_covar, alpha=0.1, beta=2.0, kappa=-1.0):
        """
        Initialize UKF state.

        Args:
            x_hat: Initial state estimate, shape (N,)
            covar: Initial state covariance, shape (N, N)
            prop_covar: Process noise covariance, shape (N, N)
            noise_covar: Measurement noise covariance, shape (Z, Z)
            alpha: Spread of sigma points (0 < alpha <= 1)
            beta: Prior knowledge parameter (2 is optimal for Gaussian)
            kappa: Secondary scaling parameter
        """
        self.x_hat = jnp.asarray(x_hat)
        self.covar = jnp.asarray(covar)
        self.prop_covar = jnp.asarray(prop_covar)
        self.noise_covar = jnp.asarray(noise_covar)

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Compute UKF parameters
        self.n = len(x_hat)  # State dimension
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n

        # Compute weights
        self.w_mean = self._compute_mean_weights()
        self.w_covar = self._compute_covar_weights()

    def _compute_mean_weights(self):
        """Compute weights for mean calculation."""
        n = self.n
        lambda_ = self.lambda_

        w_0 = lambda_ / (n + lambda_)
        w_i = 1.0 / (2.0 * (n + lambda_))

        weights = jnp.full(2 * n + 1, w_i)
        weights = weights.at[0].set(w_0)

        return weights

    def _compute_covar_weights(self):
        """Compute weights for covariance calculation."""
        n = self.n
        lambda_ = self.lambda_
        alpha = self.alpha
        beta = self.beta

        w_0 = lambda_ / (n + lambda_) + (1.0 - alpha**2 + beta)
        w_i = 1.0 / (2.0 * (n + lambda_))

        weights = jnp.full(2 * n + 1, w_i)
        weights = weights.at[0].set(w_0)

        return weights

    def _compute_sigma_points(self, x, covar):
        """
        Generate sigma points using Merwe's scaled sigma point algorithm.

        Args:
            x: State vector, shape (N,)
            covar: Covariance matrix, shape (N, N)

        Returns:
            Sigma points matrix of shape (2*N+1, N)
        """
        n = self.n
        lambda_ = self.lambda_

        # Cholesky decomposition: L @ L.T = (n + lambda) * covar
        try:
            L = jnp.linalg.cholesky((n + lambda_) * covar)
        except Exception:
            # If Cholesky fails, add small regularization
            L = jnp.linalg.cholesky((n + lambda_) * covar + jnp.eye(n) * 1e-9)

        # Generate sigma points
        sigma_points = jnp.zeros((2 * n + 1, n))
        sigma_points = sigma_points.at[0].set(x)  # First point is the mean

        for i in range(n):
            sigma_points = sigma_points.at[i + 1].set(x + L[:, i])
            sigma_points = sigma_points.at[i + 1 + n].set(x - L[:, i])

        return sigma_points

    def predict(self, prop_fn):
        """
        Prediction step: propagate sigma points through dynamics.

        Args:
            prop_fn: Function mapping state x to next state, x' = f(x)

        Returns:
            Tuple of (propagated_points, predicted_mean, predicted_covar)
        """
        # Generate sigma points
        sigma_points = self._compute_sigma_points(self.x_hat, self.covar)

        # Propagate each sigma point through dynamics
        points_prop = jnp.array([prop_fn(sigma_points[i]) for i in range(len(sigma_points))])

        # Compute mean and covariance from propagated points
        x_hat_pred, covar_pred = unscented_transform(points_prop, self.w_mean, self.w_covar)

        # Add process noise
        covar_pred = covar_pred + self.prop_covar

        return points_prop, x_hat_pred, covar_pred

    def update(self, z, prop_fn, measure_fn):
        """
        Full UKF update: predict and correct.

        Args:
            z: Measurement vector
            prop_fn: Propagation function, x' = f(x)
            measure_fn: Measurement function, z = h(x, z_measurement)

        Returns:
            None (updates internal state in-place)
        """
        z = jnp.asarray(z)

        # Predict step
        sigma_points = self._compute_sigma_points(self.x_hat, self.covar)
        points_x = jnp.array([prop_fn(sigma_points[i]) for i in range(len(sigma_points))])
        x_hat_pred, covar_pred = unscented_transform(points_x, self.w_mean, self.w_covar)
        covar_pred = covar_pred + self.prop_covar

        # Update step
        # Transform sigma points through measurement function
        points_z = jnp.array([measure_fn(points_x[i], z) for i in range(len(points_x))])
        z_hat, meas_covar = unscented_transform(points_z, self.w_mean, self.w_covar)
        meas_covar = meas_covar + self.noise_covar

        # Compute cross-covariance
        cross_cov = cross_covar(x_hat_pred, z_hat, points_x, points_z, self.w_covar)

        # Kalman gain
        meas_covar_inv = jnp.linalg.inv(meas_covar)
        kalman_gain = cross_cov @ meas_covar_inv

        # Update state
        innovation = z - z_hat
        self.x_hat = x_hat_pred + kalman_gain @ innovation
        self.covar = covar_pred - kalman_gain @ meas_covar @ kalman_gain.T
