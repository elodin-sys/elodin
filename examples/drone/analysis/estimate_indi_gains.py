#!/usr/bin/env uv run

import math
import os

import numpy as np
import polars as pl


def lpf(cutoff_freq: float, sample_freq: float, data: np.ndarray) -> np.ndarray:
    """
    Implements a second-order (biquad) low-pass filter.
    Based on the cookbook formulas from:
    https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html

    Args:
        cutoff_freq: Corner/cutoff frequency in Hz
        sample_freq: Sampling frequency in Hz
        data: Input data array of shape (N, C) where N is number of samples, C is channels

    Returns:
        filtered_data: Filtered output with same shape as input
    """
    assert cutoff_freq > 0
    assert sample_freq > 0

    Q = 1 / math.sqrt(2)
    omega = 2 * math.pi * cutoff_freq / sample_freq
    alpha = math.sin(omega) / (2 * Q)
    a0 = 1 + alpha

    b0 = (1 - math.cos(omega)) / 2
    b1 = 1 - math.cos(omega)
    b2 = b0
    a1 = -2 * math.cos(omega)
    a2 = 1 - alpha

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    N, C = data.shape  # N is number of samples, C is number of channels
    filtered_data = np.zeros_like(data)
    # delay is [x_n-1, x_n-2, y_n-1, y_n-2]
    delay = np.zeros([4, C])
    for i in range(N):
        x_n = data[i]
        # assert that the shape of delay is (4, <shape of x_n>):
        assert delay.shape == (4, *x_n.shape)
        x_n1, x_n2, y_n1, y_n2 = delay
        y_n = b0 * x_n + b1 * x_n1 + b2 * x_n2 - a1 * y_n1 - a2 * y_n2
        delay = np.array([x_n, x_n1, y_n, y_n1])
        filtered_data[i] = delay[2]
    return filtered_data


def load_flight_data() -> tuple[np.ndarray, np.ndarray]:
    telemetry = os.path.join(os.path.dirname(__file__), "../telemetry.csv")
    df = pl.read_csv(telemetry)
    body_ang_rate = df.select(["body_ang_vel_x", "body_ang_vel_y", "body_ang_vel_z"]).to_numpy()
    motor_ang_rate = df.select(
        ["motor_ang_vel_1", "motor_ang_vel_2", "motor_ang_vel_3", "motor_ang_vel_4"]
    ).to_numpy()
    return body_ang_rate, motor_ang_rate


if __name__ == "__main__":
    dt = 1.0 / 300.0
    body_ang_rate, motor_ang_rate = load_flight_data()

    T = body_ang_rate.shape[0]
    M = T - 2

    body_ang_rate_f = lpf(20, 1 / dt, body_ang_rate)
    motor_ang_rate_f = lpf(20, 1 / dt, motor_ang_rate)

    body_ang_accel = np.diff(body_ang_rate_f, axis=0) / dt  # Shape: (T-1, 3)
    d_body_ang_accel = np.diff(body_ang_accel, axis=0)  # Shape: (T-2, 3)

    d_motor_ang_rate_f = np.diff(motor_ang_rate_f, axis=0)  # Shape: (T-1, 4)
    d_motor_ang_rate_f_t = d_motor_ang_rate_f[1:]  # Shape: (T-2, 4)
    d_motor_ang_rate_f_t_1 = d_motor_ang_rate_f[:-1]  # Shape: (T-2, 4)
    motor_ang_rate_f_t = motor_ang_rate_f[2:]  # Shape: (T-2, 4)

    phi1 = motor_ang_rate_f_t * d_motor_ang_rate_f_t  # Shape: (T-2, 4)
    phi2 = d_motor_ang_rate_f_t - d_motor_ang_rate_f_t_1  # Shape: (T-2, 4)

    Y = d_body_ang_accel
    Y = Y.T
    Y_flat = Y.reshape(-1)

    X_design = np.zeros((3 * M, 24))
    phi1_kron = np.kron(np.eye(3), phi1)
    phi2_kron = np.kron(np.eye(3), phi2)
    X_design[:, :12] = phi1_kron
    X_design[:, 12:] = phi2_kron

    theta_hat, residuals, rank, s = np.linalg.lstsq(X_design, Y_flat, rcond=None)
    G1_hat = theta_hat[:12].reshape(3, 4)
    G2_hat = theta_hat[12:].reshape(3, 4)

    Y_pred = X_design.dot(theta_hat)
    residuals = Y_flat - Y_pred
    MSE = np.mean(residuals**2)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((Y_flat - np.mean(Y_flat)) ** 2)
    R_squared = 1 - (SS_res / SS_tot)

    print(f"Estimated G1:\n{G1_hat}")
    print(f"Estimated G2:\n{G2_hat}")
    print(f"Mean Squared Error: {MSE}")
    print(f"R^2 Score: {R_squared}")

    print(f"g1=np.array({np.array2string(G1_hat, separator=',')}),")
    print(f"g2=np.array({np.array2string(G2_hat, separator=',')}),")
