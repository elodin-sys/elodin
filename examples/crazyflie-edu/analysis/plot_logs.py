#!/usr/bin/env python3
"""
Log Analysis Script for Crazyflie Educational Labs

This script provides utilities for analyzing simulation data exported
from the Elodin editor.

Usage:
    python analysis/plot_logs.py <log_file.csv>

Or import functions in your own analysis:
    from analysis.plot_logs import load_log, plot_sensors
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_log(filepath: str) -> pd.DataFrame:
    """
    Load a CSV log file exported from Elodin.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with log data
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Columns: {list(df.columns)}")
    return df


def plot_gyro(df: pd.DataFrame, time_col: str = "time", save_path: str = None):
    """
    Plot gyroscope data from log.

    Args:
        df: DataFrame with gyro columns
        time_col: Name of time column
        save_path: If provided, save plot to this path
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    gyro_cols = ["gyro.x", "gyro.y", "gyro.z"]
    labels = ["Roll Rate (X)", "Pitch Rate (Y)", "Yaw Rate (Z)"]
    colors = ["r", "g", "b"]

    for i, (col, label, color) in enumerate(zip(gyro_cols, labels, colors)):
        if col in df.columns:
            axes[i].plot(df[time_col], df[col], color=color, linewidth=0.5)
            axes[i].set_ylabel(f"{label}\n(rad/s)")
            axes[i].grid(True, alpha=0.3)

            # Add statistics
            mean = df[col].mean()
            std = df[col].std()
            axes[i].axhline(mean, color=color, linestyle="--", alpha=0.5)
            axes[i].text(
                0.02,
                0.95,
                f"μ={mean:.4f}, σ={std:.4f}",
                transform=axes[i].transAxes,
                verticalalignment="top",
            )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Gyroscope Data")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_accel(df: pd.DataFrame, time_col: str = "time", save_path: str = None):
    """
    Plot accelerometer data from log.

    Args:
        df: DataFrame with accel columns
        time_col: Name of time column
        save_path: If provided, save plot to this path
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    accel_cols = ["accel.x", "accel.y", "accel.z"]
    labels = ["Forward (X)", "Left (Y)", "Up (Z)"]
    colors = ["r", "g", "b"]

    for i, (col, label, color) in enumerate(zip(accel_cols, labels, colors)):
        if col in df.columns:
            axes[i].plot(df[time_col], df[col], color=color, linewidth=0.5)
            axes[i].set_ylabel(f"{label}\n(g)")
            axes[i].grid(True, alpha=0.3)

            mean = df[col].mean()
            axes[i].axhline(mean, color=color, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Accelerometer Data")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_motors(df: pd.DataFrame, time_col: str = "time", save_path: str = None):
    """
    Plot motor data from log.

    Args:
        df: DataFrame with motor columns
        time_col: Name of time column
        save_path: If provided, save plot to this path
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # PWM commands
    pwm_cols = ["motor_pwm.m1", "motor_pwm.m2", "motor_pwm.m3", "motor_pwm.m4"]
    for col in pwm_cols:
        if col in df.columns:
            axes[0].plot(df[time_col], df[col], linewidth=0.8, label=col.split(".")[-1])
    axes[0].set_ylabel("PWM Command")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # RPM
    rpm_cols = ["motor_rpm.m1", "motor_rpm.m2", "motor_rpm.m3", "motor_rpm.m4"]
    for col in rpm_cols:
        if col in df.columns:
            axes[1].plot(df[time_col], df[col], linewidth=0.8, label=col.split(".")[-1])
    axes[1].set_ylabel("RPM")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Motor Data")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def compute_statistics(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Compute statistics for specified columns.

    Args:
        df: DataFrame with data
        columns: List of column names to analyze

    Returns:
        DataFrame with statistics
    """
    stats = []
    for col in columns:
        if col in df.columns:
            stats.append(
                {
                    "column": col,
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                }
            )

    return pd.DataFrame(stats)


def compare_noise(
    df_noise_on: pd.DataFrame, df_noise_off: pd.DataFrame, columns: list
) -> pd.DataFrame:
    """
    Compare statistics between noise-on and noise-off runs.

    Args:
        df_noise_on: DataFrame with sensor noise enabled
        df_noise_off: DataFrame with sensor noise disabled
        columns: List of columns to compare

    Returns:
        DataFrame with comparison
    """
    comparison = []
    for col in columns:
        if col in df_noise_on.columns and col in df_noise_off.columns:
            comparison.append(
                {
                    "column": col,
                    "noise_on_mean": df_noise_on[col].mean(),
                    "noise_on_std": df_noise_on[col].std(),
                    "noise_off_mean": df_noise_off[col].mean(),
                    "noise_off_std": df_noise_off[col].std(),
                }
            )

    return pd.DataFrame(comparison)


def fit_pwm_to_speed(pwm: np.ndarray, rpm: np.ndarray) -> tuple:
    """
    Fit affine model: omega = a + b * pwm

    Args:
        pwm: Array of PWM commands
        rpm: Array of measured RPM values

    Returns:
        Tuple of (a, b) coefficients
    """
    # Convert RPM to rad/s
    omega = rpm * 2 * np.pi / 60

    # Fit: omega = a + b * pwm
    A = np.vstack([np.ones_like(pwm), pwm]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, omega, rcond=None)

    a, b = coeffs
    print(f"PWM to Speed fit: omega = {a:.2f} + {b:.4f} * PWM")
    return a, b


def fit_speed_to_force(omega: np.ndarray, force: np.ndarray) -> float:
    """
    Fit quadratic model: F = k * omega^2

    Args:
        omega: Array of angular velocities (rad/s)
        force: Array of measured forces (N)

    Returns:
        Thrust constant k
    """
    # Fit: F = k * omega^2 (through origin)
    omega_sq = omega**2
    k = np.sum(force * omega_sq) / np.sum(omega_sq**2)

    print(f"Speed to Force fit: F = {k:.2e} * omega^2")
    return k


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Analyze Crazyflie simulation logs")
    parser.add_argument("logfile", help="Path to CSV log file")
    parser.add_argument(
        "--plot",
        choices=["gyro", "accel", "motors", "all"],
        default="all",
        help="What to plot",
    )
    parser.add_argument("--save", help="Save plots to this directory")
    args = parser.parse_args()

    # Load data
    df = load_log(args.logfile)

    # Create save directory if needed
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    if args.plot in ["gyro", "all"]:
        save_path = str(save_dir / "gyro.png") if save_dir else None
        plot_gyro(df, save_path=save_path)

    if args.plot in ["accel", "all"]:
        save_path = str(save_dir / "accel.png") if save_dir else None
        plot_accel(df, save_path=save_path)

    if args.plot in ["motors", "all"]:
        save_path = str(save_dir / "motors.png") if save_dir else None
        plot_motors(df, save_path=save_path)

    # Print statistics
    print("\nGyroscope Statistics:")
    gyro_stats = compute_statistics(df, ["gyro.x", "gyro.y", "gyro.z"])
    print(gyro_stats.to_string(index=False))

    print("\nAccelerometer Statistics:")
    accel_stats = compute_statistics(df, ["accel.x", "accel.y", "accel.z"])
    print(accel_stats.to_string(index=False))


if __name__ == "__main__":
    main()
