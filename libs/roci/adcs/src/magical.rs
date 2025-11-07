//! MAG.I.CAL magnotometer calibration algorithm
//!
//! Source: https://doi.org/10.1109/JSEN.2019.2919179

use nalgebra::{DMatrix, Matrix3, SMatrix, Vector3};

fn calibration_step<const N: usize>(
    y: SMatrix<f64, 3, N>,
    m: SMatrix<f64, 3, N>,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let g = DMatrix::from_fn(4, N, |i, j| if i < 3 { m[(i, j)] } else { 1.0 });
    let g_t = g.transpose();
    let g_g_t_inv = (g * g_t.clone()).try_inverse()?;
    let y_g_t = DMatrix::from_fn(3, N, |i, j| y[(i, j)]) * g_t;
    let l = y_g_t * g_g_t_inv;
    let t: Matrix3<f64> = l.fixed_view::<3, 3>(0, 0).into_owned();
    let h: Vector3<f64> = l.fixed_view::<3, 1>(0, 3).into_owned();
    Some((t, h))
}

pub struct Result<const N: usize> {
    pub m: SMatrix<f64, 3, N>,
    pub t: Matrix3<f64>,
    pub h: Vector3<f64>,
}

pub fn calibrate<const N: usize>(y: [Vector3<f64>; N]) -> Option<Result<N>> {
    let m: [Vector3<f64>; N] = y.map(|y| y.normalize());
    let mut m_mat = SMatrix::<f64, 3, N>::from_fn(|i, j| m[j][i]);
    let y_mat = SMatrix::<f64, 3, N>::from_fn(|i, j| y[j][i]);

    for _ in 0..32 {
        let (t, h) = calibration_step(y_mat, m_mat)?;
        let t_inv = t.try_inverse()?;
        let m_tilde: [Vector3<f64>; N] = y.map(|y_vec| t_inv * (y_vec - h));
        let j: f64 = m_tilde
            .iter()
            .map(|m_k| (m_k.norm_squared() - 1.0).powi(2))
            .sum();
        m_mat = SMatrix::<f64, 3, N>::from_fn(|i, j| m_tilde[j].normalize()[i]);
        if j < 0.00001 {
            return Some(Result { m: m_mat, t, h });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::tests::test_mag_readings;

    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::vector;

    #[test]
    fn test_calibrate() {
        let readings = test_mag_readings();
        let Result { m, h, t } = calibrate(readings).expect("failed to converge");
        assert_relative_eq!(h, vector![2.0, 10.0, 40.0], epsilon = 1e-2); // original value from matlab
        let t_normalized: Matrix3<f64> = {
            let x: Vector3<f64> = t.fixed_view::<3, 1>(0, 0).into_owned().normalize();
            let y: Vector3<f64> = t.fixed_view::<3, 1>(0, 1).into_owned().normalize();
            let z: Vector3<f64> = t.fixed_view::<3, 1>(0, 2).into_owned().normalize();
            Matrix3::from_columns(&[x, y, z])
        };
        // The matlab value was in column-major format, so we transpose it
        let expected_t = Matrix3::from_row_slice(&[
            0.9738617339145248,
            0.11686340806974296,
            0.19477234678290495,
            0.14762034939153687,
            0.9841356626102459,
            0.0984135662610246,
            0.16404467873119666,
            0.06561787149247868,
            0.9842680723871801,
        ])
        .transpose();
        assert_relative_eq!(t_normalized, expected_t, epsilon = 1e-2); // value normalized from matlab

        let expected_mat = SMatrix::<f64, 3, 500>::from_fn(|i, j| {
            crate::tests::normalized_cal_mag_readings()[j][i]
        });
        assert_relative_eq!(m, expected_mat, epsilon = 1e-3);
    }
}
