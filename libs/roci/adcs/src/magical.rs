//! MAG.I.CAL magnotometer calibration algorithm
//!
//! Source: https://doi.org/10.1109/JSEN.2019.2919179

use nox::{ArrayRepr, Const, Tensor};
use nox::{Field, Matrix, Matrix3, Vector};

fn calibration_step<const N: usize>(
    y: Matrix<f64, 3, N, ArrayRepr>,
    m: Matrix<f64, 3, N, ArrayRepr>,
) -> Option<(Matrix3<f64, ArrayRepr>, Vector<f64, 3, ArrayRepr>)> {
    let ones: Tensor<f64, (Const<1>, Const<N>), ArrayRepr> = f64::one().broadcast();
    let g: Tensor<f64, (Const<4>, Const<N>), ArrayRepr> = m.concat(ones);
    let g_t = g.transpose();
    let g_g_t_inv = g.dot(&g_t).try_inverse().ok()?;
    let y_g_t = y.dot(&g_t);
    let l = y_g_t.dot(&g_g_t_inv);
    let t: Matrix<f64, 3, 3, ArrayRepr> = l.fixed_slice(&[0, 0]);
    let h: Matrix<f64, 1, 3, ArrayRepr> = l.fixed_slice(&[0, 3]).transpose();
    let h = Vector::from_buf(h.into_buf()[0]);
    Some((t, h))
}

pub struct Result<const N: usize> {
    pub m: Matrix<f64, 3, N, ArrayRepr>,
    pub t: Matrix<f64, 3, 3, ArrayRepr>,
    pub h: Vector<f64, 3, ArrayRepr>,
}

pub fn calibrate<const N: usize>(y: [Vector<f64, 3, ArrayRepr>; N]) -> Option<Result<N>> {
    let m = y.map(|y| y.normalize());
    let mut m = Matrix::from_buf(m.map(Tensor::into_buf)).transpose();
    let y_mat = Matrix::from_buf(y.map(Tensor::into_buf)).transpose();
    for _ in 0..32 {
        let (t, h) = calibration_step(y_mat, m)?;
        let t_inv = t.try_inverse().ok()?;
        let m_tilde = y.map(|y| t_inv.dot(&(y - h)));
        let j: f64 = m_tilde
            .iter()
            .map(|m_k| (m_k.norm_squared().into_buf() - 1.0).powi(2))
            .sum();
        m = Matrix::from_buf(m_tilde.map(|m| m.normalize().into_buf())).transpose();
        if j < 0.00001 {
            return Some(Result { m, t, h });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::tests::test_mag_readings;

    use super::*;
    use approx::assert_relative_eq;
    use nox::tensor;

    #[test]
    fn test_calibrate() {
        let readings = test_mag_readings();
        let Result { m, h, t } = calibrate(readings).expect("failed to converge");
        assert_relative_eq!(h, tensor![2.0, 10.0, 40.0], epsilon = 1e-2); // original value from matlab
        let t_normalized: Matrix3<f64, ArrayRepr> = {
            let x: Matrix<f64, 1, 3, _> = t.fixed_slice(&[0, 0]).transpose();
            let x: Vector<f64, 3, _> = Vector::from_buf(x.into_buf()[0]).normalize();
            let y: Matrix<f64, 1, 3, _> = t.fixed_slice(&[0, 1]).transpose();
            let y = Vector::from_buf(y.into_buf()[0]).normalize();
            let z: Matrix<f64, 1, 3, _> = t.fixed_slice(&[0, 2]).transpose();
            let z = Vector::from_buf(z.into_buf()[0]).normalize();
            Matrix::from_buf([x, y, z].map(Tensor::into_buf))
        };
        assert_relative_eq!(
            tensor![
                [0.9738617339145248, 0.11686340806974296, 0.19477234678290495],
                [0.14762034939153687, 0.9841356626102459, 0.0984135662610246],
                [0.16404467873119666, 0.06561787149247868, 0.9842680723871801]
            ],
            t_normalized,
            epsilon = 1e-2
        ); // value normalized from matlab

        let expected =
            Tensor::from_buf(crate::tests::normalized_cal_mag_readings().map(|v| v.into_buf()));
        assert_relative_eq!(m.transpose(), expected, epsilon = 1e-3);
    }
}
