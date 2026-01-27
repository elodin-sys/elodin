//! Provides a Matrix type alias with convenience functions for converting to various representations.
use core::marker::PhantomData;

use crate::{ArrayRepr, Const};
use crate::{
    DefaultRepr, Dim, Error, NonScalarDim, NonTupleDim, OwnedRepr, RealField, SquareDim, Tensor,
    Vector,
};

/// Type alias for a tensor that specifically represents a matrix.
pub type Matrix<T, const R: usize, const C: usize, P = DefaultRepr> =
    Tensor<T, (Const<R>, Const<C>), P>;

pub type Matrix3<T, R = DefaultRepr> = Matrix<T, 3, 3, R>;
pub type Matrix3x6<T, R = DefaultRepr> = Matrix<T, 3, 6, R>;
pub type Matrix4<T, R = DefaultRepr> = Matrix<T, 4, 4, R>;
pub type Matrix5<T, R = DefaultRepr> = Matrix<T, 5, 5, R>;
pub type Matrix6<T, R = DefaultRepr> = Matrix<T, 6, 6, R>;
pub type Matrix6x3<T, R = DefaultRepr> = Matrix<T, 6, 3, R>;

impl<const R: usize, const C: usize, T: RealField, Rep: OwnedRepr> Matrix<T, R, C, Rep> {
    pub fn from_rows(rows: [Vector<T, C, Rep>; R]) -> Self {
        let arr = rows.map(|x| x.inner);
        let inner = Rep::concat_many(arr, 0);
        Matrix {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: RealField, D: Dim, R: OwnedRepr> Tensor<T, (D, D), R>
where
    D: NonTupleDim + NonScalarDim,
    (D, D): Dim + SquareDim<SideDim = D>,
{
    pub fn try_inverse(&self) -> Result<Self, Error> {
        let shape = R::shape(&self.inner);
        match shape.as_ref().first() {
            Some(2) => {
                let m11 = self.get([0, 0]);
                let m12 = self.get([0, 1]);
                let m21 = self.get([1, 0]);
                let m22 = self.get([1, 1]);

                let determinant = &m11 * &m22 - &m21 * &m12;

                let parts = [
                    m22 / &determinant,
                    -m12 / &determinant,
                    -m21 / &determinant,
                    m11 / &determinant,
                ];
                Ok(Tensor::from_scalars_with_shape(parts, &[2, 2]))
            }
            Some(3) => {
                let m11 = self.get([0, 0]);
                let m12 = self.get([0, 1]);
                let m13 = self.get([0, 2]);

                let m21 = self.get([1, 0]);
                let m22 = self.get([1, 1]);
                let m23 = self.get([1, 2]);

                let m31 = self.get([2, 0]);
                let m32 = self.get([2, 1]);
                let m33 = self.get([2, 2]);

                let minor_m12_m23 = &m22 * &m33 - &m32 * &m23;
                let minor_m11_m23 = &m21 * &m33 - &m31 * &m23;
                let minor_m11_m22 = &m21 * &m32 - &m31 * &m22;

                let determinant =
                    &m11 * &minor_m12_m23 - &m12 * &minor_m11_m23 + &m13 * &minor_m11_m22;

                let parts = [
                    &minor_m12_m23 / &determinant,
                    (&m13 * &m32 - &m33 * &m12) / &determinant,
                    (&m12 * &m23 - &m22 * &m13) / &determinant,
                    -&minor_m11_m23 / &determinant,
                    (&m11 * &m33 - &m31 * &m13) / &determinant,
                    (&m13 * &m21 - &m23 * &m11) / &determinant,
                    &minor_m11_m22 / &determinant,
                    (&m12 * &m31 - &m32 * &m11) / &determinant,
                    (m11 * m22 - m21 * m12) / &determinant,
                ];
                Ok(Tensor::from_scalars_with_shape(parts, &[3, 3]))
            }
            _ => R::try_lu_inverse(&self.inner).map(Tensor::from_inner),
        }
    }
}

impl<T: RealField> Matrix3<T, ArrayRepr> {
    pub fn look_at_rh(
        dir: impl Into<Vector<T, 3, ArrayRepr>>,
        up: impl Into<Vector<T, 3, ArrayRepr>>,
    ) -> Self {
        let dir = dir.into();
        let up = up.into();
        // apply gram-schmidt orthogonalization to create a rot matrix
        let f = dir.normalize();
        let up = if up.dot(&dir).abs() == T::one() {
            Vector::y_axis()
        } else {
            up
        };
        let s = f.cross(&up).normalize();
        let u = s.cross(&f);
        Self::from_rows([s, f, u]).transpose()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Vector3, tensor};

    use super::*;

    // NOTE: XLA-based execution tests commented out during IREE migration
    // These tests used Client::cpu() and .build().compile() which are no longer available
    // TODO: Re-implement tests using IREE execution path or ArrayRepr local execution

    #[test]
    fn test_look_at() {
        // source: nalgebra
        // `Rotation3::look_at_rh(&Vector3::new(1.0,2.0,3.0).normalize(), &Vector3::y_axis())`
        let expected = tensor![
            [
                -0.9486832980505139,
                0.2672612419124244,
                -0.16903085094570333
            ],
            [0.0, 0.5345224838248488, 0.8451542547285167],
            [0.31622776601683794, 0.8017837257372732, -0.50709255283711]
        ];
        assert_eq!(
            Matrix3::look_at_rh(Vector3::new(1.0, 2.0, 3.0).normalize(), Vector3::y_axis()),
            expected
        );

        let m: Matrix3<f64, crate::ArrayRepr> =
            Matrix3::look_at_rh(Vector3::new(0.0, 1.0, 0.0).normalize(), Vector3::z_axis());
        assert_eq!(m, Matrix3::eye());

        let m: Matrix3<f64, crate::ArrayRepr> =
            Matrix3::look_at_rh(Vector3::new(1.0, 0.0, 0.0).normalize(), Vector3::y_axis());
        assert_eq!(m.dot(&tensor![0.0, 0.0, -1.0]), tensor![0.0, -1.0, 0.0]);
    }
}
