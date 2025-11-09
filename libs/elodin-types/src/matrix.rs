//! Provides matrix operations and types, supporting various arithmetic and linear algebra operations on fixed-size matrices.

use crate::{
    Const, DefaultRepr, Dim, Error, OwnedRepr, RealField, SquareDim, Tensor, TensorDim, Vector,
};

// The array macro is exported at crate root  
// Used in tests
#[allow(unused_imports)]
use crate::array as tensor;

/// 2D Matrix of shape RxC
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
        let inner = Rep::stack(rows.into_iter().map(|v| v.inner), 0);
        Self::from_inner(inner)
    }
}

impl<T: RealField, D: Dim, R: OwnedRepr> Tensor<T, (D, D), R>
where
    (D, D): SquareDim,
    D: TensorDim,
{
    pub fn try_inverse(&self) -> Result<Self, Error> {
        R::try_lu_inverse(&self.inner).map(Tensor::from_inner)
    }
}

// TODO: This is really important but will require us to modify
// the representation.
// impl<T: RealField, const R: usize, const C: usize, Rep: OwnedRepr> Transpose
//     for Matrix<T, R, C, Rep>
// {
//     type Transposed = Matrix<T, C, R, Rep>;

//     fn transpose(&self) -> Self::Transposed {
//         let inner = <Rep as Repr>::transpose(&self.inner);

//         Matrix::<T, C, R, Rep>::from_inner(inner)
//     }
// }

// /// Performs matrix multiplication between two matrices, resulting in a new matrix with the specified dimensions.
// impl<T: RealField, const R: usize, const M: usize, const C: usize, P: OwnedRepr>
//     Mul<Matrix<T, M, C, P>> for Matrix<T, R, M, P>
// {
//     type Output = Matrix<T, R, C, P>;

//     fn mul(self, rhs: Matrix<T, M, C, P>) -> Self::Output {
//         let inner = <P as Repr>::dot(&self.inner, &rhs.inner);
//         Matrix::<T, R, C, P>::from_inner(inner)
//     }
// }

// /// Performs matrix multiplication with a vector, treating the vector as a column matrix.
// impl<T: RealField, const R: usize, const C: usize, P: OwnedRepr> Mul<Vector<T, C, P>>
//     for Matrix<T, R, C, P>
// {
//     type Output = Vector<T, R, P>;

//     fn mul(self, rhs: Vector<T, C, P>) -> Self::Output {
//         let inner = <P as Repr>::dot(&self.inner, &rhs.inner);
//         Vector::<T, R, P>::from_inner(inner)
//     }
// }

// impl<
//     const R: usize,
//     const C: usize,
//     T: RealField + Mul<Output = T> + Div<Output = T>,
//     Rep: OwnedRepr,
// > Div<T> for Matrix<T, R, C, Rep>
// {
//     type Output = Matrix<T, R, C, Rep>;

//     fn div(self, rhs: T) -> Self::Output {
//         self * (T::one_prim() / rhs)
//     }
// }

// TODO: Implement these in a repr-agnostic way (i.e. not requiring ArrayRepr specifically)
// For now just use the ArrayRepr implementation directly

impl<T: RealField> Matrix3<T, crate::array::ArrayRepr> {
    pub fn look_at_rh(
        dir: impl Into<Vector<T, 3, crate::array::ArrayRepr>>,
        up: impl Into<Vector<T, 3, crate::array::ArrayRepr>>,
    ) -> Self {
        let dir = dir.into();
        let up = up.into();
        // TODO: This needs a better orthogonalization process. Gram-Schdmit?
        let f = dir.normalize();
        let s = f.cross(&up).normalize();
        let u = s.cross(&f);

        let m = Self::from_rows([s, f, u]);
        m.transpose()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Vector3};
    use super::*;

    #[test]
    fn test_add() {
        let a: Matrix<f32, 1, 2> = tensor![[1.0f32, 2.0]].into();
        let b: Matrix<f32, 1, 2> = tensor![[2.0, 3.0]].into();
        let c = a + b;
        assert_eq!(c, tensor![[3.0, 5.0]].into());
    }

    #[test]
    fn test_sub() {
        let a: Matrix<f32, 1, 2> = tensor![[1.0f32, 2.0]].into();
        let b: Matrix<f32, 1, 2> = tensor![[2.0, 3.0]].into();
        let c = a - b;
        assert_eq!(c, tensor![[-1.0, -1.0]].into());
    }

    #[test]
    fn test_mul() {
        let a: Matrix<f32, 2, 2> = tensor![[1.0f32, 2.0], [2.0, 3.0]].into();
        let b: Matrix<f32, 2, 2> = tensor![[2.0, 3.0], [4.0, 5.0]].into();
        let c = a * b;
        assert_eq!(c, tensor![[2., 6.], [8., 15.]].into());
    }

    #[test]
    fn test_fixed_slice() {
        let mat: Matrix<f32, 1, 4> = tensor![[1.0f32, 2.0, 3.0, 4.0]].into();
        let slice = mat.fixed_slice::<(Const<1>, Const<1>)>(&[0, 2]);
        assert_eq!(slice, tensor![[3.0]].into());
    }

    #[test]
    fn test_eye() {
        let m: Matrix<f32, 3, 3> = Matrix::eye();
        assert_eq!(
            m,
            tensor![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into()
        );
    }

    #[test]
    fn test_diag() {
        let m: Matrix<f32, 3, 3> = Matrix::from_diag(tensor![1.0, 4.0, 8.0].into());
        assert_eq!(
            m,
            tensor![[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]].into()
        );
    }

    #[test]
    fn test_from_rows() {
        let m: Matrix<f32, 3, 3> = Matrix::from_rows([
            tensor![1.0, 2.0, 3.0].into(),
            tensor![4.0, 5.0, 6.0].into(),
            tensor![7.0, 8.0, 9.0].into(),
        ]);
        assert_eq!(
            m,
            tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into()
        );
    }

    #[test]
    fn test_look_at() {
        // source: nalgebra
        // `Rotation3::look_at_rh(&Vector3::new(1.0,2.0,3.0).normalize(), &Vector3::y_axis())`
        let expected: Matrix3<f64, crate::array::ArrayRepr> = tensor![
            [
                -0.9486832980505139,
                0.2672612419124244,
                -0.16903085094570333
            ],
            [0.0, 0.5345224838248488, 0.8451542547285167],
            [0.31622776601683794, 0.8017837257372732, -0.50709255283711]
        ].into();
        
        let v: Vector3<f64, crate::array::ArrayRepr> = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(
            Matrix3::look_at_rh(v.normalize(), Vector3::y_axis()),
            expected
        );

        let v2: Vector3<f64, crate::array::ArrayRepr> = Vector3::new(0.0, 1.0, 0.0);
        let m: Matrix3<f64, crate::array::ArrayRepr> =
            Matrix3::look_at_rh(v2.normalize(), Vector3::z_axis());
        assert_eq!(m, Matrix3::eye());

        let v3: Vector3<f64, crate::array::ArrayRepr> = Vector3::new(1.0, 0.0, 0.0);
        let m: Matrix3<f64, crate::array::ArrayRepr> =
            Matrix3::look_at_rh(v3.normalize(), Vector3::y_axis());
        let test_vec: Vector3<f64, crate::array::ArrayRepr> = tensor![0.0, 0.0, -1.0].into();
        assert_eq!(m.dot(&test_vec), tensor![0.0, -1.0, 0.0].into());
    }
}