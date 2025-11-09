//! Contains vector operations for fixed-size vectors, supporting basic arithmetic, dot products, and cross products.

use crate::{
    Const, DefaultRepr, Dim, DimMul, Field, Matrix, OwnedRepr, RealField, Scalar, Tensor, ToTypenum,
};

// The array macro is exported at crate root
use crate::array as tensor;

/// N-dimensional Vector
pub type Vector<T, const N: usize, P = DefaultRepr> = Tensor<T, Const<N>, P>;

pub type Vector3<T, R = DefaultRepr> = Vector<T, 3, R>;

impl<T: Field, R: OwnedRepr> Vector<T, 3, R> {
    pub fn new(
        x: impl Into<Scalar<T, R>>,
        y: impl Into<Scalar<T, R>>,
        z: impl Into<Scalar<T, R>>,
    ) -> Self {
        Self::from_arr([x.into(), y.into(), z.into()])
    }

    pub fn x(&self) -> Scalar<T, R> {
        self.get(0)
    }

    pub fn y(&self) -> Scalar<T, R> {
        self.get(1)
    }

    pub fn z(&self) -> Scalar<T, R> {
        self.get(2)
    }
}

impl<T: Field> Vector<T, 3, crate::array::ArrayRepr> {
    pub fn x_axis() -> Self {
        tensor![T::one_prim(), T::zero_prim(), T::zero_prim()].into()
    }
    pub fn y_axis() -> Self {
        tensor![T::zero_prim(), T::one_prim(), T::zero_prim()].into()
    }
    pub fn z_axis() -> Self {
        tensor![T::zero_prim(), T::zero_prim(), T::one_prim()].into()
    }
}

impl<T: Field, const N: usize, R: OwnedRepr> Vector<T, N, R> {
    /// Creates a vector from an array of scalar references.
    pub fn from_arr(arr: [Scalar<T, R>; N]) -> Self
    where
        Const<N>: Dim,
        Const<N>: ToTypenum,
        Const<1>: DimMul<Const<N>, Output = Const<N>>,
        <Const<1> as DimMul<Const<N>>>::Output: Dim,
    {
        Self::from_inner(R::concat_many(arr.into_iter().map(|x| x.inner), 0))
    }
}

impl<T: Field, const N: usize, R: OwnedRepr> Vector<T, N, R> {
    /// Returns the individual scalar components of the vector as an array.
    pub fn parts(&self) -> [Scalar<T, R>; N] {
        let mut arr = [(); N].map(|_| None);
        for (idx, val) in arr.iter_mut().enumerate() {
            *val = Some(self.get(idx));
        }
        arr.map(|x| x.unwrap())
    }
}

impl<T: RealField, R: OwnedRepr> Vector<T, 3, R> {
    /// Computes the cross product of two 3-dimensional vectors.
    pub fn cross(&self, other: &Self) -> Self {
        let u = self.parts();
        let v = other.parts();

        Self::new(
            u[1].clone() * v[2].clone() - u[2].clone() * v[1].clone(),
            u[2].clone() * v[0].clone() - u[0].clone() * v[2].clone(),
            u[0].clone() * v[1].clone() - u[1].clone() * v[0].clone(),
        )
    }

    /// Returns the skew-symmetric matrix representing the cross product as a linear operator.
    pub fn skew(&self) -> Matrix<T, 3, 3, R> {
        let v = self.parts();
        let zero = T::zero::<R>();

        // Build skew-symmetric matrix directly from scalars
        // [  0  -z   y ]
        // [  z   0  -x ]
        // [ -y   x   0 ]
        Tensor::from_inner(R::from_scalars(
            [
                zero.clone(),
                -v[2].clone(),
                v[1].clone(),
                v[2].clone(),
                zero.clone(),
                -v[0].clone(),
                -v[1].clone(),
                v[0].clone(),
                zero,
            ]
            .into_iter()
            .map(|s| s.inner),
            &[3, 3],
        ))
    }
}

impl<T: Field + RealField, const N: usize, R: OwnedRepr> Vector<T, N, R> {
    /// Computes the norm squared of the vector.
    pub fn norm_squared(&self) -> Scalar<T, R> {
        self.dot(self)
    }

    /// Computes the norm (magnitude) of the vector.
    pub fn norm(&self) -> Scalar<T, R> {
        self.norm_squared().sqrt()
    }

    /// Returns a normalized (unit length) version of the vector.
    pub fn normalize(&self) -> Self {
        self.clone() / self.norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::FRAC_PI_4;
    // Import specific items to avoid ambiguity
    use crate::{Scalar, Vector3};

    #[test]
    fn test_vector_from_arr() {
        let a = Scalar::from(1.0f32);
        let b = Scalar::from(2.0);
        let c = Scalar::from(3.0);
        let vec: Vector<f32, 3> = Vector::from_arr([a, b, c]);
        assert_eq!(vec, tensor![1.0, 2.0, 3.0].into());
    }

    #[test]
    fn test_vector_scalar_mult() {
        let vec: Vector<f32, 3> = tensor![1.0f32, 2.0, 3.0].into();
        let result = vec * 2.0;
        assert_eq!(result, tensor![2.0, 4.0, 6.0].into());
    }

    #[test]
    fn test_vector_dot() {
        let a: Vector<f32, 3> = tensor![1.0f32, 2.0, 3.0].into();
        let b: Vector<f32, 3> = tensor![2.0, 3.0, 4.0].into();
        let result = a.dot(&b);
        assert_eq!(result, 20.0.into());
    }

    #[test]
    fn test_norm_squared() {
        let a: Vector<f32, 3> = tensor![1.0f32, 2.0, 3.0].into();
        let result = a.norm_squared();
        assert_eq!(result, 14.0.into());
    }

    #[test]
    fn test_norm() {
        let a: Vector<f32, 3> = tensor![1.0f32, 2.0, 3.0].into();
        let result = a.norm();
        assert_eq!(result, 14.0f32.sqrt().into());
    }

    #[test]
    fn test_vector_mul() {
        let a: Vector<f32, 3> = tensor![1.0f32, 2.0, 3.0].into();
        let b: Vector<f32, 3> = tensor![2.0, 3.0, 4.0].into();
        let c = a * b;
        assert_eq!(c, tensor![2.0, 6.0, 12.0].into());
    }

    #[test]
    fn test_abs() {
        let a: Vector<f32, 3> = tensor![-1.0f32, -2.0, -3.0].into();
        let result = a.abs();
        assert_eq!(result, tensor![1.0, 2.0, 3.0].into());
    }

    #[test]
    fn test_atan2() {
        let a: Vector<f64, 3> = tensor![1.0f64; 3].into();
        let b: Vector<f64, 3> = tensor![1.0; 3].into();
        let result = a.atan2(&b);
        assert_eq!(result, tensor![FRAC_PI_4; 3].into());
    }

    #[test]
    fn test_cross() {
        let a: Vector<f64, 3> = tensor![1.0, 0.0, 0.0].into();
        let b: Vector<f64, 3> = tensor![0.0, 1.0, 0.0].into();
        let c = a.cross(&b);
        assert_eq!(c, tensor![0.0, 0.0, 1.0].into());
    }

    #[test]
    fn test_skew() {
        let a: Vector3<f64> = tensor![1.0, 2.0, 3.0].into();
        let skew_mat = a.skew();
        assert_eq!(
            skew_mat,
            tensor![[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]].into()
        );
    }

    #[test]
    fn test_outer() {
        let a: Vector<f64, 3> = tensor![1.0, 2.0, 3.0].into();
        let b: Vector<f64, 2> = tensor![4.0, 5.0].into();
        let result = a.outer(&b);
        assert_eq!(
            result,
            tensor![[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]].into()
        );
    }
}
