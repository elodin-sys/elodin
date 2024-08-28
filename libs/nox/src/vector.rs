//! Provides functionality for handling vectors in computational tasks, supporting conversion between host and Nox-specific representations, and enabling vector operations like extension, normalization, and cross products.
use crate::{
    tensor, ArrayRepr, DefaultRepr, Dim, Field, Matrix, RealField, Repr, Scalar, Tensor, TensorItem,
};
use nalgebra::{Const, DimMul, ToTypenum};

use std::marker::PhantomData;

/// Type alias for a tensor that specifically represents a vector.
pub type Vector<T, const N: usize, P = DefaultRepr> = Tensor<T, Const<N>, P>;

pub type Vector3<T, R = DefaultRepr> = Vector<T, 3, R>;

impl<T: Field, R: Repr> Vector<T, 3, R> {
    pub fn new(
        x: impl Into<Scalar<T, R>>,
        y: impl Into<Scalar<T, R>>,
        z: impl Into<Scalar<T, R>>,
    ) -> Self {
        Self::from_arr([x.into(), y.into(), z.into()])
    }
}

impl<T: Field, R: Repr> Vector<T, 3, R>
where
    Self: From<Vector<T, 3, ArrayRepr>>,
{
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

impl<T: TensorItem + Field, const N: usize, R: Repr> Vector<T, N, R> {
    /// Creates a vector from an array of scalar references.
    pub fn from_arr(arr: [Scalar<T, R>; N]) -> Self
    where
        Const<N>: Dim,
        Const<N>: ToTypenum,
        Const<1>: DimMul<Const<N>, Output = Const<N>>,
        <Const<1> as DimMul<Const<N>>>::Output: Dim,
    {
        let arr = arr.map(|x| x.inner);
        let inner = R::concat_many(arr, 0);
        Vector {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: TensorItem + Field, const N: usize, R: Repr> Vector<T, N, R> {
    /// Returns the individual scalar components of the vector as an array.
    pub fn parts(&self) -> [Scalar<T, R>; N] {
        let mut i = 0;
        [0; N].map(|_| {
            let elem = self.get(i);
            i += 1;
            elem
        })
    }
}

impl<T: RealField, R: Repr> Vector<T, 3, R> {
    /// Computes the cross product of two 3-dimensional vectors.
    pub fn cross(&self, other: &Self) -> Self {
        let [ax, ay, az] = self.parts();
        let [bx, by, bz] = other.parts();
        let x = &ay * &bz - &az * &by;
        let y = &az * &bx - &ax * &bz;
        let z = &ax * &by - &ay * &bx;
        Vector::from_arr([x, y, z])
    }

    /// Computes the skew-symmetric matrix representation of the vector.
    /// This is the matrix that, when multiplied with another vector, computes the cross product of the two vectors.
    pub fn skew(&self) -> Matrix<T, 3, 3, R> {
        let [x, y, z] = self.parts();
        Matrix::from_rows(
            [
                [T::zero(), -z.clone(), y.clone()],
                [z.clone(), T::zero(), -x.clone()],
                [-y.clone(), x.clone(), T::zero()],
            ]
            .map(Vector::from_arr),
        )
    }
}

impl<T: Field + RealField, const N: usize, R: Repr> Vector<T, N, R> {
    /// Computes the norm squared of the vector.
    pub fn norm_squared(&self) -> Scalar<T, R> {
        self.dot(self)
    }

    /// Computes the norm of the vector, which is the square root of the norm squared.
    pub fn norm(&self) -> Scalar<T, R> {
        self.dot(self).sqrt()
    }

    /// Normalizes the vector to a unit vector.
    pub fn normalize(&self) -> Self {
        self / self.norm()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_4;

    use nalgebra::{matrix, vector};

    use super::*;
    use crate::ToHost;
    use crate::*;

    #[test]
    fn test_vector_from_arr() {
        let client = Client::cpu().unwrap();
        fn from_arr(a: Scalar<f32>, b: Scalar<f32>, c: Scalar<f32>) -> Vector<f32, 3> {
            Vector::from_arr([a, b, c])
        }
        let comp = from_arr.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec.run(&client, 1.0f32, 2.0, 3.0).unwrap().to_host();
        assert_eq!(out, vector![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_scalar_mult() {
        let client = Client::cpu().unwrap();
        let comp = (|vec: Vector<f32, 3>| 2.0 * vec).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![1.0f32, 2.0, 3.0])
            .unwrap()
            .to_host();
        assert_eq!(out, vector![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vector_dot() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>, b| a.dot(&b)).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![1.0f32, 2.0, 3.0], vector![2.0, 3.0, 4.0])
            .unwrap()
            .to_host();
        assert_eq!(out, 20.0)
    }

    #[test]
    fn test_norm_squared() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>| a.norm_squared()).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![1.0f32, 2.0, 3.0])
            .unwrap()
            .to_host();
        assert_eq!(out, 14.0)
    }

    #[test]
    fn test_norm() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>| a.norm()).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![1.0f32, 2.0, 3.0])
            .unwrap()
            .to_host();
        assert_eq!(out, 14.0f32.sqrt())
    }

    #[test]
    fn test_vector_mul() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 1>, b: Vector<f32, 1>| a * b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![2.0f32], vector![2.0])
            .unwrap()
            .to_host();
        assert_eq!(out, vector![4.0f32])
    }

    #[test]
    fn test_extend() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>| a.extend(1.0)).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![2.0f32, 1.0, 2.0])
            .unwrap()
            .to_host();
        assert_eq!(out, vector![2.0, 1.0, 2.0, 1.0])
    }

    #[test]
    fn test_abs() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>| a.abs()).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![-2.0f32, 1.0, -2.0])
            .unwrap()
            .to_host();
        assert_eq!(out, vector![2.0, 1.0, 2.0])
    }

    #[test]
    fn test_atan2() {
        let client = Client::cpu().unwrap();
        let comp = (|x: Vector<f64, 2>, y: Vector<f64, 2>| y.atan2(&x))
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![3.0, -3.0], vector![-3.0, 3.0])
            .unwrap()
            .to_host();
        assert_eq!(out, vector![-FRAC_PI_4, 3.0 * FRAC_PI_4])
    }

    #[test]
    fn test_cross() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>, b: Vector<f32, 3>| a.cross(&b))
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![1.0, 0.0, 0.0], vector![0.0, 1.0, 0.0])
            .unwrap()
            .to_host();
        assert_eq!(out, vector![0.0, 0.0, 1.0])
    }

    #[test]
    fn test_skew() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>| a.skew()).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec.run(&client, vector![1.0, 2.0, 3.0]).unwrap().to_host();
        assert_eq!(
            out,
            matrix![
                0.0, -3.0, 2.0;
                3.0, 0.0, -1.0;
                -2.0, 1.0, 0.0
            ]
        )
    }

    #[test]
    fn test_outer() {
        let a = tensor![1.0, 2.0, 3.0];
        let b = tensor![1.0, 2.0, 3.0];
        let out = a.outer(&b);
        let expected = tensor![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        assert_eq!(out, expected);

        let a = tensor![1.0, 0.0, 3.0];
        let b = tensor![2.0, 2.0, 0.0];
        let out = a.outer(&b);
        let expected = tensor![[2.0, 2.0, 0.0], [0.0, 0.0, 0.0], [6.0, 6.0, 0.0]];
        assert_eq!(out, expected);

        let a = tensor![1.0, 2.0];
        let b = tensor![1.0, 2.0, 3.0, 4.0];
        let out = a.outer(&b);
        assert_eq!(out, tensor![[1., 2., 3., 4.], [2., 4., 6., 8.]]);
    }
}
