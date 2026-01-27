//! Provides functionality for handling vectors in computational tasks, supporting conversion between host and Nox-specific representations, and enabling vector operations like extension, normalization, and cross products.
use crate::{
    ArrayRepr, DefaultRepr, Dim, Field, Matrix, OwnedRepr, RealField, Scalar, Tensor, TensorItem,
    tensor,
};
use crate::{Const, DimMul, ToTypenum};

use core::marker::PhantomData;

/// Type alias for a tensor that specifically represents a vector.
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

impl<T: Field, R: OwnedRepr> Vector<T, 3, R>
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

impl<T: TensorItem + Field, const N: usize, R: OwnedRepr> Vector<T, N, R> {
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

impl<T: TensorItem + Field, const N: usize, R: OwnedRepr> Vector<T, N, R> {
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

impl<T: RealField, R: OwnedRepr> Vector<T, 3, R> {
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

impl<T: Field + RealField, const N: usize, R: OwnedRepr> Vector<T, N, R> {
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
    use super::*;
    use crate::*;

    // NOTE: XLA-based execution tests commented out during IREE migration
    // These tests used Client::cpu() and .build().compile() which are no longer available
    // TODO: Re-implement tests using IREE execution path or ArrayRepr local execution

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
