use crate::{ArrayTy, Matrix, Noxpr, Quaternion, Scalar, Vector};
use nalgebra::{Const, IsContiguous, Storage};
use smallvec::smallvec;
use xla::{ArrayElement, NativeType};

pub trait ConstantExt<Out> {
    fn constant(&self) -> Out;
}

impl<T: NativeType + ArrayElement + Copy> ConstantExt<Scalar<T>> for T {
    fn constant(&self) -> Scalar<T> {
        let lit = T::literal(*self);
        Scalar::from_op(Noxpr::constant(
            lit,
            ArrayTy {
                element_type: T::TY,
                shape: smallvec![],
            },
        ))
    }
}

impl<T: NativeType + ArrayElement, const N: usize, S> ConstantExt<Vector<T, N>>
    for nalgebra::Vector<T, Const<N>, S>
where
    S: Storage<T, Const<N>, Const<1>>,
    S: IsContiguous,
{
    fn constant(&self) -> Vector<T, N> {
        let shape = smallvec![N as i64];
        let lit = T::create_r1(self.as_slice()).reshape(&shape).unwrap();
        let constant = Noxpr::constant(
            lit,
            ArrayTy {
                element_type: T::TY,
                shape,
            },
        );
        Vector::from_op(constant)
    }
}

impl<T: NativeType + ArrayElement, const R: usize, const C: usize, S> ConstantExt<Matrix<T, R, C>>
    for nalgebra::Matrix<T, Const<R>, Const<C>, S>
where
    S: Storage<T, Const<R>, Const<C>>,
    S: IsContiguous,
{
    fn constant(&self) -> Matrix<T, R, C> {
        let shape = smallvec![R as i64, C as i64];
        let lit = T::create_r1(self.as_slice()).reshape(&shape).unwrap();
        let constant = Noxpr::constant(
            lit,
            ArrayTy {
                element_type: T::TY,
                shape,
            },
        );
        Matrix::from_op(constant)
    }
}

impl<T: NativeType + ArrayElement + nalgebra::Scalar> ConstantExt<Quaternion<T>>
    for nalgebra::Quaternion<T>
{
    fn constant(&self) -> Quaternion<T> {
        Quaternion(self.coords.into())
    }
}

impl<T: NativeType + ArrayElement + nalgebra::Scalar> From<nalgebra::Quaternion<T>>
    for Quaternion<T>
{
    fn from(val: nalgebra::Quaternion<T>) -> Self {
        val.constant()
    }
}

impl<T, const N: usize, S> From<nalgebra::Vector<T, Const<N>, S>> for Vector<T, N>
where
    S: Storage<T, Const<N>, Const<1>> + IsContiguous,
    T: NativeType + ArrayElement,
{
    fn from(val: nalgebra::Vector<T, Const<N>, S>) -> Self {
        val.constant()
    }
}

impl<T, const R: usize, const C: usize, S> From<nalgebra::Matrix<T, Const<R>, Const<C>, S>>
    for Matrix<T, R, C>
where
    S: Storage<T, Const<R>, Const<C>> + IsContiguous,
    T: NativeType + ArrayElement,
{
    fn from(val: nalgebra::Matrix<T, Const<R>, Const<C>, S>) -> Self {
        val.constant()
    }
}
