use crate::{ArrayTy, Matrix, Noxpr, Vector};
use nalgebra::{Const, IsContiguous, Storage};
use smallvec::smallvec;
use xla::{ArrayElement, NativeType};

pub trait ConstantExt<Out> {
    fn constant(&self) -> Out;
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
