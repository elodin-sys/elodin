use crate::{AsOp, Op, Param};
use nalgebra::{
    constraint::ShapeConstraint, ClosedAdd, ClosedDiv, ClosedMul, ClosedSub,
    Scalar as NalgebraScalar,
};
use simba::scalar::ClosedNeg;
use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};
use xla::XlaOp;

pub trait TensorLike: Sized + AsOp {
    fn from_op(op: XlaOp) -> Self;

    fn sqrt(&self) -> Self {
        Self::from_op(self.as_op().sqrt().unwrap())
    }
}

pub struct Tensor<T, D: TensorDim, P: Param = Op> {
    inner: Arc<P::Inner>,
    phantom: PhantomData<(T, D)>,
}

pub trait TensorDim {}
pub trait NonScalarDim {}

pub struct ScalarDim;
impl TensorDim for ScalarDim {}
impl TensorDim for nalgebra::Dyn {}
impl NonScalarDim for nalgebra::Dyn {}
impl<const N: usize> TensorDim for nalgebra::Const<N> {}
impl<const N: usize> NonScalarDim for nalgebra::Const<N> {}

// This macro allows us to implement `TensorDim` for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
macro_rules! impl_tensor_dim {
      ($($ty:tt),+) => {
        impl<$($ty,)*> TensorDim for ($($ty,)*)
              where $($ty: TensorDim, )*
        {
        }

        impl<$($ty,)*> NonScalarDim for ($($ty,)*)
              where $($ty: NonScalarDim, )*
        {
        }

      }
}

impl_tensor_dim!(T1);
impl_tensor_dim!(T1, T2);
impl_tensor_dim!(T1, T2, T3);
impl_tensor_dim!(T1, T2, T3, T4);
impl_tensor_dim!(T1, T2, T3, T4, T5);
impl_tensor_dim!(T1, T2, T3, T4, T5, T6);
impl_tensor_dim!(T1, T2, T3, T4, T5, T6, T7);
impl_tensor_dim!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_tensor_dim!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_tensor_dim!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_tensor_dim!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_tensor_dim!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

pub trait DimAdd<D1: TensorDim, D2: TensorDim> {}
pub trait DimSub<D1: TensorDim, D2: TensorDim> {}

pub trait DimMul<D1: TensorDim, D2: TensorDim> {}
pub trait DimDiv<D1: TensorDim, D2: TensorDim> {}

impl<D: TensorDim> DimAdd<D, D> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimAdd<ScalarDim, D> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimAdd<D, ScalarDim> for ShapeConstraint {}

impl<D: TensorDim> DimSub<D, D> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimSub<ScalarDim, D> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimSub<D, ScalarDim> for ShapeConstraint {}

impl<D1, D2> DimMul<(D1, D2), (D2, D1)> for ShapeConstraint
where
    D1: NonScalarDim + TensorDim,
    D2: NonScalarDim + TensorDim,
{
}

impl<D: NonScalarDim + TensorDim> DimDiv<D, ScalarDim> for ShapeConstraint {}

impl<T: NalgebraScalar + ClosedAdd, D1: TensorDim, D2: TensorDim> Add<Tensor<T, D2>>
    for Tensor<T, D1>
where
    ShapeConstraint: DimAdd<D1, D2>,
{
    type Output = Self;

    fn add(self, rhs: Tensor<T, D2>) -> Self::Output {
        Tensor {
            inner: Arc::new((self.inner.as_ref() + rhs.inner.as_ref()).expect("xla build error")),
            phantom: PhantomData,
        }
    }
}

impl<T: NalgebraScalar + ClosedMul, D1: TensorDim, D2: TensorDim> Mul<Tensor<T, D2>>
    for Tensor<T, D1>
where
    ShapeConstraint: DimMul<D1, D2>,
{
    type Output = Self;

    fn mul(self, rhs: Tensor<T, D2>) -> Self::Output {
        Tensor {
            inner: Arc::new((self.inner.as_ref() * rhs.inner.as_ref()).expect("xla build error")),
            phantom: PhantomData,
        }
    }
}

impl<T: NalgebraScalar + ClosedDiv, D1: TensorDim, D2: TensorDim> Div<Tensor<T, D2>>
    for Tensor<T, D1>
where
    ShapeConstraint: DimDiv<D1, D2>,
{
    type Output = Self;

    fn div(self, rhs: Tensor<T, D2>) -> Self::Output {
        Tensor {
            inner: Arc::new((self.inner.as_ref() / rhs.inner.as_ref()).expect("xla build error")),
            phantom: PhantomData,
        }
    }
}

impl<T: NalgebraScalar + ClosedSub, D1: TensorDim, D2: TensorDim> Sub<Tensor<T, D2>>
    for Tensor<T, D1>
where
    ShapeConstraint: DimSub<D1, D2>,
{
    type Output = Self;

    fn sub(self, rhs: Tensor<T, D2>) -> Self::Output {
        Tensor {
            inner: Arc::new((self.inner.as_ref() - rhs.inner.as_ref()).expect("xla build error")),
            phantom: PhantomData,
        }
    }
}

impl<T: NalgebraScalar + ClosedNeg, D: TensorDim> Neg for Tensor<T, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Tensor {
            inner: Arc::new(self.inner.as_ref().neg().expect("xla build error")),
            phantom: PhantomData,
        }
    }
}
