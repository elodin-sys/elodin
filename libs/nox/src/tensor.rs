use crate::{AsBuffer, Buffer, IntoOp, Op, Param};
use nalgebra::{
    constraint::ShapeConstraint, ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, Const,
    Scalar as NalgebraScalar,
};
use simba::scalar::ClosedNeg;
use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};
use xla::{NativeType, XlaOp};

#[repr(transparent)]
pub struct Tensor<T, D: TensorDim, P: Param = Op> {
    pub(crate) inner: P::Inner,
    pub(crate) phantom: PhantomData<(T, D)>,
}

impl<T, D: TensorDim> Tensor<T, D, Op> {
    fn from_op(op: XlaOp) -> Self {
        Self {
            inner: op,
            phantom: PhantomData,
        }
    }

    pub fn sqrt(&self) -> Self {
        Self::from_op(self.inner.sqrt().unwrap())
    }

    pub fn log(&self) -> Self {
        Self::from_op(self.inner.log().unwrap())
    }

    /// *Safety*: This function is memory safe, it is marked unsafe because you could introduce crashes and/or
    /// other weirdness in XLA using it. Its only intended use case is as a bogan form of type-erasure for Tensors.
    /// Essentially you must guarentee that the Op you are casting from is of the correct dimension and type
    pub(crate) unsafe fn unsafe_mut_cast<NT, ND: TensorDim>(
        &mut self,
    ) -> &'_ mut Tensor<NT, ND, Op> {
        // Safety: this is safe because we are casting between two `Tensor<Op>` types,
        // which are marked `repr(transparent)`, since the only two types are the inner `Op` and a
        // ZST PhantomData
        std::mem::transmute(self)
    }
}

impl<T, D: TensorDim> IntoOp for Tensor<T, D, Op> {
    fn into_op(self, _builder: &xla::XlaBuilder) -> xla::XlaOp {
        self.inner
    }
}

pub trait TensorDim {}
pub trait NonScalarDim {}

pub struct ScalarDim;
impl TensorDim for ScalarDim {}
impl TensorDim for nalgebra::Dyn {}
impl NonScalarDim for nalgebra::Dyn {}
impl<const N: usize> TensorDim for nalgebra::Const<N> {}
impl<const N: usize> NonScalarDim for nalgebra::Const<N> {}

pub trait ConstDim<const RANK: usize> {
    const RANK: usize = RANK;
    fn dims() -> [usize; RANK];
}

pub trait XlaDim {
    type Array;
    fn dims() -> Self::Array;
}

pub trait DimRank<const RANK: usize> {
    const RANK: usize = RANK;
}

impl ConstDim<0> for ScalarDim {
    fn dims() -> [usize; 0] {
        []
    }
}

impl XlaDim for ScalarDim {
    type Array = [i64; 0];
    fn dims() -> [i64; 0] {
        []
    }
}

impl DimRank<0> for ScalarDim {}

impl<const N: usize> ConstDim<1> for Const<N> {
    fn dims() -> [usize; 1] {
        [N]
    }
}

impl<const N: usize> XlaDim for Const<N> {
    type Array = [i64; 1];
    fn dims() -> [i64; 1] {
        [N as i64]
    }
}

impl XlaDim for nalgebra::Dyn {
    type Array = [i64; 1];
    fn dims() -> [i64; 1] {
        [-1]
    }
}

impl<const N: usize> DimRank<1> for Const<N> {}

// This macro allows us to implement `TensorDim` for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
macro_rules! impl_tensor_dim {
      ($num:literal; $($ty:tt),+) => {
        impl<$($ty,)*> TensorDim for ($($ty,)*)
              where $($ty: TensorDim, )*
        {
        }

        impl<$($ty,)*> NonScalarDim for ($($ty,)*)
              where $($ty: NonScalarDim, )*
        {
        }


        impl<$($ty,)*> DimRank<$num> for ($($ty,)*)
              where $($ty: NonScalarDim, )*
        {
        }

        impl<$($ty,)*> ConstDim<$num> for ($($ty,)*)
              where $($ty: ConstDim<1>, )*
        {
            fn dims() -> [usize; $num] {
                [$($ty::dims()[0],)*]
            }
        }

        impl<$($ty,)*> XlaDim for ($($ty,)*)
              where $($ty: XlaDim<Array = [i64; 1]>, )*
        {
            type Array = [i64; $num];
            fn dims() -> [i64; $num] {
                [$($ty::dims()[0],)*]
            }
        }

      }
}

impl_tensor_dim!(1; T1);
impl_tensor_dim!(2; T1, T2);
impl_tensor_dim!(3; T1, T2, T3);
impl_tensor_dim!(4; T1, T2, T3, T4);
impl_tensor_dim!(5; T1, T2, T3, T4, T5);
impl_tensor_dim!(6; T1, T2, T3, T4, T5, T6);
impl_tensor_dim!(7; T1, T2, T3, T4, T5, T6, T7);
impl_tensor_dim!(8; T1, T2, T3, T4, T5, T6, T7, T8);
impl_tensor_dim!(9; T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_tensor_dim!(10; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_tensor_dim!(11; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_tensor_dim!(12; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

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

impl DimMul<Const<1>, Const<1>> for ShapeConstraint {}

impl<D: NonScalarDim + TensorDim> DimDiv<D, ScalarDim> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimMul<D, ScalarDim> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimMul<ScalarDim, D> for ShapeConstraint {}

macro_rules! impl_op {
    ($op: tt, $op_fn:tt, $constraint: tt, $ret: tt, $inner: tt, $($t_bound:tt),+) => {
        impl<T, D1: TensorDim, D2: TensorDim> $op<Tensor<T, D2>>
            for Tensor<T, D1>
        where
            $(T: $t_bound,)+
            ShapeConstraint: $constraint<D1, D2>,
        {
            type Output = Self;

            fn $op_fn(self, rhs: Tensor<T, D2>) -> Self::Output {
                Tensor::from_op((&self.inner $inner &rhs.inner).expect("xla build error"))
            }
        }

        impl<'a, T, D1: TensorDim, D2: TensorDim> $op<&'a Tensor<T, D2>>
            for Tensor<T, D1>
        where
            $(T: $t_bound,)+
            ShapeConstraint: $constraint<D1, D2>,
        {
            type Output = Self;

            fn $op_fn(self, rhs: &'a Tensor<T, D2>) -> Self::Output {
                Tensor::from_op((&self.inner $inner &rhs.inner).expect("xla build error"))
            }
        }

        impl<'a, T, D1: TensorDim, D2: TensorDim> $op<Tensor<T, D2>>
            for &'a Tensor<T, D1>
        where
            $(T: $t_bound,)+
            ShapeConstraint: $constraint<D1, D2>,
        {
            type Output = $ret<T, D1>;

            fn $op_fn(self, rhs: Tensor<T, D2>) -> Self::Output {
                Tensor::from_op((&self.inner $inner &rhs.inner).expect("xla build error"))
            }
        }


        impl<'a, 'b, T, D1: TensorDim, D2: TensorDim> $op<&'b Tensor<T, D2>>
            for &'a Tensor<T, D1>
        where
            $(T: $t_bound,)+
            ShapeConstraint: $constraint<D1, D2>,
        {
            type Output = $ret<T, D1>;

            fn $op_fn(self, rhs: &'b Tensor<T, D2>) -> Self::Output {
                Tensor::from_op((&self.inner $inner &rhs.inner).expect("xla build error"))
            }
        }

    };
}

impl_op! {Add, add, DimAdd, Tensor, +, ClosedAdd, NalgebraScalar}
impl_op! {Mul, mul, DimMul, Tensor, *, ClosedMul, NalgebraScalar}
impl_op! {Div, div, DimDiv, Tensor, /, ClosedDiv, NalgebraScalar}
impl_op! {Sub, sub, DimSub, Tensor, -, ClosedSub, NalgebraScalar}

impl<T: NalgebraScalar + ClosedNeg, D: TensorDim> Neg for Tensor<T, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Tensor::from_op(self.inner.neg().expect("xla build error"))
    }
}

impl<'a, T: NalgebraScalar + ClosedNeg, D: TensorDim> Neg for &'a Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn neg(self) -> Self::Output {
        Tensor::from_op(self.inner.neg().expect("xla build error"))
    }
}

impl<T, D: TensorDim + DimRank<R>, const R: usize> FixedSliceExt<T, D, R> for Tensor<T, D, Op> {
    fn fixed_slice<ND: TensorDim + ConstDim<R>>(&self, offsets: [usize; R]) -> Tensor<T, ND, Op> {
        let offsets = offsets.map(|o| o as i64);
        let mut new_offsets = [0; R];
        for (i, (a, b)) in offsets.iter().zip(ND::dims().into_iter()).enumerate() {
            new_offsets[i] = a + b as i64;
        }
        Tensor::from_op(
            self.inner
                .slice(&offsets, &new_offsets, &[1i64; R])
                .unwrap(),
        )
    }
}

pub trait FixedSliceExt<T, D: TensorDim, const R: usize> {
    fn fixed_slice<ND: TensorDim + ConstDim<R>>(&self, offsets: [usize; R]) -> Tensor<T, ND, Op>;
}

impl<T: NalgebraScalar + ClosedMul + NativeType, D1: TensorDim> Mul<T> for Tensor<T, D1>
where
    ShapeConstraint: DimMul<D1, ScalarDim>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_op(
            (&self.inner * self.inner.builder().c0(rhs).unwrap()).expect("xla build error"),
        )
    }
}

impl<'a, T: NalgebraScalar + ClosedMul + NativeType, D1: TensorDim> Mul<T> for &'a Tensor<T, D1>
where
    ShapeConstraint: DimMul<D1, ScalarDim>,
{
    type Output = Tensor<T, D1>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_op(
            (&self.inner * self.inner.builder().c0(rhs).unwrap()).expect("xla build error"),
        )
    }
}

macro_rules! impl_prim {
    ($ty:tt) => {
        impl<D: TensorDim> Mul<Tensor<$ty, D>> for $ty {
            type Output = Tensor<$ty, D>;

            fn mul(self, rhs: Tensor<$ty, D>) -> Self::Output {
                Tensor::from_op((rhs.inner.builder().c0(self).unwrap() * rhs.inner).unwrap())
            }
        }

        impl<'a, D: TensorDim> Mul<&'a Tensor<$ty, D>> for $ty {
            type Output = Tensor<$ty, D>;

            fn mul(self, rhs: &Tensor<$ty, D>) -> Self::Output {
                Tensor::from_op((rhs.inner.builder().c0(self).unwrap() * &rhs.inner).unwrap())
            }
        }
    };
}

impl_prim!(f64);
impl_prim!(f32);
impl_prim!(u64);
impl_prim!(u32);
impl_prim!(i64);
impl_prim!(i32);

impl<T, D: TensorDim> AsBuffer for Tensor<T, D, Buffer> {
    fn as_buffer(&self) -> &xla::PjRtBuffer {
        &self.inner
    }
}
