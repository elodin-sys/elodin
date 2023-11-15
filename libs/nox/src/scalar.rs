use crate::{
    AsOp, Builder, FromBuilder, FromPjrtBuffer, Literal, Op, Param, ScalarDim, Tensor, TensorLike,
    ToHost,
};
use nalgebra::ClosedAdd;
use std::{
    marker::PhantomData,
    ops::Add,
    sync::{atomic::Ordering, Arc},
};
use xla::{ArrayElement, NativeType, XlaOp};

pub type Scalar<T, P = Op> = Tensor<T, ScalarDim, P>;

impl<T> TensorLike for Scalar<T, Op> {
    fn from_op(op: XlaOp) -> Self {
        Self {
            inner: Arc::new(op),
            phantom: PhantomData,
        }
    }
}

impl<T> AsOp for Scalar<T, Op> {
    fn as_op(&self) -> &XlaOp {
        self.inner.as_ref()
    }
}

impl<T, P: Param> ToHost for Scalar<T, P> {
    type HostTy = T;
}

macro_rules! impl_prim_buffer {
    ($ty:tt) => {
        impl FromPjrtBuffer for $ty {
            fn from_pjrt(pjrt: Vec<Vec<xla::PjRtBuffer>>) -> Self {
                let buf = &pjrt[0][0];
                let literal = buf.to_literal_sync().unwrap();
                literal.get_first_element().unwrap()
            }
        }
    };
}

impl_prim_buffer!(f64);
impl_prim_buffer!(f32);
impl_prim_buffer!(u64);
impl_prim_buffer!(u32);
impl_prim_buffer!(i64);
impl_prim_buffer!(i32);

impl<T: ClosedAdd + ArrayElement + NativeType> Add<T> for Scalar<T, Op> {
    type Output = Scalar<T, Op>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = self.inner.builder().c0(rhs).unwrap();
        Scalar {
            inner: Arc::new((self.inner.as_ref() + rhs).unwrap()),
            phantom: PhantomData,
        }
    }
}

pub trait ScalarExt: Sized {
    fn literal(self) -> Scalar<Self, Literal>;
    fn constant(self, builder: &Builder) -> Scalar<Self, Op>;
}

impl<T> ScalarExt for T
where
    T: ArrayElement + Sized + NativeType,
{
    fn literal(self) -> Scalar<Self, Literal> {
        Scalar {
            inner: Arc::new(xla::Literal::scalar(self)),
            phantom: PhantomData,
        }
    }

    fn constant(self, builder: &Builder) -> Scalar<Self, Op> {
        let inner = Arc::new(
            builder
                .inner
                .constant_r0(self)
                .expect("constant creation failed"),
        );

        Scalar {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: xla::ArrayElement> FromBuilder for Scalar<T, Op> {
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        let i = builder.param_count.fetch_add(1, Ordering::SeqCst);
        Scalar {
            inner: Arc::new(
                builder
                    .inner
                    .parameter(i, T::TY, &[], &format!("param_{}", i))
                    .expect("parameter create failed"),
            ),
            phantom: PhantomData,
        }
    }
}
