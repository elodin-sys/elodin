use std::{
    marker::PhantomData,
    ops::Mul,
    sync::{atomic::Ordering, Arc},
};

use nalgebra::{ArrayStorage, ClosedMul, Const, Scalar as NalgebraScalar};
use xla::{ArrayElement, NativeType, XlaOp};

use crate::{
    AsBuffer, AsOp, Buffer, BufferForm, Builder, Client, FromBuilder, FromHost, Op, Param, Scalar,
    Tensor, TensorLike, ToHost,
};

pub type Vector<T, const N: usize, P = Op> = Tensor<T, Const<N>, P>;

impl<T: NativeType> Vector<T, 3, Op> {
    pub fn extend(&self, elem: T) -> Vector<T, 4, Op> {
        Vector {
            inner: Arc::new(
                self.inner
                    .concat_in_dim(
                        &[self
                            .inner
                            .builder()
                            .c0(elem)
                            .unwrap()
                            .reshape(&[1])
                            .unwrap()],
                        0,
                    )
                    .unwrap(),
            ),
            phantom: PhantomData,
        }
    }
}

impl<T, const N: usize, P: Param> Clone for Vector<T, N, P> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            phantom: PhantomData,
        }
    }
}

impl<T, const N: usize> TensorLike for Vector<T, N, Op> {
    fn from_op(op: XlaOp) -> Self {
        Self {
            inner: Arc::new(op),
            phantom: PhantomData,
        }
    }
}

impl<T: NalgebraScalar + ClosedMul> Mul for Vector<T, 1> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Vector {
            inner: Arc::new((self.inner.as_ref() * rhs.inner.as_ref()).expect("xla build error")),
            phantom: PhantomData,
        }
    }
}

impl<T: NalgebraScalar + ClosedMul, const N: usize> Mul<Vector<T, N>> for Scalar<T> {
    type Output = Vector<T, N>;

    fn mul(self, rhs: Vector<T, N>) -> Self::Output {
        Vector {
            inner: Arc::new((self.inner.as_ref() * rhs.inner.as_ref()).expect("xla build error")),
            phantom: PhantomData,
        }
    }
}

macro_rules! impl_prim {
    ($ty:tt) => {
        impl<const N: usize> Mul<Vector<$ty, N>> for $ty {
            type Output = Vector<$ty, N>;

            fn mul(self, rhs: Vector<$ty, N>) -> Self::Output {
                Vector {
                    inner: Arc::new((rhs.inner.builder().c0(self).unwrap() * rhs.as_op()).unwrap()),
                    phantom: PhantomData,
                }
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

impl<T, const N: usize> AsOp for Vector<T, N, Op> {
    fn as_op(&self) -> &XlaOp {
        &self.inner
    }
}

impl<T, const R: usize> AsBuffer for Vector<T, R, Buffer> {
    fn as_buffer(&self) -> &xla::PjRtBuffer {
        self.inner.as_ref()
    }
}

impl<T: xla::ArrayElement, const R: usize> FromBuilder for Vector<T, R, Op> {
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        let i = builder.param_count.fetch_add(1, Ordering::SeqCst);
        Vector {
            inner: Arc::new(
                builder
                    .inner
                    .parameter(i, T::TY, &[R as i64], &format!("param_{}", i))
                    .expect("parameter create failed"),
            ),
            phantom: PhantomData,
        }
    }
}

impl<T, const R: usize> FromHost for Vector<T, R, Buffer>
where
    T: NativeType + NalgebraScalar + ArrayElement,
{
    type HostTy = nalgebra::Vector<T, Const<R>, ArrayStorage<T, R, 1>>;

    fn from_host(client: &Client, native: Self::HostTy) -> Self {
        let inner = client
            .0
            .buffer_from_host_buffer(native.as_slice(), &[R], None)
            .unwrap();
        Vector {
            inner: Arc::new(inner),
            phantom: PhantomData,
        }
    }
}

impl<T, const R: usize> Vector<T, R, Op> {
    pub fn from_arr(arr: [Vector<T, 1, Op>; R]) -> Self {
        let arr = arr.map(|v| v.inner);
        let op = arr[0].concat_in_dim(&arr[1..], 0).unwrap();
        Vector {
            inner: Arc::new(op),
            phantom: PhantomData,
        }
    }

    pub fn fixed_slice<const NR: usize>(&self, offset: usize) -> Vector<T, NR> {
        let offset = offset as i64;
        Vector {
            inner: Arc::new(
                self.as_op()
                    .slice(&[offset], &[offset + (NR as i64)], &[1])
                    .unwrap(),
            ),
            phantom: PhantomData,
        }
    }

    pub fn dot(&self, other: &Self) -> Scalar<T> {
        Scalar {
            inner: Arc::new(self.as_op().dot(&other.inner).unwrap()),
            phantom: PhantomData,
        }
    }

    pub fn norm_squared(&self) -> Scalar<T> {
        self.dot(self)
    }

    pub fn norm(&self) -> Scalar<T> {
        self.dot(self).sqrt()
    }
}

impl<T, const R: usize, P: Param> ToHost for Vector<T, R, P> {
    type HostTy = nalgebra::Matrix<T, Const<R>, Const<1>, ArrayStorage<T, R, 1>>;
}

impl<T, const R: usize> BufferForm for Vector<T, R, Op> {
    type BufferTy = Vector<T, R, Buffer>;
}

#[cfg(test)]
mod tests {
    use nalgebra::vector;

    use crate::CompFn;

    use super::*;

    #[test]
    fn test_vector_from_arr() {
        let client = Client::cpu().unwrap();
        fn from_arr(a: Vector<f32, 1>, b: Vector<f32, 1>, c: Vector<f32, 1>) -> Vector<f32, 3> {
            Vector::from_arr([a, b, c])
        }
        let comp = from_arr.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![1.0f32], vector![2.0], vector![3.0])
            .unwrap();
        assert_eq!(out, vector![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_scalar_mult() {
        let client = Client::cpu().unwrap();
        let comp = (|vec: Vector<f32, 3>| 2.0 * vec).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec.run(&client, vector![1.0f32, 2.0, 3.0]).unwrap();
        assert_eq!(out, vector![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vector_dot() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>, b| a.dot(&b)).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, vector![1.0f32, 2.0, 3.0], vector![2.0, 3.0, 4.0])
            .unwrap();
        assert_eq!(out, 20.0)
    }

    #[test]
    fn test_norm_squared() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>| a.norm_squared()).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec.run(&client, vector![1.0f32, 2.0, 3.0]).unwrap();
        assert_eq!(out, 14.0)
    }

    #[test]
    fn test_norm() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>| a.norm()).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec.run(&client, vector![1.0f32, 2.0, 3.0]).unwrap();
        assert_eq!(out, 14.0f32.sqrt())
    }

    #[test]
    fn test_vector_mul() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 1>, b: Vector<f32, 1>| a * b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec.run(&client, vector![2.0f32], vector![2.0]).unwrap();
        assert_eq!(out, vector![4.0f32])
    }

    #[test]
    fn test_extend() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Vector<f32, 3>| a.extend(1.0)).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec.run(&client, vector![2.0f32, 1.0, 2.0]).unwrap();
        assert_eq!(out, vector![2.0, 1.0, 2.0, 1.0])
    }
}
