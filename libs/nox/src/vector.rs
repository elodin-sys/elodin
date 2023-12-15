use std::marker::PhantomData;

use nalgebra::{ArrayStorage, Const, Scalar as NalgebraScalar};
use num_traits::Zero;
use xla::{ArrayElement, NativeType};

use crate::{Buffer, BufferArg, Client, FromHost, MaybeOwned, Op, Scalar, Tensor, ToHost};

pub type Vector<T, const N: usize, P = Op> = Tensor<T, Const<N>, P>;

impl<T: NativeType> Vector<T, 3, Op> {
    pub fn extend(&self, elem: T) -> Vector<T, 4, Op> {
        Vector {
            inner: self
                .inner
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
            phantom: PhantomData,
        }
    }
}

impl<T, const N: usize> ToHost for Vector<T, N, Buffer>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement,
{
    type HostTy = nalgebra::Vector<T, Const<N>, ArrayStorage<T, N, 1>>;

    fn to_host(&self) -> Self::HostTy {
        let literal = self.inner.to_literal_sync().unwrap();
        let mut out = Self::HostTy::zeros();
        literal.copy_raw_to(out.as_mut_slice()).unwrap();
        out
    }
}

// impl<T, const N: usize, P: Param> Clone for Vector<T, N, P> {
//     fn clone(&self) -> Self {
//         Self {
//             inner: self.inner.clone(),
//             phantom: PhantomData,
//         }
//     }
// }

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
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T, const R: usize> BufferArg<Vector<T, R, Buffer>>
    for nalgebra::Vector<T, Const<R>, ArrayStorage<T, R, 1>>
where
    T: NativeType + NalgebraScalar + ArrayElement,
{
    fn as_buffer(&self, client: &Client) -> MaybeOwned<'_, xla::PjRtBuffer> {
        let inner = client
            .0
            .buffer_from_host_buffer(self.as_slice(), &[R], None)
            .unwrap();
        MaybeOwned::Owned(inner)
    }
}

impl<T, const R: usize> Vector<T, R, Op> {
    pub fn from_arr(arr: [Vector<T, 1, Op>; R]) -> Self {
        let arr = arr.map(|v| v.inner);
        let op = arr[0].concat_in_dim(&arr[1..], 0).unwrap();
        Vector {
            inner: op,
            phantom: PhantomData,
        }
    }

    pub fn dot(&self, other: &Self) -> Scalar<T> {
        Scalar {
            inner: self.inner.dot(&other.inner).unwrap(),
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

// impl<T, const R: usize, P: Param> ToHost for Vector<T, R, Buffer> {
//     type HostTy = nalgebra::Matrix<T, Const<R>, Const<1>, ArrayStorage<T, R, 1>>;

//     fn to_host(&self) -> HostTy {
//         todo!()
//     }
// }

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
            .unwrap()
            .to_host();
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
}
