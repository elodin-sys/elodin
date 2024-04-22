use nalgebra::{ArrayStorage, Const, DimMul, Scalar as NalgebraScalar, ToTypenum};
use num_traits::Zero;
use smallvec::smallvec;
use std::{marker::PhantomData, mem::MaybeUninit};
use xla::{ArrayElement, NativeType};

use crate::{
    ArrayBufUnit, ArrayDim, ArrayTy, Buffer, BufferArg, Client, ConcatManyDim, Dim, Field,
    FromHost, MaybeOwned, Noxpr, Op, Repr, Scalar, Tensor, TensorItem, ToHost,
};

pub type Vector<T, const N: usize, P = Op> = Tensor<T, Const<N>, P>;

impl<T: NativeType + ArrayElement> Vector<T, 3, Op> {
    pub fn extend(&self, elem: T) -> Vector<T, 4, Op> {
        let elem = elem.literal();
        let constant = Noxpr::constant(elem, ArrayTy::new(T::TY, smallvec![1]));
        let inner = Noxpr::concat_in_dim(vec![self.inner.clone(), constant], 0);
        Vector {
            inner,
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
        out.as_mut_slice()
            .copy_from_slice(literal.typed_buf::<T>().unwrap());
        out
    }
}

impl<T, const R: usize> FromHost for Vector<T, R, Buffer>
where
    T: NativeType + Field + ArrayElement,
{
    type HostTy = nalgebra::Vector<T, Const<R>, ArrayStorage<T, R, 1>>;

    fn from_host(client: &Client, native: Self::HostTy) -> Self {
        let inner = client
            .copy_host_buffer(native.as_slice(), &[R as i64])
            .unwrap();
        Vector {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: TensorItem, const R: usize> BufferArg<Vector<T, R, Buffer>>
    for nalgebra::Vector<T, Const<R>, ArrayStorage<T, R, 1>>
where
    T: NativeType + Field + ArrayElement,
{
    fn as_buffer(&self, client: &Client) -> MaybeOwned<'_, xla::PjRtBuffer> {
        let inner = client
            .copy_host_buffer(self.as_slice(), &[R as i64])
            .unwrap();
        MaybeOwned::Owned(inner)
    }
}

impl<T: TensorItem + Field, const N: usize, R: Repr> Vector<T, N, R> {
    pub fn from_arr(arr: [&Scalar<T, R>; N]) -> Self
    where
        Const<N>: Dim,
        Const<N>: ToTypenum,
        Const<1>: DimMul<Const<N>, Output = Const<N>>,
        <Const<1> as DimMul<Const<N>>>::Output: Dim,
        <ConcatManyDim<Const<1>, N> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <ConcatManyDim<Const<1>, N> as ArrayDim>::Buf<T>>,
    {
        let args = arr.map(|v| &v.inner);
        let inner = R::concat_many(args);
        //let nodes = arr.map(|v| v.inner).to_vec();
        //let inner = Noxpr::concat_in_dim(nodes, 0);
        Vector {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: TensorItem + Field, const N: usize, R: Repr> Vector<T, N, R> {
    pub fn parts(&self) -> [Scalar<T, R>; N] {
        let mut i = 0;
        [0; N].map(|_| {
            let elem = self.get(i);
            i += 1;
            elem
        })
    }
}

impl<T: Field> Vector<T, 3, Op> {
    pub fn cross(&self, other: &Self) -> Self {
        let [ax, ay, az] = self.parts();
        let [bx, by, bz] = other.parts();
        let x = &ay * &bz - &az * &by;
        let y = &az * &bx - &ax * &bz;
        let z = &ax * &by - &ay * &bx;
        Vector::from_arr([&x, &y, &z])
    }
}

impl<T: Field, const N: usize> Vector<T, N, Op> {
    pub fn norm_squared(&self) -> Scalar<T> {
        self.dot(self)
    }

    pub fn norm(&self) -> Scalar<T> {
        self.dot(self).sqrt()
    }

    pub fn normalize(&self) -> Self {
        self.clone() / self.norm()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::vector;

    use crate::CompFn;

    use super::*;

    #[test]
    fn test_vector_from_arr() {
        let client = Client::cpu().unwrap();
        fn from_arr(a: Scalar<f32>, b: Scalar<f32>, c: Scalar<f32>) -> Vector<f32, 3> {
            Vector::from_arr([&a, &b, &c])
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
}
