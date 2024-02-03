use crate::{
    ArrayTy, Buffer, BufferArg, Client, FromHost, Literal, MaybeOwned, Noxpr, Op, Tensor, ToHost,
};
use nalgebra::{ArrayStorage, Const, IsContiguous, Scalar as NalgebraScalar, Storage};
use num_traits::Zero;
use smallvec::smallvec;
use std::marker::PhantomData;
use xla::{ArrayElement, NativeType};

pub type Matrix<T, const R: usize, const C: usize, P = Op> = Tensor<T, (Const<R>, Const<C>), P>;

impl<T, const R: usize, const C: usize> ToHost for Matrix<T, R, C, Buffer>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement,
{
    type HostTy = nalgebra::Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;

    fn to_host(&self) -> Self::HostTy {
        let literal = self.inner.to_literal_sync().unwrap();
        let buf = literal.typed_buf().unwrap();
        // XLA uses row-major form while nalgebra uses column-major form,
        // so we need to use from_row_iterator to copy the data out
        Self::HostTy::from_row_iterator(buf.iter().copied())
    }
}

impl<T, S, const R: usize, const C: usize> BufferArg<Matrix<T, R, C, Buffer>>
    for nalgebra::Matrix<T, Const<R>, Const<C>, S>
where
    T: xla::NativeType + NalgebraScalar + ArrayElement,
    S: Storage<T, Const<R>, Const<C>>,
    S: IsContiguous,
{
    fn as_buffer(&self, client: &Client) -> MaybeOwned<'_, xla::PjRtBuffer> {
        MaybeOwned::Owned(self.buffer(client).inner)
    }
}

impl<T: ArrayElement, const R: usize, const C: usize> Matrix<T, R, C, Literal> {
    pub fn constant(self) -> Matrix<T, R, C, Op> {
        let inner = Noxpr::constant(
            self.inner,
            ArrayTy {
                element_type: T::TY,
                shape: smallvec![R as i64, C as i64],
            },
        );

        Matrix {
            inner,
            phantom: PhantomData,
        }
    }
}

pub trait MatrixExt<T, const R: usize, const C: usize> {
    fn constant(&self) -> Matrix<T, R, C, Op>;
    fn literal(&self) -> Matrix<T, R, C, Literal>;
    fn buffer(&self, client: &Client) -> Matrix<T, R, C, Buffer>;
}

impl<T, S, const R: usize, const C: usize> MatrixExt<T, R, C>
    for nalgebra::Matrix<T, Const<R>, Const<C>, S>
where
    T: xla::NativeType + NalgebraScalar + ArrayElement,
    S: Storage<T, Const<R>, Const<C>>,
    S: IsContiguous,
{
    fn constant(&self) -> Matrix<T, R, C, Op> {
        self.literal().constant()
    }

    fn buffer(&self, client: &Client) -> Matrix<T, R, C, Buffer> {
        let mut buf: Vec<T> = Vec::with_capacity(C * R);
        for row in self.row_iter() {
            buf.extend(row.iter())
        }
        let inner = client.0.copy_host_buffer(&buf, &[R, C]).unwrap();
        Matrix {
            inner,
            phantom: PhantomData,
        }
    }

    fn literal(&self) -> Matrix<T, R, C, Literal> {
        let inner = T::create_r1(self.as_slice())
            .reshape(&[R as i64, C as i64])
            .unwrap();
        Matrix {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T, const R: usize, const C: usize> FromHost for Matrix<T, R, C, Buffer>
where
    T: NativeType + NalgebraScalar + ArrayElement,
{
    type HostTy = nalgebra::Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;

    fn from_host(client: &Client, native: Self::HostTy) -> Self {
        native.buffer(client)
    }
}

impl<T, const R: usize, const C: usize, const N: usize> FromHost
    for Tensor<T, (Const<R>, Const<C>, Const<N>), Buffer>
where
    T: NativeType + NalgebraScalar + ArrayElement,
{
    type HostTy = [nalgebra::Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>; N];

    fn from_host(client: &Client, native: Self::HostTy) -> Self {
        let mut buf = Vec::with_capacity(C * R * N);
        for mat in native.iter() {
            buf.extend_from_slice(mat.as_slice());
        }
        let inner = client.0.copy_host_buffer(&buf, &[N, R, C]).unwrap();
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T, S, const R: usize, const C: usize, const N: usize>
    BufferArg<Tensor<T, (Const<R>, Const<C>, Const<N>), Buffer>>
    for [nalgebra::Matrix<T, Const<R>, Const<C>, S>; N]
where
    T: xla::NativeType + NalgebraScalar + ArrayElement,
    S: Storage<T, Const<R>, Const<C>>,
    S: IsContiguous,
{
    fn as_buffer(&self, client: &Client) -> MaybeOwned<'_, xla::PjRtBuffer> {
        let mut buf = Vec::with_capacity(C * R * N);
        for mat in self.iter() {
            buf.extend_from_slice(mat.as_slice());
        }
        let inner = client.0.copy_host_buffer(&buf, &[N, R, C]).unwrap();
        MaybeOwned::Owned(inner)
    }
}

pub trait Dot<Rhs = Self> {
    type Output;

    fn dot(self, rhs: Rhs) -> Self::Output;
}

impl<T, const R: usize, const C: usize> Dot for Matrix<T, R, C, Op>
where
    T: NativeType + NalgebraScalar + ArrayElement,
{
    type Output = Matrix<T, R, C, Op>;

    fn dot(self, rhs: Self) -> Self::Output {
        let inner = Noxpr::dot(self.inner, &rhs.inner);
        Matrix {
            inner,
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{matrix, vector};

    use crate::{CompFn, FixedSliceExt, Vector};

    use super::*;

    #[test]
    fn test_add() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Matrix<f32, 1, 2>, b: Matrix<f32, 1, 2>| a + b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, matrix![1.0f32, 2.0], matrix![2.0, 3.0])
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![3.0, 5.0]);
    }

    #[test]
    fn test_sub() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Matrix<f32, 1, 2>, b: Matrix<f32, 1, 2>| a - b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, matrix![1.0f32, 2.0], matrix![2.0, 3.0])
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![-1.0, -1.0]);
    }

    #[test]
    fn test_mul() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Matrix<f32, 2, 2>, b: Matrix<f32, 2, 2>| a * b)
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(
                &client,
                matrix![1.0f32, 2.0; 2.0, 3.0],
                matrix![2.0, 3.0; 4.0, 5.0],
            )
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![2., 6.; 8., 15.]);
    }

    #[test]
    fn test_fixed_slice() {
        let client = Client::cpu().unwrap();
        fn slice(mat: Matrix<f32, 1, 4>) -> Matrix<f32, 1, 1> {
            mat.fixed_slice::<(Const<1>, Const<1>)>([0, 2])
        }
        let comp = slice.build().unwrap();
        let exec = match comp.compile(&client) {
            Ok(exec) => exec,
            Err(xla::Error::XlaError { msg, .. }) => {
                println!("{}", msg);
                panic!();
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        };
        let out = exec
            .run(&client, matrix![1.0f32, 2.0, 3.0, 4.0])
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![3.0])
    }

    #[test]
    fn test_index() {
        let client = Client::cpu().unwrap();
        fn index(mat: Matrix<f32, 4, 3>, index: Vector<u32, 2>) -> Matrix<f32, 2, 3> {
            mat.index(index)
        }
        let comp = index.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let a = matrix![0., 1., 2.;
                        2., 3., 4.;
                        4., 5., 6.;
                        7., 8., 9.];
        let out: nalgebra::Matrix<f32, Const<2>, Const<3>, ArrayStorage<f32, 2, 3>> =
            exec.run(&client, a, vector![1, 2]).unwrap().to_host();
        assert_eq!(
            out,
            matrix![2., 3., 4.;
                    4., 5., 6.]
        );

        let out: nalgebra::Matrix<f32, Const<2>, Const<3>, ArrayStorage<f32, 2, 3>> =
            exec.run(&client, a, vector![0, 3]).unwrap().to_host();
        assert_eq!(
            out,
            matrix![0., 1., 2.;
                    7., 8., 9.]
        );
    }
}
