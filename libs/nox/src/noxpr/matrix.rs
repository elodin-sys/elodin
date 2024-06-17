use crate::{
    ArrayTy, Buffer, BufferArg, Client, FromHost, Literal, Matrix, MaybeOwned, Noxpr, Op, Tensor,
    ToHost,
};
use nalgebra::{ArrayStorage, Const, IsContiguous, Scalar as NalgebraScalar, Storage};
use num_traits::Zero;
use smallvec::smallvec;
use std::marker::PhantomData;
use xla::{ArrayElement, NativeType};

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

impl<T: ArrayElement + NativeType, const R: usize, const C: usize> Matrix<T, R, C, Literal> {
    /// Converts a literal matrix to a constant matrix for operations within Nox.
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

/// Extension trait to convert between different representations.
pub trait MatrixExt<T: ArrayElement + NativeType, const R: usize, const C: usize> {
    /// Converts the matrix to a constant matrix representation.
    fn constant(&self) -> Matrix<T, R, C, Op>;

    /// Converts the matrix to a literal matrix representation.
    fn literal(&self) -> Matrix<T, R, C, Literal>;

    /// Converts the matrix to a buffer for client-side operations.
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
        let inner = client
            .copy_host_buffer(&buf, &[R as i64, C as i64])
            .unwrap();
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
        let inner = client
            .copy_host_buffer(&buf, &[N as i64, R as i64, C as i64])
            .unwrap();
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
        let inner = client
            .copy_host_buffer(&buf, &[N as i64, R as i64, C as i64])
            .unwrap();
        MaybeOwned::Owned(inner)
    }
}
