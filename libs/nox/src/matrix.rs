use crate::{
    AsBuffer, AsOp, Buffer, BufferForm, Builder, Client, CompFn, FromBuilder, FromHost,
    FromPjrtBuffer, Literal, Op, Param, Scalar, Tensor, ToHost,
};
use nalgebra::{ArrayStorage, ClosedAdd, Const, IsContiguous, Scalar as NalgebraScalar, Storage};
use num_traits::Zero;
use std::{
    marker::PhantomData,
    ops::{Add, Mul, Sub},
    sync::{atomic::Ordering, Arc},
};
use xla::{ArrayElement, NativeType, XlaOp};

pub struct Matrix<T, const R: usize, const C: usize, P: Param = Op> {
    pub(crate) inner: Arc<P::Inner>,
    phantom: PhantomData<T>,
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C, Op> {
    pub fn fixed_slice<const NR: usize, const NC: usize>(
        &self,
        row_offset: usize,
        col_offset: usize,
    ) -> Matrix<T, NR, NC> {
        let row_offset = row_offset as i64;
        let col_offset = col_offset as i64;
        Matrix {
            inner: Arc::new(
                self.as_op()
                    .slice(
                        &[row_offset, col_offset],
                        &[row_offset + (NR as i64), col_offset + (NC as i64)],
                        &[1, 1],
                    )
                    .unwrap(),
            ),
            phantom: PhantomData,
        }
    }
}

impl<T, const R: usize, const C: usize> Tensor for Matrix<T, R, C, Op> {
    fn from_op(op: XlaOp) -> Self {
        Self {
            inner: Arc::new(op),
            phantom: PhantomData,
        }
    }
}

impl<T, const R: usize, const C: usize, P: Param> ToHost for Matrix<T, R, C, P> {
    type HostTy = nalgebra::Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;
}

impl<T, const R: usize, const C: usize> AsBuffer for Matrix<T, R, C, Buffer> {
    fn as_buffer(&self) -> &xla::PjRtBuffer {
        self.inner.as_ref()
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C, Literal> {
    fn constant(self, builder: &Builder) -> Matrix<T, R, C, Op> {
        let inner = Arc::new(
            builder
                .inner
                .constant_literal(&self.inner)
                .expect("constant creation failed"),
        );

        Matrix {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: NalgebraScalar + ClosedAdd, const R: usize, const C: usize> Add for Matrix<T, R, C> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Matrix {
            inner: Arc::new((self.inner.as_ref() + rhs.inner.as_ref()).expect("xla build error")),
            phantom: PhantomData,
        }
    }
}

impl<T: NalgebraScalar + ClosedAdd, const R: usize, const C: usize> Sub for Matrix<T, R, C> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Matrix {
            inner: Arc::new((self.inner.as_ref() - rhs.inner.as_ref()).expect("xla build error")),
            phantom: PhantomData,
        }
    }
}

impl<T: NalgebraScalar + ClosedAdd, const R: usize, const C: usize> Mul for Matrix<T, R, C> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Matrix {
            inner: Arc::new((self.inner.as_ref() * rhs.inner.as_ref()).expect("xla build error")),
            phantom: PhantomData,
        }
    }
}

pub trait MatrixExt<T, const R: usize, const C: usize> {
    fn constant(&self, builder: &Builder) -> Matrix<T, R, C, Op>;
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
    fn constant(&self, builder: &Builder) -> Matrix<T, R, C, Op> {
        self.literal().constant(builder)
    }

    fn buffer(&self, client: &Client) -> Matrix<T, R, C, Buffer> {
        let inner = client
            .0
            .buffer_from_host_buffer(self.as_slice(), &[R, C], None)
            .unwrap();
        Matrix {
            inner: Arc::new(inner),
            phantom: PhantomData,
        }
    }

    fn literal(&self) -> Matrix<T, R, C, Literal> {
        Matrix {
            inner: Arc::new(
                xla::Literal::vec1(self.as_slice())
                    .reshape(&[R as i64, C as i64])
                    .expect("reshape failed"),
            ),
            phantom: PhantomData,
        }
    }
}

impl<T, const R: usize, const C: usize> FromPjrtBuffer
    for nalgebra::Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement,
{
    fn from_pjrt(pjrt: Vec<Vec<xla::PjRtBuffer>>) -> Self {
        let buf = &pjrt[0][0];
        let literal = buf.to_literal_sync().unwrap();
        let mut out = Self::zeros();
        literal.copy_raw_to(out.as_mut_slice()).unwrap();
        out
    }
}

impl<T: xla::ArrayElement, const R: usize, const C: usize> FromBuilder for Matrix<T, R, C, Op> {
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        let i = builder.param_count.fetch_add(1, Ordering::SeqCst);
        Matrix {
            inner: Arc::new(
                builder
                    .inner
                    .parameter(i, T::TY, &[R as i64, C as i64], &format!("param_{}", i))
                    .expect("parameter create failed"),
            ),
            phantom: PhantomData,
        }
    }
}

impl<T, const R: usize, const C: usize> AsOp for Matrix<T, R, C, Op> {
    fn as_op(&self) -> &XlaOp {
        &self.inner
    }
}

impl<T, const R: usize, const C: usize> BufferForm for Matrix<T, R, C, Op> {
    type BufferTy = Matrix<T, R, C, Buffer>;
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

pub trait MapExt {
    type Elem;
    type Output<R>;

    fn map<O>(
        &self,
        func: impl CompFn<(Self::Elem,), Scalar<O>>,
    ) -> Result<Self::Output<O>, xla::Error>
    where
        Scalar<O>: AsOp;
}

impl<T: ArrayElement, const R: usize, const C: usize> MapExt for Matrix<T, R, C> {
    type Elem = Scalar<T>;

    type Output<O> = Matrix<O, R, C>;

    fn map<O>(
        &self,
        func: impl CompFn<(Self::Elem,), Scalar<O>>,
    ) -> Result<Self::Output<O>, xla::Error>
    where
        Scalar<O>: AsOp,
    {
        let comp = func.build()?;
        Ok(Matrix {
            inner: Arc::new(self.as_op().map(comp.comp, &[0, 1])?),
            phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    #[test]
    fn test_add() {
        let client = Client::cpu().unwrap();
        let comp = Matrix::add.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, matrix![1.0f32, 2.0], matrix![2.0, 3.0])
            .unwrap();
        assert_eq!(out, matrix![3.0, 5.0]);
    }

    #[test]
    fn test_sub() {
        let client = Client::cpu().unwrap();
        let comp = Matrix::sub.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, matrix![1.0f32, 2.0], matrix![2.0, 3.0])
            .unwrap();
        assert_eq!(out, matrix![-1.0, -1.0]);
    }

    #[test]
    fn test_mul() {
        let client = Client::cpu().unwrap();
        let comp = Matrix::mul.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(
                &client,
                matrix![1.0f32, 2.0; 2.0, 3.0],
                matrix![2.0, 3.0; 4.0, 5.0],
            )
            .unwrap();
        assert_eq!(out, matrix![2., 6.; 8., 15.]);
    }

    #[test]
    fn test_map() {
        let client = Client::cpu().unwrap();
        fn add_one(mat: Matrix<f32, 1, 4>) -> Matrix<f32, 1, 4> {
            mat.map(|x: Scalar<f32>| x + 1f32).unwrap()
        }
        let comp = add_one.build().unwrap();
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
        let out = exec.run(&client, matrix![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(out, matrix![2.0, 3.0, 4.0, 5.0])
    }

    #[test]
    fn test_fixed_slice() {
        let client = Client::cpu().unwrap();
        fn slice(mat: Matrix<f32, 1, 4>) -> Matrix<f32, 1, 1> {
            mat.fixed_slice::<1, 1>(0, 2)
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
        let out = exec.run(&client, matrix![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(out, matrix![3.0])
    }
}
