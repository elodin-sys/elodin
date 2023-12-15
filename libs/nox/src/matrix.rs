use crate::{
    Buffer, BufferArg, Builder, Client, CompFn, FromHost, IntoOp, Literal, MaybeOwned, Op, Scalar,
    Tensor, ToHost,
};
use nalgebra::{ArrayStorage, Const, IsContiguous, Scalar as NalgebraScalar, Storage};
use num_traits::Zero;
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
        let mut out = Self::HostTy::zeros();
        literal.copy_raw_to(out.as_mut_slice()).unwrap();
        out
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

impl<T, const R: usize, const C: usize> Matrix<T, R, C, Literal> {
    fn constant(self, builder: &Builder) -> Matrix<T, R, C, Op> {
        let inner = builder
            .inner
            .constant_literal(&self.inner)
            .expect("constant creation failed");

        Matrix {
            inner,
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
            inner,
            phantom: PhantomData,
        }
    }

    fn literal(&self) -> Matrix<T, R, C, Literal> {
        Matrix {
            inner: xla::Literal::vec1(self.as_slice())
                .reshape(&[R as i64, C as i64])
                .expect("reshape failed"),
            phantom: PhantomData,
        }
    }
}

// impl<T, const R: usize, const C: usize> FromPjrtBuffer
//     for nalgebra::Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>
// where
//     T: xla::NativeType + NalgebraScalar + Zero + ArrayElement,
// {
//     fn from_pjrt(pjrt: Vec<Vec<xla::PjRtBuffer>>) -> Self {
//         let buf = &pjrt[0][0];
//     }
// }

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
        Scalar<O>: IntoOp;
}

impl<T: ArrayElement, const R: usize, const C: usize> MapExt for Matrix<T, R, C> {
    type Elem = Scalar<T>;

    type Output<O> = Matrix<O, R, C>;

    fn map<O>(
        &self,
        func: impl CompFn<(Self::Elem,), Scalar<O>>,
    ) -> Result<Self::Output<O>, xla::Error>
    where
        Scalar<O>: IntoOp,
    {
        let comp = func.build()?;
        Ok(Matrix {
            inner: self.inner.map(comp.comp, &[0, 1])?,
            phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use crate::FixedSliceExt;

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
        let out = exec
            .run(&client, matrix![1.0f32, 2.0, 3.0, 4.0])
            .unwrap()
            .to_host();
        assert_eq!(out, matrix![2.0, 3.0, 4.0, 5.0])
    }

    #[test]
    fn test_map_mut() {
        let client = Client::cpu().unwrap();
        let comp = (|mat: &mut Matrix<f32, 1, 4>, b: &mut Matrix<f32, 1, 3>| {
            *mat = mat.map(|x: Scalar<f32>| x + 1f32).unwrap();
            *b = b.map(|x: Scalar<f32>| x + 2f32).unwrap();
        })
        .build()
        .unwrap();
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
        let mut a = Matrix::from_host(&client, matrix![1.0f32, 2.0, 3.0, 4.0]);
        let mut b = Matrix::from_host(&client, matrix![4.0f32, 0.0, 1.0]);
        exec.run(&client, &mut a, &mut b).unwrap();
        let a = a.to_host();
        assert_eq!(a, matrix![2.0, 3.0, 4.0, 5.0]);
        let b = b.to_host();
        assert_eq!(b, matrix![6.0, 2.0, 3.0]);
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
}
