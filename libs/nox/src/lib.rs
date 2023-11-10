#![allow(clippy::arc_with_non_send_sync)]

use num_traits::Zero;
use std::{
    any,
    marker::PhantomData,
    ops::{Add, Mul},
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc,
    },
};
use xla::{ArrayElement, NativeType, XlaBuilder, XlaOp};

use nalgebra::{ArrayStorage, ClosedAdd, Const, IsContiguous, Scalar as NalgebraScalar, Storage};

pub struct Op;

pub struct Literal;

pub struct Buffer;

pub trait Param {
    type Inner;
}

impl Param for Op {
    type Inner = XlaOp;
}

impl Param for Literal {
    type Inner = xla::Literal;
}

impl Param for Buffer {
    type Inner = xla::PjRtBuffer;
}

pub struct Matrix<T, const R: usize, const C: usize, P: Param = Op> {
    inner: Arc<P::Inner>,
    phantom: PhantomData<T>,
}

pub struct Scalar<T, P: Param = Op> {
    inner: Arc<P::Inner>,
    phantom: PhantomData<T>,
}

impl<T> AsOp for Scalar<T, Op> {
    fn as_op(&self) -> &XlaOp {
        self.inner.as_ref()
    }
}

impl<T: ClosedAdd + ArrayElement> Add for Scalar<T, Op> {
    type Output = Scalar<T, Op>;

    fn add(self, rhs: Self) -> Self::Output {
        Scalar {
            inner: Arc::new((self.inner.as_ref() + rhs.inner.as_ref()).unwrap()),
            phantom: PhantomData,
        }
    }
}

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

pub struct Builder {
    inner: XlaBuilder,
    param_count: AtomicI64,
}

impl Builder {
    pub fn new(name: &str) -> Self {
        Self {
            inner: XlaBuilder::new(name),
            param_count: AtomicI64::default(),
        }
    }
}

pub trait CompFn<T, R>: Send + Sync {
    fn compute(&self, builder: &Builder) -> R;

    fn build(&self) -> Result<Comp<T, R>, xla::Error>
    where
        R: AsOp,
    {
        let builder = Builder::new(any::type_name::<Self>());
        let res = self.compute(&builder);
        let comp = res.as_op().build()?;
        Ok(Comp {
            comp,
            phantom: PhantomData,
        })
    }
}

trait FromBuilder {
    type Item<'a>;

    fn from_builder(builder: &Builder) -> Self::Item<'_>;
}

impl<'b> FromBuilder for &'b Builder {
    type Item<'a> = &'a Builder;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        builder
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

macro_rules! impl_comp_fn {
      ($($ty:tt),+) => {
          #[allow(non_snake_case)]
          impl<F, $($ty,)* R> CompFn<($($ty, )*), R> for F
          where
              F: Sync + Send,
              F: Fn($($ty, )*) -> R,
              F: for<'a> Fn($(<$ty as FromBuilder>::Item<'a>, )*) -> R ,
              $($ty: FromBuilder, )*
          {

              fn compute(&self, builder: &Builder) -> R {

                  $(
                      let $ty = $ty::from_builder(builder);
                  )*
                  let res = (self)($($ty,)*);
                  res
              }
          }
      };
  }

impl_comp_fn!(T1);
impl_comp_fn!(T1, T2);
impl_comp_fn!(T1, T2, T3);
impl_comp_fn!(T1, T2, T3, T4);
impl_comp_fn!(T1, T2, T3, T4, T5);
impl_comp_fn!(T1, T2, T3, T4, T5, T6);
impl_comp_fn!(T1, T2, T3, T4, T5, T6, T7);
impl_comp_fn!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_comp_fn!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_comp_fn!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_comp_fn!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_comp_fn!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

pub struct Comp<T, R> {
    comp: xla::XlaComputation,
    phantom: PhantomData<(T, R)>,
}

impl<T: BufferForm, R> Comp<T, R> {
    pub fn compile(&self, client: &Client) -> Result<Exec<T::BufferTy, R>, xla::Error> {
        let exec = self.comp.compile(&client.0)?;
        Ok(Exec {
            exec,
            phantom: PhantomData,
        })
    }
}

pub struct Exec<T, R> {
    exec: xla::PjRtLoadedExecutable,
    phantom: PhantomData<(T, R)>,
}

pub trait CompTy {
    type HostTy;

    fn from_host(client: &Client, native: Self::HostTy) -> Self;
    fn to_host(self) -> Self::HostTy;
}

pub trait FromPjrtBuffer {
    fn from_pjrt(pjrt: Vec<Vec<xla::PjRtBuffer>>) -> Self;
}

pub trait AsBuffer {
    fn as_buffer(&self) -> &xla::PjRtBuffer;
}

pub trait AsOp {
    fn as_op(&self) -> &XlaOp;
}

pub trait BufferForm {
    type BufferTy;
}

impl<T, const R: usize, const C: usize> AsOp for Matrix<T, R, C, Op> {
    fn as_op(&self) -> &XlaOp {
        &self.inner
    }
}

impl<T, const R: usize, const C: usize> BufferForm for Matrix<T, R, C, Op> {
    type BufferTy = Matrix<T, R, C, Buffer>;
}

impl<T, const R: usize, const C: usize> CompTy for Matrix<T, R, C, Buffer>
where
    T: NativeType + NalgebraScalar + ArrayElement,
{
    type HostTy = nalgebra::Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;

    fn from_host(client: &Client, native: Self::HostTy) -> Self {
        native.buffer(client)
    }

    fn to_host(self) -> Self::HostTy {
        todo!()
    }
}

macro_rules! impl_exec {
      ($($ty:tt),+) => {
        #[allow(non_snake_case, clippy::too_many_arguments)]
        impl<$($ty,)* R> Exec<($($ty,)*), R>
        where
            R: ToHost,
            R::HostTy: FromPjrtBuffer,
            $($ty: CompTy + AsBuffer, )*
        {
            pub fn run(&self, client: &Client, $($ty: $ty::HostTy,)*) -> Result<R::HostTy, xla::Error> {
                $(
                let $ty = $ty::from_host(client, $ty);
                let $ty = $ty.as_buffer();
                )*
                let res = self.exec.execute_b(&[$($ty,)*])?;
                Ok(R::HostTy::from_pjrt(res))
            }
        }
      }
}

pub trait ToHost {
    type HostTy;
}

impl_exec!(T1);
impl_exec!(T1, T2);
impl_exec!(T1, T2, T3);
impl_exec!(T1, T2, T3, T4);
impl_exec!(T1, T2, T3, T4, T5);
impl_exec!(T1, T2, T3, T4, T5, T6);
impl_exec!(T1, T2, T3, T4, T5, T6, T7);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

macro_rules! impl_literal_form {
      ($($ty:tt),+) => {
        impl<$($ty,)*> BufferForm for ($($ty,)*)
              where $($ty: BufferForm, )*
        {
            type BufferTy = ($($ty::BufferTy,)*);
        }
      }
}

impl_literal_form!(T1);
impl_literal_form!(T1, T2);
impl_literal_form!(T1, T2, T3);
impl_literal_form!(T1, T2, T3, T4);
impl_literal_form!(T1, T2, T3, T4, T5);
impl_literal_form!(T1, T2, T3, T4, T5, T6);
impl_literal_form!(T1, T2, T3, T4, T5, T6, T7);
impl_literal_form!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_literal_form!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_literal_form!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_literal_form!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_literal_form!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

pub struct Client(xla::PjRtClient);

impl Client {
    pub fn cpu() -> Result<Self, xla::Error> {
        xla::PjRtClient::cpu().map(Client)
    }

    pub fn gpu() -> Result<Self, xla::Error> {
        xla::PjRtClient::gpu(0.95, false).map(Client)
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
        let client = Client::gpu().unwrap();
        let comp = Matrix::add.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(&client, matrix![1.0f32, 2.0], matrix![2.0, 3.0])
            .unwrap();
        assert_eq!(out, matrix![3.0, 5.0]);
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
}
