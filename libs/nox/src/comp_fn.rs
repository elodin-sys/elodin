use crate::{Builder, Comp, IntoOp, Op, Tensor, TensorDim, XlaDim};
use std::{any, marker::PhantomData, sync::atomic::Ordering};

pub trait CompFn<T, R>: Send + Sync {
    fn compute(&self, builder: &Builder) -> R;

    fn build(&self) -> Result<Comp<T, R>, xla::Error>
    where
        R: IntoOp,
    {
        let builder = Builder::new(any::type_name::<Self>());
        let res = self.compute(&builder);
        let res = if !builder.mut_params.is_empty() {
            let mut tuple = Vec::with_capacity(builder.mut_params.count() + 1);
            tuple.push(res.into_op(&builder.inner));
            for o in builder.mut_params.into_iter() {
                tuple.insert(1, o.into_inner().into_op(&builder.inner));
            }
            builder.inner.tuple(&tuple)?
        } else {
            res.into_op(&builder.inner)
        };
        let comp = res.build()?;
        Ok(Comp {
            comp,
            phantom: PhantomData,
        })
    }
}

pub trait FromBuilder {
    type Item<'a>;

    fn from_builder(builder: &Builder) -> Self::Item<'_>;
    fn is_mut_borrowed() -> bool {
        false
    }
}

impl<'b> FromBuilder for &'b Builder {
    type Item<'a> = &'a Builder;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        builder
    }
}

impl<T: xla::ArrayElement, D: XlaDim + TensorDim> FromBuilder for Tensor<T, D, Op>
where
    D::Array: AsRef<[i64]>,
{
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        let i = builder.param_count.fetch_add(1, Ordering::SeqCst);
        Tensor {
            inner: builder
                .inner
                .parameter(i, T::TY, D::dims().as_ref(), &format!("param_{}", i))
                .expect("parameter create failed"),
            phantom: PhantomData,
        }
    }
}

impl<'b, T: xla::ArrayElement + 'static, D: XlaDim + TensorDim + 'static> FromBuilder
    for &'b mut Tensor<T, D, Op>
where
    D::Array: AsRef<[i64]>,
{
    type Item<'a> = &'a mut Tensor<T, D, Op>;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        let i = builder.param_count.fetch_add(1, Ordering::SeqCst);
        let tensor_index = builder.mut_params.push(
            Tensor {
                inner: builder
                    .inner
                    .parameter(i, T::TY, D::dims().as_ref(), &format!("param_{}", i))
                    .expect("parameter create failed"),
                phantom: PhantomData,
            }
            .into(),
        );
        // Safety: Boxcar ensures that the pointers are fixed, since it never reallocates.
        // We also do not take a new reference of this type, until the `CompFn` has been called
        let tensor = unsafe { &mut *builder.mut_params[tensor_index].get() };
        // Safety: since we created the inner op above with the correct type and dimension, we can
        // guarentee that this is correct
        unsafe { tensor.unsafe_mut_cast() }
    }

    fn is_mut_borrowed() -> bool {
        true
    }
}

// This macro allows us to implement `CompFn` for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
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
                  let mut alias_index = 0;
                  $(
                      let param_index = builder.param_count.load(Ordering::SeqCst);
                      let $ty = $ty::from_builder(builder);
                      if $ty::is_mut_borrowed() {
                        alias_index += 1;
                        builder.inner.setup_alias(param_index, alias_index).unwrap();
                      }
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
