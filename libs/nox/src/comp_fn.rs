use crate::{AsOp, Builder, Comp};
use std::{any, marker::PhantomData};

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

pub trait FromBuilder {
    type Item<'a>;

    fn from_builder(builder: &Builder) -> Self::Item<'_>;
}

impl<'b> FromBuilder for &'b Builder {
    type Item<'a> = &'a Builder;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        builder
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
