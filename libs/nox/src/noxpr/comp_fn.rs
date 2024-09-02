//! Defines traits and structures for constructing functions and transforming them into computational graphs.
use crate::{ArrayTy, Builder, Comp, ConstDim, Noxpr, NoxprFn, NoxprTy, Op, ReprMonad};
use smallvec::SmallVec;
use std::{any, marker::PhantomData};
use xla::ArrayElement;

/// Represents a computational function that can be converted into an XLA computation.
pub trait CompFn<T, R>: Send + Sync {
    /// Computes the result of the function, using a mutable reference to a `Builder` to construct the computation.
    fn compute(&self, builder: &mut Builder) -> R;

    /// Builds the computational graph (`NoxprFn`) of the function.
    fn build_expr(&self) -> Result<NoxprFn, crate::Error>
    where
        R: ReprMonad<Op>,
    {
        let mut builder = Builder::new();
        let res = self.compute(&mut builder);
        let inner = if !builder.mut_params.is_empty() {
            let mut tuple = Vec::with_capacity(builder.mut_params.count() + 1);
            let res_op = res.into_inner();
            tuple.push(res_op);
            for o in builder.mut_params.into_iter() {
                tuple.insert(1, o.into_inner().inner);
            }
            Noxpr::tuple(tuple)
        } else {
            res.into_inner()
        };
        Ok(NoxprFn {
            inner,
            args: builder.params.into_inner(),
        })
    }

    /// Converts the computational function into a compiled `Comp` type, which is executable within an XLA environment.
    ///
    /// This method is a convenience function that builds the expression and compiles it into a ready-to-execute form.
    fn build(&self) -> Result<Comp<T, R>, crate::Error>
    where
        R: ReprMonad<Op>,
    {
        let expr = self.build_expr()?;
        let op = expr.build(any::type_name::<Self>())?;
        let comp = op.build()?;
        Ok(Comp {
            comp,
            phantom: PhantomData,
        })
    }
}

/// Provides functionality to construct an item from a `Builder`.
pub trait FromBuilder {
    /// Defines the type of item that can be constructed from a `Builder`.
    type Item<'a>;

    /// Constructs an item from the `Builder`.
    fn from_builder(builder: &Builder) -> Self::Item<'_>;

    /// Indicates whether the constructed item has mutable borrow semantics within the builder context.
    ///
    /// Default implementation returns `false`, implying no mutable borrows are made.
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

impl<M: ReprMonad<Op>> FromBuilder for M
where
    M::Elem: ArrayElement,
    M::Dim: ConstDim,
{
    type Item<'a> = M;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        let mut params = builder.params.borrow_mut();
        let i = params.len() as i64;
        let shape: SmallVec<[i64; 4]> = M::Dim::DIM.iter().map(|&x| x as i64).collect();
        let inner = Noxpr::parameter(
            i,
            NoxprTy::ArrayTy(ArrayTy {
                element_type: M::Elem::TY,
                shape,
            }),
            format!("param_{}", i),
        );
        params.push(inner.clone());
        M::from_inner(inner)
    }
}

// TODO(sphw): to make mutable params work again we will need to make some changes to this function
// In particular we need to make this function perform the alias setup itself, right now it doesn't do that
// The other complexity comes from the fact that we need to make sure that the alias setup needs to be done
// in reverse order
// impl<'b, T: xla::ArrayElement + 'static, D: XlaDim + TensorDim + 'static> FromBuilder
//     for &'b mut Tensor<T, D, Op>
// where
//     D::Array: AsRef<[i64]>,
// {
//     type Item<'a> = &'a mut Tensor<T, D, Op>;

//     fn from_builder(builder: &Builder) -> Self::Item<'_> {
//         let mut params = builder.params.borrow_mut();
//         let i = params.len() as i64;
//         let inner = Noxpr::parameter(
//             i,
//             ArrayTy {
//                 element_type: T::TY,
//                 shape: SmallVec::from_slice(D::dims().as_ref()),
//             },
//             format!("param_{}", i),
//         );

//         params.push(inner.clone());
//         let tensor_index = builder.mut_params.push(
//             Tensor {
//                 inner,
//                 phantom: PhantomData,
//             }
//             .into(),
//         );
//         // Safety: Boxcar ensures that the pointers are fixed, since it never reallocates.
//         // We also do not take a new reference of this type, until the `CompFn` has been called
//         let tensor = unsafe { &mut *builder.mut_params[tensor_index].get() };
//         // Safety: since we created the inner op above with the correct type and dimension, we can
//         // guarentee that this is correct
//         unsafe { tensor.unsafe_mut_cast() }
//     }

//     fn is_mut_borrowed() -> bool {
//         true
//     }
// }

// This macro allows us to implement `CompFn` for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
macro_rules! impl_comp_fn {
      ($($ty:tt),*) => {
          impl<$($ty,)*> FromBuilder for ($($ty,)*)
          where
              $($ty: FromBuilder,)*
          {
              type Item<'a> = ($($ty::Item<'a>,)*);

              #[allow(unused_variables, clippy::unused_unit)]
              fn from_builder(builder: &Builder) -> Self::Item<'_> {
                  ($($ty::from_builder(builder),)*)
              }
          }
      };
  }

impl_comp_fn!();
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

impl<F, T, R> CompFn<T, R> for F
where
    F: Sync + Send,
    F: for<'a> fn_traits::Fn<T, Output = R>,
    T: for<'a> FromBuilder<Item<'a> = T>,
{
    fn compute(&self, builder: &mut Builder) -> R {
        let param_index = builder.params.borrow().len();
        if T::is_mut_borrowed() {
            builder.setup_alias(param_index as u64, 1);
        }
        let arg = T::from_builder(builder);
        self.call(arg)
    }
}
