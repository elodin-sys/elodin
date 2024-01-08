use crate::{Component, ComponentArray, Error, SystemParam, ViewTy};
use nox::{CompFn, Noxpr};
use std::marker::PhantomData;

pub struct Query<Param> {
    exprs: Vec<Noxpr>,
    len: usize,
    phantom_data: PhantomData<Param>,
}

macro_rules! impl_query {
    ($num:tt; $($param:tt),*) => {
        impl<$($param),*> SystemParam for Query<($($param,)*)>
            where $(ComponentArray<$param>: SystemParam<Item = ComponentArray<$param>>),*
        {
            type Item = Self;

            fn init(builder: &mut crate::PipelineBuilder) -> Result<(), crate::Error> {
                $(
                    ComponentArray::<$param>::init(builder)?;
                )*
                Ok(())
            }

            fn from_builder(builder: &crate::PipelineBuilder) -> Self {
                let mut exprs = Vec::with_capacity($num);
                $(
                    let a = ComponentArray::<$param>::from_builder(builder);
                    let _len = a.len;
                    exprs.push(a.buffer);
                )*
                Self {
                    exprs ,
                    len: _len,
                    phantom_data: PhantomData,
                }
            }

            fn insert_into_builder(self, _builder: &mut crate::PipelineBuilder) {
            }
        }

        impl<$($param),*> Query<($($param,)*)>
            where $(ComponentArray<$param>: SystemParam<Item = ComponentArray<$param>>),*
        {
            pub fn map<O: Component>(&self, func: impl CompFn<($($param,)*), O>) -> Result<ComponentArray<O>, Error> {
                let func = func.build_expr()?;
                let buffer = Noxpr::vmap_with_axis(func, &[0; $num], &self.exprs)?;
                Ok(ComponentArray {
                    buffer,
                    view_ty: ViewTy::Full,
                    len: self.len,
                    phantom_data: PhantomData,
                })
            }

            pub fn join<B: Component>(mut self, other: &ComponentArray<B>) -> Query<($($param,)* B)> {
                self.exprs.push(other.buffer.clone());
                Query {
                    exprs: self.exprs,
                    len: self.len, // TODO(sphw/ELO-ELO-50): add support for cross archetype queries
                    phantom_data: PhantomData,
                }
            }
        }
    }
}

impl_query!(1; T1);
impl_query!(2; T1, T2);
impl_query!(3; T1, T2, T3);
impl_query!(4; T1, T2, T3, T4);
impl_query!(5; T1, T2, T3, T4, T5);
impl_query!(6; T1, T2, T3, T4, T5, T6);
impl_query!(7; T1, T2, T3, T4, T5, T6, T7);
impl_query!(8; T1, T2, T3, T4, T5, T6, T7, T8);
impl_query!(9; T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_query!(10; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_query!(11; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_query!(12; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

impl<A: Component> ComponentArray<A> {
    pub fn join<B: Component>(&self, other: &ComponentArray<B>) -> Query<(A, B)> {
        Query {
            exprs: vec![self.buffer.clone(), other.buffer.clone()],
            len: self.len, // TODO(sphw/ELO-ELO-50): add support for cross archetype queries
            phantom_data: PhantomData,
        }
    }
}
