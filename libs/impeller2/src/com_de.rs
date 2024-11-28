use crate::types::{ComponentId, ComponentView, EntityId};

pub trait Componentize {
    fn sink_columns(&self, output: &mut impl Decomponentize);

    const MAX_SIZE: usize = usize::MAX;
}

impl Componentize for () {
    fn sink_columns(&self, _output: &mut impl Decomponentize) {}

    const MAX_SIZE: usize = 0;
}

macro_rules! impl_componentize {
    ($($ty:tt),+) => {
        impl<$($ty),*> Componentize for ($($ty,)*)
        where
            $($ty: Componentize),+
        {
            #[allow(unused_parens, non_snake_case)]
            fn sink_columns(&self, output: &mut impl Decomponentize) {
                let ($($ty,)*) = self;
                $($ty.sink_columns(output);)*
            }

            const MAX_SIZE: usize = 0 $(+ $ty::MAX_SIZE)*;
        }
    };
}

impl_componentize!(T1);
impl_componentize!(T1, T2);
impl_componentize!(T1, T2, T3);
impl_componentize!(T1, T2, T3, T4);
impl_componentize!(T1, T2, T3, T4, T5);
impl_componentize!(T1, T2, T3, T4, T5, T6);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17);
impl_componentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18);

pub trait Decomponentize {
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: ComponentView<'_>,
    );
}

impl Decomponentize for () {
    fn apply_value(
        &mut self,
        _component_id: ComponentId,
        _entity_id: EntityId,
        _value: ComponentView<'_>,
    ) {
    }
}
macro_rules! impl_decomponentize {
    ($($ty:tt),+) => {
        impl<$($ty),*> Decomponentize for ($($ty,)*)
        where
            $($ty: Decomponentize),+
        {
            #[allow(unused_parens, non_snake_case)]
            fn apply_value(
                &mut self,
                component_id: ComponentId,
                entity_id: EntityId,
                value: ComponentView<'_>,
            ) {
                let ($($ty,)*) = self;
                $(
                    $ty.apply_value(component_id, entity_id, value.clone());
                )*
            }
        }
    };
}

impl_decomponentize!(T1);
impl_decomponentize!(T1, T2);
impl_decomponentize!(T1, T2, T3);
impl_decomponentize!(T1, T2, T3, T4);
impl_decomponentize!(T1, T2, T3, T4, T5);
impl_decomponentize!(T1, T2, T3, T4, T5, T6);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17);
impl_decomponentize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18);
