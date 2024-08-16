use conduit::{ComponentId, ComponentValue, ComponentValueDim, EntityId};

pub trait Decomponentize {
    fn apply_value<D: ComponentValueDim>(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: ComponentValue<'_, D>,
    );
}

impl Decomponentize for () {
    fn apply_value<D: ComponentValueDim>(
        &mut self,
        _component_id: ComponentId,
        _entity_id: EntityId,
        _value: ComponentValue<'_, D>,
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
            fn apply_value<D: ComponentValueDim>(
                &mut self,
                component_id: ComponentId,
                entity_id: EntityId,
                value: ComponentValue<'_, D>,
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
