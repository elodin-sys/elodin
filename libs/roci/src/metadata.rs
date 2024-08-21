use impeller::{ComponentId, Metadata};

pub trait Metadatatize {
    fn get_metadata(&self, component_id: ComponentId) -> Option<&Metadata>;

    fn metadata() -> impl Iterator<Item = Metadata> {
        std::iter::empty()
    }
}

macro_rules! impl_metadatatize {
    ($($ty:tt),+) => {
        impl<$($ty),*> Metadatatize for ($($ty,)*)
        where
            $($ty: Metadatatize),+
        {
            #[allow(unused_parens, non_snake_case)]
            fn get_metadata(&self, component_id: ComponentId) -> Option<&Metadata> {
                let ($($ty,)*) = self;
                None$(
                    .or_else(|| $ty.get_metadata(component_id))
                )*
            }
        }
    };
}

impl_metadatatize!(T1);
impl_metadatatize!(T1, T2);
impl_metadatatize!(T1, T2, T3);
impl_metadatatize!(T1, T2, T3, T4);
impl_metadatatize!(T1, T2, T3, T4, T5);
impl_metadatatize!(T1, T2, T3, T4, T5, T6);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17);
impl_metadatatize!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18);
