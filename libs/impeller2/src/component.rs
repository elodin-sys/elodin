use crate::types::ComponentId;
use crate::util::concat_str;
use serde::Serialize;
use serde::de::DeserializeOwned;

#[cfg(feature = "alloc")]
use crate::schema::Schema;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

pub trait Component {
    const NAME: &'static str;
    const ASSET: bool = false;
    const COMPONENT_ID: ComponentId = ComponentId::new(Self::NAME);

    #[cfg(feature = "alloc")]
    fn schema() -> Schema<Vec<u64>>;
}

#[cfg(feature = "alloc")]
pub(crate) trait PrimTypeElem {
    const PRIM_TYPE: crate::types::PrimType;
}

macro_rules! impl_prim_type_element {
    ($ty:ty, $prim_ty:ident) => {
        #[cfg(feature = "alloc")]
        impl PrimTypeElem for $ty {
            const PRIM_TYPE: crate::types::PrimType = crate::types::PrimType::$prim_ty;
        }
    };
}

impl_prim_type_element!(f32, F32);
impl_prim_type_element!(f64, F64);
impl_prim_type_element!(i8, I8);
impl_prim_type_element!(i16, I16);
impl_prim_type_element!(i32, I32);
impl_prim_type_element!(i64, I64);
impl_prim_type_element!(u8, U8);
impl_prim_type_element!(u16, U16);
impl_prim_type_element!(u32, U32);
impl_prim_type_element!(u64, U64);

pub trait Asset: DeserializeOwned + Serialize {
    const NAME: &'static str;
}

macro_rules! impl_prim_component {
    ($ty:ty) => {
        impl Component for $ty {
            const NAME: &'static str = stringify!($ty);

            #[cfg(feature = "alloc")]
            fn schema() -> Schema<Vec<u64>> {
                Schema::new(<$ty>::PRIM_TYPE, [0u64; 0]).unwrap()
            }
        }

        impl<const N: usize> Component for [$ty; N] {
            const NAME: &'static str =
                concat_str!(concat_str!(stringify!($ty), "x"), stringify!(N));

            #[cfg(feature = "alloc")]
            fn schema() -> Schema<Vec<u64>> {
                Schema::new(<$ty>::PRIM_TYPE, [N]).unwrap()
            }
        }
    };
}

impl_prim_component!(f32);
impl_prim_component!(f64);
impl_prim_component!(i8);
impl_prim_component!(i16);
impl_prim_component!(i32);
impl_prim_component!(i64);
impl_prim_component!(u8);
impl_prim_component!(u16);
impl_prim_component!(u32);
impl_prim_component!(u64);
