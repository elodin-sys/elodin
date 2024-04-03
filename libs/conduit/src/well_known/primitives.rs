use crate::{Component, ComponentType, ComponentValue, PrimitiveTy};
use ndarray::array;
use smallvec::smallvec;

macro_rules! impl_primitive {
    ($ty:tt, $prim_ty:tt) => {
        impl ComponentType {
            pub fn $ty() -> Self {
                <$ty>::component_type()
            }
        }

        impl Component for $ty {
            const NAME: &'static str = stringify!($ty);

            fn component_type() -> ComponentType {
                ComponentType {
                    primitive_ty: PrimitiveTy::$prim_ty,
                    shape: smallvec![],
                }
            }

            fn component_value<'a>(&self) -> crate::ComponentValue<'a> {
                let arr = array![*self].into_dyn();
                ComponentValue::$prim_ty(ndarray::CowArray::from(arr))
            }

            fn from_component_value(value: crate::ComponentValue<'_>) -> Option<Self>
            where
                Self: Sized,
            {
                let ComponentValue::$prim_ty(arr) = value else {
                    return None;
                };
                let arr = arr.into_dimensionality::<ndarray::Ix0>().ok()?;
                let arr = arr.as_slice()?;
                arr.get(0).copied()
            }
        }
    };
}

impl_primitive!(u8, U8);
impl_primitive!(u16, U16);
impl_primitive!(u32, U32);
impl_primitive!(u64, U64);
impl_primitive!(i8, I8);
impl_primitive!(i16, I16);
impl_primitive!(i32, I32);
impl_primitive!(i64, I64);
impl_primitive!(f32, F32);
impl_primitive!(f64, F64);
impl_primitive!(bool, Bool);
