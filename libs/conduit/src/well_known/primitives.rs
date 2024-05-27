use crate::{concat_str, ConstComponent};
use crate::{Component, ComponentType, ComponentValue, PrimitiveTy, ValueRepr};
use ndarray::{array, CowArray};
use paste::paste;
use smallvec::SmallVec;

macro_rules! impl_primitive {
    ($ty:tt, $prim_ty:tt) => {
        impl ComponentType {
            pub const fn $ty() -> Self {
                ComponentType {
                    primitive_ty: PrimitiveTy::$prim_ty,
                    shape: SmallVec::new_const(),
                }
            }

            paste! {
                pub const fn [<vec_$ty>]<const N: i64>() -> Self {
                    ComponentType {
                        primitive_ty: PrimitiveTy::$prim_ty,
                        shape: unsafe { SmallVec::from_const_with_len_unchecked([N,0,0,0], 1) }
                    }
                }
            }
        }

        impl Component for $ty {
            const NAME: &'static str = stringify!($ty);
            const ASSET: bool = false;

            fn component_type() -> ComponentType {
                ComponentType::$ty()
            }
        }

        impl ConstComponent for $ty {
            const TY: ComponentType = ComponentType::$ty();
        }

        impl ValueRepr for $ty {
            fn component_value(&self) -> crate::ComponentValue<'_> {
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

        impl<const N: usize> Component for [$ty; N] {
            const NAME: &'static str = concat_str!("f32x", stringify!(N));

            fn component_type() -> ComponentType {
                Self::TY
            }
        }

        impl<const N: usize> ConstComponent for [$ty; N] {
            const TY: ComponentType = ComponentType {
                primitive_ty: PrimitiveTy::$prim_ty,
                shape: unsafe { SmallVec::from_const_with_len_unchecked([N as i64, 0, 0, 0], 1) },
            };
        }

        impl<const N: usize> ValueRepr for [$ty; N] {
            fn component_value(&self) -> ComponentValue<'_> {
                ComponentValue::$prim_ty(CowArray::from(self).into_dyn())
            }

            fn from_component_value(value: ComponentValue<'_>) -> Option<Self>
            where
                Self: Sized,
            {
                let ComponentValue::$prim_ty(val) = value else {
                    return None;
                };
                if val.len() != N {
                    return None;
                }
                val.as_slice()?.try_into().ok()
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
