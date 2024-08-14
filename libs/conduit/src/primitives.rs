use crate::{concat_str, ConstComponent};
use crate::{Component, ComponentType, ComponentValue, PrimitiveTy, ValueRepr};
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
            const MAX_SIZE: usize = core::mem::size_of::<$ty>();
        }

        impl ValueRepr for $ty {
            type ValueDim = ndarray::Ix0;

            fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
                let arr = ndarray::arr0(*self);
                ComponentValue::$prim_ty(ndarray::CowArray::from(arr))
            }

            fn from_component_value<D: ndarray::Dimension>(
                value: crate::ComponentValue<'_, D>,
            ) -> Option<Self> {
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

            const MAX_SIZE: usize = core::mem::size_of::<[$ty; N]>();
        }

        impl<const N: usize> ValueRepr for [$ty; N] {
            type ValueDim = ndarray::Ix1;

            fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
                ComponentValue::$prim_ty(ndarray::CowArray::from(self))
            }

            fn from_component_value<D: ndarray::Dimension>(
                value: crate::ComponentValue<'_, D>,
            ) -> Option<Self> {
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
