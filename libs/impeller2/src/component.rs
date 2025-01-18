use crate::types::ComponentId;
use crate::{types::PrimType, util::concat_str};
use nox::{ConstDim, Dim, Field, OwnedRepr, Tensor};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::schema::Schema;

pub trait Component {
    const NAME: &'static str;
    const ASSET: bool = false;
    const COMPONENT_ID: ComponentId = ComponentId::new(Self::NAME);

    #[cfg(feature = "std")]
    fn schema() -> Schema<Vec<u64>>;
}

impl<T, D, R> Component for Tensor<T, D, R>
where
    T: Field + PrimTypeElem,
    D: Dim + ConstDim,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("tensor_", T::PRIM_TYPE.as_str());

    fn schema() -> Schema<Vec<u64>> {
        Schema::new(Self::COMPONENT_ID, T::PRIM_TYPE, D::DIM).unwrap()
    }
}

impl<T, R> Component for nox::SpatialTransform<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("spatial_transform_", T::PRIM_TYPE.as_str());

    fn schema() -> Schema<Vec<u64>> {
        Schema::new(Self::COMPONENT_ID, T::PRIM_TYPE, [7usize]).unwrap()
    }
}

impl<T, R> Component for nox::SpatialMotion<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("spatial_motion_", T::PRIM_TYPE.as_str());

    fn schema() -> Schema<Vec<u64>> {
        Schema::new(Self::COMPONENT_ID, T::PRIM_TYPE, [6usize]).unwrap()
    }
}

impl<T, R> Component for nox::SpatialForce<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("spatial_force_", T::PRIM_TYPE.as_str());

    fn schema() -> Schema<Vec<u64>> {
        Schema::new(Self::COMPONENT_ID, T::PRIM_TYPE, [6usize]).unwrap()
    }
}

impl<T, R> Component for nox::SpatialInertia<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("spatial_inertia_", T::PRIM_TYPE.as_str());

    fn schema() -> Schema<Vec<u64>> {
        Schema::new(Self::COMPONENT_ID, T::PRIM_TYPE, [7usize]).unwrap()
    }
}

impl<T, R> Component for nox::Quaternion<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("quaternion_", T::PRIM_TYPE.as_str());

    fn schema() -> Schema<Vec<u64>> {
        Schema::new(Self::COMPONENT_ID, T::PRIM_TYPE, [4usize]).unwrap()
    }
}

trait PrimTypeElem {
    const PRIM_TYPE: PrimType;
}

macro_rules! impl_prim_type_element {
    ($ty:ty, $prim_ty:ident) => {
        impl PrimTypeElem for $ty {
            const PRIM_TYPE: PrimType = PrimType::$prim_ty;
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

            fn schema() -> Schema<Vec<u64>> {
                Schema::new(Self::COMPONENT_ID, <$ty>::PRIM_TYPE, [1usize]).unwrap()
            }
        }

        impl<const N: usize> Component for [$ty; N] {
            const NAME: &'static str =
                concat_str!(concat_str!(stringify!($ty), "x"), stringify!(N));

            fn schema() -> Schema<Vec<u64>> {
                Schema::new(Self::COMPONENT_ID, <$ty>::PRIM_TYPE, [N]).unwrap()
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
