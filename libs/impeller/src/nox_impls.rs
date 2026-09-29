use crate::{
    com_de::{AsComponentView, FromComponentView},
    component::{Component, PrimTypeElem},
    error::Error,
    schema::Schema,
    types::ComponentView,
    util::concat_str,
};
use nox::{
    Array, ArrayRepr, ConstDim, Dim, Field, OwnedRepr, ReprMonad, Tensor, array::ArrayViewExt,
};

macro_rules! impl_array {
    ($ty:tt, $prim:tt) => {
        impl<D: nox::Dim + nox::ConstDim> AsComponentView for nox::Array<$ty, D> {
            fn as_component_view(&self) -> ComponentView<'_> {
                ComponentView::$prim(self.view())
            }
        }

        impl<D: nox::Dim + nox::ConstDim> FromComponentView for nox::Array<$ty, D> {
            fn from_component_view(view: ComponentView<'_>) -> Result<Self, Error> {
                let ComponentView::$prim(view) = view else {
                    return Err(Error::InvalidComponentData);
                };
                view.try_to_owned().ok_or(Error::InvalidComponentData)
            }
        }
    };
}

impl_array!(f32, F32);
impl_array!(f64, F64);
impl_array!(i16, I16);
impl_array!(i32, I32);
impl_array!(i64, I64);
impl_array!(u16, U16);
impl_array!(u32, U32);
impl_array!(u64, U64);
impl_array!(bool, Bool);

macro_rules! impl_spatial_ty {
    ($ty:ty; $elem:tt) => {
        impl<$elem> FromComponentView for $ty
        where
            $elem: Field,
            $ty: ReprMonad<ArrayRepr>,
            Array<<$ty as ReprMonad<ArrayRepr>>::Elem, <$ty as ReprMonad<ArrayRepr>>::Dim>:
                FromComponentView,
        {
            fn from_component_view(view: ComponentView<'_>) -> Result<Self, Error> {
                let arr = Array::from_component_view(view)?;
                Ok(<$ty>::from_inner(arr))
            }
        }

        impl<$elem> AsComponentView for $ty
        where
            $elem: Field,
            $ty: ReprMonad<ArrayRepr>,
            Array<<$ty as ReprMonad<ArrayRepr>>::Elem, <$ty as ReprMonad<ArrayRepr>>::Dim>:
                AsComponentView,
        {
            fn as_component_view(&self) -> ComponentView<'_> {
                self.inner().as_component_view()
            }
        }
    };
}

impl_spatial_ty!(nox::SpatialMotion<T, ArrayRepr>; T);
impl_spatial_ty!(nox::SpatialTransform<T, ArrayRepr>; T);
impl_spatial_ty!(nox::SpatialForce<T, ArrayRepr>; T);
impl_spatial_ty!(nox::SpatialInertia<T, ArrayRepr>; T);
impl_spatial_ty!(nox::Quaternion<T, ArrayRepr>; T);

impl<T, D> FromComponentView for nox::Tensor<T, D, ArrayRepr>
where
    T: Field,
    D: Dim + ConstDim,
    nox::Tensor<T, D, ArrayRepr>: ReprMonad<ArrayRepr>,
    Array<T, D>: FromComponentView,
{
    fn from_component_view(view: ComponentView<'_>) -> Result<Self, Error> {
        let arr = Array::from_component_view(view)?;
        Ok(<nox::Tensor<T, D, ArrayRepr>>::from_inner(arr))
    }
}

impl<T, D> AsComponentView for nox::Tensor<T, D, ArrayRepr>
where
    T: Field,
    D: Dim,
    Array<T, D>: AsComponentView,
{
    fn as_component_view(&self) -> ComponentView<'_> {
        self.inner().as_component_view()
    }
}

impl<T, R> Component for nox::SpatialTransform<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("spatial_transform_", T::PRIM_TYPE.as_str());

    #[cfg(feature = "alloc")]
    fn schema() -> Schema<Vec<u64>> {
        Schema::new(T::PRIM_TYPE, [7usize]).unwrap()
    }
}

impl<T, R> Component for nox::SpatialMotion<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("spatial_motion_", T::PRIM_TYPE.as_str());

    #[cfg(feature = "alloc")]
    fn schema() -> Schema<Vec<u64>> {
        Schema::new(T::PRIM_TYPE, [6usize]).unwrap()
    }
}

impl<T, R> Component for nox::SpatialForce<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("spatial_force_", T::PRIM_TYPE.as_str());

    #[cfg(feature = "alloc")]
    fn schema() -> Schema<Vec<u64>> {
        Schema::new(T::PRIM_TYPE, [6usize]).unwrap()
    }
}

impl<T, R> Component for nox::SpatialInertia<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("spatial_inertia_", T::PRIM_TYPE.as_str());

    #[cfg(feature = "alloc")]
    fn schema() -> Schema<Vec<u64>> {
        Schema::new(T::PRIM_TYPE, [7usize]).unwrap()
    }
}

impl<T, R> Component for nox::Quaternion<T, R>
where
    T: Field + PrimTypeElem,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("quaternion_", T::PRIM_TYPE.as_str());

    #[cfg(feature = "alloc")]
    fn schema() -> Schema<Vec<u64>> {
        Schema::new(T::PRIM_TYPE, [4usize]).unwrap()
    }
}

impl<T, D, R> Component for Tensor<T, D, R>
where
    T: Field + PrimTypeElem,
    D: Dim + ConstDim,
    R: OwnedRepr,
{
    const NAME: &'static str = concat_str!("tensor_", T::PRIM_TYPE.as_str());

    #[cfg(feature = "alloc")]
    fn schema() -> Schema<Vec<u64>> {
        Schema::new(T::PRIM_TYPE, D::DIM).unwrap()
    }
}
