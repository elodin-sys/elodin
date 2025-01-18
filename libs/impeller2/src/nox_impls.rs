use nox::{Array, ArrayRepr, ConstDim, Dim, Field, ReprMonad};

use crate::{
    com_de::{AsComponentView, FromComponentView},
    error::Error,
    types::ComponentView,
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
