use crate::{Component, ComponentType, ComponentValue, PrimitiveTy, ValueRepr};
use smallvec::smallvec;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct Camera;

impl Component for Camera {
    const NAME: &'static str = "camera";
    const ASSET: bool = false;

    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: PrimitiveTy::U64,
            shape: smallvec![0],
        }
    }
}

impl ValueRepr for Camera {
    type ValueDim = ndarray::Ix0;

    fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
        ComponentValue::U64(ndarray::CowArray::from(ndarray::arr0(0)))
    }

    fn from_component_value<D: ndarray::Dimension>(
        _value: crate::ComponentValue<'_, D>,
    ) -> Option<Self> {
        Some(Self)
    }
}
