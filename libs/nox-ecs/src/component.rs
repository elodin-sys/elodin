use conduit::{ComponentId, ComponentType, PrimitiveTy};
use nox::xla::{ArrayElement, ElementType, NativeType};
use nox::{IntoOp, Scalar, ScalarExt, Tensor, TensorDim, XlaDim};

use nox_ecs_macros::Component;
use smallvec::smallvec;

pub trait Component: IntoOp + for<'a> nox::FromBuilder<Item<'a> = Self> {
    fn name() -> String;
    fn component_type() -> ComponentType;
    fn is_asset() -> bool {
        false
    }
}

pub trait ComponentExt: Component {
    fn component_id() -> ComponentId {
        ComponentId::new(&Self::name())
    }
}

pub fn is_valid_name(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
}

impl<C: Component> ComponentExt for C {}

impl<T: ArrayElement + NativeType, D: TensorDim + XlaDim> Component for Tensor<T, D> {
    fn name() -> String {
        format!("tensor_{}", T::PRIMITIVE_TY)
    }
    fn component_type() -> ComponentType {
        // If T is an ArrayElement, then it's shape must be ().
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: D::shape(),
        }
    }
}

impl<T: ArrayElement + NativeType> Component for nox::SpatialTransform<T> {
    fn name() -> String {
        format!("spatial_transform_{}", T::PRIMITIVE_TY)
    }
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![7],
        }
    }
}

impl<T: ArrayElement + NativeType> Component for nox::SpatialMotion<T> {
    fn name() -> String {
        format!("spatial_motion_{}", T::PRIMITIVE_TY)
    }
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![6],
        }
    }
}

impl<T: ArrayElement + NativeType> Component for nox::SpatialInertia<T> {
    fn name() -> String {
        format!("spatial_inertia_{}", T::PRIMITIVE_TY)
    }
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![7],
        }
    }
}

impl<T: ArrayElement + NativeType> Component for nox::SpatialForce<T> {
    fn name() -> String {
        format!("spatial_force_{}", T::PRIMITIVE_TY)
    }
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![6],
        }
    }
}

#[derive(Component)]
pub struct WorldPos(pub nox::SpatialTransform<f64>);

#[derive(Component)]
pub struct Seed(pub Scalar<u64>);

impl Seed {
    pub fn zero() -> Self {
        Seed(0u64.constant())
    }
}

trait ArrayElementExt: ArrayElement {
    const PRIMITIVE_TY: PrimitiveTy = primitive_ty::<Self>();
}

impl<T: ArrayElement> ArrayElementExt for T {}

const fn primitive_ty<T: ArrayElement>() -> PrimitiveTy {
    match T::TY {
        ElementType::Pred => PrimitiveTy::Bool,
        ElementType::S8 => PrimitiveTy::I8,
        ElementType::S16 => PrimitiveTy::I16,
        ElementType::S32 => PrimitiveTy::I32,
        ElementType::S64 => PrimitiveTy::I64,
        ElementType::U8 => PrimitiveTy::U8,
        ElementType::U16 => PrimitiveTy::U16,
        ElementType::U32 => PrimitiveTy::U32,
        ElementType::U64 => PrimitiveTy::U64,
        ElementType::F32 => PrimitiveTy::F32,
        ElementType::F64 => PrimitiveTy::F64,
        _ => unimplemented!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_component_names() {
        assert!(is_valid_name("world_pos"));
        assert!(is_valid_name("seed"));
        assert!(is_valid_name("c_157cb4f26de"));
    }

    #[test]
    fn test_invalid_component_names() {
        assert!(!is_valid_name("WorldPos"));
        assert!(!is_valid_name("world pos"));
        assert!(!is_valid_name("world-pos"));
        assert!(!is_valid_name("world_pos!"));
        assert!(!is_valid_name("_world_pos!"));
        assert!(!is_valid_name("./world_pos"));
    }

    #[test]
    fn component_names() {
        assert_eq!(WorldPos::name(), "world_pos");
        assert_eq!(Seed::name(), "seed");
    }
}
