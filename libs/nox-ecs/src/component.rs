use conduit::{ComponentId, ComponentType, PrimitiveTy};
use nox::xla::{ArrayElement, NativeType};
use nox::{IntoOp, Scalar, ScalarExt, Tensor, TensorDim, XlaDim};

use nox_ecs_macros::Component;
use smallvec::smallvec;

pub trait Component: IntoOp + for<'a> nox::FromBuilder<Item<'a> = Self> {
    const NAME: Option<&'static str> = None;

    fn component_id() -> ComponentId {
        let name = Self::NAME.unwrap_or_else(|| std::any::type_name::<Self>());
        let ty = Self::component_type();
        ComponentId::new(&format!("{name}:{ty}"))
    }
    fn component_type() -> ComponentType;
    fn is_asset() -> bool {
        false
    }
}

impl<T: ArrayElement + NativeType + conduit::Component, D: TensorDim + XlaDim> Component
    for Tensor<T, D>
{
    fn component_type() -> ComponentType {
        let mut ty = T::component_type();
        ty.shape.insert_from_slice(0, &D::shape());
        ty
    }
}

macro_rules! impl_spatial_ty {
    ($nox_ty:ty, $prim_ty:expr, $shape:expr) => {
        impl Component for $nox_ty {
            fn component_type() -> ComponentType {
                ComponentType {
                    primitive_ty: $prim_ty,
                    shape: $shape,
                }
            }
        }
    };
}

impl_spatial_ty!(nox::SpatialTransform::<f64>, PrimitiveTy::F64, smallvec![7]);

impl_spatial_ty!(nox::SpatialMotion::<f64>, PrimitiveTy::F64, smallvec![6]);

impl_spatial_ty!(nox::SpatialInertia::<f64>, PrimitiveTy::F64, smallvec![7]);

impl_spatial_ty!(nox::SpatialForce::<f64>, PrimitiveTy::F64, smallvec![6]);

#[derive(Component)]
#[nox(name = "world_pos")]
pub struct WorldPos(pub nox::SpatialTransform<f64>);

#[derive(Component)]
#[nox(name = "seed")]
pub struct Seed(pub Scalar<u64>);

impl Seed {
    pub fn zero() -> Self {
        Seed(0u64.constant())
    }
}
