use conduit::{ComponentId, ComponentType, PrimitiveTy};
use nox::xla::{ArrayElement, NativeType};
use nox::{IntoOp, Scalar, ScalarExt, Tensor, TensorDim, XlaDim};

use nox_ecs_macros::Component;
use smallvec::smallvec;

pub trait Component: IntoOp + for<'a> nox::FromBuilder<Item<'a> = Self> {
    fn component_id() -> ComponentId;
    fn component_type() -> ComponentType;
    fn is_asset() -> bool {
        false
    }
}

impl<T: ArrayElement + NativeType + conduit::Component, D: TensorDim + XlaDim> Component
    for Tensor<T, D>
{
    fn component_id() -> ComponentId {
        let name = T::NAME;
        // TODO(Akhil): Make this more efficient
        let dims = D::shape()
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("_");
        let comp_name = format!("tensor_{name}_{dims}");
        ComponentId::new(comp_name.as_str())
    }

    fn component_type() -> ComponentType {
        let mut ty = T::component_type();
        ty.shape.insert_from_slice(0, &D::shape());
        ty
    }
}

macro_rules! impl_spatial_ty {
    ($nox_ty:ty, $prim_ty:expr, $shape:expr, $name: tt) => {
        impl Component for $nox_ty {
            fn component_id() -> ComponentId {
                conduit::ComponentId::new($name)
            }

            fn component_type() -> ComponentType {
                ComponentType {
                    primitive_ty: $prim_ty,
                    shape: $shape,
                }
            }
        }
    };
}

impl_spatial_ty!(
    nox::SpatialTransform::<f64>,
    PrimitiveTy::F64,
    smallvec![7],
    "spatial_transform_f64"
);

impl_spatial_ty!(
    nox::SpatialMotion::<f64>,
    PrimitiveTy::F64,
    smallvec![6],
    "spatial_motion_f64"
);

impl_spatial_ty!(
    nox::SpatialInertia::<f64>,
    PrimitiveTy::F64,
    smallvec![7],
    "spatial_inertia_f64"
);

impl_spatial_ty!(
    nox::SpatialForce::<f64>,
    PrimitiveTy::F64,
    smallvec![6],
    "spatial_force_f64"
);

#[derive(Component)]
pub struct WorldPos(pub nox::SpatialTransform<f64>);

#[derive(Component)]
pub struct Seed(pub Scalar<u64>);

impl Seed {
    pub fn zero() -> Self {
        Seed(0u64.constant())
    }
}
