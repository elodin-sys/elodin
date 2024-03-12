use conduit::{ComponentId, ComponentType, PrimitiveTy};
use nox::{IntoOp, Scalar};

use nox_ecs_macros::Component;
use smallvec::smallvec;

pub trait Component: IntoOp + for<'a> nox::FromBuilder<Item<'a> = Self> {
    type Inner;

    fn component_id() -> ComponentId;
    fn component_type() -> ComponentType;
    fn is_asset() -> bool {
        false
    }
}

macro_rules! impl_scalar_primitive {
    ($inner:tt) => {
        impl Component for Scalar<$inner> {
            type Inner = Self;

            fn component_id() -> ComponentId {
                use conduit::Component;
                $inner::component_id()
            }

            fn component_type() -> ComponentType {
                use conduit::Component;
                $inner::component_type()
            }
        }
    };
}

impl_scalar_primitive!(f64);
impl_scalar_primitive!(f32);
impl_scalar_primitive!(u64);
impl_scalar_primitive!(u32);
impl_scalar_primitive!(u16);
impl_scalar_primitive!(i64);
impl_scalar_primitive!(i32);
impl_scalar_primitive!(i16);

macro_rules! impl_spatial_ty {
    ($nox_ty:ty, $prim_ty:expr, $shape:expr, $name: tt) => {
        impl Component for $nox_ty {
            type Inner = Self;

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
