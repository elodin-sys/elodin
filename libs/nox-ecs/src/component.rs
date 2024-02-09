use elodin_conduit::{ComponentId, ComponentType};
use nox::{nalgebra, IntoOp, Scalar, ScalarExt};

pub trait Component: IntoOp + for<'a> nox::FromBuilder<Item<'a> = Self> {
    type Inner;
    type HostTy;

    fn host(val: Self::HostTy) -> Self;
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

            type HostTy = $inner;

            fn host(val: Self::HostTy) -> Self {
                val.constant()
            }

            fn component_id() -> ComponentId {
                use elodin_conduit::Component;
                $inner::component_id()
            }

            fn component_type() -> ComponentType {
                use elodin_conduit::Component;
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

macro_rules! impl_ty {
    ($host_ty:ty, $nox_ty:ty, $comp_ty:expr) => {
        impl Component for $nox_ty {
            type Inner = Self;
            type HostTy = $host_ty;

            fn host(val: Self::HostTy) -> Self {
                use nox::ConstantExt;
                val.constant()
            }

            fn component_id() -> ComponentId {
                elodin_conduit::ComponentId::new(stringify!($nox_ty))
            }

            fn component_type() -> ComponentType {
                $comp_ty
            }
        }
    };
}

impl_ty!(nalgebra::Vector3<f64>, nox::Vector<f64, 3>, ComponentType::Vector3F64);
impl_ty!(nalgebra::Vector3<f32>, nox::Vector<f32, 3>, ComponentType::Vector3F32);

macro_rules! impl_spatial_ty {
    ($nox_ty:ty, $comp_ty:expr, $name: tt) => {
        impl Component for $nox_ty {
            type Inner = Self;
            type HostTy = Self;

            fn host(val: Self::HostTy) -> Self {
                val
            }

            fn component_id() -> ComponentId {
                elodin_conduit::ComponentId::new($name)
            }

            fn component_type() -> ComponentType {
                $comp_ty
            }
        }
    };
}

impl_spatial_ty!(
    nox::SpatialTransform::<f64>,
    ComponentType::SpatialPosF64,
    "spatial_transform_f64"
);

impl_spatial_ty!(
    nox::SpatialMotion::<f64>,
    ComponentType::SpatialMotionF64,
    "spatial_transform_f64"
);
