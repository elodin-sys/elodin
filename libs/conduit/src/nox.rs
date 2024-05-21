use nox::{
    xla::{ArrayElement, ElementType, NativeType, PjRtBuffer},
    ArrayTy, Client, Dim, IntoOp, NoxprNode, Tensor,
};
use smallvec::smallvec;
use std::ops::Deref;

use crate::{
    types::ArchetypeName,
    well_known::{Material, Mesh, Shape},
    world::World,
    Archetype, Component, ComponentExt, ComponentType, ComponentValue, Handle, Metadata,
    PrimitiveTy,
};

impl ComponentValue<'_> {
    pub fn to_pjrt_buf(&self, client: &Client) -> Result<PjRtBuffer, nox::Error> {
        let ComponentType {
            primitive_ty,
            shape,
        } = self.ty();
        let element_ty = primitive_ty.element_type();
        let bytes = self.bytes().ok_or(nox::Error::OutOfBoundsAccess)?;
        client
            .copy_raw_host_buffer(element_ty, bytes, &shape)
            .map_err(nox::Error::from)
    }
}

impl PrimitiveTy {
    #[inline]
    pub fn element_type(self) -> ElementType {
        match self {
            PrimitiveTy::U8 => ElementType::U8,
            PrimitiveTy::U16 => ElementType::U16,
            PrimitiveTy::U32 => ElementType::U32,
            PrimitiveTy::U64 => ElementType::U64,
            PrimitiveTy::I8 => ElementType::S8,
            PrimitiveTy::I16 => ElementType::S16,
            PrimitiveTy::I32 => ElementType::S32,
            PrimitiveTy::I64 => ElementType::S64,
            PrimitiveTy::Bool => ElementType::Pred,
            PrimitiveTy::F32 => ElementType::F32,
            PrimitiveTy::F64 => ElementType::F64,
        }
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

impl From<ComponentType> for ArrayTy {
    fn from(val: ComponentType) -> Self {
        ArrayTy {
            element_type: val.primitive_ty.element_type(),
            shape: val.shape,
        }
    }
}

impl<T: Component + IntoOp + 'static> Archetype for T {
    fn name() -> ArchetypeName {
        ArchetypeName::from(T::name().as_str())
    }

    fn components() -> Vec<Metadata> {
        vec![T::metadata()]
    }

    fn insert_into_world(self, world: &mut World) {
        let mut col = world.column_mut::<T>().unwrap();
        let op = self.into_op();
        let NoxprNode::Constant(c) = op.deref() else {
            panic!("push into host column must be constant expr");
        };
        col.push_raw(c.data.raw_buf());
    }
}

impl<T: ArrayElement + NativeType, D: Dim> Component for Tensor<T, D> {
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

impl<T: ArrayElement + NativeType> Component for nox::Quaternion<T> {
    fn name() -> String {
        format!("quaternion_{}", T::PRIMITIVE_TY)
    }
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![4],
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

impl Archetype for Shape {
    fn name() -> ArchetypeName {
        ArchetypeName::from("Shape")
    }

    fn components() -> Vec<Metadata> {
        vec![Handle::<Mesh>::metadata(), Handle::<Material>::metadata()]
    }

    fn insert_into_world(self, world: &mut World) {
        self.mesh.insert_into_world(world);
        self.material.insert_into_world(world);
    }
}
