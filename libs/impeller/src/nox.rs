use core::mem;
use nalgebra::{Const, Dyn};
use nox::{
    Array, ArrayDim, ArrayRepr, ConstDim, Dim, Field, Quaternion, Repr, SpatialForce,
    SpatialInertia, SpatialMotion, SpatialTransform, Tensor,
};
use smallvec::{smallvec, SmallVec};

#[cfg(feature = "xla")]
use nox::{xla::ElementType, ArrayTy, Client, NoxprNode};

use crate::{
    concat_str, Component, ComponentType, ComponentValue, ConstComponent, PrimitiveTy, ValueRepr,
};

#[cfg(feature = "xla")]
use crate::{
    types::ArchetypeName,
    well_known::{Material, Mesh, Shape},
    ComponentExt, Handle, Metadata,
};

#[cfg(feature = "xla")]
impl ComponentValue<'_> {
    pub fn to_pjrt_buf(&self, client: &Client) -> Result<nox::xla::PjRtBuffer, nox::Error> {
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
    #[cfg(feature = "xla")]
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

trait PrimitiveTyElement {
    const PRIMITIVE_TY: PrimitiveTy;
}

macro_rules! impl_primitive_ty_element {
    ($ty:ty, $primitive_ty:ident) => {
        impl PrimitiveTyElement for $ty {
            const PRIMITIVE_TY: PrimitiveTy = PrimitiveTy::$primitive_ty;
        }
    };
}

impl_primitive_ty_element!(f32, F32);
impl_primitive_ty_element!(f64, F64);
impl_primitive_ty_element!(i8, I8);
impl_primitive_ty_element!(i16, I16);
impl_primitive_ty_element!(i32, I32);
impl_primitive_ty_element!(i64, I64);
impl_primitive_ty_element!(u8, U8);
impl_primitive_ty_element!(u16, U16);
impl_primitive_ty_element!(u32, U32);
impl_primitive_ty_element!(u64, U64);

#[cfg(feature = "xla")]
impl From<ComponentType> for ArrayTy {
    fn from(val: ComponentType) -> Self {
        ArrayTy {
            element_type: val.primitive_ty.element_type(),
            shape: val.shape,
        }
    }
}

#[cfg(feature = "xla")]
impl<T: Component + nox::IntoOp + 'static> crate::Archetype for T {
    fn name() -> ArchetypeName {
        ArchetypeName::from(T::NAME)
    }

    fn components() -> Vec<Metadata> {
        vec![T::metadata()]
    }

    fn insert_into_world(self, world: &mut crate::World) {
        use std::ops::Deref;
        let mut col = world.column_mut::<T>().unwrap();
        let op = self.into_op();
        let NoxprNode::Constant(c) = op.deref() else {
            panic!("push into host column must be constant expr");
        };
        col.push_raw(c.data.raw_buf());
    }
}

impl<T: Field + PrimitiveTyElement, D: Dim, R: Repr> Component for Tensor<T, D, R> {
    const NAME: &'static str = concat_str!("tensor_", T::PRIMITIVE_TY.display_str());
    fn component_type() -> ComponentType {
        // If T is an ArrayElement, then it's shape must be ().
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: D::shape(),
        }
    }
}

impl<T: Field + PrimitiveTyElement, D: Dim> ValueRepr for Tensor<T, D, ArrayRepr>
where
    Array<T, D>: ValueRepr,
{
    type ValueDim = <Array<T, D> as ValueRepr>::ValueDim;

    fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
        self.inner().fixed_dim_component_value()
    }

    fn component_value(&self) -> ComponentValue<'_> {
        self.inner().component_value()
    }

    fn from_component_value<Dim: ndarray::Dimension>(
        value: crate::ComponentValue<'_, Dim>,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        Array::<T, D>::from_component_value(value).map(Tensor::from_inner)
    }
}

impl<T: Field + PrimitiveTyElement, R: Repr> Component for nox::Quaternion<T, R> {
    const NAME: &'static str = concat_str!("quaternion_", T::PRIMITIVE_TY.display_str());
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![4],
        }
    }
}

impl<T: Field + PrimitiveTyElement> ValueRepr for Quaternion<T, ArrayRepr>
where
    Array<T, Const<4>>: ValueRepr,
{
    type ValueDim = <Array<T, Const<4>> as ValueRepr>::ValueDim;

    fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
        self.0.fixed_dim_component_value()
    }

    fn component_value(&self) -> ComponentValue<'_> {
        self.0.component_value()
    }

    fn from_component_value<Dim: ndarray::Dimension>(
        value: crate::ComponentValue<'_, Dim>,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        Some(Quaternion(Tensor::from_component_value(value)?))
    }
}

impl<T: Field + PrimitiveTyElement, R: Repr> Component for nox::SpatialTransform<T, R> {
    const NAME: &'static str = concat_str!("spatial_transform_", T::PRIMITIVE_TY.display_str());
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![7],
        }
    }
}

impl<T: Field + PrimitiveTyElement> ValueRepr for SpatialTransform<T, ArrayRepr>
where
    Array<T, Const<7>>: ValueRepr,
{
    type ValueDim = <Array<T, Const<7>> as ValueRepr>::ValueDim;

    fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
        self.inner.fixed_dim_component_value()
    }

    fn component_value(&self) -> ComponentValue<'_> {
        self.inner.component_value()
    }

    fn from_component_value<Dim: ndarray::Dimension>(
        value: crate::ComponentValue<'_, Dim>,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        Some(SpatialTransform {
            inner: Tensor::from_component_value(value)?,
        })
    }
}

impl<T: Field + PrimitiveTyElement, R: Repr> Component for SpatialMotion<T, R> {
    const NAME: &'static str = concat_str!("spatial_motion_", T::PRIMITIVE_TY.display_str());
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![6],
        }
    }
}

impl<T: Field + PrimitiveTyElement> ValueRepr for SpatialMotion<T, ArrayRepr>
where
    Array<T, Const<6>>: ValueRepr,
{
    type ValueDim = <Array<T, Const<6>> as ValueRepr>::ValueDim;

    fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
        self.inner.fixed_dim_component_value()
    }

    fn component_value(&self) -> ComponentValue<'_> {
        self.inner.component_value()
    }

    fn from_component_value<Dim: ndarray::Dimension>(
        value: crate::ComponentValue<'_, Dim>,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        Some(SpatialMotion {
            inner: Tensor::from_component_value(value)?,
        })
    }
}

impl<T: Field + PrimitiveTyElement, R: Repr> Component for nox::SpatialInertia<T, R> {
    const NAME: &'static str = concat_str!("spatial_inertia_", T::PRIMITIVE_TY.display_str());
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![7],
        }
    }
}

impl<T: Field + PrimitiveTyElement> ValueRepr for SpatialInertia<T, ArrayRepr>
where
    Array<T, Const<7>>: ValueRepr,
{
    type ValueDim = <Array<T, Const<7>> as ValueRepr>::ValueDim;

    fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
        self.inner.fixed_dim_component_value()
    }

    fn component_value(&self) -> ComponentValue<'_> {
        self.inner.component_value()
    }

    fn from_component_value<Dim: ndarray::Dimension>(
        value: crate::ComponentValue<'_, Dim>,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        Some(SpatialInertia {
            inner: Tensor::from_component_value(value)?,
        })
    }
}

impl<T: Field + PrimitiveTyElement, R: Repr> Component for nox::SpatialForce<T, R> {
    const NAME: &'static str = concat_str!("spatial_force_", T::PRIMITIVE_TY.display_str());
    fn component_type() -> ComponentType {
        ComponentType {
            primitive_ty: T::PRIMITIVE_TY,
            shape: smallvec![6],
        }
    }
}

impl<T: Field + PrimitiveTyElement> ValueRepr for SpatialForce<T, ArrayRepr>
where
    Array<T, Const<6>>: ValueRepr,
{
    type ValueDim = <Array<T, Const<6>> as ValueRepr>::ValueDim;

    fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
        self.inner.fixed_dim_component_value()
    }

    fn component_value(&self) -> ComponentValue<'_> {
        self.inner.component_value()
    }

    fn from_component_value<Dim: ndarray::Dimension>(
        value: crate::ComponentValue<'_, Dim>,
    ) -> Option<Self>
    where
        Self: Sized,
    {
        Some(SpatialForce {
            inner: Tensor::from_component_value(value)?,
        })
    }
}

#[cfg(feature = "xla")]
impl crate::Archetype for Shape {
    fn name() -> ArchetypeName {
        ArchetypeName::from("Shape")
    }

    fn components() -> Vec<Metadata> {
        vec![Handle::<Mesh>::metadata(), Handle::<Material>::metadata()]
    }

    fn insert_into_world(self, world: &mut crate::World) {
        self.mesh.insert_into_world(world);
        self.material.insert_into_world(world);
    }
}

pub trait ComponentValueDimable: ArrayDim {
    type ValueDim: ndarray::Dimension;

    fn value_dim(dim: Self::Dim) -> Self::ValueDim;
}

impl ComponentValueDimable for () {
    type ValueDim = ndarray::Ix0;

    fn value_dim(_dim: Self::Dim) -> Self::ValueDim {
        ndarray::Ix0()
    }
}

impl<const N: usize> ComponentValueDimable for Const<N> {
    type ValueDim = ndarray::Ix1;

    fn value_dim(_dim: Self::Dim) -> Self::ValueDim {
        ndarray::Ix1(N)
    }
}

impl<const D1: usize, const D2: usize> ComponentValueDimable for (Const<D1>, Const<D2>) {
    type ValueDim = ndarray::Ix2;

    fn value_dim(_dim: Self::Dim) -> Self::ValueDim {
        ndarray::Ix2(D1, D2)
    }
}

impl<const D1: usize, const D2: usize, const D3: usize> ComponentValueDimable
    for (Const<D1>, Const<D2>, Const<D3>)
{
    type ValueDim = ndarray::Ix3;

    fn value_dim(_dim: Self::Dim) -> Self::ValueDim {
        ndarray::Ix3(D1, D2, D3)
    }
}

impl ComponentValueDimable for Dyn {
    type ValueDim = ndarray::IxDyn;

    fn value_dim(dim: Self::Dim) -> Self::ValueDim {
        ndarray::IxDyn(dim.as_ref())
    }
}

macro_rules! impl_array_to_value_repr {
    ($ty:tt, $prim:tt) => {
        impl<D: Dim + ComponentValueDimable> ValueRepr for nox::Array<$ty, D> {
            type ValueDim = D::ValueDim;

            fn fixed_dim_component_value(&self) -> ComponentValue<'_, Self::ValueDim> {
                use nox::ArrayBuf;
                let dim = D::dim(&self.buf);
                ndarray::ArrayView::from_shape(D::value_dim(dim), &self.buf.as_buf())
                    .ok()
                    .map(ndarray::CowArray::from)
                    .map(ComponentValue::$prim)
                    .unwrap()
            }

            fn from_component_value<Dim: ndarray::Dimension>(
                value: crate::ComponentValue<'_, Dim>,
            ) -> Option<Self>
            where
                Self: Sized,
            {
                use nox::ArrayBuf;
                let ComponentValue::$prim(array) = value else {
                    return None;
                };
                let mut uninit = nox::Array::<$ty, D>::zeroed(array.shape());
                let array = array.as_slice()?;
                if array.len() != uninit.buf.as_buf().len() {
                    return None;
                }
                array
                    .iter()
                    .zip(uninit.buf.as_mut_buf())
                    .for_each(|(a, b)| {
                        *b = *a;
                    });
                Some(uninit)
            }
        }
    };
}

impl_array_to_value_repr!(f32, F32);
impl_array_to_value_repr!(f64, F64);
impl_array_to_value_repr!(i16, I16);
impl_array_to_value_repr!(i32, I32);
impl_array_to_value_repr!(i64, I64);
impl_array_to_value_repr!(u16, U16);
impl_array_to_value_repr!(u32, U32);
impl_array_to_value_repr!(u64, U64);

#[allow(clippy::len_zero)]
const fn dim_to_smallvec<Dim: ConstDim>() -> SmallVec<[i64; 4]> {
    let mut arr = [0i64; 4];
    let len = Dim::DIM.len();
    if len > 4 {
        panic!("dim length must be less than or equal to 4");
    }
    arr[0] = if Dim::DIM.len() > 0 {
        Dim::DIM[0] as i64
    } else {
        0
    };
    arr[1] = if Dim::DIM.len() > 1 {
        Dim::DIM[1] as i64
    } else {
        0
    };
    arr[2] = if Dim::DIM.len() > 2 {
        Dim::DIM[2] as i64
    } else {
        0
    };
    arr[3] = if Dim::DIM.len() > 3 {
        Dim::DIM[3] as i64
    } else {
        0
    };

    unsafe { SmallVec::from_const_with_len_unchecked(arr, len) }
}

impl<T: Field + PrimitiveTyElement, D: Dim + ConstDim, R: Repr> ConstComponent for Tensor<T, D, R> {
    const TY: ComponentType = ComponentType {
        primitive_ty: T::PRIMITIVE_TY,
        shape: dim_to_smallvec::<D>(),
    };

    const MAX_SIZE: usize = size_of_dim::<T, D>();
}

const fn size_of_dim<T: Sized, Dim: ConstDim>() -> usize {
    let size = mem::size_of::<T>();
    let len = Dim::DIM.len();
    if len > 4 {
        panic!("dim length must be less than or equal to 4");
    }
    let mut len = 0;
    if !Dim::DIM.is_empty() {
        len += Dim::DIM[0];
    }
    if Dim::DIM.len() > 1 {
        len += Dim::DIM[1];
    }
    if Dim::DIM.len() > 2 {
        len += Dim::DIM[2];
    }
    if Dim::DIM.len() > 3 {
        len += Dim::DIM[3];
    }
    len * size
}

impl<T: Field + PrimitiveTyElement, R: Repr> ConstComponent for SpatialTransform<T, R> {
    const TY: ComponentType = ComponentType {
        primitive_ty: T::PRIMITIVE_TY,
        shape: dim_to_smallvec::<Const<7>>(),
    };

    const MAX_SIZE: usize = size_of_dim::<T, Const<7>>();
}

impl<T: Field + PrimitiveTyElement, R: Repr> ConstComponent for SpatialMotion<T, R> {
    const TY: ComponentType = ComponentType {
        primitive_ty: T::PRIMITIVE_TY,
        shape: dim_to_smallvec::<Const<6>>(),
    };

    const MAX_SIZE: usize = size_of_dim::<T, Const<6>>();
}

impl<T: Field + PrimitiveTyElement, R: Repr> ConstComponent for SpatialForce<T, R> {
    const TY: ComponentType = ComponentType {
        primitive_ty: T::PRIMITIVE_TY,
        shape: dim_to_smallvec::<Const<6>>(),
    };

    const MAX_SIZE: usize = size_of_dim::<T, Const<6>>();
}

impl<T: Field + PrimitiveTyElement, R: Repr> ConstComponent for SpatialInertia<T, R> {
    const TY: ComponentType = ComponentType {
        primitive_ty: T::PRIMITIVE_TY,
        shape: dim_to_smallvec::<Const<7>>(),
    };

    const MAX_SIZE: usize = size_of_dim::<T, Const<7>>();
}

impl<T: Field + PrimitiveTyElement, R: Repr> ConstComponent for Quaternion<T, R> {
    const TY: ComponentType = ComponentType {
        primitive_ty: T::PRIMITIVE_TY,
        shape: dim_to_smallvec::<Const<4>>(),
    };

    const MAX_SIZE: usize = size_of_dim::<T, Const<4>>();
}
