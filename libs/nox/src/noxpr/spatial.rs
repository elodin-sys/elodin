use xla::NativeType;

use crate::{
    Builder, Elem, FromBuilder, FromOp, Noxpr, Op, SpatialForce, SpatialInertia, SpatialMotion,
    SpatialTransform, TensorItem, Vector,
};

impl<T: TensorItem> FromOp for SpatialMotion<T, Op> {
    fn from_op(op: Noxpr) -> Self {
        Self {
            inner: Vector::from_op(op),
        }
    }
}

impl<T: TensorItem> crate::IntoOp for SpatialMotion<T, Op> {
    fn into_op(self) -> Noxpr {
        self.inner.into_op()
    }
}

impl<T: xla::ArrayElement + NativeType + Elem> FromBuilder for SpatialMotion<T, Op> {
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        Self {
            inner: Vector::from_builder(builder),
        }
    }
}

impl<T: TensorItem> crate::IntoOp for SpatialInertia<T, Op> {
    fn into_op(self) -> Noxpr {
        self.inner.into_op()
    }
}

impl<T: TensorItem> FromOp for SpatialInertia<T, Op> {
    fn from_op(op: Noxpr) -> Self {
        Self {
            inner: Vector::from_op(op),
        }
    }
}

impl<T: xla::ArrayElement + NativeType + Elem> FromBuilder for SpatialInertia<T, Op> {
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        Self {
            inner: Vector::from_builder(builder),
        }
    }
}

impl<T: TensorItem> crate::IntoOp for SpatialForce<T, Op> {
    fn into_op(self) -> Noxpr {
        self.inner.into_op()
    }
}

impl<T: TensorItem> FromOp for SpatialForce<T, Op> {
    fn from_op(op: Noxpr) -> Self {
        Self {
            inner: Vector::from_op(op),
        }
    }
}

impl<T: xla::ArrayElement + NativeType + Elem> FromBuilder for SpatialForce<T, Op> {
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        Self {
            inner: Vector::from_builder(builder),
        }
    }
}

impl<T: TensorItem + Elem> FromOp for SpatialTransform<T, Op> {
    fn from_op(op: Noxpr) -> Self {
        Self {
            inner: Vector::from_op(op),
        }
    }
}

impl<T: xla::ArrayElement + NativeType + Elem> FromBuilder for SpatialTransform<T, Op> {
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        Self {
            inner: Vector::from_builder(builder),
        }
    }
}

impl<T: TensorItem> crate::IntoOp for SpatialTransform<T, Op> {
    fn into_op(self) -> Noxpr {
        self.inner.into_op()
    }
}
