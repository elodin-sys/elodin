//! Provides the core functionality for manipulating tensors.
use crate::{
    AsBuffer, Buffer, Field, FromOp, IntoOp, Noxpr, NoxprScalarExt, Op, Param, Scalar, Vector,
};
use nalgebra::{constraint::ShapeConstraint, ClosedMul, Const, Scalar as NalgebraScalar};
use simba::scalar::ClosedNeg;
use smallvec::{smallvec, SmallVec};
use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};
use xla::{ArrayElement, ElementType, NativeType};

/// Represents a tensor with a specific type `T`, dimensionality `D`, and underlying representation `P`.
#[repr(transparent)]
pub struct Tensor<T, D: TensorDim, P: Param = Op> {
    pub(crate) inner: P::Inner,
    pub(crate) phantom: PhantomData<(T, D)>,
}

impl<T, D: TensorDim, P: Param> std::fmt::Debug for Tensor<T, D, P>
where
    P::Inner: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("inner", &self.inner)
            .finish()
    }
}

/// Trait for items that can be represented as tensors.
/// Specifies the type of the item, its tensor representation, and dimensionality.
pub trait TensorItem {
    type Item: FromOp;
    type Tensor<D>
    where
        D: TensorDim;
    type Dim: TensorDim;
    const ELEM: ElementType;
}

impl<T: NativeType + ArrayElement> TensorItem for T {
    type Item = Scalar<T>;
    type Tensor<D> = Tensor<T, D> where D: TensorDim;
    type Dim = ();

    const ELEM: ElementType = T::TY;
}

impl<T: TensorItem, D: TensorDim> TensorItem for Tensor<T, D, Op> {
    type Item = T::Item; // NOTE: this bound might be wrong

    type Dim = D;
    type Tensor<TD: TensorDim> = Tensor<T, TD>;

    const ELEM: ElementType = T::ELEM;
}

impl<T, D: TensorDim> FromOp for Tensor<T, D> {
    fn from_op(inner: Noxpr) -> Self {
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

/// Trait for collapsing a tensor into a simpler form, typically by reducing its dimensionality.
pub trait Collapse {
    type Out;
    /// Collapses the tensor into a simpler form.
    fn collapse(self) -> Self::Out;
}

impl<T: TensorItem> Collapse for Scalar<T, Op>
where
    T::Item: IntoOp,
{
    type Out = <T as TensorItem>::Item;

    fn collapse(self) -> Self::Out {
        T::Item::from_op(self.inner)
    }
}

impl<T: TensorItem, InnerDim: TensorDim, D: TensorDim + NonScalarDim> Collapse
    for Tensor<Tensor<T, InnerDim>, D, Op>
where
    (D, InnerDim): DimConcat<D, InnerDim>,
    <(D, InnerDim) as DimConcat<D, InnerDim>>::Output: TensorDim,
{
    type Out = Tensor<T, ConcatDims<D, InnerDim>>;
    fn collapse(self) -> Self::Out {
        Tensor {
            inner: self.inner,
            phantom: PhantomData,
        }
    }
}

impl<T, D: TensorDim> Clone for Tensor<T, D, Op> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            phantom: self.phantom,
        }
    }
}

impl<T, D: TensorDim> Tensor<T, D, Op> {
    pub(crate) fn from_op(inner: Noxpr) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn sqrt(&self) -> Self {
        Self::from_op(self.inner.clone().sqrt())
    }

    pub fn log(&self) -> Self {
        Self::from_op(self.inner.clone().log())
    }

    pub fn sin(&self) -> Self {
        Self::from_op(self.inner.clone().sin())
    }

    pub fn cos(&self) -> Self {
        Self::from_op(self.inner.clone().cos())
    }
}

impl<T: Field, D: TensorDim + XlaDim> Tensor<T, D, Op> {
    pub fn zeros() -> Self {
        T::zero().broadcast()
    }
}

impl<T, D: TensorDim> IntoOp for Tensor<T, D, Op> {
    fn into_op(self) -> Noxpr {
        self.inner
    }
}

/// Represents a dimensionality of a tensor. This trait is a marker for types that can specify tensor dimensions.
pub trait TensorDim {}
/// Represents non-scalar dimensions, i.e., dimensions other than `()`.
pub trait NonScalarDim {}
/// Represents constant dimensions, specified at compile-time.
pub trait ConstDim {}

pub type ScalarDim = ();
impl TensorDim for ScalarDim {}
impl TensorDim for nalgebra::Dyn {}
impl NonScalarDim for nalgebra::Dyn {}
impl<const N: usize> TensorDim for nalgebra::Const<N> {}
impl<const N: usize> NonScalarDim for nalgebra::Const<N> {}

impl<T: TensorDim> DimDiv<T, T> for ShapeConstraint {}

pub trait XlaDim {
    fn shape() -> SmallVec<[i64; 4]>;
}

impl ConstDim for ScalarDim {}

impl XlaDim for ScalarDim {
    fn shape() -> SmallVec<[i64; 4]> {
        smallvec![]
    }
}

impl<const N: usize> ConstDim for Const<N> {}

impl<const N: usize> XlaDim for Const<N> {
    fn shape() -> SmallVec<[i64; 4]> {
        smallvec![N as i64]
    }
}

// This macro allows us to implement `TensorDim` for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
macro_rules! impl_tensor_dim {
    ($num:literal; $($ty:tt),+) => {
        impl<$($ty,)*> TensorDim for ($($ty,)*)
              where $($ty: TensorDim, )*
        {
        }

        impl<$($ty,)*> NonScalarDim for ($($ty,)*)
              where $($ty: NonScalarDim, )*
        {
        }

        impl<$($ty,)*> ConstDim for ($($ty,)*)
              where $($ty: ConstDim, )*
        {
        }

        impl<$($ty,)*> XlaDim for ($($ty,)*)
              where $($ty: XlaDim, )*
        {
            fn shape() -> SmallVec<[i64; 4]> {
                let mut shape = SmallVec::new();
                $(shape.extend_from_slice(&$ty::shape());)*
                shape
            }
        }
    }
}

impl_tensor_dim!(1; T1);
impl_tensor_dim!(2; T1, T2);
impl_tensor_dim!(3; T1, T2, T3);
impl_tensor_dim!(4; T1, T2, T3, T4);
impl_tensor_dim!(5; T1, T2, T3, T4, T5);
impl_tensor_dim!(6; T1, T2, T3, T4, T5, T6);
impl_tensor_dim!(7; T1, T2, T3, T4, T5, T6, T7);
impl_tensor_dim!(8; T1, T2, T3, T4, T5, T6, T7, T8);
impl_tensor_dim!(9; T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_tensor_dim!(10; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_tensor_dim!(11; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_tensor_dim!(12; T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

pub trait DimAdd<D1: TensorDim, D2: TensorDim> {}
pub trait DimSub<D1: TensorDim, D2: TensorDim> {}

pub trait DimMul<D1: TensorDim, D2: TensorDim> {}
pub trait DimDiv<D1: TensorDim, D2: TensorDim> {}

impl<D: TensorDim> DimAdd<D, D> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimAdd<ScalarDim, D> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimAdd<D, ScalarDim> for ShapeConstraint {}

impl<D: TensorDim> DimSub<D, D> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimSub<ScalarDim, D> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimSub<D, ScalarDim> for ShapeConstraint {}

impl<D: NonScalarDim + TensorDim> DimDiv<D, ScalarDim> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimMul<D, ScalarDim> for ShapeConstraint {}
impl<D: NonScalarDim + TensorDim> DimMul<ScalarDim, D> for ShapeConstraint {}

impl<D: TensorDim> DimMul<D, D> for ShapeConstraint {}

macro_rules! impl_op {
    ($op: tt, $op_fn:tt, $constraint: tt, $inner: tt, $($t_bound:tt),+) => {
        impl<T, D1: TensorDim, D2: TensorDim> $op<Tensor<T, D2>>
            for Tensor<T, D1>
        where
            $(T: $t_bound,)+
            ShapeConstraint: BroadcastDim<D1, D2>,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>>;

            fn $op_fn(self, rhs: Tensor<T, D2>) -> Self::Output {
                Tensor::from_op(self.inner.clone() $inner rhs.inner.clone())
            }
        }

        impl<'a, T, D1: TensorDim, D2: TensorDim> $op<&'a Tensor<T, D2>>
            for Tensor<T, D1>
        where
            $(T: $t_bound,)+
            ShapeConstraint: BroadcastDim<D1, D2>,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>>;

            fn $op_fn(self, rhs: &'a Tensor<T, D2>) -> Self::Output {
                Tensor::from_op(self.inner.clone() $inner rhs.inner.clone())
            }
        }

        impl<'a, T, D1: TensorDim, D2: TensorDim> $op<Tensor<T, D2>>
            for &'a Tensor<T, D1>
        where
            $(T: $t_bound,)+
            ShapeConstraint: BroadcastDim<D1, D2>,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>>;

            fn $op_fn(self, rhs: Tensor<T, D2>) -> Self::Output {
                Tensor::from_op(self.inner.clone() $inner rhs.inner.clone())
            }
        }


        impl<'a, 'b, T, D1: TensorDim, D2: TensorDim> $op<&'b Tensor<T, D2>>
            for &'a Tensor<T, D1>
        where
            $(T: $t_bound,)+
            ShapeConstraint: BroadcastDim<D1, D2>,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>>;

            fn $op_fn(self, rhs: &'b Tensor<T, D2>) -> Self::Output {
                Tensor::from_op((self.inner.clone() $inner rhs.inner.clone()))
            }
        }

    };
}

impl_op! {Add, add, DimAdd, +, Field}
impl_op! {Mul, mul, DimMul, *, Field}
impl_op! {Div, div, DimDiv, /, Field}
impl_op! {Sub, sub, DimSub, -, Field}

impl<T: Field, D: TensorDim> Neg for Tensor<T, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Tensor::from_op(self.inner.neg())
    }
}

impl<'a, T: NalgebraScalar + ClosedNeg, D: TensorDim> Neg for &'a Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn neg(self) -> Self::Output {
        Tensor::from_op(self.inner.clone().neg())
    }
}

impl<T, D: TensorDim + XlaDim> FixedSliceExt<T, D> for Tensor<T, D, Op> {
    fn fixed_slice<ND: TensorDim + XlaDim>(&self, offsets: &[usize]) -> Tensor<T, ND, Op> {
        let offsets: SmallVec<_> = offsets.iter().map(|o| *o as i64).collect();
        let new_offsets = offsets
            .iter()
            .zip(ND::shape())
            .map(|(a, b)| a + b)
            .collect();
        let strides = smallvec![1i64; offsets.len()];
        Tensor::from_op(self.inner.clone().slice(offsets, new_offsets, strides))
    }
}

/// Extension trait for tensors supporting fixed-size slicing operations.
pub trait FixedSliceExt<T, D: TensorDim> {
    /// Returns a tensor slice with dimensions specified by `ND`, starting at the given `offsets`.
    fn fixed_slice<ND: TensorDim + XlaDim>(&self, offsets: &[usize]) -> Tensor<T, ND, Op>;
}

impl<T: NalgebraScalar + ClosedMul + NativeType + ArrayElement, D1: TensorDim> Mul<T>
    for Tensor<T, D1>
where
    ShapeConstraint: DimMul<D1, ScalarDim>,
{
    type Output = Tensor<T, D1>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_op(self.inner.clone() * rhs.constant())
    }
}

impl<'a, T: NalgebraScalar + ClosedMul + NativeType + ArrayElement, D1: TensorDim> Mul<T>
    for &'a Tensor<T, D1>
where
    ShapeConstraint: DimMul<D1, ScalarDim>,
{
    type Output = Tensor<T, D1>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_op(self.inner.clone() * rhs.constant())
    }
}

macro_rules! impl_prim {
    ($ty:tt) => {
        impl<D: TensorDim> Mul<Tensor<$ty, D>> for $ty {
            type Output = Tensor<$ty, D>;

            fn mul(self, rhs: Tensor<$ty, D>) -> Self::Output {
                Tensor::from_op((self.constant() * rhs.inner))
            }
        }

        impl<'a, D: TensorDim> Mul<&'a Tensor<$ty, D>> for $ty {
            type Output = Tensor<$ty, D>;

            fn mul(self, rhs: &Tensor<$ty, D>) -> Self::Output {
                Tensor::from_op((self.constant() * rhs.inner.clone()))
            }
        }
    };
}

impl_prim!(f64);
impl_prim!(f32);
impl_prim!(u64);
impl_prim!(u32);
impl_prim!(i64);
impl_prim!(i32);

impl<T, D: TensorDim> AsBuffer for Tensor<T, D, Buffer> {
    fn as_buffer(&self) -> &xla::PjRtBuffer {
        &self.inner
    }
}

/// Trait for mapping dimensions in tensor operations.
/// Allows for transforming and replacing dimensions in tensor types.
pub trait MapDim<D> {
    type Item: TensorDim;
    type MappedDim: TensorDim;
    type ReplaceMappedDim<Dim: TensorDim>: TensorDim;
    const MAPPED_DIM: usize;
}

type MappedDim<T, D> = <T as MapDim<D>>::MappedDim;
type ReplaceMappedDim<T, D, R> = <T as MapDim<D>>::ReplaceMappedDim<R>;

pub struct Mapped;

impl<D: TensorDim> MapDim<D> for Mapped {
    type Item = ();
    type MappedDim = D;
    type ReplaceMappedDim<Dim: TensorDim> = Dim;

    const MAPPED_DIM: usize = 0;
}

/// Trait for default dimension mapping in tensor operations.
pub trait DefaultMap
where
    Self: Sized,
{
    /// The default dimension mapping for the implementing type.
    type DefaultMapDim: MapDim<Self>;
}

pub type DefaultMappedDim<D> = <<D as DefaultMap>::DefaultMapDim as MapDim<D>>::MappedDim;

macro_rules! impl_map {
    ($num:literal; $($ty:tt),+) => {

         impl<M, $($ty,)*> DefaultMap for (M, $($ty,)*)
         where
             M: TensorDim,
             $($ty: TensorDim, )*
         {
             type DefaultMapDim = (Mapped, $($ty,)* );
         }

        impl_map_inner!($num; TT1; $($ty),*);
        impl_map_inner!($num; TT1, TT2; $($ty),*);
        impl_map_inner!($num; TT1, TT2, TT3; $($ty),*);
        impl_map_inner!($num; TT1, TT2, TT3, TT4 ; $($ty),*);

        #[allow(unused_parens)]
        impl<M, $($ty,)*> MapDim<(M, $($ty,)*)> for (Mapped, $($ty,)*)
        where
            M: TensorDim,
            $($ty: TensorDim, )*
        {
            type Item = ($($ty),*);
            type MappedDim = M;
            type ReplaceMappedDim<Dim: TensorDim> = (Dim, $($ty),*);

            const MAPPED_DIM: usize = 0;
        }

        impl<$($ty,)* A> DimConcat<($($ty,)*), A> for (($($ty,)*), A)
        where
            $($ty: TensorDim, )*
            A: TensorDim + NonTupleDim
        {
            type Output = ($($ty),*, A);
        }

        impl<$($ty,)* A> DimConcat<A, ($($ty,)*)> for (A, ($($ty,)*))
        where
            $($ty: TensorDim, )*
            A: TensorDim + NonTupleDim
        {
            type Output = (A,$($ty),*);
        }
    };
}

macro_rules! impl_map_inner {
    ($num:literal; $($trail_ty:tt),* ; $($ty:tt),*) => {
        impl<$($ty,)* $($trail_ty,)* M> MapDim<
            ($($ty,)* Mapped, $($trail_ty,)*)
            > for ($($ty,)* M, $($trail_ty,)*)
              where $($ty: TensorDim, )*
              $($trail_ty: TensorDim, )*
            M: TensorDim
        {
            type Item = ($($ty),*, $($trail_ty),*);
            type MappedDim = M;
            type ReplaceMappedDim<Dim: TensorDim> = ($($ty),*, Dim, $($trail_ty),*);

            const MAPPED_DIM: usize = $num;
        }
    };
}

impl_map!(1; T1);
impl_map!(2; T1, T2);
impl_map!(3; T1, T2, T3);
impl_map!(4; T1, T2, T3, T4);
impl_map!(5; T1, T2, T3, T4, T5);
impl_map!(6; T1, T2, T3, T4, T5, T6);
impl_map!(7; T1, T2, T3, T4, T5, T6, T7);

impl<const N: usize> DefaultMap for Const<N> {
    type DefaultMapDim = Mapped;
}

/// Trait representing the concatenation of dimensions.
pub trait DimConcat<A, B> {
    type Output;
}

impl<const A: usize> DimConcat<Const<A>, ()> for (Const<A>, ()) {
    type Output = Const<A>;
}

impl<const A: usize> DimConcat<(), Const<A>> for ((), Const<A>) {
    type Output = Const<A>;
}

impl<A: NonScalarDim + NonTupleDim, B: NonScalarDim + NonTupleDim> DimConcat<A, B> for (A, B) {
    type Output = (A, B);
}

/// Represents types that are not tuples in dimension concatenation contexts.
pub trait NonTupleDim {}

impl NonTupleDim for ScalarDim {}
impl<const N: usize> NonTupleDim for Const<N> {}

// impl<A: NonScalarDim + TensorDim, B: NonScalarDim + TensorDim> DimConcat<A, B> for (A, B) {
//     type Output = (A, B);
// }

pub type ConcatDims<A, B> = <(A, B) as DimConcat<A, B>>::Output;

impl<T, D: TensorDim> Tensor<T, D> {
    pub fn reshape<ND>(self) -> Tensor<T, ND>
    where
        ND: TensorDim + XlaDim,
    {
        Tensor {
            inner: self.inner.reshape(ND::shape()),
            phantom: PhantomData,
        }
    }

    pub fn broadcast<ND: TensorDim>(self) -> Tensor<T, ND>
    where
        ND: TensorDim + XlaDim,
    {
        let inner = self.inner.broadcast(ND::shape());
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

type AddDim<A, B> = <A as nalgebra::DimAdd<B>>::Output;

#[allow(clippy::type_complexity)]
impl<T: TensorItem, D: TensorDim + DefaultMap> Tensor<T, D, crate::Op> {
    pub fn concat<OD: TensorDim + DefaultMap>(
        &self,
        other: Tensor<T, OD>,
    ) -> Tensor<
        T,
        ReplaceMappedDim<OD::DefaultMapDim, D, AddDim<DefaultMappedDim<D>, DefaultMappedDim<OD>>>,
    >
    where
        DefaultMappedDim<D>: nalgebra::DimAdd<DefaultMappedDim<OD>> + nalgebra::Dim,
        DefaultMappedDim<OD>: nalgebra::Dim,
        OD::DefaultMapDim: MapDim<D>,
        D::DefaultMapDim: MapDim<OD>,
        AddDim<DefaultMappedDim<D>, DefaultMappedDim<OD>>: TensorDim,
        <<OD as DefaultMap>::DefaultMapDim as MapDim<D>>::MappedDim: nalgebra::Dim,
    {
        let inner = Noxpr::concat_in_dim(
            vec![self.inner.clone(), other.inner.clone()],
            <D::DefaultMapDim>::MAPPED_DIM,
        );
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn concat_with_dim<OD: TensorDim, MDim: MapDim<D> + MapDim<OD>>(
        &self,
        other: Tensor<T, OD>,
    ) -> Tensor<T, ReplaceMappedDim<MDim, D, AddDim<MappedDim<MDim, D>, MappedDim<MDim, OD>>>>
    where
        MappedDim<MDim, D>: nalgebra::DimAdd<MappedDim<MDim, OD>> + nalgebra::Dim,
        MappedDim<MDim, OD>: nalgebra::Dim,
        AddDim<MappedDim<MDim, D>, MappedDim<MDim, OD>>: TensorDim,
    {
        let inner = Noxpr::concat_in_dim(
            vec![self.inner.clone(), other.inner.clone()],
            <MDim as MapDim<D>>::MAPPED_DIM,
        );
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

/// Trait for broadcasting dimensions in tensor operations, used to unify dimensions for element-wise operations.
pub trait BroadcastDim<D1, D2> {
    type Output: TensorDim;
}

pub type BroadcastedDim<D1, D2> = <ShapeConstraint as BroadcastDim<D1, D2>>::Output;

impl<D: TensorDim> BroadcastDim<D, D> for ShapeConstraint {
    type Output = D;
}

impl<D: TensorDim + NotConst1> BroadcastDim<D, Const<1>> for ShapeConstraint {
    type Output = D;
}

impl<D: TensorDim + NotConst1> BroadcastDim<Const<1>, D> for ShapeConstraint {
    type Output = D;
}

impl<D: TensorDim + NotConst1> BroadcastDim<D, nalgebra::Dyn> for ShapeConstraint {
    type Output = nalgebra::Dyn;
}

impl<D: TensorDim + NotConst1> BroadcastDim<nalgebra::Dyn, D> for ShapeConstraint {
    type Output = nalgebra::Dyn;
}

impl<D: TensorDim + NotConst1> BroadcastDim<D, ScalarDim> for ShapeConstraint {
    type Output = D;
}

impl<D: TensorDim + NotConst1> BroadcastDim<ScalarDim, D> for ShapeConstraint {
    type Output = D;
}

/// Marker trait for types not equivalent to `Const<1>`, used in broadcasting logic.
pub trait NotConst1 {}

seq_macro::seq!(N in 2..99 {
    impl NotConst1 for Const<N> {}
});

impl<T: TensorItem, D: TensorDim> Tensor<T, D> {
    pub fn index<I: TensorIndex<T, D>>(&self, index: I) -> I::Output {
        index.index(self.clone())
    }
}

/// Trait for indexing into tensors, allowing for the extraction of sub-tensors or elements based on indices.
pub trait TensorIndex<T, D: TensorDim> {
    type Output;

    /// Performs the indexing operation on a tensor, returning the result.
    fn index(self, tensor: Tensor<T, D>) -> Self::Output;
}

impl<T: TensorItem, D: TensorDim + DefaultMap, IT: ArrayElement, const N: usize> TensorIndex<T, D>
    for Vector<IT, N>
where
    ReplaceMappedDim<D::DefaultMapDim, D, Const<1>>: XlaDim,
{
    type Output = Tensor<T, ReplaceMappedDim<D::DefaultMapDim, D, Const<N>>>;

    fn index(self, tensor: Tensor<T, D>) -> Self::Output {
        let indices = self
            .inner
            .broadcast_in_dim(smallvec![N as i64, 1], smallvec![0]);
        let slice_shape = ReplaceMappedDim::<D::DefaultMapDim, D, Const<1>>::shape();

        let offset_dims = (1..slice_shape.len() as i64).collect();
        let inner = tensor.inner.gather(
            indices,
            offset_dims,
            smallvec![0],
            smallvec![0],
            slice_shape,
            1,
        );
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}
