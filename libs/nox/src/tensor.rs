//! Provides the core functionality for manipulating tensors.
use crate::local_backend::{ArrayBufUnit, ArrayDim};
use crate::{
    AsBuffer, Buffer, Dim, DimGet, Field, FromOp, GetDim, IntoOp, MatMul, Noxpr, NoxprScalarExt,
    Op, Repr, Scalar, Vector,
};
use core::mem::MaybeUninit;
use nalgebra::{constraint::ShapeConstraint, ClosedMul, Const, Dyn, Scalar as NalgebraScalar};
use simba::scalar::ClosedNeg;
use smallvec::{smallvec, SmallVec};
use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};
use xla::{ArrayElement, ElementType, NativeType};

/// Represents a tensor with a specific type `T`, dimensionality `D`, and underlying representation `P`.
#[repr(transparent)]
pub struct Tensor<T: TensorItem, D: Dim, R: Repr = Op> {
    pub(crate) inner: R::Inner<T::Elem, D>,
    pub(crate) phantom: PhantomData<(T, D)>,
}

impl<T: TensorItem, D: Dim, P: Repr> std::fmt::Debug for Tensor<T, D, P>
where
    P::Inner<T::Elem, D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("inner", &self.inner)
            .finish()
    }
}

/// Trait for items that can be contained in Tensors (i.e the `T`, in `Tensor<T>`)
///
/// This trait allows `Tensor` to be used like a higher-order container (like a special `Vec` or slice).
/// In most use cases you only use `Tensor` to hold basic primitives like `f64`, but you can also have a tensor of [`Quaternion`] or even of
/// another `Tensor`.
pub trait TensorItem {
    /// The type used when mapping across the `Tensor`.
    /// For example, if you have a `Tensor<f64>` you will get a `Scalar<f64>` when mapping over the tensor
    type Item: FromOp;

    /// A helper type that allows you to get a new Tensor with this `TensorItem`, and the specified dimension
    type Tensor<D>
    where
        D: Dim;

    /// The dimension of the underyling item. For example `f64` will be `ScalarDim`
    type Dim: Dim;

    /// The `ElemenetType` for the underyling element. This is always a primitive (f64, f32, etc).
    const ELEM: ElementType;

    /// The primitive element that will be stored in actual memory
    type Elem: Copy;
}

impl<T: NativeType + ArrayElement + Copy> TensorItem for T {
    type Item = Scalar<T>;
    type Tensor<D> = Tensor<T, D> where D: Dim;
    type Dim = ();

    const ELEM: ElementType = T::TY;

    type Elem = T;
}

impl<T: TensorItem, D: Dim> TensorItem for Tensor<T, D, Op> {
    type Item = T::Item; // NOTE: this bound might be wrong

    type Dim = D;
    type Tensor<TD: Dim> = Tensor<T, TD>;

    const ELEM: ElementType = T::ELEM;

    type Elem = T::Elem;
}

impl<T: TensorItem, D: Dim> FromOp for Tensor<T, D> {
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

impl<T: TensorItem, InnerDim: Dim, D: Dim + NonScalarDim> Collapse
    for Tensor<Tensor<T, InnerDim>, D, Op>
where
    (D, InnerDim): DimConcat<D, InnerDim>,
    <(D, InnerDim) as DimConcat<D, InnerDim>>::Output: Dim,
{
    type Out = Tensor<T, ConcatDims<D, InnerDim>>;
    fn collapse(self) -> Self::Out {
        Tensor {
            inner: self.inner,
            phantom: PhantomData,
        }
    }
}

impl<T: TensorItem, D: Dim> Clone for Tensor<T, D, Op> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            phantom: self.phantom,
        }
    }
}

impl<T: TensorItem, D: Dim> Tensor<T, D, Op> {
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

impl<T: Field, D: Dim> Tensor<T, D, Op> {
    pub fn zeros() -> Self {
        T::zero().broadcast()
    }

    pub fn ones() -> Self {
        T::one().broadcast()
    }
}

impl<T: TensorItem, D: Dim> IntoOp for Tensor<T, D, Op> {
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

pub trait XlaDim {
    fn shape() -> SmallVec<[i64; 4]>;
}

impl XlaDim for Dyn {
    fn shape() -> SmallVec<[i64; 4]> {
        todo!()
    }
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

macro_rules! impl_op {
    ($op: tt, $op_fn:tt, $inner: tt, $($t_bound:tt),+) => {
        impl<T, D1: Dim, D2: Dim, R: Repr> $op<Tensor<T, D2, R>>
            for Tensor<T, D1, R>
        where
            $(T: $t_bound,)+
            $(T::Elem: $t_bound,)+
            T: Copy + $op<Output = T>,
            T::Elem: $op<Output = T::Elem>,
            D1: Dim + ArrayDim,
            D2: Dim + ArrayDim,
            ShapeConstraint: BroadcastDim<D1, D2>,
            <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
            <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T::Elem>>:
                ArrayBufUnit<T::Elem, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T::Elem>>,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>, R>;

            fn $op_fn(self, rhs: Tensor<T, D2, R>) -> Self::Output {
                let inner = R::$op_fn::<T::Elem, D1, D2>(&self.inner, &rhs.inner);
                Tensor { inner, phantom: PhantomData }
            }
        }

        impl<'a, T, D1: Dim, D2: Dim, R: Repr> $op<&'a Tensor<T, D2, R>>
            for Tensor<T, D1, R>
        where
            $(T: $t_bound,)+
            $(T::Elem: $t_bound,)+
            T: Copy + $op<Output = T>,
            T::Elem: $op<Output = T::Elem>,
            D1: Dim + ArrayDim,
            D2: Dim + ArrayDim,
            ShapeConstraint: BroadcastDim<D1, D2>,
            <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
            <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T::Elem>>:
                ArrayBufUnit<T::Elem, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T::Elem>>,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>, R>;

            fn $op_fn(self, rhs: &'a Tensor<T, D2, R>) -> Self::Output {
                let inner = R::$op_fn::<T::Elem, D1, D2>(&self.inner, &rhs.inner);
                Tensor { inner, phantom: PhantomData }
            }
        }

    impl<'a, 'b, T, D1: Dim, D2: Dim, R: Repr> $op<&'a Tensor<T, D2, R>>
            for &'b Tensor<T, D1, R>
        where
            $(T: $t_bound,)+
            $(T::Elem: $t_bound,)+
            T: Copy + $op<Output = T>,
            T::Elem: $op<Output = T::Elem>,
            D1: Dim + ArrayDim,
            D2: Dim + ArrayDim,
            ShapeConstraint: BroadcastDim<D1, D2>,
            <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
            <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T::Elem>>:
                ArrayBufUnit<T::Elem, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T::Elem>>,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>, R>;

            fn $op_fn(self, rhs: &'a Tensor<T, D2, R>) -> Self::Output {
                let inner = R::$op_fn::<T::Elem, D1, D2>(&self.inner, &rhs.inner);
                Tensor { inner, phantom: PhantomData }
            }
        }

    impl<'a, T, D1: Dim, D2: Dim, R: Repr> $op<Tensor<T, D2, R>>
            for &'a Tensor<T, D1, R>
        where
            $(T: $t_bound,)+
            $(T::Elem: $t_bound,)+
            T: Copy + $op<Output = T>,
            T::Elem: $op<Output = T::Elem>,
            D1: Dim + ArrayDim,
            D2: Dim + ArrayDim,
            ShapeConstraint: BroadcastDim<D1, D2>,
            <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
            <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T::Elem>>:
                ArrayBufUnit<T::Elem, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T::Elem>>,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>, R>;

            fn $op_fn(self, rhs: Tensor<T, D2, R>) -> Self::Output {
                let inner = R::$op_fn::<T::Elem, D1, D2>(&self.inner, &rhs.inner);
                Tensor { inner, phantom: PhantomData }
            }
        }
    };
}

impl_op! {Add, add,  +, Field}
impl_op! {Mul, mul, *, Field}
impl_op! {Div, div, /, Field}
impl_op! {Sub, sub, -, Field}

impl<T: Field, D: Dim> Neg for Tensor<T, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Tensor::from_op(self.inner.neg())
    }
}

impl<'a, T: TensorItem + NalgebraScalar + ClosedNeg, D: Dim> Neg for &'a Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn neg(self) -> Self::Output {
        Tensor::from_op(self.inner.clone().neg())
    }
}

impl<T: TensorItem, D: Dim + XlaDim> FixedSliceExt<T, D> for Tensor<T, D, Op> {
    fn fixed_slice<ND: Dim + XlaDim>(&self, offsets: &[usize]) -> Tensor<T, ND, Op> {
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
pub trait FixedSliceExt<T: TensorItem, D: Dim> {
    /// Returns a tensor slice with dimensions specified by `ND`, starting at the given `offsets`.
    fn fixed_slice<ND: Dim>(&self, offsets: &[usize]) -> Tensor<T, ND, Op>;
}

impl<T: NalgebraScalar + ClosedMul + NativeType + ArrayElement, D1: Dim> Mul<T> for Tensor<T, D1> {
    type Output = Tensor<T, D1>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_op(self.inner.clone() * rhs.constant())
    }
}

impl<'a, T: NalgebraScalar + ClosedMul + NativeType + ArrayElement, D1: Dim> Mul<T>
    for &'a Tensor<T, D1>
{
    type Output = Tensor<T, D1>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_op(self.inner.clone() * rhs.constant())
    }
}

macro_rules! impl_prim {
    ($ty:tt) => {
        impl<D: Dim> Mul<Tensor<$ty, D>> for $ty {
            type Output = Tensor<$ty, D>;

            fn mul(self, rhs: Tensor<$ty, D>) -> Self::Output {
                Tensor::from_op((self.constant() * rhs.inner))
            }
        }

        impl<'a, D: Dim> Mul<&'a Tensor<$ty, D>> for $ty {
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

impl<T: TensorItem, D: Dim> AsBuffer for Tensor<T, D, Buffer> {
    fn as_buffer(&self) -> &xla::PjRtBuffer {
        &self.inner
    }
}

/// Trait for mapping dimensions in tensor operations.
/// Allows for transforming and replacing dimensions in tensor types.
pub trait MapDim<D> {
    type Item: Dim;
    type MappedDim: Dim;
    type ReplaceMappedDim<ReplaceDim: Dim>;
    const MAPPED_DIM: usize;
}

pub type MappedDim<T, D> = <T as MapDim<D>>::MappedDim;
pub type ReplaceMappedDim<T, D, R> = <T as MapDim<D>>::ReplaceMappedDim<R>;

pub struct Mapped;

impl<D: Dim> MapDim<D> for Mapped {
    type Item = ();
    type MappedDim = D;
    type ReplaceMappedDim<ReplaceDim: Dim> = ReplaceDim;

    const MAPPED_DIM: usize = 0;
}

/// Trait that defines the default mapped dimension for use with `.map`
pub trait DefaultMap
where
    Self: Sized,
{
    /// The default dimension to be mapped, usually the first dim.
    type DefaultMapDim: MapDim<Self>;
}

pub type DefaultMappedDim<D> = <<D as DefaultMap>::DefaultMapDim as MapDim<D>>::MappedDim;

impl DefaultMap for ScalarDim {
    type DefaultMapDim = Const<1>;
}

impl MapDim<ScalarDim> for Const<1> {
    type Item = ();
    type MappedDim = Const<1>;
    type ReplaceMappedDim<ReplaceDim: Dim> = ReplaceDim;

    const MAPPED_DIM: usize = 0;
}

macro_rules! impl_map {
    ($num:literal; $($ty:tt),+) => {

        #[allow(unused_parens)]
         impl<M, $($ty,)*> DefaultMap for (M, $($ty,)*)
         where
             M: Dim,
            $($ty: Dim, )*
            ($($ty),*): Dim,
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
            M: Dim,
            $($ty: Dim, )*
            ($($ty),*): Dim,
        {
            type Item = ($($ty),*);
            type MappedDim = M;
            type ReplaceMappedDim<ReplaceDim: Dim> = (ReplaceDim, $($ty),*);

            const MAPPED_DIM: usize = 0;
        }

        impl<$($ty,)* A> DimConcat<($($ty,)*), A> for (($($ty,)*), A)
        where
            $($ty: Dim, )*
            A: Dim + NonTupleDim
        {
            type Output = ($($ty),*, A);
        }

        impl<$($ty,)* A> DimConcat<A, ($($ty,)*)> for (A, ($($ty,)*))
        where
            $($ty: Dim, )*
            A: Dim + NonTupleDim
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
              where $($ty: Dim, )*
              $($trail_ty: Dim, )*
            M: Dim,
            (T1, TT1): Dim,
            ($($ty),*, $($trail_ty),*): Dim,
        {
            type Item = ($($ty),*, $($trail_ty),*);
            type MappedDim = M;
            type ReplaceMappedDim<ReplaceDim: Dim> = ($($ty),*, ReplaceDim, $($trail_ty),*);

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

/// Trait implemented by all non-tuple dimensions, used by the [`DimConcat`] trait
pub trait NonTupleDim {}

impl NonTupleDim for ScalarDim {}
impl<const N: usize> NonTupleDim for Const<N> {}

// impl<A: NonScalarDim + TensorDim, B: NonScalarDim + TensorDim> DimConcat<A, B> for (A, B) {
//     type Output = (A, B);
// }

pub type ConcatDims<A, B> = <(A, B) as DimConcat<A, B>>::Output;

impl<T: TensorItem, D: Dim> Tensor<T, D> {
    pub fn reshape<ND>(self) -> Tensor<T, ND>
    where
        ND: Dim + XlaDim,
    {
        Tensor {
            inner: self.inner.reshape(ND::shape()),
            phantom: PhantomData,
        }
    }

    pub fn broadcast<ND: Dim>(self) -> Tensor<T, ND>
    where
        ND: Dim + XlaDim,
    {
        let inner = self.inner.broadcast(ND::shape());
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

pub type AddDim<A, B> = <A as nalgebra::DimAdd<B>>::Output;

#[allow(clippy::type_complexity)]
impl<T: TensorItem, D: Dim + DefaultMap> Tensor<T, D, crate::Op> {
    pub fn concat<OD: Dim + DefaultMap>(
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
        AddDim<DefaultMappedDim<D>, DefaultMappedDim<OD>>: Dim,
        <<OD as DefaultMap>::DefaultMapDim as MapDim<D>>::MappedDim: nalgebra::Dim,
        ReplaceMappedDim<OD::DefaultMapDim, D, AddDim<DefaultMappedDim<D>, DefaultMappedDim<OD>>>:
            Dim,
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

    pub fn concat_with_dim<OD: Dim, MDim: MapDim<D> + MapDim<OD>>(
        &self,
        other: Tensor<T, OD>,
    ) -> Tensor<T, ReplaceMappedDim<MDim, D, AddDim<MappedDim<MDim, D>, MappedDim<MDim, OD>>>>
    where
        MappedDim<MDim, D>: nalgebra::DimAdd<MappedDim<MDim, OD>> + nalgebra::Dim,
        MappedDim<MDim, OD>: nalgebra::Dim,
        AddDim<MappedDim<MDim, D>, MappedDim<MDim, OD>>: Dim,
        ReplaceMappedDim<MDim, D, AddDim<MappedDim<MDim, D>, MappedDim<MDim, OD>>>: Dim,
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
    type Output: Dim;
}

pub type BroadcastedDim<D1, D2> = <ShapeConstraint as BroadcastDim<D1, D2>>::Output;

impl<const A: usize> BroadcastDim<Const<A>, Const<A>> for ShapeConstraint {
    type Output = Const<A>;
}

impl BroadcastDim<ScalarDim, ScalarDim> for ShapeConstraint {
    type Output = ScalarDim;
}

impl BroadcastDim<nalgebra::Dyn, nalgebra::Dyn> for ShapeConstraint {
    type Output = nalgebra::Dyn;
}

impl<D: Dim + NotConst1> BroadcastDim<D, Const<1>> for ShapeConstraint {
    type Output = D;
}

impl<D: Dim + NotConst1> BroadcastDim<Const<1>, D> for ShapeConstraint {
    type Output = D;
}

pub trait NotDyn {}
impl<const N: usize> NotDyn for Const<N> {}
impl NotDyn for ScalarDim {}

impl<D: Dim + NotDyn + NonScalarDim> BroadcastDim<D, nalgebra::Dyn> for ShapeConstraint {
    type Output = nalgebra::Dyn;
}

impl<D: Dim + NotDyn + NonScalarDim> BroadcastDim<nalgebra::Dyn, D> for ShapeConstraint {
    type Output = nalgebra::Dyn;
}

impl<D: Dim + NonScalarDim> BroadcastDim<D, ScalarDim> for ShapeConstraint {
    type Output = D;
}

impl<D: Dim + NonScalarDim> BroadcastDim<ScalarDim, D> for ShapeConstraint {
    type Output = D;
}

impl<const A1: usize, const A2: usize, const B1: usize, const B2: usize>
    BroadcastDim<(Const<A1>, Const<A2>), (Const<B1>, Const<B2>)> for ShapeConstraint
where
    ShapeConstraint: BroadcastDim<Const<A1>, Const<B1>>,
    ShapeConstraint: BroadcastDim<Const<A2>, Const<B2>>,
    (
        BroadcastedDim<Const<A1>, Const<B1>>,
        BroadcastedDim<Const<A2>, Const<B2>>,
    ): Dim,
{
    type Output = (
        BroadcastedDim<Const<A1>, Const<B1>>,
        BroadcastedDim<Const<A2>, Const<B2>>,
    );
}

impl<
        const A1: usize,
        const A2: usize,
        const A3: usize,
        const B1: usize,
        const B2: usize,
        const B3: usize,
    > BroadcastDim<(Const<A1>, Const<A2>, Const<A3>), (Const<B1>, Const<B2>, Const<B3>)>
    for ShapeConstraint
where
    ShapeConstraint: BroadcastDim<Const<A1>, Const<B1>>,
    ShapeConstraint: BroadcastDim<Const<A2>, Const<B2>>,
    ShapeConstraint: BroadcastDim<Const<A3>, Const<B3>>,
    (
        BroadcastedDim<Const<A1>, Const<B1>>,
        BroadcastedDim<Const<A2>, Const<B2>>,
        BroadcastedDim<Const<A3>, Const<B3>>,
    ): Dim,
{
    type Output = (
        BroadcastedDim<Const<A1>, Const<B1>>,
        BroadcastedDim<Const<A2>, Const<B2>>,
        BroadcastedDim<Const<A3>, Const<B3>>,
    );
}

impl<const A2: usize, const B1: usize, const B2: usize>
    BroadcastDim<(Dyn, Const<A2>), (Const<B1>, Const<B2>)> for ShapeConstraint
where
    ShapeConstraint: BroadcastDim<Const<A2>, Const<B2>>,
    (Dyn, BroadcastedDim<Const<A2>, Const<B2>>): Dim,
{
    type Output = (Dyn, BroadcastedDim<Const<A2>, Const<B2>>);
}

pub trait DotDim<D1, D2> {
    type Output: Dim;
}

impl<const N: usize> DotDim<Const<N>, Const<N>> for ShapeConstraint {
    type Output = ScalarDim;
}

impl<D1: TensorDim + NonScalarDim, D2: TensorDim + NonScalarDim, D3: TensorDim + NonScalarDim>
    DotDim<(D1, D2), (D2, D3)> for ShapeConstraint
where
    (D1, D3): Dim,
{
    type Output = (D1, D3);
}

impl<D2: TensorDim + NonScalarDim, D3: TensorDim + NonScalarDim> DotDim<(D2, D3), D3>
    for ShapeConstraint
where
    D2: Dim,
{
    type Output = D2;
}

impl DotDim<Dyn, Dyn> for ShapeConstraint {
    type Output = Dyn;
}
pub type DottedDim<D1, D2> = <ShapeConstraint as DotDim<D1, D2>>::Output;

impl<T: Field, D1: Dim, R: Repr> Tensor<T, D1, R> {
    pub fn dot<D2>(
        &self,
        right: &Tensor<T, D2, R>,
    ) -> Tensor<T, <ShapeConstraint as DotDim<D1, D2>>::Output, R>
    where
        T: MatMul + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<D1, D2>,
        <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
        <DottedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <DottedDim<D1, D2> as ArrayDim>::Buf<T>>,
    {
        Tensor {
            inner: R::dot::<T, D1, D2>(&self.inner, &right.inner),
            phantom: PhantomData,
        }
    }

    pub fn get(&self, index: usize) -> Tensor<T, GetDim<D1>, R>
    where
        ShapeConstraint: DimGet<D1>,
        <GetDim<D1> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <GetDim<D1> as ArrayDim>::Buf<T>>,
    {
        let inner = R::get::<T, D1>(&self.inner, index);
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

/// Marker trait for types not equivalent to `Const<1>`, used in broadcasting logic.
pub trait NotConst1 {}

seq_macro::seq!(N in 2..99 {
    impl NotConst1 for Const<N> {}
});

impl<T: TensorItem, D: Dim> Tensor<T, D> {
    pub fn index<I: TensorIndex<T, D>>(&self, index: I) -> I::Output {
        index.index(self.clone())
    }
}

/// Trait for indexing into tensors, allowing for the extraction of sub-tensors or elements based on indices.
pub trait TensorIndex<T: TensorItem, D: Dim> {
    type Output;

    /// Performs the indexing operation on a tensor, returning the result.
    fn index(self, tensor: Tensor<T, D>) -> Self::Output;
}

impl<T: TensorItem, D: Dim + DefaultMap, IT: TensorItem, const N: usize> TensorIndex<T, D>
    for Vector<IT, N>
where
    ReplaceMappedDim<D::DefaultMapDim, D, Const<1>>: Dim,
    ReplaceMappedDim<D::DefaultMapDim, D, Const<N>>: Dim,
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
