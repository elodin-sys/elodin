//! Provides the core functionality for manipulating tensors.
use crate::array::prelude::*;
use crate::{
    Const, DefaultRepr, Dim, Dyn, Elem, Error, Field, OwnedRepr, RealField, Repr, ReprMonad,
    Scalar, ShapeConstraint,
};
use approx::{AbsDiffEq, RelativeEq};
use std::iter::Sum;
use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

/// Represents a tensor with a specific type `T`, dimensionality `D`, and underlying representation `P`.
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tensor<T: TensorItem, D: Dim, R: Repr = DefaultRepr> {
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            deserialize = "R::Inner<T::Elem, D>: serde::Deserialize<'de>",
            serialize = "R::Inner<T::Elem, D>: serde::Serialize"
        ))
    )]
    pub(crate) inner: R::Inner<T::Elem, D>,
    pub(crate) phantom: PhantomData<(T, D)>,
}

impl<T: Field, D: Dim, R: OwnedRepr> Copy for Tensor<T, D, R> where R::Inner<T::Elem, D>: Copy {}

impl<T: TensorItem, D: Dim, P: OwnedRepr> std::fmt::Debug for Tensor<T, D, P>
where
    P::Inner<T::Elem, D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Tensor").field(&self.inner).finish()
    }
}

impl<T: TensorItem, D: Dim, R: OwnedRepr> PartialEq for Tensor<T, D, R>
where
    R::Inner<T::Elem, D>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Trait for items that can be contained in Tensors (i.e the `T`, in `Tensor<T>`)
///
/// This trait allows `Tensor` to be used like a higher-order container (like a special `Vec` or slice).
/// In most use cases you only use `Tensor` to hold basic primitives like `f64`,
/// but you can also have a tensor of [`crate::Quaternion`] or even of another `Tensor`.
pub trait TensorItem {
    /// The type used when mapping across the `Tensor`.
    /// For example, if you have a `Tensor<f64>` you will get a `Scalar<f64>` when mapping over the tensor
    type Item;

    /// A helper type that allows you to get a new Tensor with this `TensorItem`, and the specified dimension
    type Tensor<D>
    where
        D: Dim;

    /// The dimension of the underlying item. For example `f64` will be `ScalarDim`
    type Dim: Dim;

    /// The primitive element that will be stored in actual memory
    type Elem: Elem;
}

impl<T: Elem> TensorItem for T {
    type Item = Scalar<T>;
    type Tensor<D> = Tensor<T, D> where D: Dim;
    type Dim = ();

    type Elem = T;
}

impl<T: Field, D: Dim, R: OwnedRepr> Clone for Tensor<T, D, R> {
    fn clone(&self) -> Self {
        Self {
            inner: R::noop(&self.inner),
            phantom: self.phantom,
        }
    }
}

impl<T: RealField, D: Dim, R: OwnedRepr> Tensor<T, D, R> {
    pub fn sqrt(&self) -> Self {
        Self::from_inner(R::sqrt(&self.inner))
    }

    pub fn sin(&self) -> Self {
        Self::from_inner(R::sin(&self.inner))
    }

    pub fn cos(&self) -> Self {
        Self::from_inner(R::cos(&self.inner))
    }

    pub fn abs(&self) -> Self {
        Self::from_inner(R::abs(&self.inner))
    }

    pub fn atan2(&self, other: &Self) -> Self {
        Self::from_inner(R::atan2(&self.inner, &other.inner))
    }

    pub fn try_lu_inverse(&self) -> Result<Self, Error>
    where
        D: SquareDim,
    {
        R::try_lu_inverse(&self.inner).map(Self::from_inner)
    }
}

impl<T: TensorItem, D: Dim, R: OwnedRepr> Tensor<T, D, R> {
    pub fn from_inner(inner: R::Inner<T::Elem, D>) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: Field, D: Dim + NonScalarDim + ConstDim, R: OwnedRepr> Tensor<T, D, R> {
    pub fn zeros() -> Self {
        T::zero().broadcast()
    }

    pub fn ones() -> Self {
        T::one().broadcast()
    }
}

impl<T: TensorItem, D: Dim, R: OwnedRepr> Tensor<T, D, R> {
    pub fn inner(&self) -> &R::Inner<T::Elem, D> {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut R::Inner<T::Elem, D> {
        &mut self.inner
    }
}

impl<T: TensorItem + Elem, D: Dim> Tensor<T, D, ArrayRepr> {
    pub fn from_buf(buf: D::Buf<T::Elem>) -> Self {
        Self {
            inner: Array { buf },
            phantom: PhantomData,
        }
    }

    pub fn into_buf(self) -> D::Buf<T::Elem> {
        self.inner.buf
    }
}

/// Represents a dimensionality of a tensor. This trait is a marker for types that can specify tensor dimensions.
pub trait TensorDim {
    fn name() -> Self;
}
/// Represents non-scalar dimensions, i.e., dimensions other than `()`.
pub trait NonScalarDim {}
/// Represents constant dimensions, specified at compile-time.
pub trait ConstDim {
    const DIM: &'static [usize];

    type DimArr: AsRef<[usize]> + AsMut<[usize]> + Clone;
    fn const_dim() -> Self::DimArr;

    #[cfg(feature = "noxpr")]
    fn xla_shape() -> smallvec::SmallVec<[i64; 4]> {
        Self::DIM.iter().map(|&x| x as i64).collect()
    }
}

/// Represents a scalar dimension, which is essentially dimensionless.
pub type ScalarDim = ();
impl TensorDim for ScalarDim {
    fn name() -> Self {}
}
impl TensorDim for Dyn {
    fn name() -> Self {
        Self
    }
}
impl NonScalarDim for Dyn {}
impl<const N: usize> TensorDim for Const<N> {
    fn name() -> Self {
        Self
    }
}
impl<const N: usize> NonScalarDim for Const<N> {}

impl ConstDim for ScalarDim {
    const DIM: &'static [usize] = &[];
    type DimArr = [usize; 0];
    fn const_dim() -> Self::DimArr {
        []
    }
}

impl<const N: usize> ConstDim for Const<N> {
    const DIM: &'static [usize] = &[N];

    type DimArr = [usize; 1];

    fn const_dim() -> Self::DimArr {
        [N]
    }
}

// This macro allows us to implement `TensorDim` for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
macro_rules! impl_tensor_dim {
    ($num:literal; $($ty:tt),+) => {
        impl<$($ty,)*> TensorDim for ($($ty,)*)
              where $($ty: TensorDim, )*
        {
            fn name() -> Self {
                ($($ty::name(),)*)
            }
        }

        impl<$($ty,)*> NonScalarDim for ($($ty,)*)
              where $($ty: NonScalarDim, )*
        {
        }

        impl<$(const $ty: usize,)*> ConstDim for ($(Const<$ty>,)*)
        {
            const DIM: &'static [usize] = &[$($ty,)*];

            type DimArr = [usize; $num];
            fn const_dim() -> Self::DimArr {
                [$($ty,)*]
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
        impl<T, D1: Dim, D2: Dim, R: OwnedRepr> $op<Tensor<T, D2, R>>
            for Tensor<T, D1, R>
        where
            $(T: $t_bound,)+
            $(T::Elem: $t_bound,)+
            T: Elem + $op<Output = T>,
            T::Elem: $op<Output = T::Elem>,
            D1: Dim,
            D2: Dim,
            ShapeConstraint: BroadcastDim<D1, D2>,
            <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>, R>;

            fn $op_fn(self, rhs: Tensor<T, D2, R>) -> Self::Output {
                let inner = R::$op_fn::<T::Elem, D1, D2>(&self.inner, &rhs.inner);
                Tensor { inner, phantom: PhantomData }
            }
        }

        impl<'a, T, D1: Dim, D2: Dim, R: OwnedRepr> $op<&'a Tensor<T, D2, R>>
            for Tensor<T, D1, R>
        where
            $(T: $t_bound,)+
            $(T::Elem: $t_bound,)+
            T: Elem + $op<Output = T>,
            T::Elem: $op<Output = T::Elem>,
            D1: Dim + ArrayDim,
            D2: Dim + ArrayDim,
            ShapeConstraint: BroadcastDim<D1, D2>,
            <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>, R>;

            fn $op_fn(self, rhs: &'a Tensor<T, D2, R>) -> Self::Output {
                let inner = R::$op_fn::<T::Elem, D1, D2>(&self.inner, &rhs.inner);
                Tensor { inner, phantom: PhantomData }
            }
        }

    impl<'a, 'b, T, D1: Dim, D2: Dim, R: OwnedRepr> $op<&'a Tensor<T, D2, R>>
            for &'b Tensor<T, D1, R>
        where
            $(T: $t_bound,)+
            $(T::Elem: $t_bound,)+
            T: Elem + $op<Output = T>,
            T::Elem: $op<Output = T::Elem>,
            D1: Dim + ArrayDim,
            D2: Dim + ArrayDim,
            ShapeConstraint: BroadcastDim<D1, D2>,
            <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
        {
            type Output = Tensor<T, BroadcastedDim<D1, D2>, R>;

            fn $op_fn(self, rhs: &'a Tensor<T, D2, R>) -> Self::Output {
                let inner = R::$op_fn::<T::Elem, D1, D2>(&self.inner, &rhs.inner);
                Tensor { inner, phantom: PhantomData }
            }
        }

    impl<'a, T, D1: Dim, D2: Dim, R: OwnedRepr> $op<Tensor<T, D2, R>>
            for &'a Tensor<T, D1, R>
        where
            $(T: $t_bound,)+
            $(T::Elem: $t_bound,)+
            T: Elem + $op<Output = T>,
            T::Elem: $op<Output = T::Elem>,
            D1: Dim + ArrayDim,
            D2: Dim + ArrayDim,
            ShapeConstraint: BroadcastDim<D1, D2>,
            <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
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

impl<T: Field + Neg<Output = T>, D: Dim, R: OwnedRepr> Neg for Tensor<T, D, R> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Tensor {
            inner: R::neg(&self.inner),
            phantom: PhantomData,
        }
    }
}

impl<T: Field + Neg<Output = T>, D: Dim, R: OwnedRepr> Neg for &'_ Tensor<T, D, R> {
    type Output = Tensor<T, D, R>;

    fn neg(self) -> Self::Output {
        Tensor {
            inner: R::neg(&self.inner),
            phantom: PhantomData,
        }
    }
}

impl<T: TensorItem + Field, D: Dim, R: OwnedRepr> Tensor<T, D, R> {
    pub fn fixed_slice<D2: Dim + ConstDim>(&self, offsets: &[usize]) -> Tensor<T, D2, R> {
        Tensor::from_inner(R::copy_fixed_slice(&self.inner, offsets))
    }
}

impl<T: Field, D1: Dim, R: OwnedRepr> Mul<T> for Tensor<T, D1, R>
where
    ShapeConstraint: BroadcastDim<(), D1>,
{
    type Output = Tensor<T, BroadcastedDim<(), D1>, R>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor {
            inner: R::mul(&R::scalar_from_const(rhs), &self.inner),
            phantom: PhantomData,
        }
    }
}

impl<T: Field, D1: Dim, R: OwnedRepr> Mul<T> for &'_ Tensor<T, D1, R>
where
    ShapeConstraint: BroadcastDim<(), D1>,
{
    type Output = Tensor<T, BroadcastedDim<(), D1>, R>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor {
            inner: R::mul(&R::scalar_from_const(rhs), &self.inner),
            phantom: PhantomData,
        }
    }
}

macro_rules! impl_prim {
    ($ty:tt) => {
        impl<D: Dim, R: OwnedRepr> Mul<Tensor<$ty, D, R>> for $ty
        where
            ShapeConstraint: BroadcastDim<(), D>,
        {
            type Output = Tensor<$ty, BroadcastedDim<(), D>, R>;

            fn mul(self, rhs: Tensor<$ty, D, R>) -> Self::Output {
                let inner = R::mul(&R::scalar_from_const(self), &rhs.inner);
                Tensor {
                    inner,
                    phantom: PhantomData,
                }
            }
        }

        impl<'a, D: Dim, R: OwnedRepr> Mul<&'a Tensor<$ty, D, R>> for $ty
        where
            ShapeConstraint: BroadcastDim<(), D>,
        {
            type Output = Tensor<$ty, BroadcastedDim<(), D>, R>;

            fn mul(self, rhs: &Tensor<$ty, D, R>) -> Self::Output {
                let inner = R::mul(&R::scalar_from_const(self), &rhs.inner);
                Tensor {
                    inner,
                    phantom: PhantomData,
                }
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

/// Trait for mapping dimensions in tensor operations.
/// Allows for transforming and replacing dimensions in tensor types.
pub trait ReplaceDim<D> {
    type Item: Dim;
    type MappedDim: Dim;
    type ReplaceMappedDim<ReplaceDim: Dim>;
    const MAPPED_DIM: usize;
}

/// Alias for the mapped dimension type of `T` for dimension `D`.
pub type MappedDim<T, D> = <T as ReplaceDim<D>>::MappedDim;
/// Alias for the type replacing the mapped dimension `T` for dimension `D` with `R`.
pub type ReplaceMappedDim<T, D, R> = <T as ReplaceDim<D>>::ReplaceMappedDim<R>;

/// Represents a mapped dimension used for transforming tensor dimensions.
pub struct Mapped;

impl<D: Dim> ReplaceDim<D> for Mapped {
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
    type DefaultMapDim: ReplaceDim<Self>;
}

/// Alias for the default mapped dimension of `D`.
pub type DefaultMappedDim<D> = <<D as DefaultMap>::DefaultMapDim as ReplaceDim<D>>::MappedDim;

impl DefaultMap for ScalarDim {
    type DefaultMapDim = Const<1>;
}

impl ReplaceDim<ScalarDim> for Const<1> {
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
        impl<M, $($ty,)*> ReplaceDim<(M, $($ty,)*)> for (Mapped, $($ty,)*)
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
        impl<$($ty,)* $($trail_ty,)* M> ReplaceDim<
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
impl NonTupleDim for Dyn {}
impl<const N: usize> NonTupleDim for Const<N> {}

// impl<A: NonScalarDim + TensorDim, B: NonScalarDim + TensorDim> DimConcat<A, B> for (A, B) {
//     type Output = (A, B);
// }

/// Alias for the dimension resulting from concatenating dimensions `A` and `B`.
pub type ConcatDims<A, B> = <(A, B) as DimConcat<A, B>>::Output;

impl<T1: TensorItem + Field, D1: Dim, R: OwnedRepr> Tensor<T1, D1, R> {
    pub fn reshape<D2: Dim + ConstDim>(self) -> Tensor<T1, D2, R>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        let inner = R::reshape::<T1, D1, D2>(&self.inner, D2::const_dim());
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn broadcast<D2: Dim + ConstDim>(self) -> Tensor<T1, D2, R>
    where
        ShapeConstraint: BroadcastDim<D1, D2, Output = D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
    {
        let inner = R::broadcast::<D1, D2, T1>(&self.inner, D2::const_dim());
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn broadcast_with_shape<D2: Dim>(
        self,
        shape: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Tensor<T1, D2, R>
    where
        ShapeConstraint: BroadcastDim<D1, D2, Output = D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
    {
        let inner = R::broadcast::<D1, D2, T1>(&self.inner, shape);
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn rows_iter(&self) -> impl ExactSizeIterator<Item = Tensor<T1, RowDim<D1>, R>> + '_
    where
        ShapeConstraint: DimRow<D1>,
    {
        R::rows_iter(&self.inner).map(|inner| Tensor {
            inner,
            phantom: PhantomData,
        })
    }
}

/// Alias for the result of adding dimensions `A` and `B`.
pub type AddDim<A, B> = <A as crate::DimAdd<B>>::Output;

#[allow(clippy::type_complexity)]
impl<T1: Field, D1: Dim + DefaultMap, R: OwnedRepr> Tensor<T1, D1, R> {
    pub fn concat<D2: Dim + DefaultMap>(
        &self,
        other: Tensor<T1, D2, R>,
    ) -> Tensor<
        T1,
        ReplaceMappedDim<D2::DefaultMapDim, D1, AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>>,
        R,
    >
    where
        DefaultMappedDim<D1>: crate::DimAdd<DefaultMappedDim<D2>> + crate::Dim,
        DefaultMappedDim<D2>: crate::Dim,
        D2::DefaultMapDim: ReplaceDim<D1>,
        D1::DefaultMapDim: ReplaceDim<D2>,
        D1: DefaultMap,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as ReplaceDim<D1>>::MappedDim: crate::Dim,
        ConcatDim<D1, D2>: Dim,
    {
        let inner = R::concat(&self.inner, &other.inner);
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T1: Field, D1: Dim, R: OwnedRepr> Tensor<T1, D1, R> {
    pub fn concat_in_dim<D2, I: IntoIterator<Item = Tensor<T1, D1, R>>>(
        args: I,
        dim: usize,
    ) -> Tensor<T1, D2, R>
    where
        I::IntoIter: ExactSizeIterator,
        D1: Dim,
        D2: Dim,
    {
        let inner = R::concat_many(args.into_iter().map(|t| t.inner), dim);
        Tensor::from_inner(inner)
    }

    pub fn stack<D2, I: IntoIterator<Item = Tensor<T1, D1, R>>>(
        args: I,
        dim: usize,
    ) -> Tensor<T1, D2, R>
    where
        I::IntoIter: ExactSizeIterator,
        D1: Dim,
        D2: Dim,
    {
        let inner = R::stack(args.into_iter().map(|t| t.inner), dim);
        Tensor::from_inner(inner)
    }
}

pub trait BaseBroadcastDim<D1, D2> {
    type Output: Dim;
}

/// Trait for broadcasting dimensions in tensor operations, used to unify dimensions for element-wise operations.
pub trait BroadcastDim<D1, D2> {
    type Output: Dim;
}

/// Alias for the dimension resulting from broadcasting dimensions `D1` and `D2`.
pub type BroadcastedDim<D1, D2> = <ShapeConstraint as BroadcastDim<D1, D2>>::Output;

/// Alias for the dimension resulting from broadcasting dimensions `D1` and `D2`.
pub type BaseBroadcastedDim<D1, D2> = <ShapeConstraint as BaseBroadcastDim<D1, D2>>::Output;

impl<const D: usize> BroadcastDim<Const<D>, Const<D>> for ShapeConstraint {
    type Output = Const<D>;
}

impl<const D: usize> BaseBroadcastDim<Const<D>, Const<D>> for ShapeConstraint {
    type Output = Const<D>;
}

impl BroadcastDim<ScalarDim, ScalarDim> for ShapeConstraint {
    type Output = ScalarDim;
}

impl BaseBroadcastDim<ScalarDim, ScalarDim> for ShapeConstraint {
    type Output = ScalarDim;
}

impl BaseBroadcastDim<Dyn, Dyn> for ShapeConstraint {
    type Output = Dyn;
}

impl BroadcastDim<Dyn, Dyn> for ShapeConstraint {
    type Output = Dyn;
}

impl<D: Dim + NotConst1> BroadcastDim<D, Const<1>> for ShapeConstraint {
    type Output = D;
}

impl<D: Dim + NotConst1> BroadcastDim<Const<1>, D> for ShapeConstraint {
    type Output = D;
}

impl<D: Dim + NotConst1> BaseBroadcastDim<D, Const<1>> for ShapeConstraint {
    type Output = D;
}

impl<D: Dim + NotConst1> BaseBroadcastDim<Const<1>, D> for ShapeConstraint {
    type Output = D;
}

impl<D1: Dim + NonTupleDim, D2: Dim + NonTupleDim> BroadcastDim<D1, (D2, D1)> for ShapeConstraint
where
    (D2, D1): Dim,
{
    type Output = (D2, D1);
}

impl<D1: Dim + NonTupleDim, D2: Dim + NonTupleDim> BroadcastDim<(D1, D2), D2> for ShapeConstraint
where
    (D1, D2): Dim,
{
    type Output = (D1, D2);
}

/// Marker trait for dimension types that are not dynamic, typically used for compile-time fixed dimensions.
pub trait NotDyn {}
impl<const N: usize> NotDyn for Const<N> {}
impl NotDyn for ScalarDim {}

impl<D: Dim + NotDyn + NonScalarDim> BroadcastDim<D, Dyn> for ShapeConstraint {
    type Output = Dyn;
}

impl<D: Dim + NotDyn + NonScalarDim> BroadcastDim<Dyn, D> for ShapeConstraint {
    type Output = Dyn;
}

impl<D: Dim + NonScalarDim> BroadcastDim<D, ScalarDim> for ShapeConstraint {
    type Output = D;
}

impl<D: Dim + NonScalarDim> BroadcastDim<ScalarDim, D> for ShapeConstraint {
    type Output = D;
}

impl<A1: Dim, A2: Dim, B1: Dim, B2: Dim> BroadcastDim<(A1, A2), (B1, B2)> for ShapeConstraint
where
    ShapeConstraint: BaseBroadcastDim<A1, B1>,
    ShapeConstraint: BaseBroadcastDim<A2, B2>,
    (BaseBroadcastedDim<A1, B1>, BaseBroadcastedDim<A2, B2>): Dim,
{
    type Output = (BaseBroadcastedDim<A1, B1>, BaseBroadcastedDim<A2, B2>);
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

/// Trait for determining the resulting dimension type from a dot product between two tensors with specified dimensions.
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

/// Alias for the dimension resulting from the dot product of dimensions `D1` and `D2`.
pub type DottedDim<D1, D2> = <ShapeConstraint as DotDim<D1, D2>>::Output;

impl<T: RealField, D1: Dim, R: OwnedRepr> Tensor<T, D1, R> {
    pub fn dot<D2>(
        &self,
        right: &Tensor<T, D2, R>,
    ) -> Tensor<T, <ShapeConstraint as DotDim<D1, D2>>::Output, R>
    where
        T: Elem,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<D1, D2>,
        <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        Tensor {
            inner: R::dot::<T, D1, D2>(&self.inner, &right.inner),
            phantom: PhantomData,
        }
    }
}

impl<T: Field, D1: Dim, R: OwnedRepr> Tensor<T, D1, R> {
    pub fn get(&self, index: D1::Index) -> Tensor<T, (), R>
    where
        D1: DimGet,
    {
        let inner = R::get::<T, D1>(&self.inner, index);
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn from_scalars(parts: impl IntoIterator<Item = Tensor<T, (), R>>) -> Tensor<T, D1, R>
    where
        D1: ConstDim,
    {
        Self::from_scalars_with_shape(parts, D1::DIM)
    }

    pub fn from_scalars_with_shape(
        parts: impl IntoIterator<Item = Tensor<T, (), R>>,
        shape: &[usize],
    ) -> Tensor<T, D1, R> {
        let inner = R::from_scalars(parts.into_iter().map(|t| t.inner), shape);
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn eye() -> Tensor<T, D1, R>
    where
        D1: SquareDim + ConstDim,
    {
        let inner = R::eye();
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn transpose(&self) -> Tensor<T, TransposedDim<D1>, R>
    where
        ShapeConstraint: TransposeDim<D1>,
    {
        let inner = R::transpose::<T, D1>(&self.inner);
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn from_diag(diag: Tensor<T, D1::SideDim, R>) -> Tensor<T, D1, R>
    where
        D1: SquareDim,
    {
        let inner = R::from_diag(diag.inner);
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn map<T2: Elem, D2: Dim>(
        &self,
        func: impl Fn(Tensor<T, D1::ElemDim, R>) -> Tensor<T2, D2, R>,
    ) -> Tensor<T2, D1::MappedDim<D2>, R>
    where
        D1::MappedDim<D2>: Dim,
        T: Field,
        D1: Dim + MappableDim,
    {
        let inner = R::map::<T, D1, T2, D2>(&self.inner, |arg_inner| {
            func(Tensor {
                inner: arg_inner,
                phantom: PhantomData,
            })
            .inner
        });
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn outer<D2>(&self, other: &Tensor<T, D2, R>) -> Tensor<T, (D1, D2), R>
    where
        (D1, D2): Dim,
        (D2, D1): Dim,
        D1: NonTupleDim + NonScalarDim,
        D2: Dim + NonTupleDim + NonScalarDim,
        ShapeConstraint: BaseBroadcastDim<D1, D1, Output = D1>,
        ShapeConstraint: BaseBroadcastDim<D2, D2, Output = D2>,
    {
        let a_shape = R::shape(&self.inner);
        let b_shape = R::shape(&other.inner);
        let a_dim = a_shape.as_ref().first().copied().unwrap_or(1);
        let b_dim = b_shape.as_ref().first().copied().unwrap_or(1);

        let a_broadcast_shape = [b_dim, a_dim];
        let b_broadcast_shape = [a_dim, b_dim];
        let a: Tensor<T, (D2, D1), R> = self.clone().broadcast_with_shape(a_broadcast_shape);
        let b: Tensor<T, (D1, D2), R> = other.clone().broadcast_with_shape(b_broadcast_shape);
        a.transpose() * b
    }

    pub fn try_cholesky(&self) -> Result<Self, Error>
    where
        D1: SquareDim,
        T: RealField,
    {
        R::try_cholesky(&self.inner).map(Tensor::from_inner)
    }

    pub fn row(&self, index: usize) -> Tensor<T, RowDim<D1>, R>
    where
        ShapeConstraint: DimRow<D1>,
    {
        let inner = R::row(&self.inner, index);
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

impl<T: TensorItem, D: Dim> Default for Tensor<T, D, ArrayRepr>
where
    D::Buf<T::Elem>: Default,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
            phantom: PhantomData,
        }
    }
}

impl<T: Field, D: Dim> From<Array<T, D>> for Tensor<T, D, ArrayRepr> {
    fn from(inner: Array<T, D>) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: Field, const D1: usize> From<[T; D1]> for Tensor<T, Const<D1>, ArrayRepr> {
    fn from(buf: [T; D1]) -> Self {
        Self {
            inner: buf.into(),
            phantom: PhantomData,
        }
    }
}

impl<T: Field, const D1: usize, const D2: usize> From<[[T; D2]; D1]>
    for Tensor<T, (Const<D1>, Const<D2>), ArrayRepr>
{
    fn from(buf: [[T; D2]; D1]) -> Self {
        Self {
            inner: buf.into(),
            phantom: PhantomData,
        }
    }
}

impl<T: Field, const D1: usize, const D2: usize, const D3: usize> From<[[[T; D3]; D2]; D1]>
    for Tensor<T, (Const<D1>, Const<D2>, Const<D3>), ArrayRepr>
{
    fn from(buf: [[[T; D3]; D2]; D1]) -> Self {
        Self {
            inner: buf.into(),
            phantom: PhantomData,
        }
    }
}

#[macro_export]
macro_rules! tensor {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Tensor::<_, _, $crate::array::ArrayRepr>::from([$([$([$($x,)*],)*],)*])
    }};
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::Tensor::<_, _, $crate::array::ArrayRepr>::from([$([$($x,)*],)*])
    }};
    ($($x:expr),* $(,)*) => {{
        $crate::Tensor::<_, _, $crate::array::ArrayRepr>::from([$($x,)*])
    }};

    ($elem:expr; $n:expr) => {{
        $crate::Tensor::<_, _, $crate::array::ArrayRepr>::from([$elem; $n])
    }};
}

impl<T: Field, R: OwnedRepr> From<T> for Tensor<T, (), R> {
    fn from(val: T) -> Self {
        Scalar::from_inner(R::scalar_from_const(val))
    }
}

impl<T: TensorItem, D: Dim, R: OwnedRepr> AbsDiffEq for Tensor<T, D, R>
where
    T: Elem,
    R::Inner<T::Elem, D>: AbsDiffEq,
{
    type Epsilon = <R::Inner<T::Elem, D> as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> <R::Inner<T::Elem, D> as AbsDiffEq>::Epsilon {
        <R::Inner<T::Elem, D> as AbsDiffEq>::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.inner.abs_diff_eq(&other.inner, epsilon)
    }
}

impl<T: TensorItem, D: Dim, R: OwnedRepr> RelativeEq for Tensor<T, D, R>
where
    T: Elem,
    R::Inner<T::Elem, D>: RelativeEq + AbsDiffEq,
    <R::Inner<T::Elem, D> as AbsDiffEq>::Epsilon: Elem,
{
    fn default_max_relative() -> <R::Inner<T::Elem, D> as AbsDiffEq>::Epsilon {
        <R::Inner<T::Elem, D> as RelativeEq>::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: <R::Inner<T::Elem, D> as AbsDiffEq>::Epsilon,
        max_relative: <R::Inner<T::Elem, D> as AbsDiffEq>::Epsilon,
    ) -> bool {
        self.inner.relative_eq(&other.inner, epsilon, max_relative)
    }
}

impl<T: TensorItem, D: Dim, R: OwnedRepr> Sum for Tensor<T, D, R>
where
    T: Field,
    ShapeConstraint: BroadcastDim<D, D, Output = D>,
{
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if let Some(first) = iter.next() {
            iter.fold(first, |acc, x| acc + x)
        } else {
            panic!("Cannot compute `sum` of empty iterator.")
        }
    }
}

impl<T: TensorItem, R: OwnedRepr, D: Dim> ReprMonad<R> for Tensor<T, D, R> {
    type Elem = T::Elem;
    type Dim = D;

    type Map<N: OwnedRepr> = Tensor<T, D, N>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(<R as Repr>::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        Tensor::from_inner(func(self.inner))
    }

    fn inner(&self) -> &<R as Repr>::Inner<Self::Elem, Self::Dim> {
        &self.inner
    }

    fn into_inner(self) -> <R as Repr>::Inner<Self::Elem, Self::Dim> {
        self.inner
    }

    fn from_inner(inner: <R as Repr>::Inner<Self::Elem, Self::Dim>) -> Self {
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}
