//! Provides definitions and traits for handling operations on tensor dimensions and data types.
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::array::prelude::*;
use crate::{
    AddDim, BroadcastDim, BroadcastedDim, ConstDim, DefaultMap, DefaultMappedDim, Dim, DotDim,
    Elem, Error, Field, RealField, ReplaceDim, ShapeConstraint,
};

pub trait Repr {
    type Inner<T, D: Dim>: Clone
    where
        T: Elem;

    type Shape<D: Dim>: AsRef<[usize]> + Clone;
    fn shape<T1: Elem, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Shape<D1>;
}

/// Represents the interface for data representations in tensor operations.
pub trait OwnedRepr: Repr {
    /// Performs element-wise addition of two tensors, broadcasting as necessary.
    fn add<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Add<Output = T> + Elem,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim;

    /// Performs element-wise subtraction of two tensors, broadcasting as necessary.
    fn sub<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Sub<Output = T> + Elem,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim;

    /// Performs element-wise multiplication of two tensors, broadcasting as necessary.
    fn mul<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Mul<Output = T> + Elem,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim;

    /// Performs element-wise division of two tensors, broadcasting as necessary.
    fn div<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Div<Output = T> + Elem,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim;

    /// Computes the dot product of two tensors.
    fn dot<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, <ShapeConstraint as DotDim<D1, D2>>::Output>
    where
        T: RealField + Div<Output = T> + Elem,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<D1, D2>,
        <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim;

    /// Concatenates two arrays along the first dimension.
    fn concat<T1: Field, D1, D2: Dim + DefaultMap>(
        left: &Self::Inner<T1, D1>,
        right: &Self::Inner<T1, D2>,
    ) -> Self::Inner<T1, ConcatDim<D1, D2>>
    where
        DefaultMappedDim<D1>: crate::DimAdd<DefaultMappedDim<D2>> + crate::Dim,
        DefaultMappedDim<D2>: crate::Dim,
        D2::DefaultMapDim: ReplaceDim<D1>,
        D1::DefaultMapDim: ReplaceDim<D2>,
        D1: Dim + DefaultMap,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as ReplaceDim<D1>>::MappedDim: crate::Dim,
        ConcatDim<D1, D2>: Dim;

    /// Concatenates multiple tensors along the specified dimension
    fn concat_many<T1: Field, D1: Dim, D2: Dim, I: IntoIterator<Item = Self::Inner<T1, D1>>>(
        args: I,
        dim: usize,
    ) -> Self::Inner<T1, D2>
    where
        I::IntoIter: ExactSizeIterator;

    /// Stacks multiple tensors along specified dimension, creating a new dimension if necessary
    fn stack<T1: Field, D1: Dim, D2: Dim, I: IntoIterator<Item = Self::Inner<T1, D1>>>(
        args: I,
        dim: usize,
    ) -> Self::Inner<T1, D2>
    where
        I::IntoIter: ExactSizeIterator;

    /// Retrieves a specific tensor based on an index within a dimension.
    fn get<T1: Field, D1: Dim + DimGet>(
        arg: &Self::Inner<T1, D1>,
        index: D1::Index,
    ) -> Self::Inner<T1, ()>;

    fn broadcast<D1: Dim, D2: Dim, T1: Field>(
        arg: &Self::Inner<T1, D1>,
        dim: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim;

    fn scalar_from_const<T1: Field>(value: T1) -> Self::Inner<T1, ()>;

    fn neg<T1, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>
    where
        T1: Field + Neg<Output = T1>;

    fn sqrt<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>;

    fn atan2<T1: Field + RealField, D1: Dim>(
        left: &Self::Inner<T1, D1>,
        right: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, D1>;

    fn sin<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>;

    fn cos<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>;

    fn abs<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>;

    fn copy_fixed_slice<T1: Field, D1: Dim, D2: Dim + ConstDim>(
        arg: &Self::Inner<T1, D1>,
        offsets: &[usize],
    ) -> Self::Inner<T1, D2>;

    fn reshape<T1: Field, D1: Dim, D2: Dim>(
        arg: &Self::Inner<T1, D1>,
        dim: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Self::Inner<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>;

    fn try_lu_inverse<T1: RealField, D1: Dim + SquareDim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Result<Self::Inner<T1, D1>, Error>;

    fn from_scalars<T1: Field, D1: Dim>(
        iter: impl IntoIterator<Item = Self::Inner<T1, ()>>,
        shape: &[usize],
    ) -> Self::Inner<T1, D1>;

    fn transpose<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, TransposedDim<D1>>
    where
        ShapeConstraint: TransposeDim<D1>;

    fn eye<T1: Field, D1: Dim + SquareDim + ConstDim>() -> Self::Inner<T1, D1>;

    fn from_diag<T1: Field, D1: Dim + SquareDim>(
        diag: Self::Inner<T1, D1::SideDim>,
    ) -> Self::Inner<T1, D1>;

    fn noop<T1: Field, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>;

    fn try_cholesky<T1: RealField, D1: Dim + SquareDim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Result<Self::Inner<T1, D1>, Error>;

    fn row<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
        index: usize,
    ) -> Self::Inner<T1, RowDim<D1>>
    where
        ShapeConstraint: DimRow<D1>;

    fn rows_iter<T1: Elem, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
    ) -> impl ExactSizeIterator<Item = Self::Inner<T1, RowDim<D1>>> + '_
    where
        ShapeConstraint: DimRow<D1>;

    fn map<T1, D1, T2, D2>(
        arg: &Self::Inner<T1, D1>,
        func: impl Fn(Self::Inner<T1, D1::ElemDim>) -> Self::Inner<T2, D2>,
    ) -> Self::Inner<T2, D1::MappedDim<D2>>
    where
        D1::MappedDim<D2>: Dim,
        T1: Field,
        D1: Dim + MappableDim,
        T2: Elem + 'static,
        D2: Dim;
}

pub trait ReprMonad<R: OwnedRepr> {
    type Elem: Elem;
    type Dim: Dim;
    type Map<T: OwnedRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N>;

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim>;
    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim>;

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self;
}
