//! Provides definitions and traits for handling operations on tensor dimensions and data types.
use std::{
    iter,
    mem::MaybeUninit,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{ConstDim, RealField};
use nalgebra::{constraint::ShapeConstraint, Const};
use smallvec::SmallVec;
use xla::{ArrayElement, NativeType};

use crate::{
    array::ArrayDim, AddDim, ArrayTy, BroadcastDim, BroadcastedDim, ConcatDim, ConcatManyDim,
    DefaultMap, DefaultMappedDim, DimGet, DotDim, Field, GetDim, MapDim, MulDim, Noxpr, TensorDim,
    XlaDim,
};

/// Represents a compute operation.
pub struct Op;

/// Represents a literal value.
pub struct Literal;

/// Represents a memory buffer.
pub struct Buffer;

/// Defines a trait for dimensions supporting tensor operations, XLA compatibility, and array storage.
pub trait Dim: ArrayDim + TensorDim + XlaDim {}
impl<D: ArrayDim + TensorDim + XlaDim> Dim for D {}

/// Represents the interface for data representations in tensor operations.
pub trait Repr {
    type Inner<T, D: Dim>
    where
        T: Copy;

    /// Performs element-wise addition of two tensors, broadcasting as necessary.
    fn add<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Add<Output = T> + Copy,
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
        T: Sub<Output = T> + Copy,
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
        T: Mul<Output = T> + Copy,
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
        T: Div<Output = T> + Copy,
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
        T: Field + Div<Output = T> + Copy,
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
        DefaultMappedDim<D1>: nalgebra::DimAdd<DefaultMappedDim<D2>> + nalgebra::Dim,
        DefaultMappedDim<D2>: nalgebra::Dim,
        D2::DefaultMapDim: MapDim<D1>,
        D1::DefaultMapDim: MapDim<D2>,
        D1: Dim + DefaultMap,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatDim<D1, D2>: Dim;

    /// Concatenates multiple tensors along a new dimension.
    fn concat_many<T1: Field, D1, const N: usize>(
        args: [&Self::Inner<T1, D1>; N],
    ) -> Self::Inner<T1, ConcatManyDim<D1, N>>
    where
        DefaultMappedDim<D1>: nalgebra::DimMul<Const<N>> + nalgebra::Dim,
        D1::DefaultMapDim: MapDim<D1>,
        D1: Dim + DefaultMap,
        MulDim<DefaultMappedDim<D1>, Const<N>>: Dim,
        <<D1 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatManyDim<D1, N>: Dim;

    /// Retrieves a specific tensor based on an index within a dimension.
    fn get<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
        index: usize,
    ) -> Self::Inner<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>;

    fn broadcast<D1: Dim, D2: ArrayDim + TensorDim + XlaDim, T1: Field>(
        arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim + XlaDim;

    fn scalar_from_const<T1: Field + NativeType + ArrayElement>(value: T1) -> Self::Inner<T1, ()>;

    fn neg<T1, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>
    where
        T1: Field + Neg<Output = T1>;

    fn sqrt<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>;

    fn sin<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>;

    fn cos<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>;

    fn copy_fixed_slice<T1: Field, D1: Dim, D2: Dim + ConstDim>(
        arg: &Self::Inner<T1, D1>,
        offsets: &[usize],
    ) -> Self::Inner<T1, D2>;

    fn reshape<T1: Field, D1: Dim, D2: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>;
}

impl Repr for Literal {
    type Inner<T: Copy, D: Dim> = xla::Literal;

    fn add<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Add<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn sub<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Sub<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn mul<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Mul<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn div<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Div<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn dot<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, <ShapeConstraint as DotDim<D1, D2>>::Output>
    where
        T: Field + Div<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<D1, D2>,
        <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn concat_many<T1: Field, D1, const N: usize>(
        _args: [&Self::Inner<T1, D1>; N],
    ) -> Self::Inner<T1, ConcatManyDim<D1, N>>
    where
        DefaultMappedDim<D1>: nalgebra::DimMul<Const<N>> + nalgebra::Dim,
        D1::DefaultMapDim: MapDim<D1>,
        D1::DefaultMapDim: MapDim<D1>,
        D1: Dim + DefaultMap,
        MulDim<DefaultMappedDim<D1>, Const<N>>: Dim,
        <<D1 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatManyDim<D1, N>: Dim,
    {
        todo!()
    }

    fn get<T1: Field, D1: Dim>(
        _arg: &Self::Inner<T1, D1>,
        _index: usize,
    ) -> Self::Inner<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>,
    {
        todo!()
    }

    fn broadcast<D1: Dim, D2: ArrayDim + TensorDim + XlaDim, T1: Field>(
        _arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim + XlaDim,
    {
        todo!()
    }

    fn scalar_from_const<T1: Field>(_value: T1) -> Self::Inner<T1, ()> {
        todo!()
    }

    fn concat<T1: Field, D1, D2: Dim + DefaultMap>(
        _left: &Self::Inner<T1, D1>,
        _right: &Self::Inner<T1, D2>,
    ) -> Self::Inner<T1, ConcatDim<D1, D2>>
    where
        DefaultMappedDim<D1>: nalgebra::DimAdd<DefaultMappedDim<D2>> + nalgebra::Dim,
        DefaultMappedDim<D2>: nalgebra::Dim,
        D2::DefaultMapDim: MapDim<D1>,
        D1::DefaultMapDim: MapDim<D2>,
        D1: Dim + DefaultMap,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatDim<D1, D2>: Dim,
    {
        todo!()
    }

    fn neg<T1, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>
    where
        T1: Field + Neg<Output = T1>,
    {
        todo!()
    }

    fn sqrt<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        todo!()
    }

    fn sin<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        todo!()
    }

    fn cos<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        todo!()
    }

    fn copy_fixed_slice<T1: Field, D1: Dim, D2: Dim + ConstDim>(
        _arg: &Self::Inner<T1, D1>,
        _offsets: &[usize],
    ) -> Self::Inner<T1, D2> {
        todo!()
    }

    fn reshape<T1: Field, D1: Dim, D2: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        todo!()
    }
}

impl Repr for Buffer {
    type Inner<T: Copy, D: Dim + ArrayDim> = xla::PjRtBuffer;

    fn add<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Add<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn sub<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Sub<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn mul<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Mul<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn div<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Div<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn dot<T, D1, D2>(
        _left: &Self::Inner<T, D1>,
        _right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, <ShapeConstraint as DotDim<D1, D2>>::Output>
    where
        T: Field + Div<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<D1, D2>,
        <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        todo!()
    }

    fn concat_many<T1: Field, D1, const N: usize>(
        _args: [&Self::Inner<T1, D1>; N],
    ) -> Self::Inner<T1, ConcatManyDim<D1, N>>
    where
        DefaultMappedDim<D1>: nalgebra::DimMul<Const<N>> + nalgebra::Dim,
        D1::DefaultMapDim: MapDim<D1>,
        D1::DefaultMapDim: MapDim<D1>,
        D1: Dim + DefaultMap,
        MulDim<DefaultMappedDim<D1>, Const<N>>: Dim,
        <<D1 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatManyDim<D1, N>: Dim,
    {
        todo!()
    }

    fn get<T1: Field, D1: Dim>(
        _arg: &Self::Inner<T1, D1>,
        _index: usize,
    ) -> Self::Inner<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>,
    {
        todo!()
    }

    fn broadcast<D1: Dim, D2: ArrayDim + TensorDim + XlaDim, T1: Field>(
        _arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim + XlaDim,
    {
        todo!()
    }

    fn scalar_from_const<T1: Field>(_value: T1) -> Self::Inner<T1, ()> {
        todo!()
    }

    fn concat<T1: Field, D1, D2: Dim + DefaultMap>(
        _left: &Self::Inner<T1, D1>,
        _right: &Self::Inner<T1, D2>,
    ) -> Self::Inner<T1, ConcatDim<D1, D2>>
    where
        DefaultMappedDim<D1>: nalgebra::DimAdd<DefaultMappedDim<D2>> + nalgebra::Dim,
        DefaultMappedDim<D2>: nalgebra::Dim,
        D2::DefaultMapDim: MapDim<D1>,
        D1::DefaultMapDim: MapDim<D2>,
        D1: DefaultMap + Dim,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatDim<D1, D2>: Dim,
    {
        todo!()
    }

    fn neg<T1, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>
    where
        T1: Field + Neg<Output = T1>,
    {
        todo!()
    }

    fn sqrt<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        todo!()
    }

    fn sin<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        todo!()
    }

    fn cos<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        todo!()
    }

    fn copy_fixed_slice<T1: Field, D1: Dim, D2: Dim + ConstDim>(
        _arg: &Self::Inner<T1, D1>,
        _offsets: &[usize],
    ) -> Self::Inner<T1, D2> {
        todo!()
    }

    fn reshape<T1: Field, D1: Dim, D2: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        todo!()
    }
}

impl Repr for Op {
    type Inner<T: Copy, D: TensorDim + ArrayDim + XlaDim> = Noxpr;

    fn add<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Add<Output = T> + Copy,
        D1: Dim,
        D2: Dim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
    {
        Noxpr::add(left.clone(), right.clone())
    }

    fn sub<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Sub<Output = T> + Copy,
        D1: Dim,
        D2: Dim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
    {
        Noxpr::sub(left.clone(), right.clone())
    }

    fn mul<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Mul<Output = T> + Copy,
        D1: Dim,
        D2: Dim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
    {
        Noxpr::mul(left.clone(), right.clone())
    }

    fn div<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Div<Output = T> + Copy,
        D1: Dim,
        D2: Dim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
    {
        Noxpr::div(left.clone(), right.clone())
    }

    fn dot<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, <ShapeConstraint as DotDim<D1, D2>>::Output>
    where
        T: Field + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<D1, D2>,
        <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        Noxpr::dot(left.clone(), right)
    }

    fn concat_many<T1: Field, D1, const N: usize>(
        args: [&Self::Inner<T1, D1>; N],
    ) -> Self::Inner<T1, ConcatManyDim<D1, N>>
    where
        DefaultMappedDim<D1>: nalgebra::DimMul<Const<N>> + nalgebra::Dim,
        D1::DefaultMapDim: MapDim<D1>,
        D1::DefaultMapDim: MapDim<D1>,
        D1: Dim + DefaultMap,
        MulDim<DefaultMappedDim<D1>, Const<N>>: Dim,
        <<D1 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatManyDim<D1, N>: Dim,
    {
        Noxpr::concat_in_dim(args.iter().map(|&x| x.clone()).collect(), 0)
    }

    fn get<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
        index: usize,
    ) -> Self::Inner<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>,
        <GetDim<D1> as ArrayDim>::Buf<MaybeUninit<T1>>:,
    {
        let shape = D1::shape();
        let offsets = iter::once(index as i64)
            .chain((1..shape.len()).map(|_| 0))
            .collect::<SmallVec<[i64; 4]>>();
        let new_offsets = offsets
            .iter()
            .zip(std::iter::once(&1).chain(shape.iter().skip(1)))
            .map(|(a, b)| a + b)
            .collect();
        let strides = shape
            .iter()
            .rev()
            .scan(1, |acc, &x| {
                let res = *acc;
                *acc *= x;
                Some(res)
            })
            .collect::<SmallVec<[i64; 4]>>();
        arg.clone().slice(offsets, new_offsets, strides)
    }

    fn broadcast<D1: Dim, D2: ArrayDim + TensorDim + XlaDim, T1: Field>(
        arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim + XlaDim,
    {
        arg.clone().broadcast(D2::shape())
    }

    fn scalar_from_const<T1: Field + NativeType + ArrayElement>(value: T1) -> Self::Inner<T1, ()> {
        let lit = T1::literal(value);
        Noxpr::constant(
            lit,
            ArrayTy {
                element_type: T1::TY,
                shape: smallvec::smallvec![],
            },
        )
    }

    fn concat<T1: Field, D1, D2: Dim + DefaultMap>(
        left: &Self::Inner<T1, D1>,
        right: &Self::Inner<T1, D2>,
    ) -> Self::Inner<T1, ConcatDim<D1, D2>>
    where
        DefaultMappedDim<D1>: nalgebra::DimAdd<DefaultMappedDim<D2>> + nalgebra::Dim,
        DefaultMappedDim<D2>: nalgebra::Dim,
        D2::DefaultMapDim: MapDim<D1>,
        D1::DefaultMapDim: MapDim<D2>,
        D1: DefaultMap + Dim,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatDim<D1, D2>: Dim,
    {
        Noxpr::concat_in_dim(vec![left.clone(), right.clone()], 0)
    }

    fn neg<T1: Field, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        -arg.clone()
    }

    fn sqrt<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.clone().sqrt()
    }

    fn sin<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.clone().sin()
    }

    fn cos<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.clone().cos()
    }

    fn copy_fixed_slice<T1: Field, D1: Dim, D2: Dim + ConstDim>(
        arg: &Self::Inner<T1, D1>,
        offsets: &[usize],
    ) -> Self::Inner<T1, D2> {
        let offsets: SmallVec<_> = offsets.iter().map(|o| *o as i64).collect();
        let new_offsets = offsets
            .iter()
            .zip(D2::shape())
            .map(|(a, b)| a + b)
            .collect();
        let strides = smallvec::smallvec![1i64; offsets.len()]; // TODO(sphw): fix wrong strides
        arg.clone().slice(offsets, new_offsets, strides)
    }

    fn reshape<T1: Field, D1: Dim, D2: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        arg.clone().reshape(D2::shape())
    }
}
