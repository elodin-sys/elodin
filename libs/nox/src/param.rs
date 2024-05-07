use std::{
    iter,
    mem::MaybeUninit,
    ops::{Add, Div, Mul, Sub},
};

use nalgebra::{constraint::ShapeConstraint, Const};
use smallvec::SmallVec;

use crate::{
    local_backend::{ArrayBufUnit, ArrayDim},
    BroadcastDim, BroadcastedDim, ConcatManyDim, DefaultMap, DefaultMappedDim, DimGet, DotDim,
    DottedDim, Field, GetDim, MapDim, MulDim, Noxpr, TensorDim, XlaDim,
};

pub struct Op;

pub struct Literal;

pub struct Buffer;

pub trait Dim: ArrayDim + TensorDim + XlaDim {}
impl<D: ArrayDim + TensorDim + XlaDim> Dim for D {}

pub trait Repr {
    type Inner<T, D: Dim>
    where
        T: Copy;

    fn add<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Add<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>;
    fn sub<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Sub<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>;
    fn mul<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Mul<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>;

    fn div<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Div<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim + ArrayDim,
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>;

    fn dot<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, <ShapeConstraint as DotDim<D1, D2>>::Output>
    where
        T: Field + Div<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<D1, D2>,
        <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
        <DottedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <DottedDim<D1, D2> as ArrayDim>::Buf<T>>;

    fn concat_many<T1: Field, D1, const N: usize>(
        args: [&Self::Inner<T1, D1>; N],
    ) -> Self::Inner<T1, ConcatManyDim<D1, N>>
    where
        DefaultMappedDim<D1>: nalgebra::DimMul<Const<N>> + nalgebra::Dim,
        D1::DefaultMapDim: MapDim<D1>,
        D1: Dim + DefaultMap,
        MulDim<DefaultMappedDim<D1>, Const<N>>: Dim,
        <<D1 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatManyDim<D1, N>: Dim,
        <ConcatManyDim<D1, N> as ArrayDim>::Buf<MaybeUninit<T1>>:
            ArrayBufUnit<T1, Init = <ConcatManyDim<D1, N> as ArrayDim>::Buf<T1>>;

    fn get<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
        index: usize,
    ) -> Self::Inner<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>,
        <GetDim<D1> as ArrayDim>::Buf<MaybeUninit<T1>>:
            ArrayBufUnit<T1, Init = <GetDim<D1> as ArrayDim>::Buf<T1>>;
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <DottedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <DottedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <ConcatManyDim<D1, N> as ArrayDim>::Buf<MaybeUninit<T1>>:
            ArrayBufUnit<T1, Init = <ConcatManyDim<D1, N> as ArrayDim>::Buf<T1>>,
    {
        todo!()
    }

    fn get<T1: Field, D1: Dim>(
        _arg: &Self::Inner<T1, D1>,
        _index: usize,
    ) -> Self::Inner<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>,
        <GetDim<D1> as ArrayDim>::Buf<MaybeUninit<T1>>:
            ArrayBufUnit<T1, Init = <GetDim<D1> as ArrayDim>::Buf<T1>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <DottedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <DottedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <ConcatManyDim<D1, N> as ArrayDim>::Buf<MaybeUninit<T1>>:
            ArrayBufUnit<T1, Init = <ConcatManyDim<D1, N> as ArrayDim>::Buf<T1>>,
    {
        todo!()
    }

    fn get<T1: Field, D1: Dim>(
        _arg: &Self::Inner<T1, D1>,
        _index: usize,
    ) -> Self::Inner<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>,
        <GetDim<D1> as ArrayDim>::Buf<MaybeUninit<T1>>:
            ArrayBufUnit<T1, Init = <GetDim<D1> as ArrayDim>::Buf<T1>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <BroadcastedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <BroadcastedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <DottedDim<D1, D2> as ArrayDim>::Buf<MaybeUninit<T>>:
            ArrayBufUnit<T, Init = <DottedDim<D1, D2> as ArrayDim>::Buf<T>>,
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
        <ConcatManyDim<D1, N> as ArrayDim>::Buf<MaybeUninit<T1>>:
            ArrayBufUnit<T1, Init = <ConcatManyDim<D1, N> as ArrayDim>::Buf<T1>>,
    {
        Noxpr::concat_in_dim(args.iter().map(|&x| x.clone()).collect(), 0)
    }

    fn get<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
        index: usize,
    ) -> Self::Inner<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>,
        <GetDim<D1> as ArrayDim>::Buf<MaybeUninit<T1>>:
            ArrayBufUnit<T1, Init = <GetDim<D1> as ArrayDim>::Buf<T1>>,
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
}
