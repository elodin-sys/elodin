use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{
    ConstDim, Elem, Error, NoxprFn, NoxprTy, RealField, SquareDim, TransposeDim, TransposedDim,
};
use nalgebra::constraint::ShapeConstraint;
use smallvec::{smallvec, SmallVec};

use crate::ArrayTy;
use crate::{
    array::ArrayDim, AddDim, BroadcastDim, BroadcastedDim, ConcatDim, DefaultMap, DefaultMappedDim,
    Dim, DimGet, DimRow, DotDim, Field, MappableDim, Noxpr, ReplaceDim, Repr, RowDim,
};

/// Represents a compute operation.
pub struct Op;

/// Represents a literal value.
pub struct Literal;

/// Represents a memory buffer.
pub struct Buffer;

macro_rules! dummy_impl_repr {
    ($repr_ty: tt, $inner: ty) => {
        impl Repr for $repr_ty {
            type Inner<T: Elem, D: Dim> = $inner;

            fn add<T, D1, D2>(
                _left: &Self::Inner<T, D1>,
                _right: &Self::Inner<T, D2>,
            ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
            where
                T: Add<Output = T> + Elem,
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
                T: Sub<Output = T> + Elem,
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
                T: Mul<Output = T> + Elem,
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
                T: Div<Output = T> + Elem,
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
                T: RealField + Div<Output = T> + Elem,
                D1: Dim + ArrayDim,
                D2: Dim + ArrayDim,
                ShapeConstraint: DotDim<D1, D2>,
                <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
            {
                todo!()
            }
            fn concat_many<
                T1: Field,
                D1: Dim,
                D2: Dim,
                I: IntoIterator<Item = Self::Inner<T1, D1>>,
            >(
                _args: I,
                _dim: usize,
            ) -> Self::Inner<T1, D2>
            where
                I::IntoIter: ExactSizeIterator,
            {
                todo!()
            }

            fn broadcast<D1: Dim, D2: Dim, T1: Field>(
                _arg: &Self::Inner<T1, D1>,
                _dim: impl AsMut<[usize]>,
            ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
            where
                ShapeConstraint: BroadcastDim<D1, D2>,
                <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
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
                D2::DefaultMapDim: ReplaceDim<D1>,
                D1::DefaultMapDim: ReplaceDim<D2>,
                D1: Dim + DefaultMap,
                AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
                <<D2 as DefaultMap>::DefaultMapDim as ReplaceDim<D1>>::MappedDim: nalgebra::Dim,
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

            fn atan2<T1: Field, D1: Dim>(
                _left: &Self::Inner<T1, D1>,
                _right: &Self::Inner<T1, D1>,
            ) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn sin<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn cos<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn abs<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn copy_fixed_slice<T1: Field, D1: Dim, D2: Dim + ConstDim>(
                _arg: &Self::Inner<T1, D1>,
                _offsets: &[usize],
            ) -> Self::Inner<T1, D2> {
                todo!()
            }

            fn reshape<T1: Field, D1: Dim, D2: Dim>(
                _arg: &Self::Inner<T1, D1>,
                _dim: impl AsMut<[usize]>,
            ) -> Self::Inner<T1, D2>
            where
                ShapeConstraint: BroadcastDim<D1, D2>,
            {
                todo!()
            }

            fn try_lu_inverse<T1: RealField, D1: Dim + SquareDim>(
                _arg: &Self::Inner<T1, D1>,
            ) -> Result<Self::Inner<T1, D1>, Error> {
                todo!()
            }

            fn get<T1: Field, D1: Dim + DimGet>(
                _arg: &Self::Inner<T1, D1>,
                _index: D1::Index,
            ) -> Self::Inner<T1, ()> {
                todo!()
            }

            fn from_scalars<T1: Field, D1: Dim>(
                _iter: impl IntoIterator<Item = Self::Inner<T1, ()>>,
                _shape: &[usize],
            ) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn transpose<T1: Field, D1: Dim>(
                _arg: &Self::Inner<T1, D1>,
            ) -> Self::Inner<T1, TransposedDim<D1>>
            where
                ShapeConstraint: TransposeDim<D1>,
            {
                todo!()
            }

            fn eye<T1: Field, D1: Dim + SquareDim + ConstDim>() -> Self::Inner<T1, D1> {
                todo!()
            }

            fn from_diag<T1: Field, D1: Dim + SquareDim>(
                _diag: Self::Inner<T1, D1::SideDim>,
            ) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn noop<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn try_cholesky<T1: RealField, D1: Dim + SquareDim>(
                _arg: &Self::Inner<T1, D1>,
            ) -> Result<Self::Inner<T1, D1>, Error> {
                todo!()
            }

            fn row<T1: Field, D1: Dim>(
                _arg: &Self::Inner<T1, D1>,
                _index: usize,
            ) -> Self::Inner<T1, crate::RowDim<D1>>
            where
                ShapeConstraint: crate::DimRow<D1>,
            {
                todo!()
            }

            fn rows_iter<T1: Elem, D1: Dim>(
                _arg: &Self::Inner<T1, D1>,
            ) -> impl Iterator<Item = Self::Inner<T1, RowDim<D1>>> + '_
            where
                ShapeConstraint: DimRow<D1>,
            {
                std::iter::empty()
            }

            fn map<T1, D1, T2, D2>(
                _arg: &Self::Inner<T1, D1>,
                _func: impl Fn(Self::Inner<T1, D1::ElemDim>) -> Self::Inner<T2, D2>,
            ) -> Self::Inner<T2, D1::MappedDim<D2>>
            where
                D1::MappedDim<D2>: Dim,
                T1: Copy + Default,
                D1: Dim + MappableDim,
                T2: Copy + Default,
                D2: Dim,
            {
                todo!()
            }

            type Shape<D: Dim> = [usize; 0];
            fn shape<T1: Elem, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Shape<D1> {
                []
            }
        }
    };
}

dummy_impl_repr!(Buffer, xla::PjRtBuffer);
dummy_impl_repr!(Literal, xla::Literal);

impl Repr for Op {
    type Inner<T: Elem, D: Dim> = Noxpr;

    fn add<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Add<Output = T> + Elem,
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
        T: Sub<Output = T> + Elem,
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
        T: Mul<Output = T> + Elem,
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
        T: Div<Output = T> + Elem,
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
        T: RealField + Elem,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<D1, D2>,
        <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        Noxpr::dot(left.clone(), right)
    }

    fn concat_many<T1: Field, D1: Dim, D2: Dim, I: IntoIterator<Item = Self::Inner<T1, D1>>>(
        args: I,
        dim: usize,
    ) -> Self::Inner<T1, D2>
    where
        I::IntoIter: ExactSizeIterator,
    {
        Noxpr::concat_in_dim(args.into_iter().collect(), dim)
    }

    fn get<T1: Field, D1: Dim + DimGet>(
        arg: &Self::Inner<T1, D1>,
        index: D1::Index,
    ) -> Self::Inner<T1, ()> {
        let offsets = D1::index_as_slice(&index)
            .iter()
            .map(|&x| x as i64)
            .collect::<SmallVec<[i64; 4]>>();
        let new_offsets = offsets.iter().map(|&x| x + 1).collect();
        let shape = arg.shape().unwrap();
        let strides = shape
            .iter()
            .rev()
            .scan(1, |acc, &x| {
                let res = *acc;
                *acc *= x;
                Some(res)
            })
            .collect::<SmallVec<[i64; 4]>>();
        arg.clone()
            .slice(offsets, new_offsets, strides)
            .reshape(smallvec![])
    }

    fn broadcast<D1: Dim, D2: Dim, T1: Field>(
        arg: &Self::Inner<T1, D1>,
        dim: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
    {
        let dim = dim.as_ref().iter().map(|&x| x as i64).collect();
        arg.clone().broadcast(dim)
    }

    fn scalar_from_const<T1: Field>(value: T1) -> Self::Inner<T1, ()> {
        let lit = T1::literal(value);
        Noxpr::constant(
            lit,
            ArrayTy {
                element_type: T1::ELEMENT_TY,
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
        D2::DefaultMapDim: ReplaceDim<D1>,
        D1::DefaultMapDim: ReplaceDim<D2>,
        D1: DefaultMap + Dim,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as ReplaceDim<D1>>::MappedDim: nalgebra::Dim,
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

    fn atan2<T1: Field + RealField, D1: Dim>(
        left: &Self::Inner<T1, D1>,
        right: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, D1> {
        left.clone().atan2(right.clone())
    }

    fn sin<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.clone().sin()
    }

    fn cos<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.clone().cos()
    }

    fn abs<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.clone().abs()
    }

    fn copy_fixed_slice<T1: Field, D1: Dim, D2: Dim + ConstDim>(
        arg: &Self::Inner<T1, D1>,
        offsets: &[usize],
    ) -> Self::Inner<T1, D2> {
        let offsets: SmallVec<_> = offsets.iter().map(|o| *o as i64).collect();
        let new_offsets = offsets
            .iter()
            .zip(D2::xla_shape())
            .map(|(a, b)| a + b)
            .collect();

        let shape = arg.shape().unwrap();
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

    fn reshape<T1: Field, D1: Dim, D2: Dim>(
        arg: &Self::Inner<T1, D1>,
        dim: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Self::Inner<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        arg.clone()
            .reshape(dim.as_ref().iter().map(|&x| x as i64).collect())
    }

    fn try_lu_inverse<T1: RealField, D1: Dim + SquareDim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Result<Self::Inner<T1, D1>, Error> {
        Ok(arg.lu_inverse())
    }

    fn from_scalars<T1: Field, D1: Dim>(
        iter: impl IntoIterator<Item = Self::Inner<T1, ()>>,
        shape: &[usize],
    ) -> Self::Inner<T1, D1> {
        let nodes = iter.into_iter().collect();
        Noxpr::concat_in_dim(nodes, 0).reshape(shape.iter().map(|&x| x as i64).collect())
    }

    fn transpose<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, TransposedDim<D1>>
    where
        ShapeConstraint: TransposeDim<D1>,
    {
        let shape = arg.shape().unwrap();
        let d1 = shape
            .iter()
            .enumerate()
            .map(|(i, _)| i as i64)
            .rev()
            .collect();
        arg.clone().transpose(d1)
    }

    fn eye<T1: Field, D1: Dim + SquareDim + ConstDim>() -> Self::Inner<T1, D1> {
        let shape = D1::xla_shape();
        let a = Noxpr::iota(
            ArrayTy {
                element_type: T1::ELEMENT_TY,
                shape: shape.clone(),
            },
            0,
        );
        let b = Noxpr::iota(
            ArrayTy {
                element_type: T1::ELEMENT_TY,
                shape,
            },
            1,
        );
        a.eq(b).convert(T1::ELEMENT_TY)
    }

    fn from_diag<T1: Field, D1: Dim + SquareDim>(
        diag: Self::Inner<T1, D1::SideDim>,
    ) -> Self::Inner<T1, D1> {
        let side_shape = diag.shape().unwrap();
        let shape: SmallVec<[i64; 4]> = side_shape
            .iter()
            .chain(side_shape.iter())
            .copied()
            .collect();
        //le shape = D1::shape();
        let a = Noxpr::iota(
            ArrayTy {
                element_type: T1::ELEMENT_TY,
                shape: shape.clone(),
            },
            0,
        );
        let b = Noxpr::iota(
            ArrayTy {
                element_type: T1::ELEMENT_TY,
                shape: shape.clone(),
            },
            1,
        );
        let zero = T1::zero::<Op>().inner.broadcast(shape.clone());
        let diag = diag.broadcast_in_dim(shape.clone(), smallvec![0]);
        a.eq(b).select(diag, zero)
    }

    fn noop<T1: Field, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.clone()
    }

    fn try_cholesky<T1: RealField, D1: Dim + SquareDim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Result<Self::Inner<T1, D1>, Error> {
        Ok(Noxpr::cholesky(arg, false))
        // TODO(sphw): We will need to maks out the unused triangle to ensure that it is zero,
        // since it may be uninitialized memory or the existing values
    }

    fn row<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
        index: usize,
    ) -> Self::Inner<T1, crate::RowDim<D1>>
    where
        ShapeConstraint: crate::DimRow<D1>,
    {
        let shape = arg.shape().unwrap();
        let strides = shape
            .iter()
            .rev()
            .scan(1, |acc, &x| {
                let res = *acc;
                *acc *= x;
                Some(res)
            })
            .collect::<SmallVec<[i64; 4]>>();
        let index = index as i64;
        arg.clone()
            .slice(smallvec![index, 0], smallvec![index + 1, shape[1]], strides)
            .reshape(smallvec![shape[1]])
    }

    fn map<T1, D1, T2, D2>(
        arg: &Self::Inner<T1, D1>,
        func: impl Fn(Self::Inner<T1, D1::ElemDim>) -> Self::Inner<T2, D2>,
    ) -> Self::Inner<T2, D1::MappedDim<D2>>
    where
        D1::MappedDim<D2>: Dim,
        T1: Field,
        D1: Dim + MappableDim,
        T2: Copy + Default,
        D2: Dim,
    {
        let arg_shape = arg.shape().unwrap();
        let shape: SmallVec<[i64; 4]> = arg_shape.iter().skip(1).copied().collect();
        let fn_arg = Noxpr::parameter(
            0,
            NoxprTy::ArrayTy(ArrayTy {
                element_type: T1::ELEMENT_TY,
                shape,
            }),
            "param_0".to_string(),
        );
        let inner = func(fn_arg.clone());
        let func = NoxprFn {
            inner,
            args: vec![fn_arg],
        };
        Noxpr::vmap_with_axis(func, &[0], std::slice::from_ref(arg)).unwrap()
    }

    fn rows_iter<T1: Elem, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
    ) -> impl Iterator<Item = Self::Inner<T1, RowDim<D1>>> + '_
    where
        ShapeConstraint: DimRow<D1>,
    {
        let shape = arg.shape().unwrap();
        let strides = shape
            .iter()
            .rev()
            .scan(1, |acc, &x| {
                let res = *acc;
                *acc *= x;
                Some(res)
            })
            .collect::<SmallVec<[i64; 4]>>();
        (0..shape[0]).map(move |i| {
            let mut start = shape.clone();
            let mut stop = shape.clone();

            start[0] = i;
            for x in &mut start[1..] {
                *x = 0;
            }
            stop[0] = i;
            arg.clone().slice(start, stop, strides.clone())
        })
    }

    type Shape<D: Dim> = SmallVec<[usize; 4]>;
    fn shape<T1: Elem, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Shape<D1> {
        arg.shape()
            .unwrap()
            .into_iter()
            .map(|x| x as usize)
            .collect()
    }
}
