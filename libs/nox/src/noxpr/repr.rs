use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{ConstDim, Error, RealField, SquareDim, TransposeDim, TransposedDim};
use nalgebra::constraint::ShapeConstraint;
use smallvec::{smallvec, SmallVec};

use crate::ArrayTy;
use crate::{
    array::ArrayDim, AddDim, BroadcastDim, BroadcastedDim, ConcatDim, DefaultMap, DefaultMappedDim,
    Dim, DimGet, DotDim, Field, MapDim, Noxpr, Repr, TensorDim, XlaDim,
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
            type Inner<T: Copy, D: Dim> = $inner;

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
                T: RealField + Div<Output = T> + Copy,
                D1: Dim + ArrayDim,
                D2: Dim + ArrayDim,
                ShapeConstraint: DotDim<D1, D2>,
                <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
            {
                todo!()
            }

            fn concat_many<T1: Field, D1: Dim, D2: Dim>(
                _args: &[Self::Inner<T1, D1>],
                _dim: usize,
            ) -> Self::Inner<T1, D2> {
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

            fn from_scalars<T1: Field, D1: Dim + ConstDim>(
                _iter: impl IntoIterator<Item = Self::Inner<T1, ()>>,
            ) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn transpose<T1: Field, D1: Dim>(
                _arg: &Self::Inner<T1, D1>,
            ) -> Self::Inner<T1, TransposedDim<D1>>
            where
                ShapeConstraint: TransposeDim<D1>,
                TransposedDim<D1>: ConstDim,
            {
                todo!()
            }

            fn eye<T1: Field, D1: Dim + SquareDim + ConstDim>() -> Self::Inner<T1, D1> {
                todo!()
            }

            fn from_diag<T1: Field, D1: Dim + SquareDim + ConstDim>(
                _diag: Self::Inner<T1, D1::SideDim>,
            ) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn noop<T1: Field, D1: Dim>(_arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
                todo!()
            }

            fn try_cholesky<T1: RealField, D1: Dim + SquareDim>(
                _arg: &Self::Inner<T1, D1>,
                _upper: bool,
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
        }
    };
}

dummy_impl_repr!(Buffer, xla::PjRtBuffer);
dummy_impl_repr!(Literal, xla::Literal);

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
        T: RealField + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: DotDim<D1, D2>,
        <ShapeConstraint as DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        Noxpr::dot(left.clone(), right)
    }

    fn concat_many<T1: Field, D1: Dim, D2: Dim>(
        args: &[Self::Inner<T1, D1>],
        dim: usize,
    ) -> Self::Inner<T1, D2> {
        Noxpr::concat_in_dim(args.to_vec(), dim)
    }

    fn get<T1: Field, D1: Dim + DimGet>(
        arg: &Self::Inner<T1, D1>,
        index: D1::Index,
    ) -> Self::Inner<T1, ()> {
        let offsets = D1::index_as_array(index)
            .as_ref()
            .iter()
            .map(|&x| x as i64)
            .collect::<SmallVec<[i64; 4]>>();
        let new_offsets = offsets.iter().map(|&x| x + 1).collect();
        let shape = D1::shape();
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

    fn broadcast<D1: Dim, D2: ArrayDim + TensorDim + XlaDim, T1: Field>(
        arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim + XlaDim,
    {
        arg.clone().broadcast(D2::shape())
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

    fn try_lu_inverse<T1: RealField, D1: Dim + SquareDim>(
        _arg: &Self::Inner<T1, D1>,
    ) -> Result<Self::Inner<T1, D1>, Error> {
        // TODO: We will need to use a combination of an XLA custom call and `TriangularSolve` to mimick this
        todo!()
    }

    fn from_scalars<T1: Field, D1: Dim + ConstDim>(
        iter: impl IntoIterator<Item = Self::Inner<T1, ()>>,
    ) -> Self::Inner<T1, D1> {
        let nodes = iter.into_iter().collect();
        Noxpr::concat_in_dim(nodes, 0).reshape(D1::shape())
    }

    fn transpose<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, TransposedDim<D1>>
    where
        ShapeConstraint: TransposeDim<D1>,
        TransposedDim<D1>: ConstDim,
    {
        let d1 = D1::shape()
            .iter()
            .enumerate()
            .map(|(i, _)| i as i64)
            .rev()
            .collect();
        arg.clone().transpose(d1)
    }

    fn eye<T1: Field, D1: Dim + SquareDim + ConstDim>() -> Self::Inner<T1, D1> {
        let shape = D1::shape();
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

    fn from_diag<T1: Field, D1: Dim + SquareDim + ConstDim>(
        diag: Self::Inner<T1, D1::SideDim>,
    ) -> Self::Inner<T1, D1> {
        let shape = D1::shape();
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
        upper: bool,
    ) -> Result<Self::Inner<T1, D1>, Error> {
        Ok(Noxpr::cholesky(arg, upper))
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
        let shape = D1::shape();
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
}
