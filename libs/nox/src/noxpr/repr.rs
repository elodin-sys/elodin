use core::ops::{Add, Div, Mul, Sub};

use crate::array::dims::*;
use crate::{
    AddDim, ArrayTy, BroadcastDim, BroadcastedDim, ConstDim, DefaultMap, DefaultMappedDim, Dim,
    DotDim, Elem, Error, Field, Noxpr, NoxprFn, NoxprTy, OwnedRepr, RealField, ReplaceDim, Repr,
    ShapeConstraint,
};

use smallvec::{smallvec, SmallVec};

/// Represents a compute operation.
pub struct Op;

impl Op {
    fn cobroadcast<T: Elem, D1: Dim, D2: Dim>(left: &Noxpr, right: &Noxpr) -> (Noxpr, Noxpr) {
        let d1 = Self::shape::<T, D1>(left);
        let d2 = Self::shape::<T, D2>(right);

        let broadcast_dims = match d1.as_ref().len().cmp(&d2.as_ref().len()) {
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => {
                let mut broadcast_dims = d2.clone();
                if !crate::array::cobroadcast_dims(broadcast_dims.as_mut(), d1.as_ref()) {
                    panic!(
                        "unbroadcastable dims {:?} {:?}",
                        broadcast_dims.as_mut(),
                        d2.as_ref()
                    );
                }
                broadcast_dims
            }
            std::cmp::Ordering::Greater => {
                let mut broadcast_dims = d1.clone();
                if !crate::array::cobroadcast_dims(broadcast_dims.as_mut(), d2.as_ref()) {
                    panic!(
                        "unbroadcastable dims {:?} {:?}",
                        broadcast_dims.as_mut(),
                        d2.as_ref()
                    );
                }
                broadcast_dims
            }
        };
        let broadcast_dims: SmallVec<[i64; 4]> =
            broadcast_dims.into_iter().map(|x| x as i64).collect();

        (
            left.clone().broadcast_to(broadcast_dims.clone()),
            right.clone().broadcast_to(broadcast_dims),
        )
    }
}

impl Repr for Op {
    type Inner<T: Elem, D: Dim> = Noxpr;

    type Shape<D: Dim> = SmallVec<[usize; 4]>;
    fn shape<T1: Elem, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Shape<D1> {
        arg.shape()
            .unwrap()
            .into_iter()
            .map(|x| x as usize)
            .collect()
    }
}
impl OwnedRepr for Op {
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
        let (left, right) = Self::cobroadcast::<T, D1, D2>(left, right);
        Noxpr::add(left, right)
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
        let (left, right) = Self::cobroadcast::<T, D1, D2>(left, right);

        Noxpr::sub(left, right)
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
        let (left, right) = Self::cobroadcast::<T, D1, D2>(left, right);
        Noxpr::mul(left, right)
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
        let (left, right) = Self::cobroadcast::<T, D1, D2>(left, right);
        Noxpr::div(left, right)
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

    fn stack<T1: Field, D1: Dim, D2: Dim, I: IntoIterator<Item = Self::Inner<T1, D1>>>(
        args: I,
        dim: usize,
    ) -> Self::Inner<T1, D2>
    where
        I::IntoIter: ExactSizeIterator,
    {
        Noxpr::concat_in_dim(
            args.into_iter()
                .map(|a| {
                    let mut new_shape = a.shape().unwrap();
                    new_shape.insert(dim, 1);
                    a.broadcast_to(new_shape)
                })
                .collect(),
            dim,
        )
        //Noxpr::concat_in_dim(args.into_iter().collect(), dim)
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
        arg.clone().broadcast_to(dim)
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
        DefaultMappedDim<D1>: crate::DimAdd<DefaultMappedDim<D2>> + crate::Dim,
        DefaultMappedDim<D2>: crate::Dim,
        D2::DefaultMapDim: ReplaceDim<D1>,
        D1::DefaultMapDim: ReplaceDim<D2>,
        D1: DefaultMap + Dim,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as ReplaceDim<D1>>::MappedDim: crate::Dim,
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
        let zero = T1::zero::<Op>().inner.broadcast_to(shape.clone());
        let diag = diag.broadcast_in_dim(shape.clone(), smallvec![0]);
        let case = a.eq(b);
        case.select(diag, zero)
    }

    fn noop<T1: Field, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.clone()
    }

    fn try_cholesky<T1: RealField, D1: Dim + SquareDim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Result<Self::Inner<T1, D1>, Error> {
        Ok(Noxpr::cholesky(arg, false))
        // TODO(sphw): We will need to masks out the unused triangle to ensure that it is zero,
        // since it may be uninitialized memory or the existing values
    }

    fn row<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
        index: usize,
    ) -> Self::Inner<T1, RowDim<D1>>
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
        T2: Elem,
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
    ) -> impl ExactSizeIterator<Item = Self::Inner<T1, RowDim<D1>>> + '_
    where
        ShapeConstraint: DimRow<D1>,
    {
        let shape = arg.shape().unwrap();
        let strides = std::iter::once(1)
            .chain(shape.iter().skip(1).copied())
            .rev()
            .scan(1, |acc, x| {
                let res = *acc;
                *acc *= x;
                Some(res)
            })
            .collect::<SmallVec<[i64; 4]>>();
        let row_shape: SmallVec<[i64; 4]> = shape.iter().skip(1).copied().collect();
        (0..shape[0] as usize).map(move |i| {
            let i = i as i64;
            let mut start = shape.clone();
            let mut stop = shape.clone();

            start[0] = i;
            for x in &mut start[1..] {
                *x = 0;
            }
            stop[0] = i + 1;
            arg.clone()
                .slice(start, stop, strides.clone())
                .reshape(row_shape.clone())
        })
    }
}
