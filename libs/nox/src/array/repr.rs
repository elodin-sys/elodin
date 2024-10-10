use super::*;

/// Backend implementation for local computation on arrays.
pub struct ArrayRepr;

impl Repr for ArrayRepr {
    type Inner<T, D: Dim> = Array<T, D> where T: Elem;

    type Shape<D: Dim> = D::Shape;
    fn shape<T1: Elem, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Shape<D1> {
        D1::array_shape(&arg.buf)
    }
}

impl OwnedRepr for ArrayRepr {
    fn add<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Add<Output = T> + Elem,
        D1: ArrayDim + TensorDim,
        D2: ArrayDim + TensorDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: Dim,
    {
        left.add(right)
    }

    fn sub<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Sub<Output = T> + Elem,
        D1: crate::Dim + ArrayDim,
        D2: crate::Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: crate::Dim,
    {
        left.sub(right)
    }

    fn mul<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Mul<Output = T> + Elem,
        D1: crate::Dim + ArrayDim,
        D2: crate::Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: crate::Dim,
    {
        left.mul(right)
    }

    fn div<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Div<Output = T> + Elem,
        D1: crate::Dim + ArrayDim,
        D2: crate::Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: crate::Dim,
    {
        left.div(right)
    }

    fn dot<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, <ShapeConstraint as crate::DotDim<D1, D2>>::Output>
    where
        T: RealField + Div<Output = T> + Elem,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: crate::DotDim<D1, D2>,
        <ShapeConstraint as crate::DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        left.dot(right)
    }
    fn concat_many<T1: Field, D1: Dim, D2: Dim, I: IntoIterator<Item = Self::Inner<T1, D1>>>(
        args: I,
        dim: usize,
    ) -> Self::Inner<T1, D2>
    where
        I::IntoIter: ExactSizeIterator,
    {
        Array::concat_many(args, dim).unwrap()
    }

    fn stack<T1: Field, D1: Dim, D2: Dim, I: IntoIterator<Item = Self::Inner<T1, D1>>>(
        args: I,
        dim: usize,
    ) -> Self::Inner<T1, D2>
    where
        I::IntoIter: ExactSizeIterator,
    {
        Array::concat_many(args, dim).unwrap()
    }

    fn get<T1: Field, D1: Dim + DimGet>(
        arg: &Self::Inner<T1, D1>,
        index: D1::Index,
    ) -> Self::Inner<T1, ()> {
        arg.get(index)
    }

    fn broadcast<D1: Dim, D2: Dim, T1: Field>(
        arg: &Self::Inner<T1, D1>,
        dim: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim,
    {
        arg.broadcast_with_shape(dim)
    }

    fn scalar_from_const<T1: Field>(value: T1) -> Self::Inner<T1, ()> {
        Array { buf: value }
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
        D1: Dim + DefaultMap,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as ReplaceDim<D1>>::MappedDim: crate::Dim,
        ConcatDim<D1, D2>: Dim,
    {
        left.concat(right)
    }

    fn neg<T1, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1>
    where
        T1: Field + Neg<Output = T1>,
    {
        arg.neg()
    }

    fn sqrt<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.sqrt()
    }

    fn atan2<T1: Field + RealField, D1: Dim>(
        left: &Self::Inner<T1, D1>,
        right: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, D1> {
        left.atan2(right)
    }

    fn sin<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.sin()
    }

    fn cos<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.cos()
    }

    fn abs<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.abs()
    }

    fn copy_fixed_slice<T1: Field, D1: Dim, D2: Dim + ConstDim>(
        arg: &Self::Inner<T1, D1>,
        offsets: &[usize],
    ) -> Self::Inner<T1, D2> {
        arg.copy_fixed_slice(offsets)
    }

    fn reshape<T1: Field, D1: Dim, D2: Dim>(
        arg: &Self::Inner<T1, D1>,
        dim: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Self::Inner<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        arg.reshape_with_shape(dim)
    }

    fn try_lu_inverse<T1: RealField, D1: Dim + SquareDim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Result<Self::Inner<T1, D1>, Error> {
        arg.try_lu_inverse()
    }

    fn from_scalars<T1: Field, D1: Dim>(
        iter: impl IntoIterator<Item = Self::Inner<T1, ()>>,
        shape: &[usize],
    ) -> Self::Inner<T1, D1> {
        Array::from_scalars(iter, shape)
    }

    fn transpose<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, TransposedDim<D1>>
    where
        ShapeConstraint: TransposeDim<D1>,
    {
        arg.transpose()
    }

    fn eye<T1: Field, D1: Dim + SquareDim + ConstDim>() -> Self::Inner<T1, D1> {
        Array::eye()
    }

    fn from_diag<T1: Field, D1: Dim + SquareDim>(
        diag: Self::Inner<T1, D1::SideDim>,
    ) -> Self::Inner<T1, D1> {
        Array::from_diag(diag)
    }

    fn noop<T1: Field, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.clone()
    }

    fn try_cholesky<T1: RealField, D1: Dim + SquareDim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Result<Self::Inner<T1, D1>, Error> {
        arg.try_cholesky()
    }

    fn row<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
        index: usize,
    ) -> Self::Inner<T1, crate::RowDim<D1>>
    where
        ShapeConstraint: crate::DimRow<D1>,
    {
        arg.row(index)
    }

    fn map<T1, D1, T2, D2>(
        arg: &Self::Inner<T1, D1>,
        func: impl Fn(Self::Inner<T1, D1::ElemDim>) -> Self::Inner<T2, D2>,
    ) -> Self::Inner<T2, D1::MappedDim<D2>>
    where
        D1::MappedDim<D2>: Dim,
        T1: Elem + 'static,
        D1: Dim + MappableDim,
        T2: Elem + 'static,
        D2: Dim,
    {
        arg.map(func)
    }

    fn rows_iter<T1: Elem, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
    ) -> impl ExactSizeIterator<Item = Self::Inner<T1, RowDim<D1>>> + '_
    where
        ShapeConstraint: DimRow<D1>,
    {
        arg.rows_iter()
    }
}
