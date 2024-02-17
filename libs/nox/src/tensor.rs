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

pub trait TensorItem {
    type Item;
    type Tensor<D>
    where
        D: TensorDim;
    type Dim: TensorDim;
    const ELEM: ElementType;

    fn from_op(op: Noxpr) -> Self::Item;
}

impl<T: NativeType + ArrayElement> TensorItem for T {
    type Item = Scalar<T>;
    type Tensor<D> = Tensor<T, D> where D: TensorDim;
    type Dim = ();

    const ELEM: ElementType = T::TY;

    fn from_op(inner: Noxpr) -> Self::Item {
        Scalar::from_op(inner)
    }
}

impl<T: TensorItem, D: TensorDim> TensorItem for Tensor<T, D, Op> {
    type Item = T::Item; // NOTE: this bound might be wrong

    type Dim = D;
    type Tensor<TD: TensorDim> = Tensor<T, TD>;

    const ELEM: ElementType = T::ELEM;

    fn from_op(op: Noxpr) -> Self::Item {
        T::from_op(op)
    }
}

impl<T, D: TensorDim> FromOp for Tensor<T, D> {
    fn from_op(inner: Noxpr) -> Self {
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

pub trait Collapse {
    type Out;
    fn collapse(self) -> Self::Out;
}

impl<T: TensorItem> Collapse for Scalar<T, Op>
where
    T::Item: IntoOp,
{
    type Out = <T as TensorItem>::Item;

    fn collapse(self) -> Self::Out {
        T::from_op(self.inner)
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

pub trait TensorDim {}
pub trait NonScalarDim {}

pub type ScalarDim = ();
impl TensorDim for ScalarDim {}
impl TensorDim for nalgebra::Dyn {}
impl NonScalarDim for nalgebra::Dyn {}
impl<const N: usize> TensorDim for nalgebra::Const<N> {}
impl<const N: usize> NonScalarDim for nalgebra::Const<N> {}

impl<T: TensorDim> DimDiv<T, T> for ShapeConstraint {}

pub trait ConstDim<const RANK: usize> {
    const RANK: usize = RANK;
    fn dims() -> [usize; RANK];
}

pub trait XlaDim {
    type Array: AsRef<[i64]>;
    fn dims() -> Self::Array;
}

pub trait DimRank<const RANK: usize> {
    const RANK: usize = RANK;
}

impl ConstDim<0> for ScalarDim {
    fn dims() -> [usize; 0] {
        []
    }
}

impl XlaDim for ScalarDim {
    type Array = [i64; 0];
    fn dims() -> [i64; 0] {
        []
    }
}

impl DimRank<0> for ScalarDim {}

impl<const N: usize> ConstDim<1> for Const<N> {
    fn dims() -> [usize; 1] {
        [N]
    }
}

impl<const N: usize> XlaDim for Const<N> {
    type Array = [i64; 1];
    fn dims() -> [i64; 1] {
        [N as i64]
    }
}

impl XlaDim for nalgebra::Dyn {
    type Array = [i64; 1];
    fn dims() -> [i64; 1] {
        [-1]
    }
}

impl<const N: usize> DimRank<1> for Const<N> {}

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


        impl<$($ty,)*> DimRank<$num> for ($($ty,)*)
              where $($ty: NonScalarDim, )*
        {
        }

        impl<$($ty,)*> ConstDim<$num> for ($($ty,)*)
              where $($ty: ConstDim<1>, )*
        {
            fn dims() -> [usize; $num] {
                [$($ty::dims()[0],)*]
            }
        }

        impl<$($ty,)*> XlaDim for ($($ty,)*)
              where $($ty: XlaDim<Array = [i64; 1]>, )*
        {
            type Array = [i64; $num];
            fn dims() -> [i64; $num] {
                [$($ty::dims()[0],)*]
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

impl<T, D: TensorDim + DimRank<R>, const R: usize> FixedSliceExt<T, D, R> for Tensor<T, D, Op> {
    fn fixed_slice<ND: TensorDim + ConstDim<R>>(&self, offsets: [usize; R]) -> Tensor<T, ND, Op> {
        let offsets: SmallVec<_> = offsets.into_iter().map(|o| o as i64).collect();
        let new_offsets = offsets
            .iter()
            .zip(ND::dims())
            .map(|(a, b)| a + b as i64)
            .collect();
        Tensor::from_op(
            self.inner
                .clone()
                .slice(offsets, new_offsets, smallvec![1i64; R]),
        )
    }
}

pub trait FixedSliceExt<T, D: TensorDim, const R: usize> {
    fn fixed_slice<ND: TensorDim + ConstDim<R>>(&self, offsets: [usize; R]) -> Tensor<T, ND, Op>;
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

pub trait DefaultMap
where
    Self: Sized,
{
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
        ND::Array: AsRef<[i64]>,
    {
        Tensor {
            inner: self
                .inner
                .reshape(SmallVec::from_slice(ND::dims().as_ref())),
            phantom: PhantomData,
        }
    }

    pub fn broadcast<ND: TensorDim>(self) -> Tensor<T, ND>
    where
        ND: TensorDim + XlaDim,
        ND::Array: AsRef<[i64]>,
    {
        let inner = self
            .inner
            .broadcast(SmallVec::from_slice(ND::dims().as_ref()));
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

pub trait NotConst1 {}

seq_macro::seq!(N in 2..99 {
    impl NotConst1 for Const<N> {}
});

impl<T: TensorItem, D: TensorDim> Tensor<T, D> {
    pub fn index<I: TensorIndex<T, D>>(&self, index: I) -> I::Output {
        index.index(self.clone())
    }
}

pub trait TensorIndex<T, D: TensorDim> {
    type Output;

    fn index(self, tensor: Tensor<T, D>) -> Self::Output;
}

impl<T: TensorItem, D: TensorDim + DefaultMap, IT: ArrayElement, const N: usize> TensorIndex<T, D>
    for Vector<IT, N>
where
    ReplaceMappedDim<D::DefaultMapDim, D, Const<1>>: XlaDim,
    <ReplaceMappedDim<D::DefaultMapDim, D, Const<1>> as XlaDim>::Array: AsRef<[i64]>,
{
    type Output = Tensor<T, ReplaceMappedDim<D::DefaultMapDim, D, Const<N>>>;

    fn index(self, tensor: Tensor<T, D>) -> Self::Output {
        let indices = self
            .inner
            .broadcast_in_dim(smallvec![N as i64, 1], smallvec![0]);
        let slice_shape = SmallVec::from_slice(
            ReplaceMappedDim::<D::DefaultMapDim, D, Const<1>>::dims().as_ref(),
        );

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
