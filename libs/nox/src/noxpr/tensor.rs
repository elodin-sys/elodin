use crate::{
    AddDim, Array, ArrayBuf, ArrayRepr, ArrayTy, ConcatDims, Const, ConstDim, DefaultMap, Dim,
    DimConcat, Field, MappedDim, NonScalarDim, Noxpr, Op, ReplaceDim, ReplaceMappedDim, ReprMonad,
    Scalar, Tensor, TensorItem, Vector,
};
use smallvec::{smallvec, SmallVec};
use std::marker::PhantomData;
use xla::{ArrayElement, NativeType};

impl<T: Field + ArrayElement + NativeType, D: Dim> From<Array<T, D>> for Tensor<T, D, Op> {
    fn from(arr: Array<T, D>) -> Self {
        let shape = D::array_shape(&arr.buf);
        let shape: SmallVec<[i64; 4]> = shape.as_ref().iter().map(|&x| x as i64).collect();
        let lit = T::create_r1(arr.buf.as_buf())
            .reshape(&shape)
            .expect("reshape failed");
        let inner = Noxpr::constant(
            lit,
            ArrayTy {
                element_type: T::TY,
                shape,
            },
        );
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: Field + ArrayElement + NativeType, D: Dim> From<Tensor<T, D, ArrayRepr>>
    for Tensor<T, D, Op>
{
    fn from(value: Tensor<T, D, ArrayRepr>) -> Self {
        value.inner.into()
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
    T::Item: ReprMonad<Op>,
{
    type Out = <T as TensorItem>::Item;

    fn collapse(self) -> Self::Out {
        T::Item::from_inner(self.inner)
    }
}

impl<T: TensorItem + Copy, InnerDim: Dim, D: Dim + NonScalarDim> Collapse
    for Tensor<Tensor<T, InnerDim, Op>, D, Op>
where
    (D, InnerDim): DimConcat<D, InnerDim>,
    <(D, InnerDim) as DimConcat<D, InnerDim>>::Output: Dim,
{
    type Out = Tensor<T, ConcatDims<D, InnerDim>, Op>;
    fn collapse(self) -> Self::Out {
        Tensor {
            inner: self.inner,
            phantom: PhantomData,
        }
    }
}

impl<T: TensorItem, D: Dim> TensorItem for Tensor<T, D, Op> {
    type Item = T::Item; // NOTE: this bound might be wrong

    type Dim = D;
    type Tensor<TD: Dim> = Tensor<T, TD, Op>;

    type Elem = T::Elem;
}

/// Trait for indexing into tensors, allowing for the extraction of sub-tensors or elements based on indices.
pub trait TensorIndex<T: TensorItem, D: Dim> {
    type Output;

    /// Performs the indexing operation on a tensor, returning the result.
    fn index(self, tensor: Tensor<T, D, Op>) -> Self::Output;
}

impl<T: TensorItem, D: Dim + DefaultMap, IT: TensorItem, const N: usize> TensorIndex<T, D>
    for Vector<IT, N, Op>
where
    ReplaceMappedDim<D::DefaultMapDim, D, Const<1>>: Dim + ConstDim,
    ReplaceMappedDim<D::DefaultMapDim, D, Const<N>>: Dim,
{
    type Output = Tensor<T, ReplaceMappedDim<D::DefaultMapDim, D, Const<N>>, Op>;

    fn index(self, tensor: Tensor<T, D, Op>) -> Self::Output {
        let indices = self
            .inner
            .broadcast_in_dim(smallvec![N as i64, 1], smallvec![0]);
        let slice_shape = ReplaceMappedDim::<D::DefaultMapDim, D, Const<1>>::xla_shape();

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

#[allow(clippy::type_complexity)]
impl<T: TensorItem, D: Dim + DefaultMap> Tensor<T, D, crate::Op> {
    pub fn concat_with_dim<OD: Dim, MDim: ReplaceDim<D> + ReplaceDim<OD>>(
        &self,
        other: Tensor<T, OD, Op>,
    ) -> Tensor<T, ReplaceMappedDim<MDim, D, AddDim<MappedDim<MDim, D>, MappedDim<MDim, OD>>>, Op>
    where
        MappedDim<MDim, D>: crate::DimAdd<MappedDim<MDim, OD>> + crate::Dim,
        MappedDim<MDim, OD>: crate::Dim,
        AddDim<MappedDim<MDim, D>, MappedDim<MDim, OD>>: Dim,
        ReplaceMappedDim<MDim, D, AddDim<MappedDim<MDim, D>, MappedDim<MDim, OD>>>: Dim,
    {
        let inner = Noxpr::concat_in_dim(
            vec![self.inner.clone(), other.inner.clone()],
            <MDim as ReplaceDim<D>>::MAPPED_DIM,
        );
        Tensor {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: Field, D: Dim> Tensor<T, D, Op> {
    pub fn index<I: TensorIndex<T, D>>(&self, index: I) -> I::Output {
        index.index(self.clone())
    }
}

impl<T: TensorItem, D: Dim> Tensor<T, D, Op> {
    pub fn log(&self) -> Self {
        Self::from_inner(self.inner.clone().log())
    }
}
