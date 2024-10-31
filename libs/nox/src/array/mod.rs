//! Provides a local, non-XLA backend for operating on Tensors.
use crate::{
    AddDim, BroadcastDim, BroadcastedDim, ConstDim, DefaultMap, DefaultMappedDim, Dim, DottedDim,
    Elem, Error, Field, OwnedRepr, RealField, ReplaceDim, ReplaceMappedDim, Repr, ScalarDim,
    TensorDim,
};
use crate::{Const, Dyn, ShapeConstraint};
use alloc::{vec, vec::Vec};
use approx::{AbsDiffEq, RelativeEq};
use core::default::Default;
use core::{cmp, fmt, iter};
use core::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};
use faer::{
    linalg::{
        cholesky::llt::compute::{cholesky_in_place, cholesky_in_place_req},
        lu::{
            full_pivoting::compute::lu_in_place_req, partial_pivoting::inverse::invert_in_place_req,
        },
    },
    reborrow::ReborrowMut,
    Parallelism,
};
use smallvec::SmallVec;

mod dynamic;
mod repr;
mod view;
pub use repr::*;
pub use view::*;

pub type Vector<T, const N: usize> = crate::vector::Vector<T, N, ArrayRepr>;
pub type Vector3<T> = Vector<T, 3>;
pub type Vec3<T> = Vector3<T>;

pub type Matrix<T, const R: usize, const C: usize> = crate::matrix::Matrix<T, R, C, ArrayRepr>;
pub type Matrix3<T> = Matrix<T, 3, 3>;
pub type Mat3<T> = Matrix3<T>;

pub mod dims {
    pub use super::{
        ArrayDim, ConcatDim, DimGet, DimRow, MappableDim, RowDim, SquareDim, TransposeDim,
        TransposedDim,
    };
}
pub mod prelude {
    pub use super::{dims::*, Array, ArrayBuf, ArrayDim, ArrayRepr, ArrayView, Mat3, Vec3};
}

/// A struct representing an array with type-safe dimensions and element type.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Array<T: Elem, D: ArrayDim> {
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            deserialize = "D::Buf<T>: serde::Deserialize<'de>",
            serialize = "D::Buf<T>: serde::Serialize"
        ))
    )]
    pub buf: D::Buf<T>,
}

impl<T: Elem, D: ArrayDim> Clone for Array<T, D>
where
    D::Buf<T>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
        }
    }
}

impl<T: Elem, D: ArrayDim> Copy for Array<T, D> where D::Buf<T>: Copy {}

impl<T1, D1> Default for Array<T1, D1>
where
    T1: Elem,
    D1: ArrayDim,
    D1::Buf<T1>: Default,
{
    fn default() -> Self {
        Self {
            buf: Default::default(),
        }
    }
}

impl<T: Elem, D: ArrayDim> fmt::Debug for Array<T, D>
where
    D::Buf<T>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.buf.fmt(f)
    }
}

/// Defines an interface for array dimensions, associating buffer types and dimensionality metadata.
pub trait ArrayDim: TensorDim {
    type Buf<T>: ArrayBuf<T>
    where
        T: Elem;
    type Shape: ArrayShape;

    /// Returns the dimensions of the buffer.
    fn array_shape<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape;

    fn shape_slice<T: Elem>(buf: &Self::Buf<T>) -> &'_ [usize];

    /// Returns the strides of the buffer for multidimensional access.
    fn strides<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape;

    fn is_scalar() -> bool {
        false
    }
}

impl ArrayDim for ScalarDim {
    type Buf<T> = T where T: Elem;

    type Shape = [usize; 0];

    fn array_shape<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape {
        []
    }

    fn strides<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape {
        []
    }

    fn is_scalar() -> bool {
        true
    }

    fn shape_slice<T: Elem>(_buf: &Self::Buf<T>) -> &'_ [usize] {
        &[]
    }
}

impl<const D: usize> ArrayDim for Const<D> {
    type Buf<T> = [T; D] where T: Elem;

    type Shape = [usize; 1];

    #[inline]
    fn array_shape<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape {
        [D]
    }

    fn strides<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape {
        [1]
    }

    fn shape_slice<T: Elem>(_buf: &Self::Buf<T>) -> &'_ [usize] {
        &[D]
    }
}

impl<const D1: usize, const D2: usize> ArrayDim for (Const<D1>, Const<D2>) {
    type Buf<T> = [[T; D2]; D1] where T: Elem;

    type Shape = [usize; 2];

    fn array_shape<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape {
        [D1, D2]
    }

    fn strides<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape {
        [D2, 1]
    }

    fn shape_slice<T: Elem>(_buf: &Self::Buf<T>) -> &'_ [usize] {
        &[D1, D2]
    }
}

impl<const D1: usize, const D2: usize, const D3: usize> ArrayDim
    for (Const<D1>, Const<D2>, Const<D3>)
{
    type Buf<T> = [[[T; D3]; D2]; D1] where T: Elem;
    type Shape = [usize; 3];

    fn array_shape<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape {
        [D1, D2, D3]
    }

    fn strides<T: Elem>(_buf: &Self::Buf<T>) -> Self::Shape {
        [D3 * D2, D2, 1]
    }

    fn shape_slice<T: Elem>(_buf: &Self::Buf<T>) -> &'_ [usize] {
        &[D1, D2, D3]
    }
}

pub trait ArrayShape: AsRef<[usize]> + AsMut<[usize]> + Clone {
    fn from_len_elem(len: usize, axis: usize, elem: impl AsRef<[usize]>) -> Self;
}

impl<const N: usize> ArrayShape for [usize; N] {
    fn from_len_elem(len: usize, axis: usize, elem: impl AsRef<[usize]>) -> Self {
        let elem = elem.as_ref();
        if axis >= N {
            panic!("the length axis must be within the shape bounds");
        }
        if elem.len() == N {
            let mut out = [0; N];
            out.copy_from_slice(elem);
            out[axis] *= len;
            out
        } else if elem.len() + 1 == N {
            let mut out = [0; N];
            let mut i = 0;
            for (j, out) in out.iter_mut().enumerate() {
                if j == axis {
                    *out = len;
                } else {
                    *out = elem[i];
                    i += 1;
                }
            }
            out
        } else {
            panic!("the output rank must be equal or 1 higher than the element rank")
        }
        // debug_assert_eq!(
        //     N,
        //     elem.len() + 1,
        //     "the output rank must be 1 higher than the element rank"
        // );
        // out
    }
}
impl ArrayShape for SmallVec<[usize; 4]> {
    fn from_len_elem(len: usize, axis: usize, elem: impl AsRef<[usize]>) -> Self {
        let elem = elem.as_ref();
        let mut out: Self = elem.iter().copied().collect();
        out.insert(axis, len);
        out
    }
}

/// Provides buffer functionalities for a given type, allowing for safe memory operations.
pub trait ArrayBuf<T>: Clone {
    fn as_buf(&self) -> &[T];
    fn as_mut_buf(&mut self) -> &mut [T];
    fn default(dims: &[usize]) -> Self;
}

impl<T: Elem + Clone> ArrayBuf<T> for T {
    fn as_buf(&self) -> &[T] {
        core::slice::from_ref(self)
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        core::slice::from_mut(self)
    }

    fn default(_dims: &[usize]) -> Self {
        T::default()
    }
}

impl<const D: usize, T: Elem + Clone> ArrayBuf<T> for [T; D] {
    fn as_buf(&self) -> &[T] {
        self
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        self
    }

    fn default(_dims: &[usize]) -> Self {
        [T::default(); D]
    }
}

impl<T: Clone + Elem, const D1: usize, const D2: usize> ArrayBuf<T> for [[T; D1]; D2] {
    fn as_buf(&self) -> &[T] {
        self.as_flattened()
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        self.as_flattened_mut()
    }

    fn default(_dims: &[usize]) -> Self {
        [[T::default(); D1]; D2]
    }
}

impl<T: Clone + Elem, const D1: usize, const D2: usize, const D3: usize> ArrayBuf<T>
    for [[[T; D1]; D2]; D3]
{
    fn as_buf(&self) -> &[T] {
        self.as_flattened().as_flattened()
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        self.as_flattened_mut().as_flattened_mut()
    }

    fn default(_dims: &[usize]) -> Self {
        [[[T::default(); D1]; D2]; D3]
    }
}

impl<T1: Elem, D1: ArrayDim + TensorDim> Array<T1, D1> {
    pub fn zeroed(dims: &[usize]) -> Self {
        Array {
            buf: D1::Buf::<T1>::default(dims),
        }
    }
}

macro_rules! impl_op {
    ($op:tt, $op_trait:tt, $fn_name:tt) => {
        impl<T1: Elem, D1: ArrayDim + TensorDim  > Array<T1, D1> {
            #[doc = concat!("This function performs the `", stringify!($op_trait), "` operation on two arrays.")]
            pub fn $fn_name<D2: ArrayDim + TensorDim>(
                &self,
                b: &Array<T1, D2>,
            ) -> Array<T1, BroadcastedDim<D1, D2>>
            where
                T1: $op_trait<Output = T1>,
                ShapeConstraint: BroadcastDim<D1, D2>,
                <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim ,
            {
                let d1 = D1::array_shape(&self.buf);
                let d2 = D2::array_shape(&b.buf);

                match d1.as_ref().len().cmp(&d2.as_ref().len()) {
                    cmp::Ordering::Less | cmp::Ordering::Equal => {
                        let mut out: Array<T1, BroadcastedDim<D1, D2>> =
                            Array::zeroed(d2.as_ref());
                        let mut broadcast_dims = d2.clone();
                        if !cobroadcast_dims(broadcast_dims.as_mut(), d1.as_ref()) {
                            todo!("handle unbroadcastble dims {:?} {:?}", broadcast_dims.as_mut(), d1.as_ref());
                        }
                        for ((a, b), out) in self
                            .broadcast_iter(broadcast_dims.clone())
                            .unwrap()
                            .zip(b.broadcast_iter(broadcast_dims).unwrap())
                            .zip(out.buf.as_mut_buf().iter_mut())
                        {
                            *out = *a $op *b;
                        }
                        out
                    }
                    cmp::Ordering::Greater => {
                        let mut out: Array<T1, BroadcastedDim<D1, D2>> =
                            Array::zeroed(d2.as_ref());
                        let mut broadcast_dims = d1.clone();
                        if !cobroadcast_dims(broadcast_dims.as_mut(), d2.as_ref()) {
                            todo!("handle unbroadcastble dims {:?} {:?}", broadcast_dims.as_mut(), d2.as_ref());
                        }
                        for ((b, a), out) in b
                            .broadcast_iter(broadcast_dims.clone())
                            .unwrap()
                            .zip(self.broadcast_iter(broadcast_dims).unwrap())
                            .zip(out.buf.as_mut_buf().iter_mut())
                        {
                            *out = *a $op *b;
                        }
                        out
                    }
                }
            }

        }
    }
}

impl<T1: Elem, D1: Dim> Array<T1, D1> {
    pub fn reshape<D2: Dim + ConstDim>(&self) -> Array<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        let shape = D2::const_dim();
        self.reshape_with_shape(shape)
    }

    pub fn reshape_with_shape<D2: Dim>(
        &self,
        mut shape: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Array<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        let d1 = D1::array_shape(&self.buf);
        let mut out: Array<T1, D2> = Array::zeroed(shape.as_ref());
        if !cobroadcast_dims(shape.as_mut(), d1.as_ref()) {
            todo!("handle broadcastable dims");
        }
        for (a, out) in self
            .broadcast_iter(shape)
            .unwrap()
            .zip(out.buf.as_mut_buf().iter_mut())
        {
            *out = *a;
        }
        out
    }

    pub fn broadcast<D2>(&self) -> Array<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim,
        D2: Dim + ConstDim,
    {
        todo!()
    }

    pub fn broadcast_with_shape<D2>(
        &self,
        mut broadcast_dims: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Array<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim,
        D2: Dim,
    {
        let d1 = D1::array_shape(&self.buf);
        let mut out: Array<T1, BroadcastedDim<D1, D2>> = Array::zeroed(broadcast_dims.as_mut());
        if !cobroadcast_dims(broadcast_dims.as_mut(), d1.as_ref()) {
            todo!("handle broadcastable dims");
        }
        for (a, out) in self
            .broadcast_iter(broadcast_dims)
            .unwrap()
            .zip(out.buf.as_mut_buf().iter_mut())
        {
            *out = *a;
        }
        out
    }
}

impl_op!(*, Mul, mul);
impl_op!(+, Add, add);
impl_op!(-, Sub, sub);
impl_op!(/, Div, div);

macro_rules! impl_unary_op {
    ($op_trait:tt, $fn_name:tt) => {
        impl<T1: Elem, D1: Dim> Array<T1, D1> {
            pub fn $fn_name(&self) -> Array<T1, D1>
            where
                T1: $op_trait,
            {
                let d1 = D1::array_shape(&self.buf);
                let mut out: Array<T1, D1> = Array::zeroed(d1.as_ref());
                self.buf
                    .as_buf()
                    .iter()
                    .zip(out.buf.as_mut_buf().iter_mut())
                    .for_each(|(a, out)| {
                        *out = $op_trait::$fn_name(*a);
                    });
                out
            }
        }
    };
}

impl_unary_op!(RealField, sqrt);
impl_unary_op!(RealField, sin);
impl_unary_op!(RealField, cos);
impl_unary_op!(RealField, abs);

impl_unary_op!(RealField, acos);
impl_unary_op!(RealField, asin);

impl<T1: Elem, D1: Dim> Array<T1, D1> {
    pub fn neg(&self) -> Array<T1, D1>
    where
        T1: Neg<Output = T1>,
    {
        let d1 = D1::array_shape(&self.buf);
        let mut out: Array<T1, D1> = Array::zeroed(d1.as_ref());
        self.buf
            .as_buf()
            .iter()
            .zip(out.buf.as_mut_buf().iter_mut())
            .for_each(|(a, out)| {
                *out = -*a;
            });
        out
    }

    pub fn transpose(&self) -> Array<T1, TransposedDim<D1>>
    where
        ShapeConstraint: TransposeDim<D1>,
    {
        let mut dim = D1::array_shape(&self.buf);
        let dim = dim.as_mut();
        dim.swap(0, 1);
        let mut out: Array<T1, TransposedDim<D1>> = Array::zeroed(dim);
        self.transpose_iter()
            .zip(out.buf.as_mut_buf().iter_mut())
            .for_each(|(a, out)| {
                *out = *a;
            });
        out
    }

    pub fn transpose_iter(&self) -> impl Iterator<Item = &'_ T1> {
        let mut dims = D1::array_shape(&self.buf);
        dims.as_mut().reverse();
        let stride = RevStridesIter(D1::strides(&self.buf));
        let mut indexes = dims.clone();
        for index in indexes.as_mut().iter_mut() {
            *index = 0;
        }
        StrideIterator {
            buf: self.buf.as_buf(),
            stride,
            offsets: indexes.clone(),
            indexes,
            dims,
            phantom: PhantomData,
            bump_index: false,
        }
    }

    pub fn offset_iter<'o>(
        &self,
        offsets: &'o [usize],
    ) -> StrideIterator<
        '_,
        T1,
        impl StridesIter,
        impl AsRef<[usize]> + AsMut<[usize]> + Clone,
        impl AsRef<[usize]>,
        &'o [usize],
    > {
        let dims = D1::array_shape(&self.buf);
        let stride = D1::strides(&self.buf);
        let mut indexes = dims.clone();
        for (offset, index) in offsets
            .iter()
            .copied()
            .chain(iter::repeat(0))
            .zip(indexes.as_mut().iter_mut())
        {
            *index = offset;
        }
        StrideIterator {
            buf: self.buf.as_buf(),
            stride,
            indexes,
            offsets,
            dims,
            phantom: PhantomData,
            bump_index: false,
        }
    }

    pub fn offset_iter_mut<'o>(
        &mut self,
        offsets: &'o [usize],
    ) -> StrideIteratorMut<
        '_,
        T1,
        impl StridesIter,
        impl AsMut<[usize]> + AsRef<[usize]> + '_,
        impl AsRef<[usize]>,
        &'o [usize],
    > {
        let dims = D1::array_shape(&self.buf);
        let stride = D1::strides(&self.buf);
        let mut indexes = dims.clone();
        for (offset, index) in offsets
            .iter()
            .copied()
            .chain(iter::repeat(0))
            .zip(indexes.as_mut().iter_mut())
        {
            *index = offset;
        }
        StrideIteratorMut {
            buf: self.buf.as_mut_buf(),
            stride,
            indexes,
            offsets,
            dims,
            phantom: PhantomData,
            bump_index: false,
        }
    }

    /// Generates an iterator over the elements of the array after broadcasting to new dimensions.
    pub fn broadcast_iter<'a>(
        &'a self,
        new_dims: impl AsMut<[usize]> + AsRef<[usize]> + Clone + 'a,
    ) -> Option<impl Iterator<Item = &'a T1>> {
        let existing_dims = D1::array_shape(&self.buf);
        let existing_strides = D1::strides(&self.buf);
        let mut new_strides = new_dims.clone();
        let out_dims = new_dims.clone();
        let mut indexes = new_dims.clone();
        for i in indexes.as_mut().iter_mut() {
            *i = 0
        }
        for ((dim, existing_stride), (i, new_dim)) in existing_dims
            .as_ref()
            .iter()
            .rev()
            .chain(iter::repeat(&1))
            .zip(existing_strides.as_ref().iter().rev())
            .zip(new_dims.as_ref().iter().enumerate().rev())
        {
            if dim == new_dim {
                new_strides.as_mut()[i] = *existing_stride;
            } else if *dim == 1 {
                new_strides.as_mut()[i] = 0;
            } else {
                return None;
            }
        }
        for (i, _) in new_dims.as_ref()[existing_dims.as_ref().len()..]
            .iter()
            .enumerate()
        {
            new_strides.as_mut()[i] = 0;
        }
        Some(StrideIterator {
            buf: self.buf.as_buf(),
            stride: new_strides,
            offsets: indexes.clone(),
            indexes,
            dims: out_dims,
            phantom: PhantomData,
            bump_index: false,
        })
    }

    /// Performs a dot product between two arrays and returns a new array.
    fn dot<D2>(
        &self,
        right: &Array<T1, D2>,
    ) -> Array<T1, <ShapeConstraint as crate::DotDim<D1, D2>>::Output>
    where
        T1: RealField + Elem,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: crate::DotDim<D1, D2>,
        <ShapeConstraint as crate::DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        let dim_left = D1::array_shape(&self.buf);
        let dim_left = dim_left.as_ref();
        let (row_left, col_left) = match dim_left.as_ref().len() {
            2 => (dim_left[0], dim_left[1]),
            1 => (1, dim_left[0]),
            _ => unreachable!("dot is only valid for args of rank 1 or 2"),
        };
        let left = faer::mat::from_row_major_slice(self.buf.as_buf(), row_left, col_left);

        let dim_right = D2::array_shape(&right.buf);
        let dim_right = dim_right.as_ref();
        let row_right = dim_right.as_ref().first().copied().unwrap_or(1);
        let col_right = dim_right.as_ref().get(1).copied().unwrap_or(1);

        let right = faer::mat::from_row_major_slice(right.buf.as_buf(), row_right, col_right);
        let (dims, rank) = matmul_dims(dim_left, dim_right).unwrap();
        let dims = &dims[..rank];
        let mut out: Array<T1, DottedDim<D1, D2>> = Array::zeroed(dims);
        let row_right = dims.as_ref().first().copied().unwrap_or(1);
        let col_right = dims.as_ref().get(1).copied().unwrap_or(1);
        let out_mat =
            faer::mat::from_row_major_slice_mut(out.buf.as_mut_buf(), row_right, col_right);
        faer::linalg::matmul::matmul(
            out_mat,
            left,
            right,
            None,
            T1::one_prim(),
            Parallelism::None,
        );
        out
    }

    /// Concatenates two arrays along the first dimension.
    pub fn concat<D2: Dim + DefaultMap>(
        &self,
        right: &Array<T1, D2>,
    ) -> Array<T1, ConcatDim<D1, D2>>
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
        let d1 = D1::array_shape(&self.buf);
        let d2 = D2::array_shape(&right.buf);
        let mut out_dims = d2.clone();
        assert_eq!(d1.as_ref().len(), d2.as_ref().len());
        out_dims.as_mut()[0] = d1.as_ref()[0] + d2.as_ref()[0];
        let mut out: Array<T1, ConcatDim<D1, D2>> = Array::zeroed(out_dims.as_ref());
        self.buf
            .as_buf()
            .iter()
            .chain(right.buf.as_buf().iter())
            .zip(out.buf.as_mut_buf().iter_mut())
            .for_each(|(a, b)| {
                *b = *a;
            });
        out
    }

    /// Concatenates multiple arraysinto a single array along a specified dimension.
    pub fn concat_many<D2: Dim, I: IntoIterator<Item = Array<T1, D1>>>(
        args: I,
        dim: usize,
    ) -> Result<Array<T1, D2>, Error>
    where
        I::IntoIter: ExactSizeIterator,
    {
        let args = args.into_iter();
        if D1::is_scalar() {
            if dim != 0 {
                return Err(Error::InvalidConcatDims);
            }
            let mut out: Array<T1, D2> = Array::zeroed(&[args.len()]);
            let mut out_iter = out.buf.as_mut_buf().iter_mut();
            for arg in args {
                for (a, b) in arg.buf.as_buf().iter().zip(&mut out_iter) {
                    *b = *a;
                }
            }

            Ok(out)
        } else {
            let mut out: Option<Array<T1, D2>> = None;
            let len = args.len();
            let mut offset = 0;
            for arg in args {
                let arg_shape = D1::array_shape(&arg.buf);
                let out = out.get_or_insert_with(|| {
                    let out_shape = D2::Shape::from_len_elem(len, dim, arg_shape.as_ref());
                    Array::<T1, D2>::zeroed(out_shape.as_ref())
                });
                let mut current_offsets = D2::array_shape(&out.buf);
                let mut elem_shape = D2::array_shape(&out.buf);
                for x in elem_shape.as_mut() {
                    *x = 1;
                }
                let start = elem_shape.as_ref().len() - arg_shape.as_ref().len();
                elem_shape
                    .as_mut()
                    .get_mut(start..)
                    .ok_or(Error::InvalidConcatDims)?
                    .copy_from_slice(arg_shape.as_ref());
                let offset_delta = elem_shape.as_ref()[dim];

                for (i, a) in current_offsets.as_mut().iter_mut().enumerate() {
                    if i != dim {
                        *a = 0;
                    } else {
                        *a = offset;
                    }
                }
                let iter = out.offset_iter_mut(current_offsets.as_ref());
                let iter = StrideIteratorMut {
                    buf: iter.buf,
                    stride: iter.stride,
                    indexes: iter.indexes,
                    offsets: &current_offsets,
                    dims: elem_shape,
                    phantom: PhantomData,
                    bump_index: false,
                };
                iter.zip(arg.buf.as_buf().iter()).for_each(|(a, b)| {
                    *a = *b;
                });
                offset += offset_delta;
            }

            out.ok_or(Error::InvalidConcatDims)
        }
    }

    /// Retrieves a specific element from the array based on an index, effectively slicing the array.
    pub fn get(&self, index: D1::Index) -> Array<T1, ()>
    where
        D1: DimGet,
    {
        let buf = D1::get(index, &self.buf);
        Array { buf }
    }

    pub fn copy_fixed_slice<D2: Dim + ConstDim>(&self, offsets: &[usize]) -> Array<T1, D2> {
        let mut out: Array<T1, D2> = Array::zeroed(D2::DIM);
        let in_dim = D1::array_shape(&self.buf);
        assert!(
            in_dim.as_ref().len() == D2::DIM.len(),
            "source array rank is different than slice array"
        );
        for ((&out_dim, &in_dim), &offset) in D2::DIM
            .iter()
            .zip(in_dim.as_ref().iter())
            .zip(offsets.iter())
        {
            assert!(
                offset + out_dim <= in_dim,
                "slice out of bounds {:?}",
                offsets
            );
        }
        let iter = self.offset_iter(offsets);
        let iter = StrideIterator {
            buf: iter.buf,
            stride: iter.stride,
            indexes: iter.indexes,
            offsets: iter.offsets,
            dims: D2::DIM,
            phantom: PhantomData,
            bump_index: false,
        };
        for (a, out) in iter.zip(out.buf.as_mut_buf().iter_mut()) {
            *out = *a;
        }
        out
    }

    pub fn try_lu_inverse_mut(&mut self) -> Result<(), Error>
    where
        T1: RealField,
        D1: SquareDim,
    {
        let n = D1::order(&self.buf);
        let req = invert_in_place_req::<u32, T1>(n, n, Parallelism::None)
            .map_err(|_| Error::SizeOverflow)?
            .or(
                lu_in_place_req::<u32, T1>(n, n, Parallelism::None, Default::default())
                    .map_err(|_| Error::SizeOverflow)?,
            );
        inplace_it::inplace_or_alloc_array(
            req.unaligned_bytes_required(),
            |work: inplace_it::UninitializedSliceMemoryGuard<u8>| {
                let mut work = work.init(|_| 0);
                let mut stack = faer::dyn_stack::PodStack::new(&mut work);
                let mut perm = D1::ipiv(&self.buf);
                let mut perm_inv = D1::ipiv(&self.buf);
                let mut mat = faer::mat::from_row_major_slice_mut(self.buf.as_mut_buf(), n, n);
                let (_info, row_perm) = faer::linalg::lu::partial_pivoting::compute::lu_in_place(
                    mat.rb_mut(),
                    perm.as_mut(),
                    perm_inv.as_mut(),
                    Parallelism::None,
                    stack.rb_mut(),
                    Default::default(),
                );
                faer::linalg::lu::partial_pivoting::inverse::invert_in_place(
                    mat,
                    row_perm,
                    Parallelism::None,
                    stack,
                );
                Ok(())
            },
        )
    }

    pub fn try_lu_inverse(&self) -> Result<Self, Error>
    where
        T1: RealField,
        D1: SquareDim,
    {
        let mut out = self.clone();
        out.try_lu_inverse_mut()?;
        Ok(out)
    }

    pub fn from_scalars(iter: impl IntoIterator<Item = Array<T1, ()>>, shape: &[usize]) -> Self
    where
        T1: Field,
    {
        let mut out: Array<T1, D1> = Array::zeroed(shape);
        out.buf
            .as_mut_buf()
            .iter_mut()
            .zip(
                iter.into_iter()
                    .map(|a| a.buf)
                    .chain(iter::repeat(T1::zero_prim())),
            )
            .for_each(|(a, b)| {
                *a = b;
            });
        out
    }

    pub fn eye() -> Self
    where
        D1: SquareDim + ConstDim,
        T1: Field,
    {
        let mut out: Array<T1, D1> = Array::zeroed(D1::DIM);
        let len = out.buf.as_buf().len();
        out.offset_iter_mut(&[0, 0, 0])
            .enumerate()
            .take(len)
            .for_each(|(i, a)| {
                let i = i.as_ref();
                if i[0] == i[1] {
                    *a = T1::one_prim();
                } else {
                    *a = T1::zero_prim();
                }
            });
        out
    }

    pub fn from_diag<S: Dim>(diag: Array<T1, S>) -> Self
    where
        D1: SquareDim<SideDim = S>,
        T1: Field,
    {
        let mut out_dim = [0, 0];
        let diag_dim = D1::SideDim::array_shape(&diag.buf);
        out_dim[0] = diag_dim.as_ref()[0];
        out_dim[1] = diag_dim.as_ref()[0];
        let mut out: Array<T1, D1> = Array::zeroed(&out_dim);
        let len = out.buf.as_buf().len();
        out.offset_iter_mut(&[0, 0, 0])
            .enumerate()
            .take(len)
            .for_each(|(i, a)| {
                let i = i.as_ref();
                if i[0] == i[1] {
                    *a = diag.buf.as_buf()[i[0]];
                } else {
                    *a = T1::zero_prim();
                }
            });
        out
    }

    pub fn atan2(&self, other: &Self) -> Self
    where
        T1: RealField,
    {
        let mut out = self.clone();
        out.buf
            .as_mut_buf()
            .iter_mut()
            .zip(other.buf.as_buf().iter())
            .for_each(|(a, b)| {
                *a = a.atan2(*b);
            });
        out
    }

    pub fn try_cholesky_mut(&mut self) -> Result<(), Error>
    where
        T1: RealField,
        D1: SquareDim,
    {
        let n = D1::order(&self.buf);
        let req = cholesky_in_place_req::<T1>(n, Parallelism::None, Default::default())
            .map_err(|_| Error::SizeOverflow)?;
        let mat = faer::mat::from_row_major_slice_mut(self.buf.as_mut_buf(), n, n);
        inplace_it::inplace_or_alloc_array(
            req.unaligned_bytes_required(),
            |work: inplace_it::UninitializedSliceMemoryGuard<u8>| {
                let mut work = work.init(|_| 0);
                let stack = faer::dyn_stack::PodStack::new(&mut work);
                cholesky_in_place(
                    mat,
                    Default::default(),
                    Parallelism::None,
                    stack,
                    Default::default(),
                )
            },
        )?;

        for i in 0..n {
            for j in i + 1..n {
                self.buf.as_mut_buf()[i * n + j] = T1::zero_prim();
            }
        }
        Ok(())
    }

    pub fn try_cholesky(&self) -> Result<Self, Error>
    where
        T1: RealField,
        D1: SquareDim,
    {
        let mut out = self.clone();
        out.try_cholesky_mut()?;
        Ok(out)
    }

    pub fn row(&self, index: usize) -> Array<T1, RowDim<D1>>
    where
        ShapeConstraint: DimRow<D1>,
    {
        let dim = D1::array_shape(&self.buf);
        let dim = dim.as_ref();
        let out_size = if dim.len() == 1 { 1 } else { dim[1] };
        let mut out: Array<T1, RowDim<D1>> = Array::zeroed(&[out_size]);
        let offsets = &[index, 0];
        let iter = self.offset_iter(offsets);
        let iter = StrideIterator {
            buf: iter.buf,
            stride: iter.stride,
            indexes: iter.indexes,
            offsets: iter.offsets,
            dims: &[1, out_size],
            phantom: PhantomData,
            bump_index: false,
        };
        for (a, out) in iter.zip(out.buf.as_mut_buf().iter_mut()) {
            *out = *a;
        }
        out
    }

    pub fn rows_iter(&self) -> impl ExactSizeIterator<Item = Array<T1, RowDim<D1>>> + '_
    where
        ShapeConstraint: DimRow<D1>,
    {
        let dim = D1::array_shape(&self.buf);
        let len = dim.as_ref()[0];
        (0..len).map(|i| self.row(i))
    }

    pub fn map<T2: Elem, D2: Dim>(
        &self,
        func: impl Fn(Array<T1, D1::ElemDim>) -> Array<T2, D2>,
    ) -> Array<T2, D1::MappedDim<D2>>
    where
        D1: MappableDim,
        D1::MappedDim<D2>: Dim,
    {
        let dim = D1::array_shape(&self.buf);
        let elem_dim = &dim.as_ref()[1..];
        let elem_size: usize = elem_dim.as_ref().iter().sum();
        let mut out: Option<(<D2 as ArrayDim>::Shape, _)> = None;
        let len = dim.as_ref()[1];
        for (i, chunk) in self.buf.as_buf().chunks_exact(elem_size).enumerate() {
            let mut elem = Array::<T1, D1::ElemDim>::zeroed(elem_dim);
            elem.buf.as_mut_buf().copy_from_slice(chunk);
            let new_elem = func(elem);
            let new_elem_dim = D2::array_shape(&new_elem.buf);
            let (out_elem_dim, arr) = out.get_or_insert_with(|| {
                let out_dim = <D1::MappedDim<D2> as ArrayDim>::Shape::from_len_elem(
                    len,
                    0,
                    new_elem_dim.as_ref(),
                );
                (
                    new_elem_dim.clone(),
                    Array::<T2, D1::MappedDim<D2>>::zeroed(out_dim.as_ref()),
                )
            });
            if out_elem_dim.as_ref() != new_elem_dim.as_ref() {
                panic!("map must return the same dimension elements")
            }
            let out_elem_size: usize = out_elem_dim.as_ref().iter().sum();
            arr.buf.as_mut_buf()[i * out_elem_size..((i + 1) * out_elem_size)]
                .copy_from_slice(new_elem.buf.as_buf());
        }
        let (_, arr) = out.unwrap();
        arr
    }

    pub fn to_dyn(&self) -> Array<T1, Dyn> {
        let shape = D1::array_shape(&self.buf);
        let shape = SmallVec::from_slice(shape.as_ref());
        let buf = dynamic::DynArray::from_shape_vec(shape, self.buf.as_buf().to_vec()).unwrap();
        Array { buf }
    }

    pub fn cast_dyn<D2>(self) -> Array<T1, D2>
    where
        D2: ArrayDim<Buf<T1> = D1::Buf<T1>>,
    {
        Array { buf: self.buf }
    }

    pub fn view(&self) -> ArrayView<'_, T1> {
        let shape = D1::shape_slice(&self.buf);
        ArrayView {
            buf: self.buf.as_buf(),
            shape,
        }
    }
}

pub trait SquareDim: ArrayDim {
    type SideDim: Dim;
    type IPIV: AsMut<[u32]>;
    fn ipiv<T: Elem>(_buf: &Self::Buf<T>) -> Self::IPIV;
    fn order<T: Elem>(buf: &Self::Buf<T>) -> usize;
}

impl SquareDim for Dyn {
    type SideDim = Dyn;
    type IPIV = Vec<u32>;

    fn ipiv<T: Elem>(buf: &Self::Buf<T>) -> Self::IPIV {
        let n = buf.shape()[0];
        vec![0; n]
    }

    fn order<T: Elem>(buf: &Self::Buf<T>) -> usize {
        buf.shape()[0]
    }
}

impl<const N: usize> SquareDim for (Const<N>, Const<N>) {
    type SideDim = Const<N>;
    type IPIV = [u32; N];

    fn ipiv<T: Elem>(_buf: &Self::Buf<T>) -> Self::IPIV {
        [0; N]
    }

    fn order<T: Elem>(_buf: &Self::Buf<T>) -> usize {
        N
    }
}

impl SquareDim for (Dyn, Dyn) {
    type SideDim = Dyn;
    type IPIV = Vec<u32>;

    fn ipiv<T: Elem>(buf: &Self::Buf<T>) -> Self::IPIV {
        let n = buf.shape()[0];
        vec![0; n]
    }

    fn order<T: Elem>(buf: &Self::Buf<T>) -> usize {
        buf.shape()[0]
    }
}

/// Represents a type resulting from combining dimensions of two arrays during concatenation operations.
pub type ConcatDim<D1, D2> = ReplaceMappedDim<
    <D2 as DefaultMap>::DefaultMapDim,
    D1,
    AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>,
>;

/// Represents a type resulting from multiplying a dimension by a constant, used in expanding arrays.
pub type ConcatManyDim<D1, const N: usize> =
    ReplaceMappedDim<<D1 as DefaultMap>::DefaultMapDim, D1, MulDim<DefaultMappedDim<D1>, Const<N>>>;

/// Represents a type resulting from multiplication operations between two dimensions.
pub type MulDim<A, B> = <A as crate::DimMul<B>>::Output;

pub trait TransposeDim<D: Dim> {
    type Output: Dim;
}

pub type TransposedDim<D> = <ShapeConstraint as TransposeDim<D>>::Output;

impl<D1: Dim, D2: Dim> TransposeDim<(D1, D2)> for ShapeConstraint
where
    (D2, D1): Dim,
    (D1, D2): Dim,
{
    type Output = (D2, D1);
}

/// Trait to enable retrieving a specified dimension type from a composite dimension.
pub trait DimGet: Dim {
    type Index;

    fn get<T: Elem>(index: Self::Index, buf: &Self::Buf<T>) -> T;

    fn index_as_slice(index: &Self::Index) -> &[usize];
}

impl<D: Dim + crate::NonTupleDim + crate::NonScalarDim> DimGet for D {
    type Index = usize;

    fn get<T: Elem>(index: Self::Index, buf: &Self::Buf<T>) -> T {
        buf.as_buf()[index]
    }

    fn index_as_slice(index: &Self::Index) -> &[usize] {
        core::slice::from_ref(index)
    }
}

impl<
        D1: Dim + crate::NonTupleDim + crate::NonScalarDim,
        D2: crate::NonTupleDim + crate::NonScalarDim,
    > DimGet for (D1, D2)
where
    (D1, D2): Dim,
{
    type Index = [usize; 2];

    fn get<T: Elem>([a, b]: Self::Index, buf: &Self::Buf<T>) -> T {
        let dim = Self::array_shape(buf);
        let index = a * dim.as_ref()[1] + b;
        buf.as_buf()[index]
    }

    fn index_as_slice(index: &Self::Index) -> &[usize] {
        index
    }
}

pub trait DimColumn<D1> {
    type ColumnDim: Dim;
    const COLUMN_SIZE: usize;
}

pub type ColumnDim<D1> = <ShapeConstraint as DimColumn<D1>>::ColumnDim;

impl<D1: Dim, const N2: usize> DimColumn<(D1, Const<N2>)> for ShapeConstraint {
    type ColumnDim = D1;
    const COLUMN_SIZE: usize = N2;
}

pub type RowDim<D1> = <ShapeConstraint as DimRow<D1>>::RowDim;

pub trait DimRow<D1> {
    type RowDim: Dim;
}

impl<D1: Dim, D2: Dim> DimRow<(D1, D2)> for ShapeConstraint {
    type RowDim = D2;
}

impl<D1: Dim + crate::NonTupleDim> DimRow<D1> for ShapeConstraint {
    type RowDim = ();
}

pub trait MappableDim {
    type MappedDim<D>
    where
        D: Dim;

    type ElemDim: Dim;
}

impl<D1: Dim, D2: Dim> MappableDim for (D1, D2) {
    type MappedDim<D> = (D1, D) where D: Dim;

    type ElemDim = D2;
}

pub(crate) fn cobroadcast_dims(output: &mut [usize], other: &[usize]) -> bool {
    for (output, other) in output.iter_mut().rev().zip(other.iter().rev()) {
        if *output == *other || *other == 1 {
            continue;
        }
        if *output == 1 {
            *output = *other;
        } else {
            return false;
        }
    }
    true
}

pub trait StridesIter {
    fn stride_iter(&self) -> impl DoubleEndedIterator<Item = usize>;
}

impl<S: AsRef<[usize]>> StridesIter for S {
    fn stride_iter(&self) -> impl DoubleEndedIterator<Item = usize> {
        self.as_ref().iter().copied()
    }
}

struct RevStridesIter<S>(S);
impl<S: StridesIter> StridesIter for RevStridesIter<S> {
    fn stride_iter(&self) -> impl DoubleEndedIterator<Item = usize> {
        self.0.stride_iter().rev()
    }
}

/// An iterator for striding over an array buffer, providing element-wise access according to specified strides.
pub struct StrideIterator<
    'a,
    T,
    S: StridesIter,
    I: AsMut<[usize]>,
    D: AsRef<[usize]>,
    O: AsRef<[usize]>,
> {
    buf: &'a [T],
    stride: S,
    indexes: I,
    dims: D,
    phantom: PhantomData<&'a T>,
    offsets: O,
    bump_index: bool,
}

impl<'a, T, S, I, D, O> Iterator for StrideIterator<'a, T, S, I, D, O>
where
    S: StridesIter,
    I: AsMut<[usize]> + AsRef<[usize]> + 'a,
    D: AsRef<[usize]>,
    O: AsRef<[usize]>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let indexes = self.indexes.as_mut();
        let dims = self.dims.as_ref();
        if self.bump_index {
            let mut carry = true;
            for ((&dim, index), offset) in dims
                .iter()
                .zip(indexes.iter_mut())
                .zip(self.offsets.as_ref().iter())
                .rev()
            {
                if carry {
                    *index += 1;
                }
                carry = *index >= dim + offset;
                if carry {
                    *index = *offset;
                }
            }
        }
        self.bump_index = true;
        let i: usize = indexes
            .iter()
            .zip(self.stride.stride_iter())
            .map(|(&i, s): (&usize, usize)| i * s)
            .sum();
        self.buf.get(i)
    }
}

/// An iterator for striding over an array buffer, providing element-wise access according to specified strides.
pub struct StrideIteratorMut<
    'a,
    T,
    S: StridesIter,
    I: AsMut<[usize]> + AsRef<[usize]> + 'a,
    D: AsRef<[usize]>,
    O: AsRef<[usize]>,
> {
    buf: &'a mut [T],
    stride: S,
    indexes: I,
    dims: D,
    phantom: PhantomData<&'a T>,
    offsets: O,
    bump_index: bool,
}

impl<'a, T, S, I, D, O> Iterator for StrideIteratorMut<'a, T, S, I, D, O>
where
    S: StridesIter,
    I: AsMut<[usize]> + AsRef<[usize]> + 'a,
    D: AsRef<[usize]>,
    O: AsRef<[usize]>,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let indexes = self.indexes.as_mut();
        let dims = self.dims.as_ref();
        if self.bump_index {
            let mut carry = true;
            for ((&dim, index), offset) in dims
                .iter()
                .zip(indexes.iter_mut())
                .zip(self.offsets.as_ref().iter())
                .rev()
            {
                if carry {
                    *index += 1;
                }
                carry = *index >= dim + offset;
                if carry {
                    *index = *offset;
                }
            }
        }
        self.bump_index = true;
        let i: usize = indexes
            .iter()
            .zip(self.stride.stride_iter())
            .map(|(&i, s): (&usize, usize)| i * s)
            .sum();
        unsafe { self.buf.get_mut(i).map(|p| &mut *(p as *mut T)) }
    }
}

impl<
        'a,
        T,
        S: StridesIter,
        I: AsMut<[usize]> + AsRef<[usize]> + 'a,
        D: AsRef<[usize]>,
        O: AsRef<[usize]>,
    > StrideIteratorMut<'a, T, S, I, D, O>
{
    fn enumerate(self) -> impl Iterator<Item = (&'a I, &'a mut T)> {
        EnumerateStrideIteratorMut(self)
    }
}

pub struct EnumerateStrideIteratorMut<
    'a,
    T,
    S: StridesIter,
    I: AsMut<[usize]> + AsRef<[usize]> + 'a,
    D: AsRef<[usize]>,
    O: AsRef<[usize]>,
>(StrideIteratorMut<'a, T, S, I, D, O>);

impl<
        'a,
        T,
        S: StridesIter,
        I: AsMut<[usize]> + AsRef<[usize]> + 'a,
        D: AsRef<[usize]>,
        O: AsRef<[usize]>,
    > Iterator for EnumerateStrideIteratorMut<'a, T, S, I, D, O>
{
    type Item = (&'a I, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| {
            (
                unsafe { core::mem::transmute::<&I, &'a I>(&self.0.indexes) },
                x,
            )
        })
    }
}

fn matmul_dims(a: &'_ [usize], b: &'_ [usize]) -> Option<([usize; 2], usize)> {
    let mut out = [0; 2];
    match (a.len(), b.len()) {
        (0, _) => {
            for (out, b) in out.iter_mut().zip(b.iter()) {
                *out = *b
            }
            Some((out, 2))
        }
        (1, 1) => Some((out, 0)),
        (2, 1) => {
            if a[1] != b[0] {
                return None;
            };
            out[0] = a[0];
            Some((out, 1))
        }
        (2, 2) => {
            if a[1] != b[0] {
                return None;
            };
            out[0] = a[0];
            out[1] = b[1];
            Some((out, 2))
        }
        _ => None,
    }
}

impl<T: Field> From<T> for Array<T, ()> {
    fn from(buf: T) -> Self {
        Array { buf }
    }
}

impl<T: Field, const D1: usize> From<[T; D1]> for Array<T, Const<D1>> {
    fn from(buf: [T; D1]) -> Self {
        Array { buf }
    }
}

impl<T: Field, const D1: usize, const D2: usize> From<[[T; D2]; D1]>
    for Array<T, (Const<D1>, Const<D2>)>
{
    fn from(buf: [[T; D2]; D1]) -> Self {
        Array { buf }
    }
}

impl<T: Field, const D1: usize, const D2: usize, const D3: usize> From<[[[T; D3]; D2]; D1]>
    for Array<T, (Const<D1>, Const<D2>, Const<D3>)>
{
    fn from(buf: [[[T; D3]; D2]; D1]) -> Self {
        Array { buf }
    }
}

#[macro_export]
macro_rules! array {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Array::from([$([$([$($x,)*],)*],)*])
    }};
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::Array::from([$([$($x,)*],)*])
    }};
    ($($x:expr),* $(,)*) => {{
        $crate::Array::from([$($x,)*])
    }};

    ($elem:expr; $n:expr) => {{
        $crate::Array::from([$elem; $n])
    }};
}

impl<T: Elem + PartialEq, D: Dim> PartialEq for Array<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.buf.as_buf() == other.buf.as_buf()
    }
}

impl<T: AbsDiffEq, D: Dim> AbsDiffEq for Array<T, D>
where
    T: Elem,

    T::Epsilon: Elem,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.buf.as_buf().abs_diff_eq(other.buf.as_buf(), epsilon)
    }
}

impl<T: RelativeEq, D: Dim> RelativeEq for Array<T, D>
where
    T: Elem,
    T::Epsilon: Elem,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        self.buf
            .as_buf()
            .relative_eq(other.buf.as_buf(), epsilon, max_relative)
    }
}

#[cfg(test)]
mod tests {

    use core::f64::consts::FRAC_PI_4;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_add_broadcast() {
        let a = array![1.];
        let b = array![1.0; 2];
        let c: Array<f32, Const<2>> = a.add(&b);
        assert_eq!(c.buf, [2.0; 2]);

        let a: Array<f32, (Const<1>, Const<2>)> = array![[1.0, 2.0]];
        let b: Array<f32, (Const<2>, Const<2>)> = array![[1.0, 1.0], [2.0, 2.0]];
        let c: Array<f32, (Const<2>, Const<2>)> = a.add(&b);
        assert_eq!(c.buf, [[2.0, 3.0], [3.0, 4.0]]);

        let a = array![[[1.0, 2.0]], [[1.0, 2.0]]];
        let b = array![[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]];
        let c: Array<f32, (Const<2>, Const<2>, Const<2>)> = a.add(&b);
        assert_eq!(c.buf, [[[2.0, 3.0], [3.0, 4.0]], [[2.0, 3.0], [3.0, 4.0]]]);
    }

    #[test]
    fn test_matmul_3x3() {
        let eye: Array<f32, (Const<3>, Const<3>)> = Array {
            buf: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let mat: Array<f32, (Const<3>, Const<3>)> = Array {
            buf: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        };
        assert_eq!(eye.dot(&mat).buf, mat.buf);
        let a: Array<f32, (Const<3>, Const<3>)> = Array {
            buf: [
                [0.7068251811053661, 0.0, -0.7073882691671998],
                [0.7073882691671998, 0.0, 0.7068251811053661],
                [0.0, -1.0, 0.0],
            ],
        };
        let b: Array<f32, (Const<3>, Const<3>)> = Array {
            buf: [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        };
        let expected_out: Array<f32, (Const<3>, Const<3>)> = Array {
            buf: [
                [0.70738827, 0.70682518, 0.],
                [-0.70682518, 0.70738827, 0.],
                [0., 0., 1.],
            ],
        };
        assert_eq!(a.dot(&b).buf, expected_out.buf);
    }

    #[test]
    fn test_matmul_broadcast() {
        let a: Array<f32, (Const<2>, Const<2>)> = Array {
            buf: [[0.0, 1.0], [4.0, 2.0]],
        };

        let b: Array<f32, (Const<2>, Const<2>)> = Array {
            buf: [[1.0, 1.0], [2.0, 2.0]],
        };
        let c: Array<f32, (Const<2>, Const<2>)> = a.dot(&b);
        assert_eq!(c.buf, [[2.0, 2.0], [8.0, 8.0]]);

        let a: Array<f32, (Const<3>, Const<3>)> = Array {
            buf: [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        };
        let b: Array<f32, (Const<3>, Const<1>)> = Array {
            buf: [[0.0], [1.0], [1.0]],
        };
        let c: Array<f32, (Const<3>, Const<1>)> = a.dot(&b);
        assert_eq!(c.buf, [[2.0], [4.0], [6.0]])
    }

    #[test]
    fn test_concat() {
        let a: Array<f32, Const<2>> = Array { buf: [0.0, 1.0] };

        let b: Array<f32, Const<2>> = Array { buf: [2.0, 3.0] };
        let c: Array<f32, Const<4>> = a.concat(&b);
        assert_eq!(c.buf, [0., 1., 2., 3.]);

        let a: Array<f32, (Const<2>, Const<2>)> = Array {
            buf: [[0.0, 1.0], [4.0, 2.0]],
        };

        let b: Array<f32, (Const<2>, Const<2>)> = Array {
            buf: [[1.0, 1.0], [2.0, 2.0]],
        };
        let c: Array<f32, (Const<4>, Const<2>)> = a.concat(&b);
        assert_eq!(c.buf, [[0., 1.], [4., 2.], [1., 1.], [2., 2.]]);

        let a: Array<f32, ()> = Array { buf: 1.0 };
        let b: Array<f32, ()> = Array { buf: 2.0 };
        let c: Array<f32, ()> = Array { buf: 3.0 };
        let d: Array<f32, Const<3>> = Array::concat_many([a, b, c], 0).unwrap();
        assert_eq!(d.buf, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_concat_many() {
        let a: Array<f32, Const<3>> = Array {
            buf: [1.0, 2.0, 3.0],
        };
        let b: Array<f32, Const<3>> = Array {
            buf: [4.0, 5.0, 6.0],
        };
        let c: Array<f32, Const<3>> = Array {
            buf: [7.0, 8.0, 9.0],
        };
        let d: Array<f32, (Const<3>, Const<3>)> = Array::concat_many([a, b, c], 0).unwrap();
        assert_eq!(d.buf, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],]);
    }

    #[test]
    fn test_transpose() {
        let a = array![
            [0.0, -0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ];
        assert_eq!(
            a.transpose(),
            array![
                [0., 0., -1.],
                [-0., 0., 0.],
                [1., -0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]
            ]
        );
    }

    #[test]
    fn test_concat_many_dim() {
        let a: Array<f32, (Const<2>, Const<2>)> = array![[0.0, 1.0], [2.0, 3.0]];
        let b: Array<f32, (Const<2>, Const<2>)> = array![[0.0, 1.0], [2.0, 3.0]];
        let c: Array<f32, (Const<2>, Const<4>)> = Array::concat_many([a, b], 1).unwrap();
        assert_eq!(c, array![[0.0, 1.0, 0.0, 1.0], [2.0, 3.0, 2.0, 3.0]]);

        let a: Array<f32, (Const<2>, Const<2>)> = array![[0.0, 1.0], [2.0, 3.0]];
        let b: Array<f32, (Const<2>, Const<2>)> = array![[0.0, 1.0], [2.0, 3.0]];
        let c: Array<f32, (Const<4>, Const<2>)> = Array::concat_many([a, b], 0).unwrap();
        assert_eq!(c, array![[0., 1.], [2., 3.], [0., 1.], [2., 3.]]);

        let a: Array<f32, Const<3>> = Array {
            buf: [1.0, 2.0, 3.0],
        };
        let b: Array<f32, Const<3>> = Array {
            buf: [4.0, 5.0, 6.0],
        };
        let c: Array<f32, Const<3>> = Array {
            buf: [7.0, 8.0, 9.0],
        };
        let d: Array<f32, (Const<3>, Const<3>)> = Array::concat_many([a, b, c], 0).unwrap();
        assert_eq!(d.buf, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],]);
    }

    #[test]
    fn test_eye() {
        assert_eq!(Array::eye(), array![[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(
            Array::eye(),
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn test_from_diag() {
        assert_eq!(
            Array::from_diag(array![1.0, 4.0]),
            array![[1.0, 0.0], [0.0, 4.0]]
        );
        assert_eq!(
            Array::from_diag(array![1.0, 4.0, 5.0]),
            array![[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
        );
    }

    #[test]
    fn test_abs() {
        let a = array![[1.0, -2.0], [-3.0, 4.0]];
        assert_eq!(a.abs(), array![[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn test_atan2() {
        let x = array![3.0, -3.0];
        let y = array![-3.0, 3.0];
        assert_relative_eq!(y.atan2(&x), array![-FRAC_PI_4, 3.0 * FRAC_PI_4]);
    }

    #[test]
    fn test_lu_inverse() {
        let mut a = array![[1.0, 2.0], [3.0, 4.0]];
        a.try_lu_inverse_mut().unwrap();
        assert_relative_eq!(a, array![[-2.0, 1.0], [1.5, -0.5]]);

        let mut a = array![[1.0, 0.0], [0.0, 1.0]];
        a.try_lu_inverse_mut().unwrap();
        assert_eq!(a, array![[1.0, 0.0], [0.0, 1.0]]);

        #[rustfmt::skip]
         let mut a: Array<f64, (Const<10>, Const<10>)> = array![
             [24570.1, 76805.1, 43574.6, 18894.2, -7640.97, 2261.34, 22776.7, 24861.4, 81641., 34255.6],
             [12354.1, 78957.6, 5642.45, -4702.81, 63301.7, 35105.5, 35568.9, 58708.1, 45157., 65454.5],
             [8302.27, 65510.6, 20473.5, 55808.1, 39832.5, 92954.5, 79581.3, 35383.3, 96110.3, 34361.5],
             [30932.6, 67202.2, 21617.7, 75088.7, 71295.8, 42937.1, 26957.5, 59796.5, 35418.6, 26217.4],
             [77307.7, 39452.7, 75145.5, 44098.6, 12566.9, 16471.8, 71774.6, -4106.6, 53838.2, 36685.3],
             [83757.9, 17360., 8921.7, 65612.8, 90126.7, 86641.8, 21293.4, 20590.5, 13033.9, 76379.3],
             [83768.9, 46348.9, 16581.3, 31374.9, 9137.27, 37604.4, 32564., 15644.9, -4805.73, 49756.],
             [12081.9, 85443.3, 88681.9, 64841.1, 51603.8, 53034.5, 7805.68, 39358.2, -140.273, 84237.4],
             [40253.6, 69906.9, 38533.1, 60614., 57636.5, 82128.6, 68686.8, 37255.3, 33246.1, 52798.4],
             [16576.6, 37261.4, 38658.7, 91431.4, 40354.5, 9395.03, 62509.4, 28617.7, 33828.6, 60181.7]
         ];
        a.try_lu_inverse_mut().unwrap();

        #[rustfmt::skip]
         let  expected_inverse: Array<f64, (Const<10>, Const<10>)> = array![
             [-0.0000100226, 3.34899e-6, 8.88045e-6, 9.55701e-6, 0.0000103527, -4.59573e-7, 0.000013572, 2.24482e-6, -0.0000248319, -5.47633e-6],
             [0.0000858703, -0.000028404, -0.0000867496, -0.0000302066, -0.0000451287, 0.0000168553, -0.0000504144, -0.000040719, 0.000163731, 5.85461e-6],
             [-0.000043415, 0.0000132544, 0.0000398965, 0.0000175089, 0.0000308841, -0.0000127083, 0.0000196943, 0.0000276474, -0.0000772964, -9.97593e-6],
             [0.0000195235, -0.0000152707, -0.0000151017, -2.35467e-6, -0.0000144411, 5.44407e-6, -8.03035e-6, -8.26637e-6, 0.0000293996, 9.4548e-6],
             [0.0000395083, -9.01866e-6, -0.0000518759, -0.0000126334, -0.0000150054, 0.0000164758, -0.0000424713, -0.0000239907, 0.0000884494, 1.77583e-6],
             [-0.0000213967, 2.21527e-6, 0.0000281531, 4.94374e-6, 6.72697e-6, -3.5336e-6, 0.000015341, 0.0000136699, -0.0000350585, -9.13423e-6],
             [-9.60105e-6, 5.4443e-6, 7.88212e-7, -3.35387e-6, 5.25609e-6, -7.54644e-6, 2.13287e-6, -4.36139e-6, 6.954e-6, 5.16871e-6],
             [-0.000125344, 0.0000464253, 0.000126716, 0.0000579426, 0.0000603612, -0.0000382841, 0.0000876006, 0.0000614865, -0.000240183, -0.0000127178],
             [9.00549e-7, 2.22589e-6, 0.0000120239, 2.6913e-6, 5.10871e-6, 3.9984e-6, -1.12099e-6, 1.35222e-6, -0.0000214447, -1.312e-6],
             [-6.26086e-6, 8.06916e-6, 0.0000111363, -9.48246e-6, -8.31803e-7, 2.88266e-6, 9.86769e-6, 7.87647e-6, -0.0000243396, 8.19543e-6]
         ];
        assert_relative_eq!(a, expected_inverse, epsilon = 1e-4);
    }

    #[test]
    fn test_cholesky() {
        let mut a = array![[1.0, 0.0], [0.0, 1.0]];
        a.try_cholesky_mut().unwrap();
        assert_eq!(a, array![[1.0, 0.0], [0.0, 1.0]]);
        let mut a = array![[4.0, 0.0], [0.0, 16.0]];
        a.try_cholesky_mut().unwrap();
        assert_eq!(a, array![[2.0, 0.0], [0.0, 4.0]]);
        let a = array![[1., 2., 3.], [0., 2., 3.], [0., 0., 1.]];
        let b = a.dot(&a.transpose());
        assert_relative_eq!(
            b.try_cholesky().unwrap().transpose(),
            array![
                [3.7416575, 3.474396, 0.8017837],
                [0., 0.9636248, 0.22237495],
                [0., 0., 0.5547002]
            ],
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_broadcast_more_dims() {
        let a = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
        let b = array![5.0, 0.0];
        let out = a.sub(&b);
        assert_eq!(out, array![[-4., 2.], [-4., 2.], [-4., 2.]])
    }

    #[test]
    fn test_row() {
        let a = array![[1.0, 2.0], [5.0, 8.0], [9.0, 9.0]];
        assert_eq!(array![1., 2.], a.row(0));
        assert_eq!(array![5., 8.], a.row(1));
        assert_eq!(array![9., 9.], a.row(2));
    }

    #[test]
    fn test_partial_dyn_mat() {
        let a: Array<f64, (Dyn, Dyn)> = array![[1.0, 2.0], [3.0, 4.0]].to_dyn().cast_dyn();
        let b: Array<f64, (Dyn, Dyn)> = array![[1.0, 2.0], [3.0, 4.0]].to_dyn().cast_dyn();
        let c = a.dot(&b);
        let expected: Array<f64, (Dyn, Dyn)> =
            array![[7.0, 10.0], [15.0, 22.0]].to_dyn().cast_dyn();

        assert_eq!(c, expected)
    }

    #[test]
    fn test_map() {
        let a = array![[1.0, 2.0], [5.0, 8.0], [9.0, 9.0]];
        let out: Array<f64, (Const<3>, Const<2>)> =
            a.map(|x: Array<f64, Const<2>>| array![2.0f64, 3.0].add(&x));
        assert_eq!(out, array![[3.0, 5.0], [7.0, 11.0], [11.0, 12.0]]);

        let a = array![[1.0, 2.0], [5.0, 8.0], [9.0, 9.0]];
        let out: Array<f64, (Const<3>, Const<1>)> =
            a.map(|x: Array<f64, Const<2>>| x.copy_fixed_slice::<Const<1>>(&[0]));
        assert_eq!(out, array![[1.0,], [5.0,], [9.0,]]);
    }

    #[test]
    fn test_rows_iter() {
        let a = array![[1.0, 2.0], [5.0, 8.0], [9.0, 9.0]];
        let rows: Vec<_> = a.rows_iter().collect();
        assert_eq!(
            rows,
            vec![array![1.0, 2.0], array![5.0, 8.0], array![9.0, 9.0]]
        );
    }
}
