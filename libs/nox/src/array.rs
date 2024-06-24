//! Provides a local, non-XLA backend for operating on Tensors.
use approx::{AbsDiffEq, RelativeEq};
use nalgebra::{constraint::ShapeConstraint, Const, Dyn};
use smallvec::SmallVec;
use std::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    AddDim, BroadcastDim, BroadcastedDim, ConstDim, DefaultMap, DefaultMappedDim, Dim, DottedDim,
    Error, Field, MapDim, RealField, ReplaceMappedDim, Repr, ScalarDim, TensorDim, XlaDim,
};

/// A struct representing an array with type-safe dimensions and element type.
pub struct Array<T: Copy, D: ArrayDim> {
    pub buf: D::Buf<T>,
}

impl<T: Copy, D: ArrayDim> Clone for Array<T, D>
where
    D::Buf<T>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
        }
    }
}

impl<T: Copy, D: ArrayDim> Copy for Array<T, D> where D::Buf<T>: Copy {}

impl<T1, D1> Default for Array<T1, D1>
where
    T1: Copy,
    D1: ArrayDim,
    D1::Buf<T1>: Default,
{
    fn default() -> Self {
        Self {
            buf: Default::default(),
        }
    }
}

impl<T: Copy, D: ArrayDim> std::fmt::Debug for Array<T, D>
where
    D::Buf<T>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.buf.fmt(f)
    }
}

/// Defines an interface for array dimensions, associating buffer types and dimensionality metadata.
pub trait ArrayDim: TensorDim {
    type Buf<T>: ArrayBuf<T, Uninit = Self::Buf<MaybeUninit<T>>>
    where
        T: Copy;
    type Dim: AsRef<[usize]> + AsMut<[usize]> + Clone;

    /// Returns the dimensions of the buffer.
    fn dim<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim;

    /// Returns the strides of the buffer for multidimensional access.
    fn strides<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim;
}

impl ArrayDim for ScalarDim {
    type Buf<T> = T where T: Copy;

    type Dim = [usize; 0];

    fn dim<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim {
        []
    }

    fn strides<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim {
        []
    }
}

impl<const D: usize> ArrayDim for Const<D> {
    type Buf<T> = [T; D] where T: Copy;

    type Dim = [usize; 1];

    #[inline]
    fn dim<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim {
        [D]
    }

    fn strides<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim {
        [1]
    }
}

impl<const D1: usize, const D2: usize> ArrayDim for (Const<D1>, Const<D2>) {
    type Buf<T> = [[T; D2]; D1] where T: Copy;

    type Dim = [usize; 2];

    fn dim<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim {
        [D1, D2]
    }

    fn strides<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim {
        [D2, 1]
    }
}

impl<const D1: usize, const D2: usize, const D3: usize> ArrayDim
    for (Const<D1>, Const<D2>, Const<D3>)
{
    type Buf<T> = [[[T; D3]; D2]; D1] where T: Copy;
    type Dim = [usize; 3];

    fn dim<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim {
        [D1, D2, D3]
    }

    fn strides<T: Copy>(_buf: &Self::Buf<T>) -> Self::Dim {
        [D3 * D2, D2, 1]
    }
}

/// Provides buffer functionalities for a given type, allowing for safe memory operations.
pub trait ArrayBuf<T>: Clone {
    fn as_buf(&self) -> &[T];
    fn as_mut_buf(&mut self) -> &mut [T];

    type Uninit: ArrayBuf<MaybeUninit<T>>;
    fn uninit(dims: &[usize]) -> Self::Uninit;

    /// Transitions the unitilized buffer to an initialized state
    ///
    /// # Safety
    /// This function will cause undefined behavior
    /// unless you ensure that every value in the underyling buffer is initialized
    /// See [`MaybeUninit::assume_init`] for more information.
    unsafe fn assume_init(uninit: Self::Uninit) -> Self;
}

// Size-heterogeneous transmutation (seriously unsafe!)
// We use this for transmuting a [MaybeUninit<T>; N] to a [T; N]
// without forcing an unnecessary copy.s
// source: https://crates.io/crates/transmute/0.1.1
unsafe fn transmute_no_size_check<A, B>(a: A) -> B {
    debug_assert_eq!(mem::size_of::<A>(), mem::size_of::<B>());
    debug_assert_eq!(mem::align_of::<A>(), mem::align_of::<B>());
    let b = core::ptr::read(&a as *const A as *const B);
    core::mem::forget(a);
    b
}

impl<T: Clone + Copy> ArrayBuf<T> for ndarray::ArrayD<T> {
    fn as_buf(&self) -> &[T] {
        self.as_slice().expect("ndarray in non-standard order")
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        self.as_slice_mut().expect("ndarray in non-standard order")
    }

    type Uninit = ndarray::ArrayD<MaybeUninit<T>>;

    fn uninit(dims: &[usize]) -> Self::Uninit {
        unsafe { ndarray::ArrayD::uninit(dims).assume_init() }
    }

    unsafe fn assume_init(uninit: Self::Uninit) -> Self {
        uninit.assume_init()
    }
}

impl<T: Copy + Clone> ArrayBuf<T> for T {
    fn as_buf(&self) -> &[T] {
        core::slice::from_ref(self)
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        core::slice::from_mut(self)
    }

    type Uninit = MaybeUninit<T>;

    fn uninit(_dims: &[usize]) -> Self::Uninit {
        MaybeUninit::uninit()
    }

    unsafe fn assume_init(uninit: Self::Uninit) -> Self {
        unsafe { uninit.assume_init() }
    }
}

impl<const D: usize, T: Copy + Clone> ArrayBuf<T> for [T; D] {
    fn as_buf(&self) -> &[T] {
        self
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        self
    }

    type Uninit = [MaybeUninit<T>; D];

    fn uninit(_dims: &[usize]) -> Self::Uninit {
        // Safety: This code is inlined from `MaybeUninit::uninit_array()`, and is
        // safe because an unitilized array of `MaybeUninit<T>` is valid.
        unsafe { MaybeUninit::<[MaybeUninit<T>; D]>::uninit().assume_init() }
    }

    unsafe fn assume_init(uninit: Self::Uninit) -> Self {
        unsafe { transmute_no_size_check(uninit) }
    }
}

impl<T: Clone + Copy, const D1: usize, const D2: usize> ArrayBuf<T> for [[T; D1]; D2] {
    fn as_buf(&self) -> &[T] {
        let ptr = self.as_ptr();
        let len = D1 * D2;

        unsafe { std::slice::from_raw_parts(ptr as *const T, len) }
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        let ptr = self.as_ptr();
        let len = D1 * D2;

        unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, len) }
    }

    type Uninit = [[MaybeUninit<T>; D1]; D2];

    fn uninit(_dims: &[usize]) -> Self::Uninit {
        // Safety: This code is similar to the code in `MaybeUnint::unint_array()`, and is
        // safe because an unitilized array of `MaybeUninit<T>` is valid.
        unsafe { MaybeUninit::<[[MaybeUninit<T>; D1]; D2]>::uninit().assume_init() }
    }

    unsafe fn assume_init(uninit: Self::Uninit) -> Self {
        unsafe { transmute_no_size_check(uninit) }
    }
}

impl<T: Clone + Copy, const D1: usize, const D2: usize, const D3: usize> ArrayBuf<T>
    for [[[T; D1]; D2]; D3]
{
    fn as_buf(&self) -> &[T] {
        let ptr = self.as_ptr();
        let len = D1 * D2 * D3;

        unsafe { std::slice::from_raw_parts(ptr as *const T, len) }
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        let ptr = self.as_ptr();
        let len = D1 * D2 * D3;

        unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, len) }
    }

    type Uninit = [[[MaybeUninit<T>; D1]; D2]; D3];

    fn uninit(_dims: &[usize]) -> Self::Uninit {
        unsafe { MaybeUninit::<[[[MaybeUninit<T>; D1]; D2]; D3]>::uninit().assume_init() }
    }

    unsafe fn assume_init(uninit: Self::Uninit) -> Self {
        unsafe { transmute_no_size_check(uninit) }
    }
}

impl ArrayDim for Dyn {
    type Buf<T> = ndarray::ArrayD<T> where T: Clone + Copy;

    type Dim = SmallVec<[usize; 4]>;

    fn dim<T: Copy>(buf: &Self::Buf<T>) -> Self::Dim {
        buf.shape().iter().copied().collect()
    }

    fn strides<T: Copy>(buf: &Self::Buf<T>) -> Self::Dim {
        buf.strides().iter().map(|x| *x as usize).collect()
    }
}

impl<T1: Copy, D1: ArrayDim + TensorDim + XlaDim> Array<MaybeUninit<T1>, D1> {
    /// Constructs an uninitialized array given dimensions.
    pub fn uninit(dims: &[usize]) -> Self {
        Self {
            buf: D1::Buf::<T1>::uninit(dims),
        }
    }

    /// Transitions the array from uninitialized to initialized state, assuming all values are initialized.
    ///
    /// # Safety
    /// This function will cause undefined behavior unless you ensure that every value in the underyling buffer is initialized
    /// See [`MaybeUninit::assume_init`] for more information.
    pub unsafe fn assume_init(self) -> Array<T1, D1> {
        unsafe {
            Array {
                buf: <D1::Buf<T1>>::assume_init(self.buf),
            }
        }
    }
}

macro_rules! impl_op {
    ($op:tt, $op_trait:tt, $fn_name:tt) => {
        impl<T1: Copy, D1: ArrayDim + TensorDim + XlaDim> Array<T1, D1> {
            #[doc = concat!("This function performs the `", stringify!($op_trait), "` operation on two arrays.")]
            pub fn $fn_name<D2: ArrayDim + TensorDim + XlaDim>(
                &self,
                b: &Array<T1, D2>,
            ) -> Array<T1, BroadcastedDim<D1, D2>>
            where
                T1: $op_trait<Output = T1>,
                ShapeConstraint: BroadcastDim<D1, D2>,
                <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim + XlaDim,
            {
                let d1 = D1::dim(&self.buf);
                let d2 = D2::dim(&b.buf);

                match d1.as_ref().len().cmp(&d2.as_ref().len()) {
                    std::cmp::Ordering::Less | std::cmp::Ordering::Equal => {
                        let mut out: Array<MaybeUninit<T1>, BroadcastedDim<D1, D2>> =
                            Array::uninit(d2.as_ref());
                        let mut broadcast_dims = d2.clone();
                        if !cobroadcast_dims(broadcast_dims.as_mut(), d1.as_ref()) {
                            todo!("handle unbroadcastble dims");
                        }
                        for ((a, b), out) in self
                            .broadcast_iter(broadcast_dims.clone())
                            .unwrap()
                            .zip(b.broadcast_iter(broadcast_dims).unwrap())
                            .zip(out.buf.as_mut_buf().iter_mut())
                        {
                            out.write(*a $op *b);
                        }
                        unsafe { out.assume_init() }
                    }
                    std::cmp::Ordering::Greater => {
                        let mut out: Array<MaybeUninit<T1>, BroadcastedDim<D1, D2>> =
                            Array::uninit(d2.as_ref());
                        let mut broadcast_dims = d1.clone();
                        if !cobroadcast_dims(broadcast_dims.as_mut(), d2.as_ref()) {
                            todo!("handle unbroadcastble dims");
                        }
                        for ((b, a), out) in b
                            .broadcast_iter(broadcast_dims.clone())
                            .unwrap()
                            .zip(self.broadcast_iter(broadcast_dims).unwrap())
                            .zip(out.buf.as_mut_buf().iter_mut())
                        {
                            out.write(*a $op *b);
                        }
                        unsafe { out.assume_init() }
                    }
                }
            }

        }
    }
}

impl<T1: Copy, D1: ArrayDim + TensorDim + XlaDim> Array<T1, D1> {
    pub fn reshape<D2: ArrayDim + TensorDim + XlaDim>(&self) -> Array<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        let d1 = D1::dim(&self.buf);

        let mut out: Array<MaybeUninit<T1>, D2> = Array::uninit(d1.as_ref());
        let mut broadcast_dims = d1.clone();
        if !cobroadcast_dims(broadcast_dims.as_mut(), d1.as_ref()) {
            todo!("handle unbroadcastable dims");
        }
        for (a, out) in self
            .broadcast_iter(broadcast_dims)
            .unwrap()
            .zip(out.buf.as_mut_buf().iter_mut())
        {
            out.write(*a);
        }
        unsafe { out.assume_init() }
    }

    pub fn broadcast<D2: ArrayDim + TensorDim + XlaDim>(&self) -> Array<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim + XlaDim,
    {
        let d1 = D1::dim(&self.buf);

        let mut out: Array<MaybeUninit<T1>, BroadcastedDim<D1, D2>> = Array::uninit(d1.as_ref());
        let mut broadcast_dims = d1.clone();
        if !cobroadcast_dims(broadcast_dims.as_mut(), d1.as_ref()) {
            todo!("handle unbroadcastble dims");
        }
        for (a, out) in self
            .broadcast_iter(broadcast_dims)
            .unwrap()
            .zip(out.buf.as_mut_buf().iter_mut())
        {
            out.write(*a);
        }
        unsafe { out.assume_init() }
    }
}

impl_op!(*, Mul, mul);
impl_op!(+, Add, add);
impl_op!(-, Sub, sub);
impl_op!(/, Div, div);

macro_rules! impl_unary_op {
    ($op_trait:tt, $fn_name:tt) => {
        impl<T1: Copy, D1: ArrayDim + TensorDim + XlaDim> Array<T1, D1> {
            pub fn $fn_name(&self) -> Array<T1, D1>
            where
                T1: $op_trait,
            {
                let d1 = D1::dim(&self.buf);
                let mut out: Array<MaybeUninit<T1>, D1> = Array::uninit(d1.as_ref());
                self.buf
                    .as_buf()
                    .iter()
                    .zip(out.buf.as_mut_buf().iter_mut())
                    .for_each(|(a, out)| {
                        out.write($op_trait::$fn_name(*a));
                    });
                unsafe { out.assume_init() }
            }
        }
    };
}

impl_unary_op!(RealField, sqrt);
impl_unary_op!(RealField, sin);
impl_unary_op!(RealField, cos);

impl<T1: Copy, D1: ArrayDim + TensorDim + XlaDim> Array<T1, D1> {
    pub fn neg(&self) -> Array<T1, D1>
    where
        T1: Neg<Output = T1>,
    {
        let d1 = D1::dim(&self.buf);
        let mut out: Array<MaybeUninit<T1>, D1> = Array::uninit(d1.as_ref());
        self.buf
            .as_buf()
            .iter()
            .zip(out.buf.as_mut_buf().iter_mut())
            .for_each(|(a, out)| {
                out.write(-*a);
            });
        unsafe { out.assume_init() }
    }

    pub fn transpose(&self) -> Array<T1, TransposedDim<D1>>
    where
        ShapeConstraint: TransposeDim<D1>,
        TransposedDim<D1>: ConstDim,
    {
        let mut out: Array<MaybeUninit<T1>, TransposedDim<D1>> =
            Array::uninit(<TransposedDim<D1> as ConstDim>::DIM);
        self.transpose_iter()
            .zip(out.buf.as_mut_buf().iter_mut())
            .for_each(|(a, out)| {
                out.write(*a);
            });
        unsafe { out.assume_init() }
    }
}

impl<T1: Copy, D1: ArrayDim + TensorDim + XlaDim> Array<T1, D1> {
    pub fn transpose_iter(&self) -> impl Iterator<Item = &'_ T1> {
        let mut dims = D1::dim(&self.buf);
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
        let dims = D1::dim(&self.buf);
        let stride = D1::strides(&self.buf);
        let mut indexes = dims.clone();
        for (offset, index) in offsets
            .iter()
            .copied()
            .chain(std::iter::repeat(0))
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
        let dims = D1::dim(&self.buf);
        let stride = D1::strides(&self.buf);
        let mut indexes = dims.clone();
        for (offset, index) in offsets
            .iter()
            .copied()
            .chain(std::iter::repeat(0))
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
        let existing_dims = D1::dim(&self.buf);
        let existing_strides = D1::strides(&self.buf);
        let mut new_strides = new_dims.clone();
        let out_dims = new_dims.clone();
        let mut indexes = new_dims.clone();
        for i in indexes.as_mut().iter_mut() {
            *i = 0
        }
        for (i, ((dim, existing_stride), new_dim)) in existing_dims
            .as_ref()
            .iter()
            .zip(existing_strides.as_ref().iter())
            .zip(new_dims.as_ref().iter())
            .enumerate()
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
        T1: RealField + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: crate::DotDim<D1, D2>,
        <ShapeConstraint as crate::DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        let left = self;
        let dim_left = D1::dim(&left.buf);
        let stride_left = D1::strides(&left.buf);
        let dim_right = D2::dim(&right.buf);
        let m = dim_left.as_ref().first().copied().unwrap_or(0);
        let k = dim_right.as_ref().first().copied().unwrap_or(0);
        let n = dim_right.as_ref().get(1).copied().unwrap_or(1);
        let stride_right = D2::strides(&right.buf);
        let (dims, rank) = matmul_dims(dim_left.as_ref(), dim_right.as_ref()).unwrap();
        let dims = &dims[..rank];
        let mut out: Array<MaybeUninit<T1>, DottedDim<D1, D2>> = Array::uninit(dims);
        let stride_out = <crate::DottedDim<D1, D2>>::strides(&out.buf);
        let alpha = T1::one_prim();
        let a = left.buf.as_buf().as_ref().as_ptr();
        let b = right.buf.as_buf().as_ref().as_ptr();
        let rsa = stride_left.as_ref().first().copied().unwrap_or(1) as isize;
        let csa = stride_left.as_ref().get(1).copied().unwrap_or(1) as isize;
        let rsb = stride_right.as_ref().first().copied().unwrap_or(1) as isize;
        let csb = stride_right.as_ref().get(1).copied().unwrap_or(1) as isize;
        let c = out.buf.as_mut_buf().as_mut().as_mut_ptr();
        let rsc = stride_out.as_ref().first().copied().unwrap_or(1) as isize;
        let csc = stride_out.as_ref().get(1).copied().unwrap_or(1) as isize;

        unsafe {
            T1::gemm(
                m,
                k,
                n,
                alpha,
                a,
                rsa,
                csa,
                b,
                rsb,
                csb,
                T1::zero_prim(),
                c as *mut T1,
                rsc,
                csc,
            );
            out.assume_init()
        }
    }

    /// Concatenates two arrays along the first dimension.
    pub fn concat<D2: Dim + DefaultMap>(
        &self,
        right: &Array<T1, D2>,
    ) -> Array<T1, ConcatDim<D1, D2>>
    where
        DefaultMappedDim<D1>: nalgebra::DimAdd<DefaultMappedDim<D2>> + nalgebra::Dim,
        DefaultMappedDim<D2>: nalgebra::Dim,
        D2::DefaultMapDim: MapDim<D1>,
        D1::DefaultMapDim: MapDim<D2>,
        D1: DefaultMap,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatDim<D1, D2>: Dim,
    {
        let d1 = D1::dim(&self.buf);
        let d2 = D2::dim(&right.buf);
        let mut out_dims = d2.clone();
        assert_eq!(d1.as_ref().len(), d2.as_ref().len());
        out_dims.as_mut()[0] = d1.as_ref()[0] + d2.as_ref()[0];
        let mut out: Array<MaybeUninit<T1>, ConcatDim<D1, D2>> = Array::uninit(out_dims.as_ref());
        self.buf
            .as_buf()
            .iter()
            .chain(right.buf.as_buf().iter())
            .zip(out.buf.as_mut_buf().iter_mut())
            .for_each(|(a, b)| {
                b.write(*a);
            });
        unsafe { out.assume_init() }
    }

    /// Concatenates multiple arraysinto a single array along a specified dimension.
    pub fn concat_many<D2: Dim>(
        args: &[Array<T1, D1>],
        dim: usize,
    ) -> Result<Array<T1, D2>, Error> {
        let mut out_dims = D1::dim(&args[0].buf);
        if out_dims.as_ref().is_empty() {
            if dim != 0 {
                return Err(Error::InvalidConcatDims);
            }
            let mut out: Array<MaybeUninit<T1>, D2> = Array::uninit(&[args.len()]);
            args.iter()
                .flat_map(|a| a.buf.as_buf().iter())
                .zip(out.buf.as_mut_buf().iter_mut())
                .for_each(|(a, b)| {
                    b.write(*a);
                });

            Ok(unsafe { out.assume_init() })
        } else {
            for x in out_dims.as_mut() {
                *x = 0;
            }
            for arg in args {
                let d = D1::dim(&arg.buf);
                if d.as_ref().len() != out_dims.as_ref().len() {
                    return Err(Error::InvalidConcatDims);
                }
                for (i, (a, b)) in out_dims
                    .as_mut()
                    .iter_mut()
                    .zip(d.as_ref().iter())
                    .enumerate()
                {
                    if i != dim {
                        *a = *b
                    } else {
                        *a += b;
                    }
                }
            }
            let mut out: Array<MaybeUninit<T1>, D2> = Array::uninit(out_dims.as_ref());
            let mut current_offsets = out_dims.clone();
            let current_offsets = current_offsets.as_mut();
            current_offsets.iter_mut().for_each(|a| *a = 0);
            if dim >= current_offsets.len() {
                return Err(Error::InvalidConcatDims);
            }
            for arg in args.iter() {
                let d = D1::dim(&arg.buf);
                if d.as_ref().len() != current_offsets.len() {
                    return Err(Error::InvalidConcatDims);
                }
                for (i, (a, b)) in out_dims.as_ref().iter().zip(d.as_ref().iter()).enumerate() {
                    if dim != i && a != b {
                        return Err(Error::InvalidConcatDims);
                    }
                }
                for (i, a) in current_offsets.iter_mut().enumerate() {
                    if i != dim {
                        *a = 0;
                    }
                }
                let offset = d.as_ref()[dim];
                let iter = out.offset_iter_mut(current_offsets.as_ref());
                let iter = StrideIteratorMut {
                    buf: iter.buf,
                    stride: iter.stride,
                    indexes: iter.indexes,
                    offsets: &current_offsets,
                    dims: d,
                    phantom: PhantomData,
                    bump_index: false,
                };
                iter.zip(arg.buf.as_buf().iter()).for_each(|(a, b)| {
                    a.write(*b);
                });
                current_offsets[dim] += offset;
            }
            if current_offsets[dim] != out_dims.as_ref()[dim] {
                return Err(Error::InvalidConcatDims);
            }

            Ok(unsafe { out.assume_init() })
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
        let mut out: Array<MaybeUninit<T1>, D2> = Array::uninit(D2::DIM);
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
            out.write(*a);
        }
        unsafe { out.assume_init() }
    }

    pub fn try_lu_inverse_mut(&mut self) -> Result<(), Error>
    where
        T1: RealField,
        D1: SquareDim,
    {
        let mut work: Array<MaybeUninit<T1>, D1> = Array::uninit(D1::dim(&self.buf).as_ref());
        work.buf.as_mut_buf().iter_mut().for_each(|a| {
            a.write(T1::zero_prim());
        });
        let mut work = unsafe { work.assume_init() };
        let mut ipiv = D1::ipiv(&self.buf);
        let mut info = 0;
        let n = D1::order(&self.buf) as i32;
        unsafe {
            T1::getrf(n, n, self.buf.as_mut_buf(), n, ipiv.as_mut(), &mut info);
        }
        if info < 0 {
            return Err(Error::InvertFailed(-info));
        }
        unsafe {
            T1::getri(
                n,
                self.buf.as_mut_buf(),
                n,
                ipiv.as_mut(),
                work.buf.as_mut_buf(),
                n * n,
                &mut info,
            );
        }
        if info < 0 {
            return Err(Error::InvertFailed(-info));
        }
        Ok(())
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

    pub fn from_scalars(iter: impl IntoIterator<Item = Array<T1, ()>>) -> Self
    where
        D1: ConstDim,
        T1: Field,
    {
        let mut out: Array<MaybeUninit<T1>, D1> = Array::uninit(D1::DIM);
        out.buf
            .as_mut_buf()
            .iter_mut()
            .zip(
                iter.into_iter()
                    .map(|a| a.buf)
                    .chain(std::iter::repeat(T1::zero_prim())),
            )
            .for_each(|(a, b)| {
                a.write(b);
            });
        unsafe { out.assume_init() }
    }

    pub fn eye() -> Self
    where
        D1: SquareDim + ConstDim,
        T1: Field,
    {
        let mut out: Array<MaybeUninit<T1>, D1> = Array::uninit(D1::DIM);
        let len = out.buf.as_buf().len();
        out.offset_iter_mut(&[0, 0, 0])
            .enumerate()
            .take(len)
            .for_each(|(i, a)| {
                let i = i.as_ref();
                if i[0] == i[1] {
                    a.write(T1::one_prim());
                } else {
                    a.write(T1::zero_prim());
                }
            });
        unsafe { out.assume_init() }
    }

    pub fn from_diag(diag: Array<T1, D1::SideDim>) -> Self
    where
        D1: SquareDim + ConstDim,
        T1: Field,
    {
        let mut out: Array<MaybeUninit<T1>, D1> = Array::uninit(D1::DIM);
        let len = out.buf.as_buf().len();
        out.offset_iter_mut(&[0, 0, 0])
            .enumerate()
            .take(len)
            .for_each(|(i, a)| {
                let i = i.as_ref();
                if i[0] == i[1] {
                    a.write(diag.buf.as_buf()[i[0]]);
                } else {
                    a.write(T1::zero_prim());
                }
            });
        unsafe { out.assume_init() }
    }
}

pub trait SquareDim: ArrayDim {
    type SideDim: Dim;
    type IPIV: AsMut<[i32]>;
    fn ipiv<T: Copy>(_buf: &Self::Buf<T>) -> Self::IPIV;
    fn order<T: Copy>(buf: &Self::Buf<T>) -> usize;
}

impl SquareDim for Dyn {
    type SideDim = Dyn;
    type IPIV = Vec<i32>;

    fn ipiv<T: Copy>(buf: &Self::Buf<T>) -> Self::IPIV {
        let n = buf.shape()[0];
        vec![0; n]
    }

    fn order<T: Copy>(buf: &Self::Buf<T>) -> usize {
        buf.shape()[0]
    }
}

impl<const N: usize> SquareDim for (Const<N>, Const<N>) {
    type SideDim = Const<N>;
    type IPIV = [i32; N];

    fn ipiv<T: Copy>(_buf: &Self::Buf<T>) -> Self::IPIV {
        [0; N]
    }

    fn order<T: Copy>(_buf: &Self::Buf<T>) -> usize {
        N
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
pub type MulDim<A, B> = <A as nalgebra::DimMul<B>>::Output;

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

    fn get<T: Copy>(index: Self::Index, buf: &Self::Buf<T>) -> T;

    type IndexArray: AsRef<[usize]>;
    fn index_as_array(index: Self::Index) -> Self::IndexArray;
}

impl<const N: usize> DimGet for Const<N> {
    type Index = usize;

    fn get<T: Copy>(index: Self::Index, buf: &Self::Buf<T>) -> T {
        buf[index]
    }

    type IndexArray = [usize; 1];
    fn index_as_array(index: Self::Index) -> Self::IndexArray {
        [index]
    }
}

impl<const A: usize, const B: usize> DimGet for (Const<A>, Const<B>) {
    type Index = (usize, usize);

    fn get<T: Copy>((a, b): Self::Index, buf: &Self::Buf<T>) -> T {
        buf[a][b]
    }

    type IndexArray = [usize; 2];
    fn index_as_array(index: Self::Index) -> Self::IndexArray {
        [index.0, index.1]
    }
}

fn cobroadcast_dims(output: &mut [usize], other: &[usize]) -> bool {
    for (output, other) in output.iter_mut().zip(other.iter()) {
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
        // let indexes = self.indexes.as_mut();
        // let dims = self.dims.as_ref();
        // let i: usize = indexes
        //     .iter()
        //     .zip(self.stride.stride_iter())
        //     .map(|(&i, s): (&usize, usize)| i * s)
        //     .sum();
        // let mut carry = true;
        // for (&dim, index) in dims.iter().zip(indexes.iter_mut()).rev() {
        //     if carry {
        //         *index += 1;
        //     }
        //     carry = *index >= dim;
        //     if carry {
        //         *index = 0;
        //     }
        // }

        // self.buf.get(i)
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
        unsafe { std::mem::transmute(self.buf.get_mut(i)) }
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
                unsafe { std::mem::transmute::<&I, &'a I>(&self.0.indexes) },
                x,
            )
        })
    }
}

/// Backend implementation for local computation on arrays.
pub struct ArrayRepr;

impl Repr for ArrayRepr {
    type Inner<T, D: Dim> = Array<T, D> where T: Copy;

    fn add<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Add<Output = T> + Copy,
        D1: ArrayDim + TensorDim + XlaDim,
        D2: ArrayDim + TensorDim + XlaDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim + XlaDim,
    {
        left.add(right)
    }

    fn sub<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Sub<Output = T> + Copy,
        D1: crate::Dim + ArrayDim,
        D2: crate::Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: crate::Dim + ArrayDim,
    {
        left.sub(right)
    }

    fn mul<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Mul<Output = T> + Copy,
        D1: crate::Dim + ArrayDim,
        D2: crate::Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: crate::Dim + ArrayDim,
    {
        left.mul(right)
    }

    fn div<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, BroadcastedDim<D1, D2>>
    where
        T: Div<Output = T> + Copy,
        D1: crate::Dim + ArrayDim,
        D2: crate::Dim + ArrayDim,
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: crate::Dim + ArrayDim,
    {
        left.div(right)
    }

    fn dot<T, D1, D2>(
        left: &Self::Inner<T, D1>,
        right: &Self::Inner<T, D2>,
    ) -> Self::Inner<T, <ShapeConstraint as crate::DotDim<D1, D2>>::Output>
    where
        T: RealField + Div<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: crate::DotDim<D1, D2>,
        <ShapeConstraint as crate::DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        left.dot(right)
    }

    fn concat_many<T1: Field, D1: Dim, D2: Dim>(
        args: &[Self::Inner<T1, D1>],
        dim: usize,
    ) -> Self::Inner<T1, D2> {
        Array::concat_many(args, dim).unwrap()
    }

    fn get<T1: Field, D1: Dim + DimGet>(
        arg: &Self::Inner<T1, D1>,
        index: D1::Index,
    ) -> Self::Inner<T1, ()> {
        arg.get(index)
    }

    fn broadcast<D1: Dim, D2: ArrayDim + TensorDim + XlaDim, T1: Field>(
        arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, BroadcastedDim<D1, D2>>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
        <ShapeConstraint as BroadcastDim<D1, D2>>::Output: ArrayDim + XlaDim,
    {
        arg.broadcast()
    }

    fn scalar_from_const<T1: Field>(value: T1) -> Self::Inner<T1, ()> {
        Array { buf: value }
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
        D1: Dim + DefaultMap,
        AddDim<DefaultMappedDim<D1>, DefaultMappedDim<D2>>: Dim,
        <<D2 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
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

    fn sin<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.sin()
    }

    fn cos<T1: Field + RealField, D1: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D1> {
        arg.cos()
    }

    fn copy_fixed_slice<T1: Field, D1: Dim, D2: Dim + ConstDim>(
        arg: &Self::Inner<T1, D1>,
        offsets: &[usize],
    ) -> Self::Inner<T1, D2> {
        arg.copy_fixed_slice(offsets)
    }

    fn reshape<T1: Field, D1: Dim, D2: Dim>(arg: &Self::Inner<T1, D1>) -> Self::Inner<T1, D2>
    where
        ShapeConstraint: BroadcastDim<D1, D2>,
    {
        arg.reshape()
    }

    fn try_lu_inverse<T1: RealField, D1: Dim + SquareDim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Result<Self::Inner<T1, D1>, Error> {
        arg.try_lu_inverse()
    }

    fn from_scalars<T1: Field, D1: Dim + ConstDim>(
        iter: impl IntoIterator<Item = Self::Inner<T1, ()>>,
    ) -> Self::Inner<T1, D1> {
        Array::from_scalars(iter)
    }

    fn transpose<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
    ) -> Self::Inner<T1, TransposedDim<D1>>
    where
        ShapeConstraint: TransposeDim<D1>,
        TransposedDim<D1>: ConstDim,
    {
        arg.transpose()
    }

    fn eye<T1: Field, D1: Dim + SquareDim + ConstDim>() -> Self::Inner<T1, D1> {
        Array::eye()
    }

    fn from_diag<T1: Field, D1: Dim + SquareDim + ConstDim>(
        diag: Self::Inner<T1, D1::SideDim>,
    ) -> Self::Inner<T1, D1> {
        Array::from_diag(diag)
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

impl<T: Copy + PartialEq, D: Dim> PartialEq for Array<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.buf.as_buf() == other.buf.as_buf()
    }
}

impl<T: AbsDiffEq, D: Dim> AbsDiffEq for Array<T, D>
where
    T: Copy,

    T::Epsilon: Copy,
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
    T: Copy,
    T::Epsilon: Copy,
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

        let a: Array<f32, Const<1>> = Array { buf: [1.0] };
        let b: Array<f32, Const<1>> = Array { buf: [2.0] };
        let c: Array<f32, Const<1>> = Array { buf: [3.0] };
        let d: Array<f32, Const<3>> = Array::concat_many(&[a, b, c], 0).unwrap();
        assert_eq!(d.buf, [1.0, 2.0, 3.0]);
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
        let c: Array<f32, (Const<2>, Const<4>)> = Array::concat_many(&[a, b], 1).unwrap();
        assert_eq!(c, array![[0.0, 1.0, 0.0, 1.0], [2.0, 3.0, 2.0, 3.0]]);
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

    // #[test]
    // fn test_lu_inverse() {
    //     let mut a = array![[1.0, 2.0], [3.0, 4.0]];
    //     a.try_lu_inverse_mut().unwrap();
    //     assert_eq!(a, array![[-2.0, 1.0], [1.5, -0.5]]);

    //     let mut a = array![[1.0, 0.0], [0.0, 1.0]];
    //     a.try_lu_inverse_mut().unwrap();
    //     assert_eq!(a, array![[1.0, 0.0], [0.0, 1.0]]);

    //     #[rustfmt::skip]
    //     let mut a: Array<f64, (Const<10>, Const<10>)> = array![
    //         [24570.1, 76805.1, 43574.6, 18894.2, -7640.97, 2261.34, 22776.7, 24861.4, 81641., 34255.6],
    //         [12354.1, 78957.6, 5642.45, -4702.81, 63301.7, 35105.5, 35568.9, 58708.1, 45157., 65454.5],
    //         [8302.27, 65510.6, 20473.5, 55808.1, 39832.5, 92954.5, 79581.3, 35383.3, 96110.3, 34361.5],
    //         [30932.6, 67202.2, 21617.7, 75088.7, 71295.8, 42937.1, 26957.5, 59796.5, 35418.6, 26217.4],
    //         [77307.7, 39452.7, 75145.5, 44098.6, 12566.9, 16471.8, 71774.6, -4106.6, 53838.2, 36685.3],
    //         [83757.9, 17360., 8921.7, 65612.8, 90126.7, 86641.8, 21293.4, 20590.5, 13033.9, 76379.3],
    //         [83768.9, 46348.9, 16581.3, 31374.9, 9137.27, 37604.4, 32564., 15644.9, -4805.73, 49756.],
    //         [12081.9, 85443.3, 88681.9, 64841.1, 51603.8, 53034.5, 7805.68, 39358.2, -140.273, 84237.4],
    //         [40253.6, 69906.9, 38533.1, 60614., 57636.5, 82128.6, 68686.8, 37255.3, 33246.1, 52798.4],
    //         [16576.6, 37261.4, 38658.7, 91431.4, 40354.5, 9395.03, 62509.4, 28617.7, 33828.6, 60181.7]
    //     ];
    //     a.try_lu_inverse_mut().unwrap();

    //     #[rustfmt::skip]
    //     let  expected_inverse: Array<f64, (Const<10>, Const<10>)> = array![
    //         [-0.0000100226, 3.34899e-6, 8.88045e-6, 9.55701e-6, 0.0000103527, -4.59573e-7, 0.000013572, 2.24482e-6, -0.0000248319, -5.47633e-6],
    //         [0.0000858703, -0.000028404, -0.0000867496, -0.0000302066, -0.0000451287, 0.0000168553, -0.0000504144, -0.000040719, 0.000163731, 5.85461e-6],
    //         [-0.000043415, 0.0000132544, 0.0000398965, 0.0000175089, 0.0000308841, -0.0000127083, 0.0000196943, 0.0000276474, -0.0000772964, -9.97593e-6],
    //         [0.0000195235, -0.0000152707, -0.0000151017, -2.35467e-6, -0.0000144411, 5.44407e-6, -8.03035e-6, -8.26637e-6, 0.0000293996, 9.4548e-6],
    //         [0.0000395083, -9.01866e-6, -0.0000518759, -0.0000126334, -0.0000150054, 0.0000164758, -0.0000424713, -0.0000239907, 0.0000884494, 1.77583e-6],
    //         [-0.0000213967, 2.21527e-6, 0.0000281531, 4.94374e-6, 6.72697e-6, -3.5336e-6, 0.000015341, 0.0000136699, -0.0000350585, -9.13423e-6],
    //         [-9.60105e-6, 5.4443e-6, 7.88212e-7, -3.35387e-6, 5.25609e-6, -7.54644e-6, 2.13287e-6, -4.36139e-6, 6.954e-6, 5.16871e-6],
    //         [-0.000125344, 0.0000464253, 0.000126716, 0.0000579426, 0.0000603612, -0.0000382841, 0.0000876006, 0.0000614865, -0.000240183, -0.0000127178],
    //         [9.00549e-7, 2.22589e-6, 0.0000120239, 2.6913e-6, 5.10871e-6, 3.9984e-6, -1.12099e-6, 1.35222e-6, -0.0000214447, -1.312e-6],
    //         [-6.26086e-6, 8.06916e-6, 0.0000111363, -9.48246e-6, -8.31803e-7, 2.88266e-6, 9.86769e-6, 7.87647e-6, -0.0000243396, 8.19543e-6]
    //     ];
    //     assert_relative_eq!(a, expected_inverse, epsilon = 1e-4);
    // }
}
