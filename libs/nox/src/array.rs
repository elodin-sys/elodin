//! Provides a local, non-XLA backend for operating on Tensors.
use crate::{ConstDim, RealField};
use nalgebra::{constraint::ShapeConstraint, Const, Dyn};
use smallvec::SmallVec;
use std::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    AddDim, BroadcastDim, BroadcastedDim, DefaultMap, DefaultMappedDim, Dim, DottedDim, Field,
    MapDim, ReplaceMappedDim, Repr, ScalarDim, TensorDim, XlaDim,
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
        let dims = D1::dim(&self.buf);
        let stride = RevStridesIter(D1::strides(&self.buf));
        let mut indexes = dims.clone();
        for index in indexes.as_mut().iter_mut() {
            *index = 0;
        }
        StrideIterator {
            buf: self.buf.as_buf(),
            stride,
            indexes,
            dims,
            phantom: PhantomData,
        }
    }

    pub fn offset_iter(&self, offsets: &[usize]) -> impl Iterator<Item = &'_ T1> {
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
            dims,
            phantom: PhantomData,
        }
    }

    /// Generates an iterator over the elements of the array after broadcasting to new dimensions.
    pub fn broadcast_iter(
        &self,
        new_dims: impl AsMut<[usize]> + AsRef<[usize]> + Clone,
    ) -> Option<impl Iterator<Item = &'_ T1>> {
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
            indexes,
            dims: out_dims,
            phantom: PhantomData,
        })
    }

    /// Performs a dot product between two arrays and returns a new array.
    fn dot<D2>(
        &self,
        right: &Array<T1, D2>,
    ) -> Array<T1, <ShapeConstraint as crate::DotDim<D1, D2>>::Output>
    where
        T1: Field + Copy,
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
    pub fn concat_many<const N: usize>(args: [&Array<T1, D1>; N]) -> Array<T1, ConcatManyDim<D1, N>>
    where
        DefaultMappedDim<D1>: nalgebra::DimMul<Const<N>> + nalgebra::Dim,
        D1::DefaultMapDim: MapDim<D1>,
        D1::DefaultMapDim: MapDim<D1>,
        D1: DefaultMap,
        MulDim<DefaultMappedDim<D1>, Const<N>>: Dim,
        <<D1 as DefaultMap>::DefaultMapDim as MapDim<D1>>::MappedDim: nalgebra::Dim,
        ConcatManyDim<D1, N>: Dim,
    {
        let mut out_dims = D1::dim(&args[0].buf);
        if out_dims.as_ref().is_empty() {
            let mut out: Array<MaybeUninit<T1>, ConcatManyDim<D1, N>> = Array::uninit(&[N]);
            args.into_iter()
                .flat_map(|a| a.buf.as_buf().iter())
                .zip(out.buf.as_mut_buf().iter_mut())
                .for_each(|(a, b)| {
                    b.write(*a);
                });

            unsafe { out.assume_init() }
        } else {
            for arg in args[1..].iter() {
                let d = D1::dim(&arg.buf);
                out_dims.as_mut()[0] += d.as_ref().first().unwrap_or(&1);
            }
            let mut out: Array<MaybeUninit<T1>, ConcatManyDim<D1, N>> =
                Array::uninit(out_dims.as_ref());
            args.into_iter()
                .flat_map(|a| a.buf.as_buf().iter())
                .zip(out.buf.as_mut_buf().iter_mut())
                .for_each(|(a, b)| {
                    b.write(*a);
                });

            unsafe { out.assume_init() }
        }
    }

    /// Retrieves a specific element from the array based on an index, effectively slicing the array.
    pub fn get(&self, index: usize) -> Array<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>,
    {
        let arg_dims = D1::dim(&self.buf);
        let stride_dims = D1::strides(&self.buf);
        let stride = stride_dims.as_ref().last().copied().unwrap();
        let out_dims = &arg_dims.as_ref()[1..];
        let buf = &self.buf.as_buf()[index * stride..];
        let mut out: Array<MaybeUninit<T1>, GetDim<D1>> = Array::uninit(out_dims);
        for (a, b) in buf.iter().zip(out.buf.as_mut_buf().iter_mut()) {
            b.write(*a);
        }
        unsafe { out.assume_init() }
    }

    pub fn copy_fixed_slice<D2: Dim + ConstDim>(&self, offsets: &[usize]) -> Array<T1, D2> {
        let mut out: Array<MaybeUninit<T1>, D2> = Array::uninit(D2::DIM);
        for (a, out) in self
            .offset_iter(offsets)
            .zip(out.buf.as_mut_buf().iter_mut())
        {
            out.write(*a);
        }
        unsafe { out.assume_init() }
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

/// Trait to enable retrieving a specified dimension type from a composite dimension.
pub trait DimGet<D: Dim> {
    type Output: Dim;
}

/// Type alias for the result of a get operation, providing a sliced view of the array.
pub type GetDim<D> = <ShapeConstraint as DimGet<D>>::Output;

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

impl<const N: usize> DimGet<Const<N>> for ShapeConstraint {
    type Output = ();
}

macro_rules! impl_dim_get {
    ($($dim:tt),*) => {
        #[allow(unused_parens)]
        impl<D: Dim, $($dim: Dim),*> DimGet<(D, $($dim,)*)> for ShapeConstraint
        where (D, $($dim,)*): Dim,
        ($($dim,)*): Dim,
        {
            type Output = ($($dim),*);
        }
    };
}

impl_dim_get!(D1);
impl_dim_get!(D1, D2);

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
struct StrideIterator<'a, T, S: StridesIter, I: AsMut<[usize]>, D: AsRef<[usize]>> {
    buf: &'a [T],
    stride: S,
    indexes: I,
    dims: D,
    phantom: PhantomData<&'a T>,
}

impl<'a, T, S: StridesIter, I: AsMut<[usize]>, D: AsRef<[usize]>> Iterator
    for StrideIterator<'a, T, S, I, D>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let indexes = self.indexes.as_mut();
        let dims = self.dims.as_ref();
        let i: usize = indexes
            .iter()
            .zip(self.stride.stride_iter())
            .map(|(&i, s): (&usize, usize)| i * s)
            .sum();
        let mut carry = true;
        for (&dim, index) in dims.iter().zip(indexes.iter_mut()).rev() {
            if carry {
                *index += 1;
            }
            carry = *index >= dim;
            if carry {
                *index = 0;
            }
        }

        self.buf.get(i)
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
        T: Field + Div<Output = T> + Copy,
        D1: Dim + ArrayDim,
        D2: Dim + ArrayDim,
        ShapeConstraint: crate::DotDim<D1, D2>,
        <ShapeConstraint as crate::DotDim<D1, D2>>::Output: Dim + ArrayDim,
    {
        left.dot(right)
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
        Array::concat_many(args)
    }

    fn get<T1: Field, D1: Dim>(
        arg: &Self::Inner<T1, D1>,
        index: usize,
    ) -> Self::Inner<T1, GetDim<D1>>
    where
        ShapeConstraint: DimGet<D1>,
    {
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
        let d: Array<f32, Const<3>> = Array::concat_many([&a, &b, &c]);
        assert_eq!(d.buf, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_transpose() {
        let a: Array<f32, (Const<2>, Const<2>)> = Array {
            buf: [[0.0, 1.0], [2.0, 3.0]],
        };
        let b: Array<f32, (Const<2>, Const<2>)> = a.transpose();
        assert_eq!(b.buf, [[0.0, 2.0], [1.0, 3.0]]);
    }
}
