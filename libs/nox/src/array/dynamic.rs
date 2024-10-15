use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

use smallvec::SmallVec;

use crate::{ArrayBuf, ArrayDim, Const, Dyn, Elem};

#[derive(Clone, Debug)]
pub struct DynArray<T: Elem, S = Vec<T>> {
    storage: S,
    shape: SmallVec<[usize; 4]>,
    strides: SmallVec<[usize; 4]>,
    _phantom: PhantomData<T>,
}

impl<T: Elem, S> DynArray<T, S> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T: Elem> DynArray<T, Vec<T>> {
    pub fn from_shape_vec(shape: SmallVec<[usize; 4]>, storage: Vec<T>) -> Option<Self> {
        let expected_len: usize = shape.iter().copied().sum();
        if expected_len != storage.len() {
            return None;
        }
        let strides = crate::utils::calculate_strides(&shape).collect::<SmallVec<[usize; 4]>>();
        Some(DynArray {
            storage,
            shape,
            strides,
            _phantom: PhantomData,
        })
    }
}

impl<T: Elem> ArrayBuf<T> for DynArray<T, Vec<T>> {
    fn as_buf(&self) -> &[T] {
        self.storage.as_ref()
    }

    fn as_mut_buf(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    fn default(dims: &[usize]) -> Self {
        let len: usize = dims.iter().copied().sum();
        let strides = crate::utils::calculate_strides(dims).collect::<SmallVec<[usize; 4]>>();

        let shape = SmallVec::from_slice(dims);
        let storage = vec![T::default(); len];

        Self {
            storage,
            strides,
            shape,
            _phantom: PhantomData,
        }
    }
}

impl ArrayDim for Dyn {
    type Buf<T> = DynArray<T> where T: Clone + Elem;

    type Shape = SmallVec<[usize; 4]>;

    fn array_shape<T: Elem>(buf: &Self::Buf<T>) -> Self::Shape {
        buf.shape.clone()
    }

    fn strides<T: Elem>(buf: &Self::Buf<T>) -> Self::Shape {
        buf.strides.clone()
    }

    fn shape_slice<T: Elem>(buf: &Self::Buf<T>) -> &'_ [usize] {
        &buf.shape
    }
}

impl ArrayDim for (Dyn, Dyn) {
    type Buf<T> = DynArray<T> where T: Clone + Elem;

    type Shape = SmallVec<[usize; 4]>;

    fn array_shape<T: Elem>(buf: &Self::Buf<T>) -> Self::Shape {
        buf.shape.clone()
    }

    fn strides<T: Elem>(buf: &Self::Buf<T>) -> Self::Shape {
        buf.strides.clone()
    }

    fn shape_slice<T: Elem>(buf: &Self::Buf<T>) -> &'_ [usize] {
        &buf.shape
    }
}

impl ArrayDim for (Dyn, Dyn, Dyn) {
    type Buf<T> = DynArray<T> where T: Clone + Elem;

    type Shape = SmallVec<[usize; 4]>;

    fn array_shape<T: Elem>(buf: &Self::Buf<T>) -> Self::Shape {
        buf.shape.clone()
    }

    fn strides<T: Elem>(buf: &Self::Buf<T>) -> Self::Shape {
        buf.strides.clone()
    }

    fn shape_slice<T: Elem>(buf: &Self::Buf<T>) -> &'_ [usize] {
        &buf.shape
    }
}

macro_rules! impl_dyn_dim {
    ($($generics: tt),+; $($dim: ty),+) => {
        impl<$(const $generics: usize,)*> ArrayDim for ($($dim,)+) {
            type Buf<T> = DynArray<T> where T: Clone + Elem;

            type Shape = SmallVec<[usize; 4]>;

            fn array_shape<T: Elem>(buf: &Self::Buf<T>) -> Self::Shape {
                buf.shape.clone()
            }

            fn strides<T: Elem>(buf: &Self::Buf<T>) -> Self::Shape {
                buf.strides.clone()
            }

        fn shape_slice<T: Elem>(buf: &Self::Buf<T>) -> &'_ [usize] {
            &buf.shape
        }
        }
    };
}

impl_dyn_dim!(N1; Dyn, Const<N1>);
impl_dyn_dim!(N1; Const<N1>, Dyn);
impl_dyn_dim!(N1; Const<N1>, Dyn, Dyn);
impl_dyn_dim!(N1; Dyn, Const<N1>, Dyn);
impl_dyn_dim!(N1; Dyn, Dyn, Const<N1>);
impl_dyn_dim!(N1, N2; Const<N1>, Const<N2>, Dyn);
impl_dyn_dim!(N1, N2; Const<N1>, Dyn, Const<N2>);
impl_dyn_dim!(N1, N2; Dyn, Const<N1>, Const<N2>);
