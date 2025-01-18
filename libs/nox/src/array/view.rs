use core::marker::PhantomData;
use smallvec::SmallVec;
use zerocopy::{Immutable, TryFromBytes};

use crate::{utils::calculate_strides, Array, ConstDim, Dim, DimGet, Dyn, Elem, Repr};

use super::{dynamic::DynArray, ArrayBuf};

pub struct ViewRepr<'a> {
    _phantom: PhantomData<&'a ()>,
}

impl<'a> Repr for ViewRepr<'a> {
    type Inner<T, D: crate::Dim>
        = ArrayView<'a, T>
    where
        T: crate::Elem;

    type Shape<D: crate::Dim> = &'a [usize];

    fn shape<T1: crate::Elem, D1: crate::Dim>(arg: &Self::Inner<T1, D1>) -> Self::Shape<D1> {
        arg.shape
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ArrayView<'a, T> {
    pub(crate) buf: &'a [T],
    pub(crate) shape: &'a [usize],
}

impl<'a, T: Elem> ArrayView<'a, T> {
    pub fn from_bytes_shape_unchecked(buf: &'a [u8], shape: &'a [usize]) -> Option<Self>
    where
        [T]: TryFromBytes + Immutable,
    {
        let count = shape.iter().product();
        let buf = <[T]>::try_ref_from_bytes_with_elems(buf, count).ok()?;
        Some(ArrayView { buf, shape })
    }

    pub fn from_buf_shape_unchecked(buf: &'a [T], shape: &'a [usize]) -> Self {
        ArrayView { buf, shape }
    }

    /// Retrieves a specific element from the array based on an index, effectively slicing the array.
    pub fn get(&self, index: <Dyn as DimGet>::Index) -> T {
        let index = <Dyn as DimGet>::index_as_slice(&index);
        let i: usize = calculate_strides(self.shape)
            .zip(index.iter())
            .map(|(s, i)| s * i)
            .sum();
        self.buf[i]
    }

    pub fn as_bytes(&self) -> &[u8] {
        // Safe because we're only reading the bytes and T is guaranteed to be properly aligned
        unsafe {
            core::slice::from_raw_parts(
                self.buf.as_ptr() as *const u8,
                core::mem::size_of_val(self.buf),
            )
        }
    }

    pub fn to_dyn_owned(&self) -> Array<T, Dyn> {
        Array {
            buf: DynArray::from_shape_vec(SmallVec::from_slice(self.shape), self.buf.to_vec())
                .unwrap(),
        }
    }

    pub fn try_to_owned<D: ConstDim + Dim>(&self) -> Option<Array<T, D>> {
        if self.shape != D::DIM {
            return None;
        }
        let mut arr = Array::<T, D>::zeroed(D::DIM);
        arr.buf.as_mut_buf().copy_from_slice(self.buf);
        Some(arr)
    }

    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    pub fn buf(&self) -> &[T] {
        self.buf
    }
}
