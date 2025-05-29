use super::{ArrayBuf, dynamic::DynArray};
use crate::{Array, ConstDim, Dim, DimGet, Dyn, Elem, Repr, utils::calculate_strides};
use core::marker::PhantomData;
use smallvec::SmallVec;

pub use nox_array::ArrayView;

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

pub trait ArrayViewExt<T: Elem> {
    fn get(&self, index: <Dyn as DimGet>::Index) -> T;
    fn to_dyn_owned(&self) -> Array<T, Dyn>;
    fn try_to_owned<D: ConstDim + Dim>(&self) -> Option<Array<T, D>>;
}

impl<T: Elem> ArrayViewExt<T> for ArrayView<'_, T> {
    /// Retrieves a specific element from the array based on an index, effectively slicing the array.
    fn get(&self, index: <Dyn as DimGet>::Index) -> T {
        let index = <Dyn as DimGet>::index_as_slice(&index);
        let i: usize = calculate_strides(self.shape)
            .iter()
            .zip(index.iter())
            .map(|(s, i)| s * i)
            .sum();
        self.buf[i]
    }

    fn to_dyn_owned(&self) -> Array<T, Dyn> {
        Array {
            buf: DynArray::from_shape_vec(SmallVec::from_slice(self.shape), self.buf.to_vec())
                .unwrap(),
        }
    }

    fn try_to_owned<D: ConstDim + Dim>(&self) -> Option<Array<T, D>> {
        if self.shape != D::DIM {
            return None;
        }
        let mut arr = Array::<T, D>::zeroed(D::DIM);
        arr.buf.as_mut_buf().copy_from_slice(self.buf);
        Some(arr)
    }
}
