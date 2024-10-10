use std::marker::PhantomData;

use crate::{utils::calculate_strides, DimGet, Dyn, Elem, Repr};

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
        arg.dim
    }
}

#[derive(Clone, Copy)]
pub struct ArrayView<'a, T> {
    pub(crate) buf: &'a [T],
    pub(crate) dim: &'a [usize],
}

impl<'a, T: Elem> ArrayView<'a, T> {
    /// Retrieves a specific element from the array based on an index, effectively slicing the array.
    pub fn get(&self, index: <Dyn as DimGet>::Index) -> T {
        let index = <Dyn as DimGet>::index_as_slice(&index);
        let i: usize = calculate_strides(self.dim)
            .zip(index.iter())
            .map(|(s, i)| s * i)
            .sum();
        self.buf[i]
    }
}
