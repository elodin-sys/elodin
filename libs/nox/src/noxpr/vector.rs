use core::marker::PhantomData;

use crate::ArrayTy;
use crate::Elem;
use crate::Noxpr;
use crate::Op;
use crate::Vector;
use smallvec::smallvec;
use xla::{ArrayElement, NativeType};

impl<T: NativeType + ArrayElement + Elem> Vector<T, 3, Op> {
    /// Extends a 3-dimensional vector to a 4-dimensional vector by appending a given element.
    pub fn extend(&self, elem: T) -> Vector<T, 4, Op> {
        let elem = elem.literal();
        let constant = Noxpr::constant(elem, ArrayTy::new(T::TY, smallvec![1]));
        let inner = Noxpr::concat_in_dim(vec![self.inner.clone(), constant], 0);
        Vector {
            inner,
            phantom: PhantomData,
        }
    }
}
