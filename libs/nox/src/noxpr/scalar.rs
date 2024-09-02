use std::{marker::PhantomData, ops::Add};

use crate::{Field, Repr, Scalar};

impl<T: Field, R: Repr> Add<T> for Scalar<T, R> {
    type Output = Scalar<T, R>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = R::scalar_from_const(rhs);
        Scalar {
            inner: R::add(&self.inner, &rhs),
            phantom: PhantomData,
        }
    }
}
