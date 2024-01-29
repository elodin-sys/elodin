use crate::{Scalar, TensorItem};

pub trait Field: TensorItem {
    fn zero() -> Scalar<Self>
    where
        Self: Sized;
    fn one() -> Scalar<Self>
    where
        Self: Sized;
    fn two() -> Scalar<Self>
    where
        Self: Sized;
}

macro_rules! impl_real_closed_field {
    ($t:ty, $zero:tt, $one:tt, $two:tt) => {
        impl Field for $t {
            fn zero() -> Scalar<Self> {
                use crate::ConstantExt;
                $zero.constant()
            }

            fn one() -> Scalar<Self> {
                use crate::ConstantExt;
                $one.constant()
            }

            fn two() -> Scalar<Self> {
                use crate::ConstantExt;
                $one.constant()
            }
        }
    };
}

impl_real_closed_field!(f32, 0.0, 1.0, 2.0);
impl_real_closed_field!(f64, 0.0, 1.0, 2.0);

impl_real_closed_field!(i16, 0, 1, 2);
impl_real_closed_field!(i32, 0, 1, 2);
impl_real_closed_field!(i64, 0, 1, 2);
impl_real_closed_field!(u16, 0, 1, 2);
impl_real_closed_field!(u32, 0, 1, 2);
impl_real_closed_field!(u64, 0, 1, 2);
