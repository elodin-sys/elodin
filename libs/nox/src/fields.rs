//! Defines the `Field` trait for scalar operations and constants, supporting basic arithmetic, matrix multiplication, and associated utilities for numerical types.
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(feature = "xla")]
use xla::Literal;

use crate::{OwnedRepr, Scalar, TensorItem};

pub trait Elem: Copy + Default + 'static {}

impl<T> Elem for T where T: Copy + Default + 'static {}

/// Represents a mathematical field, supporting basic arithmetic operations,
/// matrix multiplication, and the generation of standard constants.

pub trait Field:
    TensorItem<Elem = Self>
    + Copy
    + Elem
    + Mul<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
{
    /// Returns a scalar tensor representing the additive identity (zero).
    fn zero<R: OwnedRepr>() -> Scalar<Self, R>
    where
        Self: Sized;

    /// Returns a scalar tensor representing the multiplicative identity (one).
    fn one<R: OwnedRepr>() -> Scalar<Self, R>
    where
        Self: Sized;

    /// Returns a scalar tensor representing the integer two.
    fn two<R: OwnedRepr>() -> Scalar<Self, R>
    where
        Self: Sized;

    /// Returns the primitive type representing zero.
    fn zero_prim() -> Self
    where
        Self: Sized;

    /// Returns the primitive type representing one.
    fn one_prim() -> Self
    where
        Self: Sized;

    /// Returns the primitive type representing two.
    fn two_prim() -> Self
    where
        Self: Sized;

    #[cfg(feature = "xla")]
    fn literal(self) -> Literal;

    #[cfg(feature = "xla")]
    const ELEMENT_TY: xla::ElementType;
}

pub trait RealField:
    Elem + Field + Neg<Output = Self> + faer::SimpleEntity + faer::ComplexField
{
    fn sqrt(self) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn abs(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn copysign(self, sign: Self) -> Self;
    fn neg_one() -> Self;
    fn acos(self) -> Self;
    fn asin(self) -> Self;
}

#[cfg(feature = "std")]
macro_rules! impl_real_field {
    ($t:ty) => {
        impl RealField for $t {
            fn sqrt(self) -> Self {
                self.sqrt()
            }

            fn cos(self) -> Self {
                self.cos()
            }

            fn sin(self) -> Self {
                self.sin()
            }

            fn abs(self) -> Self {
                self.abs()
            }

            fn atan2(self, other: Self) -> Self {
                self.atan2(other)
            }

            fn max(self, other: Self) -> Self {
                self.max(other)
            }

            fn copysign(self, sign: Self) -> Self {
                self.copysign(sign)
            }

            fn neg_one() -> Self {
                -1.0
            }

            fn acos(self) -> Self {
                self.acos()
            }

            fn asin(self) -> Self {
                self.asin()
            }
        }
    };
}

#[cfg(not(feature = "std"))]
macro_rules! impl_real_field {
    ($t:ty) => {
        impl RealField for $t {
            fn sqrt(self) -> Self {
                libm::Libm::<$t>::sqrt(self)
            }

            fn cos(self) -> Self {
                libm::Libm::<$t>::cos(self)
            }

            fn sin(self) -> Self {
                libm::Libm::<$t>::sin(self)
            }

            fn abs(self) -> Self {
                libm::Libm::<$t>::fabs(self)
            }

            fn atan2(self, other: Self) -> Self {
                libm::Libm::<$t>::atan2(self, other)
            }

            fn max(self, other: Self) -> Self {
                libm::Libm::<$t>::fmax(self, other)
            }

            fn copysign(self, sign: Self) -> Self {
                libm::Libm::<$t>::copysign(self, sign)
            }

            fn neg_one() -> Self {
                -1.0
            }

            fn acos(self) -> Self {
                libm::Libm::<$t>::acos(self)
            }

            fn asin(self) -> Self {
                libm::Libm::<$t>::asin(self)
            }
        }
    };
}

impl_real_field!(f32);
impl_real_field!(f64);

macro_rules! impl_real_closed_field {
    ($t:ty, $zero:tt, $one:tt, $two:tt) => {
        impl Field for $t {
            fn zero<R: OwnedRepr>() -> Scalar<Self, R> {
                let inner = R::scalar_from_const($zero);
                Scalar {
                    inner,
                    phantom: PhantomData,
                }
            }

            fn one<R: OwnedRepr>() -> Scalar<Self, R> {
                let inner = R::scalar_from_const($one);
                Scalar {
                    inner,
                    phantom: PhantomData,
                }
            }

            fn two<R: OwnedRepr>() -> Scalar<Self, R> {
                let inner = R::scalar_from_const($two);
                Scalar {
                    inner,
                    phantom: PhantomData,
                }
            }

            fn zero_prim() -> Self {
                $zero
            }

            fn one_prim() -> Self {
                $one
            }

            fn two_prim() -> Self {
                $two
            }

            #[cfg(feature = "xla")]
            fn literal(self) -> Literal {
                xla::NativeType::literal(self)
            }

            #[cfg(feature = "xla")]
            const ELEMENT_TY: xla::ElementType = <$t as xla::ArrayElement>::TY;
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
