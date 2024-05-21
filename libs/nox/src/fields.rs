//! Defines the `Field` trait for scalar operations and constants, supporting basic arithmetic, matrix multiplication, and associated utilities for numerical types.
use std::ops::{Add, Div, Mul, Sub};

use crate::{Scalar, TensorItem};

/// Represents a mathematical field, supporting basic arithmetic operations,
/// matrix multiplication, and the generation of standard constants.
pub trait Field:
    TensorItem<Elem = Self>
    + Copy
    + Mul<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
    + MatMul
{
    /// Returns a scalar tensor representing the additive identity (zero).
    fn zero() -> Scalar<Self>
    where
        Self: Sized;

    /// Returns a scalar tensor representing the multiplicative identity (one).
    fn one() -> Scalar<Self>
    where
        Self: Sized;

    /// Returns a scalar tensor representing the integer two.
    fn two() -> Scalar<Self>
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

            fn zero_prim() -> Self {
                $zero
            }

            fn one_prim() -> Self {
                $one
            }

            fn two_prim() -> Self {
                $one
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

/// Trait for performing matrix multiplication.
pub trait MatMul {
    /// Perform a matrix multiplication.
    ///
    /// # Safety
    /// Please see [`matrixmultiply::dgemm`] for safety info
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    );
}

impl MatMul for f64 {
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        matrixmultiply::dgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
    }
}

impl MatMul for f32 {
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        matrixmultiply::sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
    }
}

impl MatMul for i16 {
    unsafe fn gemm(
        _m: usize,
        _k: usize,
        _n: usize,
        _alpha: Self,
        _a: *const Self,
        _rsa: isize,
        _csa: isize,
        _b: *const Self,
        _rsb: isize,
        _csb: isize,
        _beta: Self,
        _c: *mut Self,
        _rsc: isize,
        _csc: isize,
    ) {
        todo!()
    }
}

impl MatMul for i32 {
    unsafe fn gemm(
        _m: usize,
        _k: usize,
        _n: usize,
        _alpha: Self,
        _a: *const Self,
        _rsa: isize,
        _csa: isize,
        _b: *const Self,
        _rsb: isize,
        _csb: isize,
        _beta: Self,
        _c: *mut Self,
        _rsc: isize,
        _csc: isize,
    ) {
        todo!()
    }
}

impl MatMul for i64 {
    unsafe fn gemm(
        _m: usize,
        _k: usize,
        _n: usize,
        _alpha: Self,
        _a: *const Self,
        _rsa: isize,
        _csa: isize,
        _b: *const Self,
        _rsb: isize,
        _csb: isize,
        _beta: Self,
        _c: *mut Self,
        _rsc: isize,
        _csc: isize,
    ) {
        todo!()
    }
}

impl MatMul for u16 {
    unsafe fn gemm(
        _m: usize,
        _k: usize,
        _n: usize,
        _alpha: Self,
        _a: *const Self,
        _rsa: isize,
        _csa: isize,
        _b: *const Self,
        _rsb: isize,
        _csb: isize,
        _beta: Self,
        _c: *mut Self,
        _rsc: isize,
        _csc: isize,
    ) {
        todo!()
    }
}

impl MatMul for u32 {
    unsafe fn gemm(
        _m: usize,
        _k: usize,
        _n: usize,
        _alpha: Self,
        _a: *const Self,
        _rsa: isize,
        _csa: isize,
        _b: *const Self,
        _rsb: isize,
        _csb: isize,
        _beta: Self,
        _c: *mut Self,
        _rsc: isize,
        _csc: isize,
    ) {
        todo!()
    }
}

impl MatMul for u64 {
    unsafe fn gemm(
        _m: usize,
        _k: usize,
        _n: usize,
        _alpha: Self,
        _a: *const Self,
        _rsa: isize,
        _csa: isize,
        _b: *const Self,
        _rsb: isize,
        _csb: isize,
        _beta: Self,
        _c: *mut Self,
        _rsc: isize,
        _csc: isize,
    ) {
        todo!()
    }
}
