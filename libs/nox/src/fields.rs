//! Defines the `Field` trait for scalar operations and constants, supporting basic arithmetic, matrix multiplication, and associated utilities for numerical types.
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(feature = "xla")]
use xla::Literal;

use crate::{Repr, Scalar, TensorItem};

/// Represents a mathematical field, supporting basic arithmetic operations,
/// matrix multiplication, and the generation of standard constants.
pub trait Field:
    TensorItem<Elem = Self>
    + Copy
    + Mul<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
{
    /// Returns a scalar tensor representing the additive identity (zero).
    fn zero<R: Repr>() -> Scalar<Self, R>
    where
        Self: Sized;

    /// Returns a scalar tensor representing the multiplicative identity (one).
    fn one<R: Repr>() -> Scalar<Self, R>
    where
        Self: Sized;

    /// Returns a scalar tensor representing the integer two.
    fn two<R: Repr>() -> Scalar<Self, R>
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

pub trait RealField: Field + Neg<Output = Self> + MatMul + LU + Cholskey {
    fn sqrt(self) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn abs(self) -> Self;
    fn atan2(self, other: Self) -> Self;
}

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
        }
    };
}

impl_real_field!(f32);
impl_real_field!(f64);

macro_rules! impl_real_closed_field {
    ($t:ty, $zero:tt, $one:tt, $two:tt) => {
        impl Field for $t {
            fn zero<R: Repr>() -> Scalar<Self, R> {
                let inner = R::scalar_from_const($zero);
                Scalar {
                    inner,
                    phantom: PhantomData,
                }
            }

            fn one<R: Repr>() -> Scalar<Self, R> {
                let inner = R::scalar_from_const($one);
                Scalar {
                    inner,
                    phantom: PhantomData,
                }
            }

            fn two<R: Repr>() -> Scalar<Self, R> {
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

pub trait LU: Sized {
    /// # Safety
    /// When using these functions you need to ensure that the n, lda, work, and ipiv values are correct
    /// or else there could be weird memory issues.
    unsafe fn getrf(m: i32, n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
    /// # Safety
    /// When using these functions you need to ensure that the n, lda, work, and ipiv values are correct
    /// or else there could be weird memory issues.
    unsafe fn getri(
        n: i32,
        a: &mut [Self],
        lda: i32,
        ipiv: &[i32],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );
}

impl LU for f64 {
    unsafe fn getrf(m: i32, n: i32, a: &mut [f64], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        lapack::dgetrf(m, n, a, lda, ipiv, info)
    }

    unsafe fn getri(
        n: i32,
        a: &mut [Self],
        lda: i32,
        ipiv: &[i32],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    ) {
        lapack::dgetri(n, a, lda, ipiv, work, lwork, info)
    }
}

impl LU for f32 {
    unsafe fn getrf(m: i32, n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        lapack::sgetrf(m, n, a, lda, ipiv, info)
    }

    unsafe fn getri(
        n: i32,
        a: &mut [Self],
        lda: i32,
        ipiv: &[i32],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    ) {
        lapack::sgetri(n, a, lda, ipiv, work, lwork, info)
    }
}

/// Trait for getting the choleskey decomposition of a matrix
pub trait Cholskey: Sized {
    /// See [`lapack::dpotrf`] or [`lapack::spotrf`] for more information
    /// # Safety
    /// When using these functions you need to ensure that the n, lda, and uplo values are correct
    /// or else there could be weird memory issues.
    unsafe fn potrf(uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32);
}

impl Cholskey for f64 {
    unsafe fn potrf(uplo: u8, n: i32, a: &mut [f64], lda: i32, info: &mut i32) {
        // Because the LAPACK functions are written in Fortran,
        // they expect the input to be in column-major order,
        // but we use row-major order.
        // So we need to ask LAPACk to use the opposite triangle.
        let uplo = if uplo == b'U' { b'L' } else { b'U' };
        lapack::dpotrf(uplo, n, a, lda, info)
    }
}

impl Cholskey for f32 {
    unsafe fn potrf(uplo: u8, n: i32, a: &mut [f32], lda: i32, info: &mut i32) {
        // Because the LAPACK functions are written in Fortran,
        // they expect the input to be in column-major order,
        // but we use row-major order.
        // So we need to ask LAPACk to use the opposite triangle.
        let uplo = if uplo == b'U' { b'L' } else { b'U' };
        lapack::spotrf(uplo, n, a, lda, info)
    }
}
