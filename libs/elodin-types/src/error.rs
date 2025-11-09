//! Provides error definitions for math operations.
use thiserror::Error;

use alloc::borrow::Cow;

/// Enumerates possible error types that can occur within math operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Error when matrix inversion failed
    #[error("matrix inversion failed with {0} arg illegal")]
    InvertFailed(i32),

    #[error("concat dim failed with dims")]
    InvalidConcatDims,

    /// Error when matrix cholesky failed
    #[error("matrix cholesky failed with {0} arg illegal")]
    Cholesky(#[from] faer::linalg::cholesky::llt::CholeskyError),

    /// faer stack overflow error
    #[error("size overflow")]
    SizeOverflow,

    #[error("expected argument {0:?}")]
    ExpectedArgument(Cow<'static, str>),

    #[error("unsupported {0}")]
    Unsupported(Cow<'static, str>),

    /// Error for out-of-bounds access in indexing operations.
    #[error("out of bounds access")]
    OutOfBoundsAccess,
}
