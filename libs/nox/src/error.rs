//! Provides error definitions.
use thiserror::Error;

/// Enumerates possible error types that can occur within the Nox tensor operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Error propagated from underlying XLA library.
    #[cfg(feature = "xla")]
    #[error("xla error {0}")]
    Xla(#[from] xla::Error),

    /// Error indicating that the length of the vmap axis does not match the number of inputs.
    #[error("vmap axis len must be same as input len")]
    WrongAxisLen,

    /// Error when an argument passed to vmap is not batchable.
    #[error("vmap arguments must be batchable")]
    UnbatchableArgument,

    /// Error when vmap is called without any arguments.
    #[error("vmap requires at least one argument")]
    VmapArgsEmpty,

    /// Error when the length of in_axes does not match the number of vmap arguments.
    #[error("vmap requires in axis length to equal arguments length")]
    VmapInAxisMismatch,

    /// Error when a JAX primitive operation encounters incompatible data types.
    #[error("this jaxpr has an incompatible dtype")]
    IncompatibleDType,

    /// Error when attempting to extract a tuple element from a non-tuple type.
    #[error("get tuple element can only be called on a tuple")]
    GetTupleElemWrongType,

    /// Error for out-of-bounds access in indexing operations.
    #[error("out of bounds access")]
    OutOfBoundsAccess,

    /// Error propagated from Python operations via PyO3.
    #[cfg(feature = "jax")]
    #[error("pyo3 error {0}")]
    PyO3(#[from] pyo3::PyErr),

    /// Error when the scan operation does not receive exactly two arguments.
    #[error("scan must have two arguments")]
    ScanWrongArgCount,

    /// Error when no arguments are provided to a scan operation.
    #[error("scan must have at least one input")]
    ScanMissingArg,

    /// Error when the dimensions of all scan arguments do not match.
    #[error("all scan arguments must have the same first dim")]
    ScanShapeMismatch,

    /// Error when matrix inversion failed
    #[error("matrix inversion failed with {0} arg illegal")]
    InvertFailed(i32),
}
