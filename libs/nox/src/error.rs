use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("xla error {0}")]
    Xla(#[from] xla::Error),
    #[error("vmap axis len must be same as input len")]
    WrongAxisLen,
    #[error("vmap arguments must be batchable")]
    UnbatchableArgument,
    #[error("vmap requires at least one argument")]
    VmapArgsEmpty,
    #[error("vmap requires in axis length to equal arguments length")]
    VmapInAxisMismatch,
    #[error("this jaxpr has an incompatible dtype")]
    IncompatibleDType,
    #[error("get tuple element can only be called on a tuple")]
    GetTupleElemWrongType,
    #[error("out of bounds access")]
    OutOfBoundsAccess,
    #[error("pyo3 error {0}")]
    PyO3(#[from] pyo3::PyErr),
    #[error("scan must have two arguments")]
    ScanWrongArgCount,
    #[error("scan must have at least one input")]
    ScanMissingArg,
    #[error("all scan arguments must have the same first dim")]
    ScanShapeMismatch,
}
