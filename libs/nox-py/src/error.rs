use pyo3::{
    PyErr,
    exceptions::{PyRuntimeError, PyValueError},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Nox(#[from] nox::Error),
    #[error("{0}")]
    PyErr(#[from] PyErr),
    #[error("hlo module was not PyBytes")]
    HloModuleNotBytes,
    #[error("unexpected input")]
    UnexpectedInput,
    #[error("unknown command: {0}")]
    UnknownCommand(String),
    #[error("{0}")]
    MissingArg(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid time step: {0:?}")]
    InvalidTimeStep(std::time::Duration),
    #[error("impeller error {0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("elodin db error {0}")]
    DB(#[from] elodin_db::Error),
    #[error("component not found")]
    ComponentNotFound,
    #[error("component value had wrong size")]
    ValueSizeMismatch,
    #[error("channel closed")]
    ChannelClosed,
    #[error("serde_json {0}")]
    Json(#[from] serde_json::Error),
    #[error("stellarator error {0}")]
    Stellar(#[from] stellarator::Error),
    #[error("arrow error {0}")]
    Arrow(#[from] ::arrow::error::ArrowError),
}

impl From<nox::xla::Error> for Error {
    fn from(value: nox::xla::Error) -> Self {
        Error::Nox(nox::Error::Xla(value))
    }
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::ComponentNotFound => {
                PyValueError::new_err("component not found")
            }
            Error::ValueSizeMismatch => {
                PyValueError::new_err("value size mismatch")
            }
            Error::PyErr(err) => err,
            err => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
