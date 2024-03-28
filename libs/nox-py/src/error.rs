use nox_ecs::nox;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    PyErr,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Nox(#[from] nox::Error),
    #[error("{0}")]
    NoxEcs(#[from] nox_ecs::Error),
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
    #[error("conduit error {0}")]
    Conduit(#[from] conduit::Error),
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::NoxEcs(nox_ecs::Error::ComponentNotFound) => {
                PyValueError::new_err("component not found")
            }
            Error::NoxEcs(nox_ecs::Error::ValueSizeMismatch) => {
                PyValueError::new_err("value size mismatch")
            }
            Error::NoxEcs(nox_ecs::Error::PyO3(err)) | Error::PyErr(err) => err,
            err => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
