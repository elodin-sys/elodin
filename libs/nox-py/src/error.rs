use pyo3::{
    PyErr,
    exceptions::{PyRuntimeError, PyValueError},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    NoxEcs(#[from] nox_ecs::Error),
    #[error("nox error: {0}")]
    Nox(#[from] nox_ecs::nox::Error),
    #[error("{0}")]
    PyErr(#[from] PyErr),
    #[error("downcast error")]
    Downcast(String),
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
    #[error("invalid log level: {0}")]
    InvalidLogLevel(String),
    #[error("impeller error {0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("elodin db error {0}")]
    DB(#[from] elodin_db::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("stellarator error: {0}")]
    Stellarator(#[from] stellarator::Error),
}

impl From<pyo3::DowncastError<'_, '_>> for Error {
    fn from(err: pyo3::DowncastError<'_, '_>) -> Self {
        Error::Downcast(err.to_string())
    }
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
            Error::InvalidLogLevel(level) => {
                PyValueError::new_err(format!("invalid log level: {level}"))
            }
            Error::NoxEcs(nox_ecs::Error::PyO3(err)) | Error::PyErr(err) => err,
            err => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
