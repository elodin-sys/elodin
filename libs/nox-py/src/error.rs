use nox;
use pyo3::{
    PyErr,
    exceptions::{PyRuntimeError, PyValueError},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Nox(#[from] nox::Error),
    #[error("component not found")]
    ComponentNotFound,
    #[error("component value had wrong size")]
    ValueSizeMismatch,
    #[error("impeller error: {0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("channel closed")]
    ChannelClosed,
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("serde_json {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    PyO3(#[from] pyo3::PyErr),
    #[error("db {0}")]
    DB(#[from] elodin_db::Error),
    #[error("stellarator error {0}")]
    Stellar(#[from] stellarator::Error),
    #[error("arrow error {0}")]
    Arrow(#[from] ::arrow::error::ArrowError),
    #[error("unexpected input")]
    UnexpectedInput,
    #[error("unknown command: {0}")]
    UnknownCommand(String),
    #[error("{0}")]
    MissingArg(String),
    #[error("invalid time step: {0:?}")]
    InvalidTimeStep(std::time::Duration),
    #[error("invalid log level: {0}")]
    InvalidLogLevel(String),
    #[error("unsupported element type for JAX backend: {0}")]
    UnsupportedDtype(String),
    #[error("cranelift backend: {0}")]
    CraneliftBackend(String),
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::ComponentNotFound => PyValueError::new_err("component not found"),
            Error::ValueSizeMismatch => PyValueError::new_err("value size mismatch"),
            Error::InvalidLogLevel(level) => {
                PyValueError::new_err(format!("invalid log level: {level}"))
            }
            Error::PyO3(err) => err,
            err => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_pyo3_error_preserves_message() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = py
                .run(
                    std::ffi::CString::new("raise RuntimeError('detailed compile error')")
                        .expect("valid python snippet")
                        .as_c_str(),
                    None,
                    None,
                )
                .expect_err("python snippet should fail");
            let display = Error::PyO3(err).to_string();
            assert!(
                display.contains("detailed compile error"),
                "display was: {display}"
            );
        });
    }
}
