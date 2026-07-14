//! Embedded `elodin-db` server: the same `elodin_db::Server` the CLI runs,
//! hosted on a dedicated stellarator thread so tests / notebooks / single-box
//! loggers don't need a separate `elodin-db run` process.

use std::net::SocketAddr;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use stellarator::struc_con::{Joinable, Thread, stellar};

/// An embedded Elodin-DB server.
///
/// `Server(path, addr)` binds `addr` immediately (errors surface here, e.g.
/// port already in use) and serves the database at `path` on a background
/// thread until `stop()` is called or the process exits.
#[pyclass]
pub struct Server {
    thread: Option<Thread<Option<()>>>,
    #[pyo3(get)]
    addr: String,
    #[pyo3(get)]
    path: String,
}

#[pymethods]
impl Server {
    #[new]
    fn new(path: &str, addr: &str) -> PyResult<Self> {
        let sock_addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid address {addr:?}: {e}")))?;
        let server = elodin_db::Server::new(path, sock_addr).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to start elodin-db at {addr}: {e}"))
        })?;
        let thread = stellar(move || async move {
            if let Err(err) = server.run().await {
                tracing::warn!(?err, "embedded elodin-db server exited with error");
            }
        });
        Ok(Self {
            thread: Some(thread),
            addr: addr.to_string(),
            path: path.to_string(),
        })
    }

    /// Stop the server (closes the listener; existing data stays on disk).
    fn stop(&mut self, py: Python<'_>) {
        if let Some(handle) = self.thread.take() {
            py.allow_threads(|| {
                super::block_on(move || async move {
                    let _ = handle.cancel().await;
                });
            });
        }
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, py: Python<'_>, _args: &Bound<'_, pyo3::types::PyTuple>) -> bool {
        self.stop(py);
        false
    }

    fn __repr__(&self) -> String {
        format!("Server(path='{}', addr='{}')", self.path, self.addr)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(handle) = self.thread.take() {
            super::block_on(move || async move {
                let _ = handle.cancel().await;
            });
        }
    }
}
