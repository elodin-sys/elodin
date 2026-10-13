//! `Recording`: read a local database directory without a server, laid out exactly as
//! `elodin-db export --join --mono-us` writes it. Backs `elodin.analysis` for `--db`.

use std::path::PathBuf;

use arrow::ipc::writer::StreamWriter;
use elodin_db::catalog::{self, Read};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(module = "elodin.db", frozen)]
pub struct Recording {
    db: elodin_db::DB,
    path: PathBuf,
}

fn read_request(
    table: String,
    columns: Vec<String>,
    start_us: Option<i64>,
    end_us: Option<i64>,
    nonnull: Option<String>,
) -> Read {
    Read {
        table,
        columns,
        start_us,
        end_us,
        nonnull,
    }
}

#[pymethods]
impl Recording {
    #[staticmethod]
    fn open(path: PathBuf) -> PyResult<Self> {
        let db = elodin_db::DB::open(path.clone())
            .map_err(|e| PyIOError::new_err(format!("open {}: {e}", path.display())))?;
        Ok(Self { db, path })
    }

    #[getter]
    fn path(&self) -> PathBuf {
        self.path.clone()
    }

    /// The export catalog (tables, columns, signals) as JSON.
    fn catalog_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| catalog::catalog(&self.db, true, None))
            .map(|c| c.to_json())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// `seq, time_us, <columns>` as Arrow IPC stream bytes, or None for an unknown table.
    #[pyo3(signature = (table, columns=Vec::new(), start_us=None, end_us=None, nonnull=None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        table: String,
        columns: Vec<String>,
        start_us: Option<i64>,
        end_us: Option<i64>,
        nonnull: Option<String>,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let req = read_request(table, columns, start_us, end_us, nonnull);
        let bytes = py.allow_threads(|| -> Result<Option<Vec<u8>>, String> {
            let Some(batch) = catalog::read(&self.db, true, &req).map_err(|e| e.to_string())?
            else {
                return Ok(None);
            };
            let mut out = Vec::new();
            let mut w =
                StreamWriter::try_new(&mut out, &batch.schema()).map_err(|e| e.to_string())?;
            w.write(&batch).map_err(|e| e.to_string())?;
            w.finish().map_err(|e| e.to_string())?;
            drop(w);
            Ok(Some(out))
        });
        let bytes = bytes.map_err(PyRuntimeError::new_err)?;
        Ok(bytes.map(|b| PyBytes::new(py, &b)))
    }

    /// Rows `read` would return, or None for an unknown table.
    #[pyo3(signature = (table, start_us=None, end_us=None, nonnull=None))]
    fn count(
        &self,
        py: Python<'_>,
        table: String,
        start_us: Option<i64>,
        end_us: Option<i64>,
        nonnull: Option<String>,
    ) -> PyResult<Option<u64>> {
        let req = read_request(table, Vec::new(), start_us, end_us, nonnull);
        py.allow_threads(|| catalog::count(&self.db, true, &req))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Write the deterministic synthetic test recording to `path`.
    #[staticmethod]
    fn write_fixture(path: PathBuf) -> PyResult<()> {
        elodin_db::fixture::write(&path).map_err(|e| PyIOError::new_err(e.to_string()))
    }
}
