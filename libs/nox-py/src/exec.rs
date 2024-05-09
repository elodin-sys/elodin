use crate::*;

use nox_ecs::ColumnStore;
use pyo3_polars::{PyDataFrame, PySeries};

#[pyclass]
pub struct Exec {
    pub exec: nox_ecs::WorldExec,
}

#[pymethods]
impl Exec {
    pub fn run(&mut self, client: &Client) -> Result<(), Error> {
        Python::with_gil(|_| self.exec.run(&client.client).map_err(Error::from))
    }

    pub fn history(&self) -> Result<PyDataFrame, Error> {
        let polars_world = self.exec.history.compact_to_world()?;
        let df = polars_world.join_archetypes()?;
        Ok(PyDataFrame(df))
    }

    fn column_array(&self, name: String) -> Result<PySeries, Error> {
        let id = ComponentId::new(&name);
        let world = self.exec.last_world();
        let series = world.column(id)?.value_series();
        Ok(PySeries(series))
    }
}
