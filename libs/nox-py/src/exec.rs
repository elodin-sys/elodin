use std::collections::HashMap;

use crate::*;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use nox_ecs::Compiled;
use pyo3_polars::{PyDataFrame, PySeries};

#[pyclass]
pub struct Exec {
    pub exec: nox_ecs::WorldExec<Compiled>,
}

#[pymethods]
impl Exec {
    #[pyo3(signature = (ticks=1, show_progress=true))]
    pub fn run(
        &mut self,
        py: Python<'_>,
        ticks: usize,
        mut show_progress: bool,
    ) -> Result<(), Error> {
        show_progress &= ticks >= 100;

        let progress_target = if show_progress {
            ProgressDrawTarget::stderr()
        } else {
            ProgressDrawTarget::hidden()
        };
        let progress_bar = ProgressBar::with_draw_target(Some(ticks as u64), progress_target)
            .with_style(
                ProgressStyle::with_template("{bar:50} {pos:>6}/{len:6} remaining: {eta}").unwrap(),
            );
        for _ in 0..ticks {
            self.exec.run()?;
            py.check_signals()?;
            progress_bar.inc(1);
        }
        progress_bar.finish_and_clear();
        Ok(())
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.exec.profile()
    }

    pub fn write_to_dir(&mut self, path: String) -> Result<(), Error> {
        self.exec.write_to_dir(path).map_err(Error::from)
    }

    pub fn history(&mut self) -> Result<PyDataFrame, Error> {
        let polars_world = self.exec.world.polars()?;
        let df = polars_world.join_archetypes()?;
        Ok(PyDataFrame(df))
    }

    fn column_array(&self, name: String) -> Result<PySeries, Error> {
        let id = ComponentId::new(&name);
        let series = self
            .exec
            .world
            .column_by_id(id)
            .ok_or(nox_ecs::Error::ComponentNotFound)?
            .series()?;
        Ok(PySeries(series))
    }
}
