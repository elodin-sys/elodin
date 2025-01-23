use std::collections::HashMap;

use crate::*;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use nox_ecs::Compiled;
use polars::{frame::DataFrame, series::Series};
use pyo3_polars::{PyDataFrame, PySeries};

#[pyclass]
pub struct Exec {
    pub exec: nox_ecs::WorldExec<Compiled>,
    pub db: elodin_db::DB,
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
        nox_ecs::impeller2_server::init_db(&self.db, &mut self.exec.world)?;
        for _ in 0..ticks {
            self.exec.run()?;
            nox_ecs::impeller2_server::commit_world_head(&self.db, &mut self.exec);
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

    pub fn history(
        &mut self,
        component_name: String,
        entity_id: EntityId,
    ) -> Result<PyDataFrame, Error> {
        let id = ComponentId::new(&component_name);
        let component = self
            .db
            .components
            .get(&id)
            .ok_or(elodin_db::Error::ComponentNotFound(id))?;
        let entity_id = entity_id.inner;
        let entity = component
            .entities
            .get(&entity_id)
            .ok_or(elodin_db::Error::EntityNotFound(entity_id))?;
        let series = nox_ecs::polars::to_series(
            entity.time_series.get(..).expect("failed to get data"),
            &entity.schema,
            &component.metadata.load(),
        )?
        .with_name(&component_name);
        let start_tick = entity.time_series.start_tick();
        let tick_series = (start_tick..start_tick + series.len() as u64)
            .collect::<Series>()
            .with_name("tick");
        Ok(PyDataFrame(DataFrame::new(vec![series, tick_series])?))
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
