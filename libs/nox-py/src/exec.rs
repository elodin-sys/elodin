use std::collections::HashMap;

use crate::*;

use impeller2::types::Timestamp;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use nox_ecs::Compiled;

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
        let mut timestamp = Timestamp::now();
        nox_ecs::impeller2_server::init_db(&self.db, &mut self.exec.world, timestamp)?;
        for _ in 0..ticks {
            self.exec.run()?;
            nox_ecs::impeller2_server::commit_world_head(&self.db, &mut self.exec, timestamp)?;
            timestamp += self.exec.world.sim_time_step().0;
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

    pub fn history<'a>(
        &mut self,
        py: Python<'a>,
        component_name: String,
        entity_id: EntityId,
    ) -> Result<Bound<'a, PyAny>, Error> {
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

        let temp_file = tempfile::NamedTempFile::with_suffix(".feather")?;
        let path = temp_file.path().to_owned();
        nox_ecs::arrow::write_ipc(
            entity
                .time_series
                .get_range(Timestamp(i64::MIN)..Timestamp(i64::MAX))
                .expect("failed to get data")
                .1,
            &entity.schema,
            &component.metadata.load(),
            path.clone(),
        )?;

        let df = py
            .import_bound("polars")?
            .call_method1("read_ipc", (path,))?;
        Ok(df)
    }

    fn column_array<'a>(
        &self,
        py: Python<'a>,
        component_name: String,
    ) -> Result<Bound<'a, PyAny>, Error> {
        let id = ComponentId::new(&component_name);
        let series = self
            .exec
            .world
            .column_by_id(id)
            .ok_or(nox_ecs::Error::ComponentNotFound)?;

        let temp_file = tempfile::NamedTempFile::with_suffix(".feather")?;
        let path = temp_file.path().to_owned();
        nox_ecs::arrow::write_ipc(
            series.column.as_ref(),
            series.schema,
            series.metadata,
            path.clone(),
        )?;

        let series = py
            .import_bound("polars")?
            .call_method1("read_ipc", (path,))?
            .get_item(component_name)?;
        Ok(series)
    }
}
