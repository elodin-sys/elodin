use std::collections::HashMap;

use crate::*;

use core::time::Duration;
use impeller2::types::{ComponentId, Timestamp};
use impeller2_wkt::ArchiveFormat;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use nox_ecs::{
    Compiled,
    impeller2_server::{self, PairId},
};
use pyo3::exceptions::PyTypeError;
use pyo3::types::IntoPyDict;

#[pyclass]
pub struct Exec {
    pub exec: nox_ecs::WorldExec<Compiled>,
    pub db: elodin_db::DB,
}

impl Exec {
    /// Wait for external control components to be updated
    fn wait_for_write_or_timeout(
        &mut self,
        py: Python<'_>,
        maybe_timeout: Option<Duration>,
        external_controls: &mut [(ComponentId, Timestamp)],
    ) -> Result<bool, Error> {
        if external_controls.is_empty() {
            return Ok(true);
        }
        // Check for updates with a timeout to avoid infinite waiting
        let start_time = std::time::Instant::now();

        loop {
            // Check if we have updates using the impeller2_server function
            if nox_ecs::impeller2_server::timestamps_changed(&self.db, external_controls)
                .unwrap_or(true)
            {
                return Ok(true);
            }

            if let Some(timeout) = maybe_timeout
                && start_time.elapsed() > timeout
            {
                return Ok(false);
            }

            // Allow Python signals to be processed
            py.check_signals()?;

            // Small sleep to avoid busy waiting
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
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
        let external_controls: HashSet<_> =
            impeller2_server::external_controls(&self.exec).collect();
        let wait_for_write: Vec<ComponentId> =
            impeller2_server::wait_for_write(&self.exec).collect();
        let wait_for_write_pair_ids: Vec<PairId> =
            impeller2_server::get_pair_ids(&self.exec, &wait_for_write).unwrap();
        let mut wait_for_write_pair_ids =
            impeller2_server::collect_timestamps(&self.db, &wait_for_write_pair_ids);

        for _ in 0..ticks {
            self.exec.run()?;
            self.db.with_state(|state| {
                nox_ecs::impeller2_server::commit_world_head(
                    state,
                    &mut self.exec,
                    timestamp,
                    Some(&external_controls),
                )
            })?;
            timestamp += self.exec.world.sim_time_step().0;
            py.check_signals()?;
            progress_bar.inc(1);
            // Wait for external control components to be updated before running the next tick
            self.wait_for_write_or_timeout(py, None, &mut wait_for_write_pair_ids)?;
        }
        progress_bar.finish_and_clear();
        Ok(())
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.exec.profile()
    }

    pub fn save_archive(&self, path: String, format: String) -> Result<(), Error> {
        let format = match format.as_str() {
            "arrow_ipc" | "arrow" => ArchiveFormat::ArrowIpc,
            "parquet" | "pq" => ArchiveFormat::Parquet,
            "csv" => ArchiveFormat::Csv,
            "native" => ArchiveFormat::Native,
            _ => return Err(Error::UnknownCommand(format)),
        };
        self.db.save_archive(path, format)?;
        Ok(())
    }

    pub fn history<'a>(
        &self,
        py: Python<'a>,
        components: ComponentsArg,
    ) -> Result<Bound<'a, PyAny>, Error> {
        let component_names = components.to_vec();

        let temp_dir = tempfile::TempDir::new()?;
        let temp_path = temp_dir.path().to_string_lossy().to_string();
        self.save_archive(temp_path.clone(), "arrow".to_string())?;
        let polars = py.import("polars")?;
        let mut dataframes = Vec::new();
        for component_name in component_names {
            let file_path = format!("{}/{}.arrow", temp_path, component_name);
            let df = polars.call_method1("read_ipc", (file_path,))?;
            dataframes.push(df);
        }

        let mut result_df = dataframes[0].clone();
        for df in dataframes.into_iter().skip(1) {
            result_df =
                result_df.call_method("join", (df,), Some(&[("on", "time")].into_py_dict(py)?))?;
        }
        Ok(result_df)
    }
}

pub enum ComponentsArg {
    Single(String),
    Multiple(Vec<String>),
}

impl ComponentsArg {
    pub fn to_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }
}

impl FromPyObject<'_> for ComponentsArg {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(single) = ob.extract::<String>() {
            Ok(Self::Single(single))
        } else if let Ok(multiple) = ob.extract::<Vec<String>>() {
            Ok(Self::Multiple(multiple))
        } else {
            Err(PyTypeError::new_err("Expected str or list[str]"))
        }
    }
}
