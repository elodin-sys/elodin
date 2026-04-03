use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::Duration;

use impeller2::types::ComponentId;
use impeller2::types::Timestamp;
use nox::{CompFn, Noxpr};
use serde::{Deserialize, Serialize};

use crate::component::Component;
use crate::error::Error;
use crate::iree_exec::IREEWorldExec;
use crate::jax_exec::JaxWorldExec;
use crate::query::ComponentArray;
use crate::world::World;

#[derive(Serialize, Deserialize, Clone)]
pub struct ExecSlotMetadata {
    pub component_id: ComponentId,
    pub shape: Vec<i64>,
    pub entity_axis_elided: bool,
}

#[derive(Clone)]
pub struct ConstantSpec {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub element_type: nox::ElementType,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ExecMetadata {
    pub arg_ids: Vec<ComponentId>,
    pub ret_ids: Vec<ComponentId>,
    pub arg_slots: Vec<ExecSlotMetadata>,
    pub ret_slots: Vec<ExecSlotMetadata>,
    pub has_singleton_lowering: bool,
    #[serde(skip)]
    pub promoted_constants: Vec<ConstantSpec>,
}

impl<C: Component> ComponentArray<C> {
    pub fn map<O: Component>(
        &self,
        func: impl CompFn<(C,), O>,
    ) -> Result<ComponentArray<O>, Error> {
        let func = func.build_expr()?;
        let buffer = if self.batch1 {
            Noxpr::substitute_params(&func, std::slice::from_ref(&self.buffer))
        } else {
            Noxpr::vmap_with_axis(func, &[0], std::slice::from_ref(&self.buffer))?
        };
        Ok(ComponentArray {
            buffer,
            len: self.len,
            phantom_data: PhantomData,
            entity_map: self.entity_map.clone(),
            component_id: O::COMPONENT_ID,
            batch1: self.batch1,
        })
    }
}

pub enum WorldExec {
    Iree(Box<IREEWorldExec>),
    Jax(Box<JaxWorldExec>),
}

impl WorldExec {
    pub fn run(&mut self) -> Result<(), Error> {
        match self {
            Self::Iree(e) => e.run(),
            Self::Jax(e) => e.run(),
        }
    }

    pub fn world(&self) -> &World {
        match self {
            Self::Iree(e) => &e.world,
            Self::Jax(e) => &e.world,
        }
    }

    pub fn world_mut(&mut self) -> &mut World {
        match self {
            Self::Iree(e) => &mut e.world,
            Self::Jax(e) => &mut e.world,
        }
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        match self {
            Self::Iree(e) => e.profile(),
            Self::Jax(e) => e.profile(),
        }
    }

    pub fn profiler_mut(&mut self) -> &mut crate::profile::Profiler {
        match self {
            Self::Iree(e) => &mut e.profiler,
            Self::Jax(e) => &mut e.profiler,
        }
    }
}

// --- Python wrapper ---

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

#[pyclass(name = "Exec")]
pub struct PyExec {
    pub exec: WorldExec,
    pub db: Box<elodin_db::DB>,
}

#[pymethods]
impl PyExec {
    #[pyo3(signature = (ticks=1, show_progress=true, is_canceled = None))]
    pub fn run(
        &mut self,
        py: Python<'_>,
        ticks: usize,
        mut show_progress: bool,
        is_canceled: Option<PyObject>,
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
        let sim_step = self.exec.world().sim_time_step().0;
        let configured_batch = self.exec.world().ticks_per_telemetry() as usize;
        let mut remaining = ticks;
        while remaining > 0 {
            let this_batch = remaining.min(configured_batch.max(1));
            // Temporarily override so the kernel runs the right number of batched ticks.
            if this_batch != configured_batch {
                self.exec.world_mut().metadata.ticks_per_telemetry = this_batch as u64;
            }
            self.exec.run()?;
            if this_batch != configured_batch {
                self.exec.world_mut().metadata.ticks_per_telemetry = configured_batch as u64;
            }
            let commit_offset = Duration::from_secs_f64(
                sim_step.as_secs_f64() * (this_batch.saturating_sub(1) as f64),
            );
            let commit_timestamp = timestamp + commit_offset;
            self.db.with_state(|state| {
                crate::impeller2_server::commit_world_head_unified(
                    state,
                    &mut self.exec,
                    commit_timestamp,
                    None,
                )
            })?;
            timestamp += Duration::from_secs_f64(sim_step.as_secs_f64() * this_batch as f64);
            remaining -= this_batch;

            if let Some(func) = &is_canceled {
                let is_canceled = Python::with_gil(|py| {
                    func.call0(py).and_then(|result| result.extract::<bool>(py))
                })?;
                if is_canceled {
                    eprintln!("exec.run canceled!");
                    return Ok(());
                }
            }
            py.check_signals()?;
            progress_bar.inc(this_batch as u64);
        }
        progress_bar.finish_and_clear();
        Ok(())
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.exec.profile()
    }

    pub fn save_archive(&self, path: String, format: String) -> Result<(), Error> {
        let format = match format.as_str() {
            "arrow_ipc" | "arrow" => impeller2_wkt::ArchiveFormat::ArrowIpc,
            "parquet" | "pq" => impeller2_wkt::ArchiveFormat::Parquet,
            "csv" => impeller2_wkt::ArchiveFormat::Csv,
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
