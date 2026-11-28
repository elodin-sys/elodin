use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use impeller::types::ComponentId;
use impeller::types::Timestamp;
use nox::{CompFn, Noxpr};
use serde::{Deserialize, Serialize};

use crate::component::Component;
use crate::cranelift_exec::CraneliftWorldExec;
use crate::error::Error;
use crate::jax_exec::JaxWorldExec;
use crate::query::ComponentArray;
use crate::tick_metrics::Timing;
use crate::world::World;

#[derive(Serialize, Deserialize, Clone)]
pub struct ExecSlotMetadata {
    pub component_id: ComponentId,
    pub shape: Vec<i64>,
    pub entity_axis_elided: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ExecMetadata {
    pub arg_ids: Vec<ComponentId>,
    pub ret_ids: Vec<ComponentId>,
    pub arg_slots: Vec<ExecSlotMetadata>,
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
    Jax(Box<JaxWorldExec>),
    Cranelift(Box<CraneliftWorldExec>),
}

impl WorldExec {
    pub fn run(&mut self) -> Result<(), Error> {
        match self {
            Self::Jax(e) => e.run(),
            Self::Cranelift(e) => e.run(),
        }
    }

    pub fn world(&self) -> &World {
        match self {
            Self::Jax(e) => &e.world,
            Self::Cranelift(e) => &e.world,
        }
    }

    pub fn world_mut(&mut self) -> &mut World {
        match self {
            Self::Jax(e) => &mut e.world,
            Self::Cranelift(e) => &mut e.world,
        }
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        match self {
            Self::Jax(e) => e.profile(),
            Self::Cranelift(e) => e.profile(),
        }
    }

    pub fn profiler_mut(&mut self) -> &mut crate::profile::Profiler {
        match self {
            Self::Jax(e) => &mut e.profiler,
            Self::Cranelift(e) => &mut e.profiler,
        }
    }

    pub fn backend_report(&self) -> crate::tick_metrics::BackendReport {
        match self {
            Self::Jax(e) => e.tick_exec.backend_report(),
            Self::Cranelift(e) => e.tick_exec.backend_report(),
        }
    }

    pub fn metadata(&self) -> &ExecMetadata {
        match self {
            Self::Jax(e) => &e.tick_exec.metadata,
            Self::Cranelift(e) => &e.tick_exec.metadata,
        }
    }

    /// Whether the run's timing level compiled probes in, on a backend that has them.
    pub fn probed(&self) -> bool {
        matches!(self, Self::Cranelift(_)) && self.world().metadata.timing > Timing::Summary
    }
}

// --- Python wrapper ---

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "Exec")]
pub struct PyExec {
    pub exec: WorldExec,
    pub db: Box<elodin_db::DB>,
    sync: crate::impeller_server::DbSync,
    /// Timestamp `init_db` recorded the initial state at.
    start: Timestamp,
    /// Ticks run so far, across `run` calls.
    ticks: u64,
    last_receipt: Option<crate::tick_metrics::Receipt>,
}

impl PyExec {
    /// `db` must already hold the world's pairs, as `init_db` leaves it.
    pub fn new(exec: WorldExec, db: elodin_db::DB, start: Timestamp) -> Self {
        let sync = db.with_state(|state| {
            let written = &exec.metadata().ret_ids;
            crate::impeller_server::DbSync::new(state, exec.world(), written, &Default::default())
        });
        Self {
            exec,
            db: Box::new(db),
            sync,
            start,
            ticks: 0,
            last_receipt: None,
        }
    }

    /// `start` plus `tick` sim steps. Wall-clock stamps would go backwards whenever the sim runs
    /// ahead of real time, and the db rejects those commits.
    fn tick_timestamp(&self, tick: u64) -> Timestamp {
        let nanos = self.exec.world().sim_time_step().0.as_nanos() * u128::from(tick);
        self.start + Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
    }
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
        let configured_batch = self.exec.world().ticks_per_telemetry() as usize;
        let mut metrics = crate::tick_metrics::TickMetrics::new();
        metrics.silent = true;
        metrics.requested_ticks = Some(ticks as u64);
        metrics.record_cycles(Some(ticks.div_ceil(configured_batch.max(1)) as u64));
        metrics.backend = Some(self.exec.backend_report());
        metrics.timing = self.exec.world().metadata.timing;
        if self.exec.probed() {
            cranelift_mlir::profile::reset_timings();
        }
        self.sync.reset_counters();
        let mut remaining = ticks;
        while remaining > 0 {
            let cycle_start = Instant::now();
            let this_batch = remaining.min(configured_batch.max(1));
            // Temporarily override so the kernel runs the right number of batched ticks.
            if this_batch != configured_batch {
                self.exec.world_mut().metadata.ticks_per_telemetry = this_batch as u64;
            }
            self.exec.run()?;
            if this_batch != configured_batch {
                self.exec.world_mut().metadata.ticks_per_telemetry = configured_batch as u64;
            }
            metrics.observe_world_run(cycle_start.elapsed());
            self.ticks += this_batch as u64;
            let commit_timestamp = self.tick_timestamp(self.ticks);
            let vtable_gen = self.db.vtable_gen.latest();
            let commit_start = Instant::now();
            self.db.with_state(|_| {
                self.sync.commit_world(
                    self.exec.world(),
                    commit_timestamp,
                    vtable_gen,
                    &Default::default(),
                )
            })?;
            metrics.observe_commit(commit_start.elapsed());
            metrics.cycles += 1;
            metrics.steps += this_batch as u64;
            metrics.observe_total_cycle(cycle_start.elapsed());
            remaining -= this_batch;

            if let Some(func) = &is_canceled {
                let is_canceled = Python::with_gil(|py| {
                    func.call0(py).and_then(|result| result.extract::<bool>(py))
                })?;
                if is_canceled {
                    eprintln!("exec.run canceled!");
                    break;
                }
            }
            py.check_signals()?;
            progress_bar.inc(this_batch as u64);
        }
        progress_bar.finish_and_clear();
        metrics.sync = Some(self.sync.counters.clone());
        metrics.systems = self.exec.probed().then(cranelift_mlir::profile::timings);
        self.last_receipt = Some(metrics.receipt());
        Ok(())
    }

    /// The receipt of the latest `run`, or `None` before the first.
    pub fn last_receipt<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.last_receipt
            .as_ref()
            .map(|receipt| receipt.to_py(py))
            .transpose()
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.exec.profile()
    }

    pub fn save_archive(&self, path: String, format: String) -> Result<(), Error> {
        let format = match format.as_str() {
            "arrow_ipc" | "arrow" => impeller_wkt::ArchiveFormat::ArrowIpc,
            "parquet" | "pq" => impeller_wkt::ArchiveFormat::Parquet,
            "csv" => impeller_wkt::ArchiveFormat::Csv,
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
            let kwargs = PyDict::new(py);
            kwargs.set_item("on", "time")?;
            kwargs.set_item("how", "full")?;
            kwargs.set_item("coalesce", true)?;
            result_df = result_df.call_method("join", (df,), Some(&kwargs))?;
        }
        result_df = result_df.call_method1("sort", ("time",))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("strategy", "forward")?;
        result_df = result_df.call_method("fill_null", (), Some(&kwargs))?;
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
