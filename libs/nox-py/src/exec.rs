use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::Instant;

use impeller2::types::ComponentId;
use impeller2::types::Timestamp;
use impeller2_wkt::ArchiveFormat;
use nox::xla::{BufferArgsRef, HloModuleProto, PjRtBuffer, PjRtLoadedExecutable};
use nox::{Client, CompFn, Noxpr};
use serde::{Deserialize, Serialize};

use crate::component::Component;
use crate::error::Error;
use crate::profile::Profiler;
use crate::query::ComponentArray;
use crate::world::{Buffers, Column, World};

// --- ECS Exec types from mod.rs ---

#[derive(Serialize, Deserialize, Clone)]
pub struct ExecMetadata {
    pub arg_ids: Vec<ComponentId>,
    pub ret_ids: Vec<ComponentId>,
}

pub trait ExecState: Clone {}

#[derive(Clone, Default)]
pub struct Uncompiled;

#[derive(Clone)]
pub struct Compiled {
    client: Client,
    exec: PjRtLoadedExecutable,
}

impl ExecState for Uncompiled {}
impl ExecState for Compiled {}

#[derive(Clone)]
pub struct Exec<S: ExecState = Uncompiled> {
    metadata: ExecMetadata,
    hlo_module: HloModuleProto,
    state: S,
}

impl Exec {
    pub fn new(metadata: ExecMetadata, hlo_module: HloModuleProto) -> Self {
        Self {
            metadata,
            hlo_module,
            state: Uncompiled,
        }
    }

    pub fn compile(self, client: Client) -> Result<Exec<Compiled>, Error> {
        let comp = self.hlo_module.computation();
        let exec = client.compile(&comp)?;
        Ok(Exec {
            metadata: self.metadata,
            hlo_module: self.hlo_module,
            state: Compiled { client, exec },
        })
    }

    pub fn read_from_dir(path: impl AsRef<std::path::Path>) -> Result<Exec, Error> {
        let path = path.as_ref();
        let mut metadata = std::fs::File::open(path.join("metadata.json"))?;
        let metadata: ExecMetadata = serde_json::from_reader(&mut metadata)?;
        let hlo_module_data = std::fs::read(path.join("hlo.binpb"))?;
        let hlo_module = HloModuleProto::parse_binary(&hlo_module_data)?;
        Ok(Exec::new(metadata, hlo_module))
    }
}

impl<S: ExecState> Exec<S> {
    pub fn metadata(&self) -> &ExecMetadata {
        &self.metadata
    }

    pub fn hlo_module(&self) -> &HloModuleProto {
        &self.hlo_module
    }
}

impl Exec<Compiled> {
    fn run(&mut self, client: &mut Buffers<PjRtBuffer>) -> Result<(), Error> {
        let mut buffers = BufferArgsRef::default().untuple_result(true);
        for id in &self.metadata.arg_ids {
            buffers.push(&client[id].buffer);
        }
        let ret_bufs = self.state.exec.execute_buffers(buffers)?;
        for (buf, comp_id) in ret_bufs.into_iter().zip(self.metadata.ret_ids.iter()) {
            let client = client.get_mut(comp_id).expect("buffer not found");
            client.buffer = buf;
        }
        Ok(())
    }
}

pub struct WorldExec<S: ExecState = Uncompiled> {
    pub world: World,
    pub client_buffers: Buffers<PjRtBuffer>,
    pub tick_exec: Exec<S>,
    pub startup_exec: Option<Exec<S>>,
    pub profiler: Profiler,
}

impl<S: ExecState> WorldExec<S> {
    pub fn new(world: World, tick_exec: Exec<S>, startup_exec: Option<Exec<S>>) -> Self {
        Self {
            world,
            client_buffers: Default::default(),
            tick_exec,
            startup_exec,
            profiler: Default::default(),
        }
    }

    pub fn tick(&self) -> u64 {
        self.world.tick()
    }

    pub fn fork(&self) -> Self {
        Self {
            world: self.world.clone(),
            client_buffers: Buffers::default(),
            tick_exec: self.tick_exec.clone(),
            startup_exec: self.startup_exec.clone(),
            profiler: self.profiler.clone(),
        }
    }
}

impl WorldExec<Uncompiled> {
    pub fn compile(mut self, client: Client) -> Result<WorldExec<Compiled>, Error> {
        let start = &mut Instant::now();
        let tick_exec = self.tick_exec.compile(client.clone())?;
        let startup_exec = self
            .startup_exec
            .map(|exec| exec.compile(client))
            .transpose()?;
        self.profiler.compile.observe(start);
        Ok(WorldExec {
            world: self.world,
            client_buffers: Default::default(),
            tick_exec,
            startup_exec,
            profiler: self.profiler,
        })
    }
}

impl WorldExec<Compiled> {
    pub fn run(&mut self) -> Result<(), Error> {
        let start = &mut Instant::now();
        self.copy_to_client()?;
        self.profiler.copy_to_client.observe(start);
        if let Some(mut startup_exec) = self.startup_exec.take() {
            startup_exec.run(&mut self.client_buffers)?;
            self.copy_to_host()?;
        }
        self.tick_exec.run(&mut self.client_buffers)?;
        self.profiler.execute_buffers.observe(start);
        self.copy_to_host()?;
        self.profiler.copy_to_host.observe(start);
        self.world.advance_tick();
        self.profiler.add_to_history.observe(start);
        Ok(())
    }

    fn copy_to_client(&mut self) -> Result<(), Error> {
        let client = &self.tick_exec.state.client;
        for id in std::mem::take(&mut self.world.dirty_components) {
            let pjrt_buf = self
                .world
                .column_by_id(id)
                .unwrap()
                .copy_to_client(client)?;
            if let Some(client) = self.client_buffers.get_mut(&id) {
                client.buffer = pjrt_buf;
            } else {
                let host = &self.world.host.get(&id).expect("missing host column");
                self.client_buffers.insert(
                    id,
                    Column {
                        buffer: pjrt_buf,
                        entity_ids: host.entity_ids.clone(),
                    },
                );
            }
        }
        Ok(())
    }

    fn copy_to_host(&mut self) -> Result<(), Error> {
        let client = &self.tick_exec.state.client;
        for (id, pjrt_buf) in self.client_buffers.iter() {
            let host_buf = self.world.host.get_mut(id).unwrap();
            client.copy_into_host_vec(&pjrt_buf.buffer, &mut host_buf.buffer)?;
        }
        Ok(())
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.profiler.profile(self.world.sim_time_step().0)
    }
}

impl<C: Component> ComponentArray<C> {
    pub fn map<O: Component>(
        &self,
        func: impl CompFn<(C,), O>,
    ) -> Result<ComponentArray<O>, Error> {
        let func = func.build_expr()?;
        let buffer = Noxpr::vmap_with_axis(func, &[0], std::slice::from_ref(&self.buffer))?;
        Ok(ComponentArray {
            buffer,
            len: self.len,
            phantom_data: PhantomData,
            entity_map: self.entity_map.clone(),
            component_id: O::COMPONENT_ID,
        })
    }
}

// --- Python wrapper ---

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

#[pyclass(name = "Exec")]
pub struct PyExec {
    pub exec: WorldExec<Compiled>,
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

        for _ in 0..ticks {
            self.exec.run()?;
            self.db.with_state(|state| {
                crate::impeller2_server::commit_world_head(state, &mut self.exec, timestamp, None)
            })?;
            timestamp += self.exec.world.sim_time_step().0;

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
            progress_bar.inc(1);
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
