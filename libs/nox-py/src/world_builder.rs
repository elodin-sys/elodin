use crate::sim_runner::{Args, SimSupervisor};
use crate::*;
use clap::Parser;
use nox_ecs::{
    conduit,
    nox::{self, ScalarExt},
    spawn_tcp_server, ArchetypeName, HostColumn, HostStore, SharedWorld, Table, World,
};
use pyo3::{exceptions::PyValueError, types::PyBytes};
use std::{collections::hash_map::Entry, path::PathBuf, time::Duration};

#[pyclass(subclass)]
#[derive(Default)]
pub struct WorldBuilder {
    pub world: World<HostStore>,
}

impl WorldBuilder {
    fn get_or_insert_archetype(
        &mut self,
        archetype: &Archetype,
    ) -> Result<&mut Table<HostStore>, Error> {
        let archetype_name = archetype.archetype_name;
        match self.world.archetypes.entry(archetype_name) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let columns = archetype
                    .component_datas
                    .iter()
                    .map(
                        |Component {
                             id,
                             ty,
                             asset,
                             metadata,
                             ..
                         }| {
                            let metadata = conduit::Metadata {
                                component_id: id.inner,
                                component_type: ty.clone().into(),
                                asset: *asset,
                                tags: metadata.clone(),
                            };
                            let col = HostColumn::new(metadata);
                            Ok((id.inner, col))
                        },
                    )
                    .collect::<Result<_, Error>>()?;
                for id in &archetype.component_ids {
                    self.world.component_map.insert(id.inner, archetype_name);
                }
                let table = Table {
                    columns,
                    ..Default::default()
                };
                Ok(entry.insert(table))
            }
        }
    }
}

#[pymethods]
impl WorldBuilder {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn(mut slf: PyRefMut<'_, Self>, archetype: Spawnable<'_>) -> Result<Entity, Error> {
        let entity_id = EntityId {
            inner: conduit::EntityId(slf.world.entity_len),
        };

        slf.spawn_with_entity_id(archetype, entity_id.clone())?;
        let world = slf.into();
        Ok(Entity {
            id: entity_id,
            world,
        })
    }

    pub fn spawn_with_entity_id(
        &mut self,
        spawnable: Spawnable,
        entity_id: EntityId,
    ) -> Result<EntityId, Error> {
        match spawnable {
            Spawnable::Archetype(archetype) => {
                let entity_id = entity_id.inner;
                let table = self.get_or_insert_archetype(&archetype)?;
                table.entity_buffer.push(entity_id.0.constant());
                for (arr, id) in archetype
                    .arrays
                    .iter()
                    .zip(archetype.component_ids.into_iter())
                {
                    let col = table
                        .columns
                        .get_mut(&id.inner)
                        .ok_or(nox_ecs::Error::ComponentNotFound)?;
                    let ty = col.component_type();
                    let size = ty.primitive_ty.element_type().element_size_in_bytes();
                    let buf = unsafe { arr.buf(size) };
                    col.push_raw(buf);
                }
                self.world.entity_len += 1;
                Ok(EntityId { inner: entity_id })
            }
            Spawnable::Asset { id, bytes } => {
                let inner = self.world.assets.insert_bytes(id, bytes.bytes);
                let component_id = conduit::ComponentId(id.0);
                let archetype = Archetype {
                    component_datas: vec![Component {
                        id: ComponentId {
                            inner: component_id,
                        },
                        ty: Python::with_gil(ComponentType::u64),
                        asset: true,
                        metadata: Default::default(),
                    }],
                    component_ids: vec![ComponentId {
                        inner: component_id,
                    }],
                    arrays: vec![],
                    archetype_name: ArchetypeName::from(format!("asset_handle_{}", id.0).as_str()),
                };

                let table = self.get_or_insert_archetype(&archetype)?;
                table.entity_buffer.push(entity_id.inner.0.constant());
                let col = table
                    .columns
                    .get_mut(&component_id)
                    .ok_or(nox_ecs::Error::ComponentNotFound)?;
                col.push_raw(&inner.id.to_le_bytes());
                self.world.entity_len += 1;
                Ok(entity_id)
            }
        }
    }

    fn insert_asset(&mut self, py: Python<'_>, asset: PyObject) -> Result<Handle, Error> {
        let asset = PyAsset::try_new(py, asset)?;
        let inner = self
            .world
            .assets
            .insert_bytes(asset.asset_id(), asset.bytes()?);
        Ok(Handle { inner })
    }

    #[cfg(feature = "server")]
    pub fn serve(
        &mut self,
        py: Python<'_>,
        sys: PyObject,
        daemon: Option<bool>,
        time_step: Option<f64>,
        client: Option<&Client>,
        addr: Option<&str>,
    ) -> Result<String, Error> {
        use self::web_socket::spawn_ws_server;
        use tokio_util::sync::CancellationToken;

        let addr = addr.unwrap_or("127.0.0.1:0").to_string();
        let daemon = daemon.unwrap_or(false);
        let _ = tracing_subscriber::fmt::fmt()
            .with_env_filter(
                EnvFilter::builder()
                    .with_default_directive("info".parse().expect("invalid filter"))
                    .from_env_lossy(),
            )
            .try_init();

        let exec = self.build(py, sys, time_step)?.exec;

        let client = match client {
            Some(c) => c.client.clone(),
            None => nox::Client::cpu()?,
        };

        let (tx, rx) = flume::unbounded();
        if daemon {
            let cancel_token = CancellationToken::new();
            std::thread::spawn(move || {
                spawn_ws_server(
                    addr.parse().unwrap(),
                    exec,
                    &client,
                    cancel_token.clone(),
                    || cancel_token.is_cancelled(),
                    tx,
                )
                .unwrap();
            });
        } else {
            let cancel_token = CancellationToken::new();
            spawn_ws_server(
                addr.parse().unwrap(),
                exec,
                &client,
                cancel_token,
                || py.check_signals().is_err(),
                tx,
            )?;
        }
        Ok(rx.recv().unwrap().to_string())
    }

    pub fn run(
        &mut self,
        py: Python<'_>,
        sys: PyObject,
        time_step: Option<f64>,
        client: Option<&Client>,
    ) -> Result<Option<String>, Error> {
        let _ = tracing_subscriber::fmt::fmt()
            .with_env_filter(
                EnvFilter::builder()
                    .with_default_directive("info".parse().expect("invalid filter"))
                    .from_env_lossy(),
            )
            .try_init();

        let pytesting = py
            .import("elodin")?
            .getattr("_called_from_test")
            .unwrap()
            .extract::<bool>()?;
        // If executed by pytest, don't run the server
        if pytesting {
            return Ok(None);
        }

        let (args, path) = if let Ok(dir) = std::env::var("ELODIN_FORCE_BUILD_DIR") {
            let dir = PathBuf::from(dir);
            (Args::Build { dir }, PathBuf::new())
        } else {
            // skip `python3`
            let mut args = std::env::args_os().skip(1);
            let path = args.next().ok_or(Error::MissingArg("path".to_string()))?;
            let path = PathBuf::from(path);
            (Args::parse_from(args), path)
        };

        match args {
            Args::Build { dir } => {
                let exec = self.build(py, sys, time_step)?.exec;
                exec.write_to_dir(dir)?;
                Ok(None)
            }
            Args::Repl { addr } => Ok(Some(addr.to_string())),
            Args::Run {
                addr,
                no_repl,
                watch,
            } => {
                if !watch {
                    let exec = self.build(py, sys, time_step)?.exec;
                    let client = match client {
                        Some(c) => c.client.clone(),
                        None => nox::Client::cpu()?,
                    };
                    if no_repl {
                        spawn_tcp_server(addr, exec, &client, || py.check_signals().is_err())?;
                        Ok(None)
                    } else {
                        std::thread::spawn(move || {
                            spawn_tcp_server(addr, exec, &client, || false).unwrap()
                        });
                        Ok(Some(addr.to_string()))
                    }
                } else if no_repl {
                    SimSupervisor::run(path).unwrap();
                    Ok(None)
                } else {
                    let _ = SimSupervisor::spawn(path);
                    Ok(Some(addr.to_string()))
                }
            }
        }
    }

    pub fn build(
        &mut self,
        py: Python<'_>,
        sys: PyObject,
        time_step: Option<f64>,
    ) -> Result<Exec, Error> {
        let start = std::time::Instant::now();
        let world = std::mem::take(&mut self.world);
        let builder = nox_ecs::PipelineBuilder::from_world(world);
        let builder = PipelineBuilder { builder };
        let py_code = "import jax
def build_expr(builder, sys):
    sys.init(builder)
    def call(args, builder):
        builder.inject_args(args)
        sys.call(builder)
        return builder.ret_vars()
    var_array = builder.var_arrays()
    xla = jax.xla_computation(lambda a: call(a, builder))(var_array)
    return xla";

        if let Some(ts) = time_step {
            let ts = Duration::from_secs_f64(ts);
            // 4ms (~240 ticks/sec) is the minimum time step
            if ts <= Duration::from_millis(4) {
                return Err(Error::InvalidTimeStep(ts));
            }
        }

        let fun: Py<PyAny> = PyModule::from_code(py, py_code, "", "")?
            .getattr("build_expr")?
            .into();
        let builder = PyCell::new(py, builder)?;
        let comp = fun
            .call1(py, (builder.borrow_mut(), sys))?
            .extract::<PyObject>(py)?;
        let comp = comp.call_method0(py, "as_serialized_hlo_module_proto")?;
        let comp = comp
            .downcast::<PyBytes>(py)
            .map_err(|_| Error::HloModuleNotBytes)?;
        let comp_bytes = comp.as_bytes();
        let hlo_module = nox::xla::HloModuleProto::parse_binary(comp_bytes)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        tracing::debug!(duration = ?start.elapsed(), "generated HLO");
        let builder = builder.replace(PipelineBuilder::default());
        let builder = builder.builder;
        let ret_ids = builder.vars.keys().copied().collect::<Vec<_>>();
        let time_step = time_step.map(Duration::from_secs_f64);
        let world = SharedWorld {
            host: builder.world,
            ..Default::default()
        };
        let metadata = nox_ecs::ExecMetadata {
            time_step,
            arg_ids: builder.param_ids,
            ret_ids,
        };
        let tick_exec = nox_ecs::Exec::new(metadata, hlo_module);
        let exec = nox_ecs::WorldExec::new(world, tick_exec, None);
        Ok(Exec { exec })
    }
}
