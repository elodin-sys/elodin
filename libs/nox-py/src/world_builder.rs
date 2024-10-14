use crate::*;
use ::s10::{cli::run_recipe, GroupRecipe, SimRecipe};
use clap::Parser;
use miette::miette;
use nox_ecs::{
    impeller, increment_sim_tick, nox, spawn_tcp_server, IntoSystem, System as _, TimeStep, World,
    WorldExt,
};
use std::{collections::HashMap, net::SocketAddr, path::PathBuf, time::Duration};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub enum Args {
    Build {
        #[arg(long)]
        dir: PathBuf,
    },
    Run {
        #[arg(default_value = "0.0.0.0:2240")]
        addr: SocketAddr,
    },
    Plan {
        out_dir: PathBuf,
        #[arg(default_value = "0.0.0.0:2240")]
        addr: SocketAddr,
    },
    #[clap(hide = true)]
    Bench {
        #[arg(long, default_value = "1000")]
        ticks: usize,
    },
}

#[pyclass(subclass)]
#[derive(Default)]
pub struct WorldBuilder {
    pub world: World,
    pub recipes: HashMap<String, ::s10::Recipe>,
}

impl WorldBuilder {
    fn insert_entity_id(&mut self, archetype: &Archetype, entity_id: EntityId) {
        let archetype_name = archetype.archetype_name;
        let columns = archetype.component_data.iter().cloned();
        for metadata in columns {
            let id = metadata.component_id();
            self.world
                .component_map
                .insert(id, (archetype_name, metadata.inner));
            self.world.host.entry(id).or_default();
        }
        self.world
            .entity_ids
            .entry(archetype_name)
            .or_default()
            .extend_from_slice(&entity_id.inner.0.to_le_bytes());
    }

    fn sim_recipe(&mut self, path: PathBuf, addr: SocketAddr) -> ::s10::Recipe {
        let sim = SimRecipe { path, addr };
        let group = GroupRecipe {
            refs: vec![],
            recipes: self
                .recipes
                .iter()
                .map(|(n, r)| (n.clone(), r.clone()))
                .chain(std::iter::once((
                    "sim".to_string(),
                    ::s10::Recipe::Sim(sim),
                )))
                .collect(),
        };
        ::s10::Recipe::Group(group)
    }
}

#[pymethods]
impl WorldBuilder {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn(&mut self, spawnable: Spawnable, name: Option<String>) -> Result<EntityId, Error> {
        let entity_id = EntityId {
            inner: impeller::EntityId(self.world.entity_len),
        };
        self.insert(entity_id, spawnable)?;
        self.world.entity_len += 1;
        if let Some(name) = name {
            let metadata = EntityMetadata::new(name, None);
            let metadata = self.world.insert_asset(metadata.inner);
            self.world.insert_with_id(metadata, entity_id.inner);
        }
        Ok(entity_id)
    }

    pub fn insert(&mut self, entity_id: EntityId, spawnable: Spawnable) -> Result<(), Error> {
        match spawnable {
            Spawnable::Archetypes(archetypes) => {
                for archetype in archetypes {
                    self.insert_entity_id(&archetype, entity_id);
                    for (arr, component) in archetype.arrays.iter().zip(archetype.component_data) {
                        let mut col = self
                            .world
                            .column_by_id_mut(ComponentId::new(&component.name))
                            .ok_or(nox_ecs::Error::ComponentNotFound)?;
                        let ty = &col.metadata.component_type;
                        let size = ty.primitive_ty.element_type().element_size_in_bytes();
                        let buf = unsafe { arr.buf(size) };
                        col.push_raw(buf);
                    }
                }
                Ok(())
            }
            Spawnable::Asset { name, bytes } => {
                let metadata = impeller::Metadata::asset(&name);
                let component_id = metadata.component_id();
                let archetype_name = metadata.component_name().into();
                let inner = self.world.assets.insert_bytes(bytes.bytes);
                let archetype = Archetype {
                    component_data: vec![Metadata { inner: metadata }],
                    arrays: vec![],
                    archetype_name,
                };

                self.insert_entity_id(&archetype, entity_id);
                let mut col = self
                    .world
                    .column_by_id_mut(component_id)
                    .ok_or(nox_ecs::Error::ComponentNotFound)?;
                col.push_raw(&inner.id.to_le_bytes());
                Ok(())
            }
        }
    }

    fn insert_asset(&mut self, py: Python<'_>, asset: PyObject) -> Result<Handle, Error> {
        let asset = PyAsset::try_new(py, asset)?;
        let inner = self.world.assets.insert_bytes(asset.bytes()?);
        Ok(Handle { inner })
    }

    fn recipe(&mut self, recipe: crate::s10::Recipe) -> PyResult<()> {
        self.recipes.insert(recipe.name(), recipe.to_rust()?);
        Ok(())
    }

    #[cfg(feature = "server")]
    pub fn serve(
        &mut self,
        py: Python<'_>,
        sys: PyObject,
        daemon: Option<bool>,
        time_step: Option<f64>,
        output_time_step: Option<f64>,
        max_ticks: Option<u64>,
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

        let exec = self.build_uncompiled(py, sys, time_step, output_time_step, max_ticks)?;

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
                    Some(cancel_token.clone()),
                    || cancel_token.is_cancelled(),
                    tx,
                )
                .unwrap();
            });
        } else {
            spawn_ws_server(
                addr.parse().unwrap(),
                exec,
                &client,
                None,
                || py.check_signals().is_err(),
                tx,
            )?;
        }
        Ok(rx.recv().unwrap().to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        py: Python<'_>,
        sys: System,
        sim_time_step: Option<f64>,
        run_time_step: Option<f64>,
        output_time_step: Option<f64>,
        max_ticks: Option<u64>,
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
            .import_bound("elodin")?
            .getattr("_called_from_test")
            .unwrap()
            .extract::<bool>()?;
        // If executed by pytest, don't run the server
        if pytesting {
            return Ok(None);
        }

        let args = py
            .import_bound("sys")?
            .getattr("argv")?
            .extract::<Vec<String>>()?;
        let path = args.first().ok_or(Error::MissingArg("path".to_string()))?;
        let path = PathBuf::from(path);
        let args = Args::parse_from(args);

        match args {
            Args::Build { dir } => {
                let mut exec = self.build_uncompiled(
                    py,
                    sys,
                    sim_time_step,
                    run_time_step,
                    output_time_step,
                    max_ticks,
                )?;
                exec.write_to_dir(dir)?;
                Ok(None)
            }
            Args::Run { addr } => {
                let exec = self.build_uncompiled(
                    py,
                    sys,
                    sim_time_step,
                    run_time_step,
                    output_time_step,
                    max_ticks,
                )?;
                let client = match client {
                    Some(c) => c.client.clone(),
                    None => nox::Client::cpu()?,
                };
                let recipes = self.recipes.clone();
                std::thread::spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .build()
                        .map_err(|err| miette!("rt err {}", err))
                        .unwrap();
                    let group = ::s10::Recipe::Group(GroupRecipe {
                        recipes,
                        ..Default::default()
                    });
                    rt.block_on(run_recipe("world".to_string(), group, false, false))
                        .unwrap();
                });
                spawn_tcp_server(addr, exec, client, || py.check_signals().is_err())?;
                Ok(None)
            }
            Args::Plan { addr, out_dir } => {
                let recipe = self.sim_recipe(path, addr);
                let toml = toml::to_string_pretty(&recipe)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                let plan_path = out_dir.join("s10.toml");
                std::fs::write(&plan_path, toml)?;
                Ok(None)
            }
            Args::Bench { ticks } => {
                let mut exec = self.build(
                    py,
                    sys,
                    sim_time_step,
                    run_time_step,
                    output_time_step,
                    max_ticks,
                    client,
                )?;
                exec.run(py, ticks, true)?;
                let tempdir = tempfile::tempdir()?;
                exec.exec.write_to_dir(tempdir)?;
                let profile = exec.profile();
                println!("copy_to_client time:  {:.3} ms", profile["copy_to_client"]);
                println!("execute_buffers time: {:.3} ms", profile["execute_buffers"]);
                println!("copy_to_host time:    {:.3} ms", profile["copy_to_host"]);
                println!("add_to_history time:  {:.3} ms", profile["add_to_history"]);
                println!("= tick time:          {:.3} ms", profile["tick"]);
                println!("compile time:         {:.3} ms", profile["compile"]);
                println!("write_to_dir time:    {:.3} ms", profile["write_to_dir"]);
                println!("real_time_factor:     {:.3}", profile["real_time_factor"]);
                Ok(None)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &mut self,
        py: Python<'_>,
        system: System,
        sim_time_step: Option<f64>,
        run_time_step: Option<f64>,
        output_time_step: Option<f64>,
        max_ticks: Option<u64>,
        client: Option<&Client>,
    ) -> Result<Exec, Error> {
        let exec = self.build_uncompiled(
            py,
            system,
            sim_time_step,
            run_time_step,
            output_time_step,
            max_ticks,
        )?;
        let client = match client {
            Some(c) => c.client.clone(),
            None => nox::Client::cpu()?,
        };
        let exec = exec.compile(client.clone())?;
        Ok(Exec { exec })
    }
}

impl WorldBuilder {
    fn build_uncompiled(
        &mut self,
        py: Python<'_>,
        sys: System,
        sim_time_step: Option<f64>,
        run_time_step: Option<f64>,
        output_time_step: Option<f64>,
        max_ticks: Option<u64>,
    ) -> Result<nox_ecs::WorldExec, Error> {
        if let Some(ts) = sim_time_step {
            let ts = Duration::from_secs_f64(ts);
            // 1ms (~1000 ticks/sec) is the minimum time step
            // if ts <= Duration::from_millis(1) {
            //     return Err(Error::InvalidTimeStep(ts));
            // }
            self.world.sim_time_step = TimeStep(ts);
            self.world.run_time_step = TimeStep(ts);
        }
        if let Some(ts) = run_time_step {
            let ts = Duration::from_secs_f64(ts);
            self.world.run_time_step = TimeStep(ts);
        }

        if let Some(ts) = output_time_step {
            let time_step = Duration::from_secs_f64(ts);
            self.world.output_time_step = Some(impeller::OutputTimeStep {
                time_step,
                last_tick: std::time::Instant::now(),
            })
        }
        if let Some(max_ticks) = max_ticks {
            self.world.max_tick = max_ticks;
        }

        self.world.add_globals();

        let world = std::mem::take(&mut self.world);
        let xla_exec = increment_sim_tick.pipe(sys).compile(&world)?;
        let tick_exec = xla_exec.compile_hlo_module(py, &world)?;

        let exec = nox_ecs::WorldExec::new(world, tick_exec, None);
        Ok(exec)
    }
}
