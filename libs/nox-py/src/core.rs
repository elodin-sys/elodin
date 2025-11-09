// Core types and functionality migrated from nox-ecs/src/lib.rs

use crate::ecs::World;
use crate::ecs::system::{CompiledSystem, IntoSystem, System, SystemBuilder};
use crate::physics::globals::increment_sim_tick;
use crate::ecs::world::{Buffers, TimeStep};
use impeller2::types::{ComponentId, EntityId};
use nox::xla::{BufferArgsRef, HloModuleProto, PjRtBuffer, PjRtLoadedExecutable};
use nox::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::time::{Duration, Instant};
use std::collections::BTreeMap;
use std::marker::PhantomData;

// Core types are re-exported from lib.rs

pub trait WorldExt {
    fn builder(self) -> WorldBuilder;
}

impl WorldExt for World {
    fn builder(self) -> WorldBuilder {
        WorldBuilder::default().world(self)
    }
}

pub struct WorldBuilder<Sys = (), StartupSys = ()> {
    world: World,
    pipe: Sys,
    startup_sys: StartupSys,
}

impl Default for WorldBuilder {
    fn default() -> Self {
        Self {
            world: World::default(),
            pipe: (),
            startup_sys: (),
        }
    }
}

impl<Sys, StartupSys> WorldBuilder<Sys, StartupSys>
where
    Sys: System,
    StartupSys: System,
{
    pub fn world(mut self, world: World) -> Self {
        self.world = world;
        self
    }

    pub fn tick_pipeline<M, A, R, N: IntoSystem<M, A, R>>(
        self,
        pipe: N,
    ) -> WorldBuilder<N::System, StartupSys> {
        WorldBuilder {
            world: self.world,
            pipe: pipe.into_system(),
            startup_sys: self.startup_sys,
        }
    }

    pub fn startup_pipeline<M, A, R, N: IntoSystem<M, A, R>>(
        self,
        startup: N,
    ) -> WorldBuilder<Sys, N::System> {
        WorldBuilder {
            world: self.world,
            pipe: self.pipe,
            startup_sys: startup.into_system(),
        }
    }

    pub fn sim_time_step(mut self, time_step: Duration) -> Self {
        self.world.metadata.sim_time_step = TimeStep(time_step);
        self
    }

    pub fn spawn(&mut self, archetype: impl crate::ecs::Archetype + 'static) -> crate::ecs::world::Entity<'_> {
        self.world.spawn(archetype)
    }

    pub fn insert_with_id(&mut self, archetype: impl crate::ecs::Archetype + 'static, entity_id: EntityId) {
        self.world.insert_with_id(archetype, entity_id);
    }

    pub fn build(mut self) -> Result<WorldExec, crate::Error> {
        self.world.set_globals();
        let tick_exec = increment_sim_tick.pipe(self.pipe).build(&mut self.world)?;
        let startup_exec = self.startup_sys.build(&mut self.world)?;
        let world_exec = WorldExec::new(self.world, tick_exec, Some(startup_exec));
        Ok(world_exec)
    }

    pub fn run(self) -> World {
        let client = Client::cpu().unwrap();
        let mut exec = self.build().unwrap().compile(client).unwrap();
        exec.run().unwrap();
        exec.world
    }
}

pub trait IntoSystemExt<Marker, Arg, Ret> {
    type System;
    fn world(self) -> WorldBuilder<Self::System>
    where
        Self: Sized,
        Self::System: Sized;
}

impl<S: IntoSystem<Marker, Arg, Ret>, Marker, Arg, Ret> IntoSystemExt<Marker, Arg, Ret> for S {
    type System = <Self as IntoSystem<Marker, Arg, Ret>>::System;

    fn world(self) -> WorldBuilder<Self::System>
    where
        Self: Sized,
        Self::System: Sized,
    {
        WorldBuilder::default().tick_pipeline(self.into_system())
    }
}

pub trait SystemExt {
    fn build(self, world: &mut World) -> Result<Exec, crate::Error>;
}

impl<S: System> SystemExt for S {
    fn build(self, world: &mut World) -> Result<Exec, crate::Error> {
        let mut system_builder = SystemBuilder {
            vars: BTreeMap::default(),
            inputs: vec![],
            world,
        };
        self.init(&mut system_builder)?;
        let CompiledSystem {
            computation,
            inputs,
            outputs,
        } = self.compile(world)?;
        let metadata = ExecMetadata {
            arg_ids: inputs,
            ret_ids: outputs,
        };
        let computation = computation.func.build("exec")?.build()?;
        Ok(Exec::new(metadata, computation.to_hlo_module()))
    }
}

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
    pub client: Client,
    pub exec: PjRtLoadedExecutable,
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

    pub fn compile(self, client: Client) -> Result<Exec<Compiled>, crate::Error> {
        let comp = self.hlo_module.computation();
        let exec = client.compile(&comp)?;
        Ok(Exec {
            metadata: self.metadata,
            hlo_module: self.hlo_module,
            state: Compiled { client, exec },
        })
    }

    pub fn read_from_dir(path: impl AsRef<Path>) -> Result<Exec, crate::Error> {
        let path = path.as_ref();
        let mut metadata = File::open(path.join("metadata.json"))?;
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
    fn run(&mut self, client: &mut Buffers<PjRtBuffer>) -> Result<(), crate::Error> {
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
    pub fn compile(mut self, client: Client) -> Result<WorldExec<Compiled>, crate::Error> {
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
    pub fn run(&mut self) -> Result<(), crate::Error> {
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

    fn copy_to_client(&mut self) -> Result<(), crate::Error> {
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
                    crate::ecs::world::Column {
                        buffer: pjrt_buf,
                        entity_ids: host.entity_ids.clone(),
                    },
                );
            }
        }
        Ok(())
    }

    fn copy_to_host(&mut self) -> Result<(), crate::Error> {
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

pub struct ErasedSystem<Sys, Arg, Ret> {
    pub system: Sys,
    phantom: PhantomData<fn(Arg, Ret) -> ()>,
}

impl<Sys, Arg, Ret> ErasedSystem<Sys, Arg, Ret> {
    pub fn new(system: Sys) -> Self {
        Self {
            system,
            phantom: PhantomData,
        }
    }
}

impl<Sys, Arg, Ret> System for ErasedSystem<Sys, Arg, Ret>
where
    Sys: System<Arg = Arg, Ret = Ret>,
{
    type Arg = ();
    type Ret = ();

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), crate::Error> {
        self.system.init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, crate::Error> {
        self.system.compile(world)
    }
}

// Profiler types
#[derive(Default, Clone, Debug)]
pub struct Profiler {
    pub build: RollingMean,
    pub compile: RollingMean,
    pub copy_to_client: RollingMean,
    pub execute_buffers: RollingMean,
    pub copy_to_host: RollingMean,
    pub add_to_history: RollingMean,
}

impl Profiler {
    pub fn tick_mean(&self) -> f64 {
        self.copy_to_client.mean()
            + self.execute_buffers.mean()
            + self.copy_to_host.mean()
            + self.add_to_history.mean()
    }

    pub fn profile(&self, time_step: Duration) -> HashMap<&'static str, f64> {
        let tick_mean = self.tick_mean();
        let time_step = time_step.as_secs_f64() * 1000.0;
        let profile = [
            ("build", self.build.mean()),
            ("compile", self.compile.mean()),
            ("copy_to_client", self.copy_to_client.mean()),
            ("execute_buffers", self.execute_buffers.mean()),
            ("copy_to_host", self.copy_to_host.mean()),
            ("add_to_history", self.add_to_history.mean()),
            ("tick", tick_mean),
            ("time_step", time_step),
            ("real_time_factor", time_step / tick_mean),
        ];
        profile.into_iter().collect()
    }
}

#[derive(Default, Clone, Debug)]
pub struct RollingMean {
    sum: Duration,
    count: u32,
}

impl RollingMean {
    pub fn observe(&mut self, start: &mut Instant) {
        let sample = start.elapsed();
        self.sum += sample;
        self.count += 1;
        *start = Instant::now();
    }

    pub fn mean(&self) -> f64 {
        self.mean_duration().as_secs_f64() * 1000.0
    }

    pub fn mean_duration(&self) -> Duration {
        self.sum / self.count.max(1)
    }
}

impl std::fmt::Display for RollingMean {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.mean_duration())
    }
}

