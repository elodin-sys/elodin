extern crate self as nox_ecs;

use crate::utils::SchemaExt;
use impeller2::types::{ComponentId, EntityId};
use impeller2_wkt::{EntityMetadata, Material, Mesh};
use nox::xla::{BufferArgsRef, HloModuleProto, PjRtBuffer, PjRtLoadedExecutable};
use nox::{ArrayTy, Client, CompFn, Noxpr};
use profile::Profiler;
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;
use std::fs::File;
use std::iter::once;
use std::path::Path;
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, marker::PhantomData};

pub use nox;

mod archetype;
mod assets;
mod component;
mod dyn_array;
mod globals;
mod integrator;
mod profile;
mod query;
mod system;
pub mod utils;

pub mod arrow;
pub mod graph;
pub mod impeller2_server;
pub mod six_dof;
pub mod world;

pub use archetype::*;
pub use assets::*;
pub use component::*;
pub use dyn_array::*;
pub use globals::*;
pub use impeller2;
pub use integrator::*;
pub use query::*;
pub use system::*;
pub use world::*;

pub use elodin_db::ComponentSchema;
pub use nox_ecs_macros::{Archetype, Component};

pub use impeller2_wkt;

pub struct ComponentArray<T> {
    pub buffer: Noxpr,
    pub len: usize,
    pub entity_map: BTreeMap<EntityId, usize>,
    pub phantom_data: PhantomData<T>,
    pub component_id: ComponentId,
}

impl<T> Clone for ComponentArray<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            len: self.len,
            entity_map: self.entity_map.clone(),
            phantom_data: PhantomData,
            component_id: self.component_id,
        }
    }
}

impl<T> ComponentArray<T> {
    // NOTE: this is not generally safe to run, you should only cast `ComponentArray`,
    // when you are sure the destination type is the actual type of the inner `Op`
    pub(crate) fn cast<D>(self) -> ComponentArray<D> {
        ComponentArray {
            buffer: self.buffer,
            phantom_data: PhantomData,
            entity_map: self.entity_map,
            len: self.len,
            component_id: self.component_id,
        }
    }

    pub fn buffer(&self) -> &Noxpr {
        &self.buffer
    }
}

impl<T: Component> ComponentArray<T> {
    pub fn get(&self, offset: i64) -> T {
        let ty: ArrayTy = T::schema().to_array_ty();
        let shape = ty.shape;

        // if shape = [3, 4], start_indices = [offset, 0, 0], stop_indices = [offset + 1, 3, 4]
        let start_indices: SmallVec<_> = once(offset).chain(shape.iter().map(|_| 0)).collect();
        let stop_indices: SmallVec<_> = once(offset + 1).chain(shape.clone()).collect();
        let strides: SmallVec<_> = smallvec![1; shape.len() + 1];

        let op = self
            .buffer
            .clone()
            .slice(start_indices, stop_indices, strides)
            .reshape(shape);
        T::from_inner(op)
    }
}

impl<T: Component + 'static> crate::system::SystemParam for ComponentArray<T> {
    type Item = Self;

    fn init(builder: &mut SystemBuilder) -> Result<(), Error> {
        let id = T::COMPONENT_ID;
        builder.init_with_column(id)?;
        Ok(())
    }

    fn param(builder: &SystemBuilder) -> Result<Self::Item, Error> {
        let id = T::COMPONENT_ID;
        if let Some(var) = builder.vars.get(&id) {
            Ok(var.clone().cast())
        } else {
            Err(Error::ComponentNotFound)
        }
    }

    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::once(T::COMPONENT_ID)
    }

    fn output(&self, builder: &mut SystemBuilder) -> Result<Noxpr, Error> {
        if let Some(var) = builder.vars.get_mut(&T::COMPONENT_ID) {
            if var.entity_map != self.entity_map {
                return Ok(update_var(
                    &var.entity_map,
                    &self.entity_map,
                    &var.buffer,
                    &self.buffer,
                ));
            }
        }
        Ok(self.buffer.clone())
    }
}

pub fn update_var(
    old_entity_map: &BTreeMap<EntityId, usize>,
    update_entity_map: &BTreeMap<EntityId, usize>,
    old_buffer: &Noxpr,
    update_buffer: &Noxpr,
) -> Noxpr {
    use nox::NoxprScalarExt;
    let (old, new, _) = intersect_ids(old_entity_map, update_entity_map);
    let shape = update_buffer.shape().unwrap();
    old.iter().zip(new.iter()).fold(
        old_buffer.clone(),
        |buffer, (existing_index, update_index)| {
            let mut start = shape.clone();
            start[0] = *update_index as i64;
            for x in start.iter_mut().skip(1) {
                *x = 0;
            }
            let mut stop = shape.clone();
            stop[0] = *update_index as i64 + 1;
            let start = std::iter::once(*update_index as i64)
                .chain(std::iter::repeat(0).take(shape.len() - 1))
                .collect();
            let existing_index = std::iter::once((*existing_index as i64).constant())
                .chain(std::iter::repeat(0i64.constant()).take(shape.len() - 1))
                .collect();
            buffer.dynamic_update_slice(
                existing_index,
                update_buffer
                    .clone()
                    .slice(start, stop, shape.iter().map(|_| 1).collect()),
            )
        },
    )
}

pub trait WorldExt {
    //fn add_globals(&mut self);
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
    Sys: crate::system::System,
    StartupSys: crate::system::System,
{
    pub fn world(mut self, world: World) -> Self {
        self.world = world;
        self
    }

    pub fn tick_pipeline<M, A, R, N: system::IntoSystem<M, A, R>>(
        self,
        pipe: N,
    ) -> WorldBuilder<N::System, StartupSys> {
        WorldBuilder {
            world: self.world,
            pipe: pipe.into_system(),
            startup_sys: self.startup_sys,
        }
    }

    pub fn startup_pipeline<M, A, R, N: system::IntoSystem<M, A, R>>(
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

    pub fn spawn(&mut self, archetype: impl Archetype + 'static) -> Entity<'_> {
        self.world.spawn(archetype)
    }

    pub fn insert_with_id(&mut self, archetype: impl Archetype + 'static, entity_id: EntityId) {
        self.world.insert_with_id(archetype, entity_id);
    }

    pub fn build(mut self) -> Result<WorldExec, Error> {
        self.world.set_globals();
        let tick_exec = increment_sim_tick.pipe(self.pipe).build(&mut self.world)?;
        let startup_exec = self.startup_sys.build(&mut self.world)?;
        let world_exec = WorldExec::new(self.world, tick_exec, Some(startup_exec));
        Ok(world_exec)
    }

    // Convenience method for building, compiling, and running the world in one go.
    // This is useful for quick prototyping and testing.
    // Panicks if any of the steps fail.
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
    fn build(self, world: &mut World) -> Result<Exec, Error>;
}

impl<S: crate::system::System> SystemExt for S {
    fn build(self, world: &mut World) -> Result<Exec, Error> {
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

impl<S: ExecState> Exec<S> {
    pub fn write_to_dir(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let mut metadata = File::create(path.join("metadata.json"))?;
        serde_json::to_writer(&mut metadata, &self.metadata)?;
        std::fs::write(path.join("hlo.binpb"), self.hlo_module.to_bytes())?;
        Ok(())
    }
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

    pub fn read_from_dir(path: impl AsRef<Path>) -> Result<Exec, Error> {
        let path = path.as_ref();
        let mut metadata = File::open(path.join("metadata.json"))?;
        let metadata: ExecMetadata = serde_json::from_reader(&mut metadata)?;
        let hlo_module_data = std::fs::read(path.join("hlo.binpb"))?;
        let hlo_module = HloModuleProto::parse_binary(&hlo_module_data)?;
        Ok(Exec::new(metadata, hlo_module))
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

    pub fn write_to_dir(&mut self, dir: impl AsRef<Path>) -> Result<(), Error> {
        let start = &mut Instant::now();
        let dir = dir.as_ref();
        let world_dir = dir.join("world");
        self.tick_exec.write_to_dir(dir.join("tick_exec"))?;
        if let Some(startup_exec) = &self.startup_exec {
            startup_exec.write_to_dir(dir.join("startup_exec"))?;
        }
        self.world.write_to_dir(&world_dir)?;
        self.profiler.write_to_dir.observe(start);
        Ok(())
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

pub struct ErasedSystem<Sys, Arg, Ret> {
    system: Sys,
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

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        self.system.init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        self.system.compile(world)
    }
}

#[derive(Archetype)]
pub struct Shape {
    pub mesh: Handle<Mesh>,
    pub material: Handle<Material>,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("nox {0}")]
    Nox(#[from] nox::Error),
    #[error("component not found")]
    ComponentNotFound,
    #[error("component value had wrong size")]
    ValueSizeMismatch,
    #[error("impeller error: {0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("channel closed")]
    ChannelClosed,
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("serde_json {0}")]
    Json(#[from] serde_json::Error),
    #[cfg(feature = "pyo3")]
    #[error("python error")]
    PyO3(#[from] pyo3::PyErr),
    #[error("db {0}")]
    DB(#[from] elodin_db::Error),
    #[error("stellarator error {0}")]
    Stellar(#[from] stellarator::Error),
    #[error("arrow error {0}")]
    Arrow(#[from] ::arrow::error::ArrowError),
}

impl From<nox::xla::Error> for Error {
    fn from(value: nox::xla::Error) -> Self {
        Error::Nox(nox::Error::Xla(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Archetype, World};
    use assets::Handle;
    use impeller2_wkt::Glb;
    use nox::{Op, OwnedRepr, Scalar, Vector, tensor};
    use nox_ecs_macros::ReprMonad;

    #[test]
    fn test_simple() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Component, ReprMonad)]
        struct B<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Component, ReprMonad)]
        struct C<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Archetype)]
        struct Body {
            a: A,
            b: B,
            c: C,
        }

        fn add_system(a: Query<(A, B)>) -> Query<C> {
            a.map(|a: A, b: B| C(a.0 + b.0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(Body {
            a: A(1.0.into()),
            b: B(2.0.into()),
            c: C((-1.0).into()),
        });

        world.spawn(Body {
            a: A(2.0.into()),
            b: B(2.0.into()),
            c: C((-1.0).into()),
        });
        let world = world.run();
        let c = world.column::<C>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[3.0, 4.0])
    }

    #[test]
    fn test_get_scalar() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Component, ReprMonad)]
        struct B<R: OwnedRepr = Op>(Scalar<f64, R>);

        fn add_system(s: ComponentArray<A>, v: ComponentArray<B>) -> ComponentArray<B> {
            v.map(|v: B| B(v.0 + s.get(0).0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(A(5.0.into()));
        world.spawn(B((-1.0).into()));
        world.spawn(B(7.0.into()));
        let world = world.run();
        let v = world.column::<B>().unwrap();
        assert_eq!(v.typed_buf::<f64>().unwrap(), &[4.0, 12.0])
    }

    #[test]
    fn test_get_tensor() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Vector<f64, 3, R>);

        #[derive(Component, ReprMonad)]
        struct B<R: OwnedRepr = Op>(Vector<f64, 3, R>);

        fn add_system(s: ComponentArray<A>, v: ComponentArray<B>) -> ComponentArray<B> {
            v.map(|v: B| B(v.0 + s.get(0).0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(A(tensor![5.0, 2.0, -3.0].into()));
        world.spawn(B(tensor![-1.0, 3.5, 6.0].into()));
        world.spawn(B(tensor![7.0, -1.0, 1.0].into()));
        let world = world.run();
        let v = world.column::<B>().unwrap();
        assert_eq!(
            v.typed_buf::<f64>().unwrap(),
            &[4.0, 5.5, 3.0, 12.0, 1.0, -2.0]
        )
    }

    #[test]
    fn test_assets() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Scalar<f64, R>);

        #[derive(Archetype)]
        struct Body {
            glb: Handle<Glb>,
            a: A,
        }
        let mut world = World::default();
        let body = Body {
            glb: world.insert_asset(Glb("foo-bar".to_string())),
            a: A(1.0.into()),
        };
        world.spawn(body);
    }

    #[test]
    fn test_startup() {
        #[derive(Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Scalar<f64, R>);

        fn startup(a: ComponentArray<A>) -> ComponentArray<A> {
            a.map(|a: A| A(a.0 * 3.0)).unwrap()
        }

        fn tick(a: ComponentArray<A>) -> ComponentArray<A> {
            a.map(|a: A| A(a.0 + 1.0)).unwrap()
        }

        let mut world = World::default();
        world.spawn(A(1.0.into()));
        let world = world
            .builder()
            .tick_pipeline(tick)
            .startup_pipeline(startup)
            .run();
        let c = world.column::<A>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[4.0]);
    }
}
