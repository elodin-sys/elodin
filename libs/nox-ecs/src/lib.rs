extern crate self as nox_ecs;

use conduit::well_known::{Material, Mesh};
use conduit::{Archetype, Builder, ComponentExt, ComponentId, ComponentType, EntityId, Handle};
use nox::xla::{BufferArgsRef, HloModuleProto, PjRtBuffer, PjRtLoadedExecutable};
use nox::{ArrayTy, Client, CompFn, FromOp, Noxpr, NoxprFn};
use profile::Profiler;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::iter::once;
use std::path::Path;
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, marker::PhantomData};

pub use conduit;
pub use nox;

mod component;
mod conduit_exec;
mod dyn_array;
mod history;
mod integrator;
mod profile;
mod query;

pub mod graph;
pub mod six_dof;

pub use component::*;
pub use conduit::{
    Buffers, ColumnRef, Entity, IntoSystem, Pipe, PolarsWorld, System, SystemParam, TimeStep, World,
};
pub use conduit_exec::*;
pub use dyn_array::*;
pub use integrator::*;
pub use query::*;

pub use nox_ecs_macros::{Archetype, Component};

pub struct ComponentArray<T> {
    pub buffer: Noxpr,
    pub len: usize,
    pub entity_map: BTreeMap<EntityId, usize>,
    pub phantom_data: PhantomData<T>,
}

impl<T> Clone for ComponentArray<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            len: self.len,
            entity_map: self.entity_map.clone(),
            phantom_data: PhantomData,
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
        }
    }

    pub fn buffer(&self) -> &Noxpr {
        &self.buffer
    }

    fn erase_ty(self) -> ComponentArray<()> {
        ComponentArray {
            buffer: self.buffer,
            phantom_data: PhantomData,
            len: self.len,
            entity_map: self.entity_map,
        }
    }
}

impl<T: conduit::Component + FromOp> ComponentArray<T> {
    pub fn get(&self, offset: i64) -> T {
        let ty: ArrayTy = T::component_type().into();
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
        T::from_op(op)
    }
}

impl<T: conduit::Component + 'static> SystemParam<PipelineBuilder> for ComponentArray<T> {
    type Item = ComponentArray<T>;

    fn init(builder: &mut PipelineBuilder) -> Result<(), Error> {
        let id = T::COMPONENT_ID;
        if builder.vars.contains_key(&id) {
            return Ok(());
        }
        let column = builder
            .world
            .column::<T>()
            .ok_or(Error::ComponentNotFound)?;
        let len = column.len();
        let mut ty: ArrayTy = T::component_type().into();
        ty.shape.insert(0, len as i64);
        let op = Noxpr::parameter(
            builder.param_ops.len() as i64,
            nox::NoxprTy::ArrayTy(ty),
            format!(
                "{}::{}",
                std::any::type_name::<T>(),
                builder.param_ops.len()
            ),
        );
        builder.param_ops.push(op.clone());
        builder.param_ids.push(id);
        let array = ComponentArray {
            buffer: op,
            phantom_data: PhantomData,
            len,
            entity_map: column.entity_map(),
        };
        builder.vars.insert(id, array.into());
        Ok(())
    }

    fn from_builder(builder: &PipelineBuilder) -> Self::Item {
        builder.vars[&T::COMPONENT_ID].borrow().clone().cast()
    }

    fn insert_into_builder(self, builder: &mut PipelineBuilder) {
        if let Some(var) = builder.vars.get_mut(&T::COMPONENT_ID) {
            let mut var = var.borrow_mut();
            if var.entity_map != self.entity_map {
                var.buffer =
                    update_var(&var.entity_map, &self.entity_map, &var.buffer, &self.buffer);
                return;
            }
        }
        builder.vars.insert(T::COMPONENT_ID, self.erase_ty().into());
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

#[derive(Default)]
pub struct PipelineBuilder {
    pub vars: BTreeMap<ComponentId, RefCell<ComponentArray<()>>>,
    pub param_ids: Vec<ComponentId>,
    pub param_ops: Vec<Noxpr>,
    pub world: World,
}

impl PipelineBuilder {
    pub fn from_world(world: World) -> Self {
        PipelineBuilder {
            vars: BTreeMap::default(),
            param_ids: vec![],
            param_ops: vec![],
            world,
        }
    }
}

impl Builder for PipelineBuilder {
    type Error = Error;
}

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
    Sys: System<PipelineBuilder>,
    StartupSys: System<PipelineBuilder>,
{
    pub fn world(mut self, world: World) -> Self {
        self.world = world;
        self
    }

    pub fn tick_pipeline<M, A, R, N: IntoSystem<PipelineBuilder, M, A, R>>(
        self,
        pipe: N,
    ) -> WorldBuilder<N::System, StartupSys> {
        WorldBuilder {
            world: self.world,
            pipe: pipe.into_system(),
            startup_sys: self.startup_sys,
        }
    }

    pub fn startup_pipeline<M, A, R, N: IntoSystem<PipelineBuilder, M, A, R>>(
        self,
        startup: N,
    ) -> WorldBuilder<Sys, N::System> {
        WorldBuilder {
            world: self.world,
            pipe: self.pipe,
            startup_sys: startup.into_system(),
        }
    }

    pub fn time_step(mut self, time_step: Duration) -> Self {
        self.world.time_step = TimeStep(time_step);
        self
    }

    pub fn spawn(&mut self, archetype: impl Archetype + 'static) -> Entity<'_> {
        self.world.spawn(archetype)
    }

    pub fn insert_with_id(&mut self, archetype: impl Archetype + 'static, entity_id: EntityId) {
        self.world.insert_with_id(archetype, entity_id);
    }

    pub fn build(mut self) -> Result<WorldExec, Error> {
        let tick_exec = self.pipe.build(&mut self.world)?;
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

impl<S: IntoSystem<PipelineBuilder, Marker, Arg, Ret>, Marker, Arg, Ret>
    IntoSystemExt<Marker, Arg, Ret> for S
{
    type System = <Self as IntoSystem<PipelineBuilder, Marker, Arg, Ret>>::System;

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

impl<S: System<PipelineBuilder>> SystemExt for S {
    fn build(self, world: &mut World) -> Result<Exec, Error> {
        let owned_world = std::mem::take(world);
        let mut builder = PipelineBuilder {
            vars: BTreeMap::default(),
            param_ids: vec![],
            param_ops: vec![],
            world: owned_world,
        };
        self.init_builder(&mut builder)?;
        self.add_to_builder(&mut builder)?;
        let ret = builder
            .vars
            .into_iter()
            .map(|(id, v)| (id, v.into_inner()))
            .collect::<Vec<_>>();
        let ret_ops = ret
            .iter()
            .map(|(_, v)| v.buffer.clone())
            .collect::<Vec<_>>();
        let ret_ids = ret.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let ret = Noxpr::tuple(ret_ops);
        let func = NoxprFn {
            args: builder.param_ops,
            inner: ret,
        };
        let op = func.build("pipeline")?;
        let comp = op.build()?;
        *world = builder.world;
        let metadata = ExecMetadata {
            arg_ids: builder.param_ids,
            ret_ids,
        };
        Ok(Exec::new(metadata, comp.to_hlo_module()))
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
            buffers.push(&client[id]);
        }
        let ret_bufs = self.state.exec.execute_buffers(buffers)?;
        for (buf, comp_id) in ret_bufs.into_iter().zip(self.metadata.ret_ids.iter()) {
            client.insert(*comp_id, buf);
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
        self.world.tick
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

    pub fn column_at_tick(
        &self,
        component_id: ComponentId,
        tick: u64,
    ) -> Option<ColumnRef<'_, &Vec<u8>>> {
        if tick == self.world.tick {
            return self.world.column_by_id(component_id);
        }
        let column = self.world.history.get(tick as usize)?.get(&component_id)?;
        let (archetype_name, metadata) = self.world.component_map.get(&component_id)?;
        let entities = self.world.entity_ids.get(archetype_name)?;
        Some(ColumnRef {
            column,
            entities,
            metadata,
        })
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

    pub fn read_from_dir(dir: impl AsRef<Path>) -> Result<WorldExec, Error> {
        let dir = dir.as_ref();
        let world_dir = dir.join("world");
        let tick_exec = Exec::read_from_dir(dir.join("tick_exec"))?;
        let startup_exec_path = dir.join("startup_exec");
        let startup_exec = if startup_exec_path.exists() {
            Some(Exec::read_from_dir(&startup_exec_path)?)
        } else {
            None
        };
        let world = World::read_from_dir(&world_dir)?;
        let world_exec = Self {
            world,
            client_buffers: Default::default(),
            tick_exec,
            startup_exec,
            profiler: Default::default(),
        };
        Ok(world_exec)
    }
}

impl WorldExec<Compiled> {
    pub fn run(&mut self) -> Result<(), Error> {
        let start = &mut Instant::now();
        self.copy_to_client()?;
        self.profiler.copy_to_client.observe(start);
        if let Some(mut startup_exec) = self.startup_exec.take() {
            startup_exec.run(&mut self.client_buffers)?;
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
            self.client_buffers.insert(id, pjrt_buf);
        }
        Ok(())
    }

    fn copy_to_host(&mut self) -> Result<(), Error> {
        let client = &self.tick_exec.state.client;
        for (id, pjrt_buf) in self.client_buffers.iter() {
            let host_buf = self.world.host.get_mut(id).unwrap();
            client.copy_into_host_vec(pjrt_buf, host_buf)?;
        }
        Ok(())
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.profiler.profile(self.world.time_step.0)
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
        })
    }
}

impl SystemParam<PipelineBuilder> for () {
    type Item = ();

    fn init(_builder: &mut PipelineBuilder) -> Result<(), Error> {
        Ok(())
    }

    fn from_builder(_builder: &PipelineBuilder) -> Self::Item {}

    fn insert_into_builder(self, _builder: &mut PipelineBuilder) {}
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

impl<Sys, Arg, Ret> System<PipelineBuilder> for ErasedSystem<Sys, Arg, Ret>
where
    Sys: System<PipelineBuilder, Arg = Arg, Ret = Ret>,
{
    type Arg = ();
    type Ret = ();

    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.system.add_to_builder(builder)
    }

    fn init_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.system.init_builder(builder)
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
    #[error("conduit error")]
    Conduit(#[from] conduit::Error),
    #[error("channel closed")]
    ChannelClosed,
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("serde_json {0}")]
    Json(#[from] serde_json::Error),
    #[cfg(feature = "pyo3")]
    #[error("python error")]
    PyO3(#[from] pyo3::PyErr),
}

impl From<nox::xla::Error> for Error {
    fn from(value: nox::xla::Error) -> Self {
        Error::Nox(nox::Error::Xla(value))
    }
}

impl<T> From<flume::SendError<T>> for Error {
    fn from(_: flume::SendError<T>) -> Self {
        Error::ChannelClosed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        six_dof::{Body, Force, Inertia, WorldAccel, WorldVel},
        Archetype, World, WorldPos,
    };
    use conduit::well_known::Glb;
    use nox::{
        nalgebra::{self, vector},
        Scalar, ScalarExt, SpatialForce, SpatialInertia, SpatialMotion, SpatialTransform, Vector,
    };
    use polars::{
        chunked_array::builder::{ListBuilderTrait, ListPrimitiveChunkedBuilder},
        datatypes::{DataType, Float64Type},
    };

    #[test]
    fn test_simple() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        #[derive(Component)]
        struct B(Scalar<f64>);

        #[derive(Component)]
        struct C(Scalar<f64>);

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
            a: A(1.0.constant()),
            b: B(2.0.constant()),
            c: C((-1.0).constant()),
        });

        world.spawn(Body {
            a: A(2.0.constant()),
            b: B(2.0.constant()),
            c: C((-1.0).constant()),
        });
        let world = world.run();
        let c = world.column::<C>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[3.0, 4.0])
    }

    #[test]
    fn test_get_scalar() {
        #[derive(Component)]
        struct Seed(Scalar<f64>);

        #[derive(Component)]
        struct Value(Scalar<f64>);

        fn add_system(s: ComponentArray<Seed>, v: ComponentArray<Value>) -> ComponentArray<Value> {
            v.map(|v: Value| Value(v.0 + s.get(0).0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(Seed(5.0.constant()));
        world.spawn(Value((-1.0).constant()));
        world.spawn(Value(7.0.constant()));
        let world = world.run();
        let v = world.column::<Value>().unwrap();
        assert_eq!(v.typed_buf::<f64>().unwrap(), &[4.0, 12.0])
    }

    #[test]
    fn test_get_vector() {
        #[derive(Component)]
        struct Seed(Vector<f64, 3>);

        #[derive(Component)]
        struct Value(Vector<f64, 3>);

        fn add_system(s: ComponentArray<Seed>, v: ComponentArray<Value>) -> ComponentArray<Value> {
            v.map(|v: Value| Value(v.0 + s.get(0).0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(Seed(vector![5.0, 2.0, -3.0].into()));
        world.spawn(Value(vector![-1.0, 3.5, 6.0].into()));
        world.spawn(Value(vector![7.0, -1.0, 1.0].into()));
        let world = world.run();
        let v = world.column::<Value>().unwrap();
        assert_eq!(
            v.typed_buf::<f64>().unwrap(),
            &[4.0, 5.5, 3.0, 12.0, 1.0, -2.0]
        )
    }

    #[test]
    fn test_assets() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        #[derive(Archetype)]
        struct Body {
            glb: Handle<Glb>,
            a: A,
        }
        let mut world = World::default();
        let body = Body {
            glb: world.insert_asset(Glb("foo-bar".to_string())),
            a: A(1.0.constant()),
        };
        world.spawn(body);
    }

    #[test]
    fn test_startup() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        fn startup(a: ComponentArray<A>) -> ComponentArray<A> {
            a.map(|a: A| A(a.0 * 3.0)).unwrap()
        }

        fn tick(a: ComponentArray<A>) -> ComponentArray<A> {
            a.map(|a: A| A(a.0 + 1.0)).unwrap()
        }

        let mut world = World::default();
        world.spawn(A(1.0.constant()));
        let world = world
            .builder()
            .tick_pipeline(tick)
            .startup_pipeline(startup)
            .run();
        let c = world.column::<A>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[4.0]);
    }

    #[test]
    fn test_write_read() {
        #[derive(Component)]
        struct A(Scalar<f64>);

        fn startup(a: ComponentArray<A>) -> ComponentArray<A> {
            a.map(|a: A| A(a.0 * 3.0)).unwrap()
        }

        fn tick(a: ComponentArray<A>) -> ComponentArray<A> {
            a.map(|a: A| A(a.0 + 1.0)).unwrap()
        }

        let mut world = World::default();
        world.spawn(A(1.0.constant()));
        let mut exec = world
            .builder()
            .tick_pipeline(tick)
            .startup_pipeline(startup)
            .build()
            .unwrap();
        let tempdir = tempfile::tempdir().unwrap();
        let tempdir = tempdir.path();
        exec.write_to_dir(tempdir).unwrap();
        let client = nox::Client::cpu().unwrap();
        let mut exec = WorldExec::read_from_dir(tempdir)
            .unwrap()
            .compile(client)
            .unwrap();
        exec.run().unwrap();
        let c = exec.world.column::<A>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[4.0]);
    }

    #[test]
    fn test_convert_to_df() {
        let mut world = World::default();

        world.spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: vector![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: vector![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
        });
        let polars = world.polars().unwrap();
        let df = polars.archetypes[&Body::name()].clone();
        let world_pos = df.column("world_pos").unwrap();
        let mut expected_world_pos =
            ListPrimitiveChunkedBuilder::<Float64Type>::new("world_pos", 8, 8, DataType::Float64);
        expected_world_pos.append_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let expected_world_pos = expected_world_pos.finish().into();
        assert_eq!(world_pos, &expected_world_pos);
    }

    #[test]
    fn test_to_world() {
        let mut world = World::default();

        world.spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: vector![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: vector![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
        });
        let polars = world.polars().unwrap();
        let buffers = &polars.history().unwrap()[0];
        assert_eq!(buffers, &world.host);
    }

    #[test]
    fn test_write_read_world() {
        let mut world = World::default();
        world.spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: vector![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: vector![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
        });
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        world.write_to_dir(dir).unwrap();
        let new_world = World::read_from_dir(dir).unwrap();
        assert_eq!(world, new_world);
    }
}
