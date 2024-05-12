extern crate self as nox_ecs;

use bytemuck::{AnyBitPattern, Pod};
use conduit::well_known::{EntityMetadata, Material, Mesh};
use conduit::{Asset, ComponentId, ComponentType, ComponentValue, EntityId, Metadata};
use history::History;
use nox::xla::{ArrayElement, BufferArgsRef, HloModuleProto, PjRtBuffer, PjRtLoadedExecutable};
use nox::{ArrayTy, Client, CompFn, FromOp, Noxpr, NoxprFn};
use polars::PolarsWorld;
use profile::Profiler;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::iter::once;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, marker::PhantomData};

pub use conduit;
pub use nox;

mod assets;
mod component;
mod conduit_exec;
mod dyn_array;
mod host_column;
mod integrator;
mod profile;
mod query;

pub mod graph;
pub mod history;
pub mod polars;
pub mod six_dof;

pub use assets::*;
pub use component::*;
pub use conduit_exec::*;
pub use dyn_array::*;
pub use host_column::*;
pub use integrator::*;
pub use query::*;

pub use nox_ecs_macros::{Archetype, Component};

pub type ArchetypeName = ustr::Ustr;

// 16.67 ms
pub const DEFAULT_TIME_STEP: Duration = Duration::from_nanos(1_000_000_000 / 120);

pub struct Table<S: WorldStore> {
    pub columns: BTreeMap<ComponentId, S::Column>,
    pub entity_buffer: S::EntityBuffer,
}

impl Default for Table<HostStore> {
    fn default() -> Self {
        Self {
            columns: Default::default(),
            entity_buffer: HostColumn::new(EntityId::metadata()),
        }
    }
}

impl Clone for Table<HostStore> {
    fn clone(&self) -> Self {
        Self {
            columns: self.columns.clone(),
            entity_buffer: self.entity_buffer.clone(),
        }
    }
}

impl Table<HostStore> {
    pub fn entity_ids(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.entity_buffer.iter::<u64>().map(EntityId)
    }
}

impl<S: WorldStore> std::fmt::Debug for Table<S>
where
    S::EntityBuffer: std::fmt::Debug,
    S::Column: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table")
            .field("columns", &self.columns)
            .field("entity_buffer", &self.entity_buffer)
            .finish()
    }
}

pub struct World<S: WorldStore = HostStore> {
    pub archetypes: ustr::UstrMap<Table<S>>,
    pub component_map: HashMap<ComponentId, ArchetypeName>,
    pub assets: AssetStore,
    pub tick: u64,
    pub entity_len: u64,
}

impl Clone for World {
    fn clone(&self) -> Self {
        Self {
            archetypes: self.archetypes.clone(),
            component_map: self.component_map.clone(),
            assets: self.assets.clone(),
            tick: 0,
            entity_len: self.entity_len,
        }
    }
}

impl<S: WorldStore> Default for World<S> {
    fn default() -> Self {
        Self {
            archetypes: Default::default(),
            component_map: Default::default(),
            assets: Default::default(),
            tick: 0,
            entity_len: 0,
        }
    }
}

impl<S: WorldStore> World<S> {
    pub fn column_mut<C: Component + 'static>(&mut self) -> Option<ColumnRefMut<'_, S>> {
        let id = self.component_map.get(&C::component_id())?;
        let archetype = self.archetypes.get_mut(id)?;
        let column = archetype.columns.get_mut(&C::component_id())?;
        Some(ColumnRefMut {
            column,
            entities: &mut archetype.entity_buffer,
        })
    }

    pub fn column<C: Component + 'static>(&self) -> Option<HostColumnRef<'_, S>> {
        self.column_by_id(C::component_id())
    }

    pub fn column_by_id(&self, id: ComponentId) -> Option<HostColumnRef<'_, S>> {
        let table_id = self.component_map.get(&id)?;
        let archetype = self.archetypes.get(table_id)?;
        let column = archetype.columns.get(&id)?;
        Some(HostColumnRef {
            column,
            entities: &archetype.entity_buffer,
        })
    }

    pub fn column_by_id_mut(&mut self, id: ComponentId) -> Option<ColumnRefMut<'_, S>> {
        let table_id = self.component_map.get(&id)?;
        let archetype = self.archetypes.get_mut(table_id)?;
        let column = archetype.columns.get_mut(&id)?;
        Some(ColumnRefMut {
            column,
            entities: &mut archetype.entity_buffer,
        })
    }

    pub fn insert_asset<C: Asset + Send + Sync + 'static>(&mut self, asset: C) -> Handle<C> {
        self.assets.insert(asset)
    }

    pub fn insert_shape(&mut self, mesh: Mesh, material: Material) -> Shape {
        let mesh = self.insert_asset(mesh);
        let material = self.insert_asset(material);
        Shape { mesh, material }
    }
}

pub struct Entity<'a> {
    id: EntityId,
    world: &'a mut World<HostStore>,
}

impl Entity<'_> {
    pub fn metadata(self, metadata: EntityMetadata) -> Self {
        let metadata = self.world.insert_asset(metadata);
        self.world.insert_with_id(metadata, self.id);
        self
    }

    pub fn insert(self, archetype: impl Archetype + 'static) -> Self {
        self.world.insert_with_id(archetype, self.id);
        self
    }

    pub fn id(&self) -> EntityId {
        self.id
    }
}

impl From<Entity<'_>> for EntityId {
    fn from(val: Entity<'_>) -> Self {
        val.id
    }
}

impl World<HostStore> {
    pub fn get_or_insert_archetype<A: Archetype + 'static>(&mut self) -> &mut Table<HostStore> {
        let archetype_name = A::name();
        self.archetypes.entry(archetype_name).or_insert_with(|| {
            let columns = A::components()
                .into_iter()
                .map(|metadata| (metadata.component_id(), HostColumn::new(metadata)))
                .collect::<BTreeMap<_, _>>();
            for id in columns.keys() {
                self.component_map.insert(*id, archetype_name);
            }
            Table {
                columns,
                ..Default::default()
            }
        })
    }

    pub fn spawn(&mut self, archetype: impl Archetype + 'static) -> Entity<'_> {
        let entity_id = EntityId(self.entity_len);
        self.spawn_with_id(archetype, entity_id);
        Entity {
            id: entity_id,
            world: self,
        }
    }
    pub fn spawn_with_id<A: Archetype + 'static>(&mut self, archetype: A, entity_id: EntityId) {
        self.insert_with_id(archetype, entity_id);
        self.entity_len += 1;
    }

    pub fn insert_with_id<A: Archetype + 'static>(&mut self, archetype: A, entity_id: EntityId) {
        use nox::ScalarExt;
        let table = self.get_or_insert_archetype::<A>();
        table.entity_buffer.push(entity_id.0.constant());
        archetype.insert_into_table(table);
    }

    pub fn copy_to_client(&self, client: &Client) -> Result<World<ClientStore>, Error> {
        let archetypes = self
            .archetypes
            .iter()
            .map(|(id, table)| {
                let columns = table
                    .columns
                    .iter()
                    .map(|(id, column)| Ok((*id, column.copy_to_client(client)?)))
                    .collect::<Result<BTreeMap<_, _>, Error>>()?;
                let table = Table {
                    columns,
                    entity_buffer: table.entity_buffer.copy_to_client(client)?,
                };
                Ok((*id, table))
            })
            .collect::<Result<_, Error>>()?;
        Ok(World {
            archetypes,
            component_map: self.component_map.clone(),
            assets: AssetStore::default(),
            tick: self.tick,
            entity_len: self.entity_len,
        })
    }

    pub fn load_column_from_client(
        &mut self,
        id: ComponentId,
        client_world: &World<ClientStore>,
    ) -> Result<(), Error> {
        let host_column = self.column_by_id_mut(id).ok_or(Error::ComponentNotFound)?;
        let client_column = client_world
            .column_by_id(id)
            .ok_or(Error::ComponentNotFound)?;
        let literal = client_column.column.to_literal_sync()?;
        host_column.column.buf.copy_from_slice(literal.raw_buf());
        Ok(())
    }

    pub fn builder(self) -> WorldBuilder {
        WorldBuilder::default().world(self)
    }
}

pub trait WorldStore {
    type Column;
    type EntityBuffer;
}

/// A dummy struct that implements WorldStore, for the client-side, i.e the gpu
///
/// Client is an overloaded term, but here it refers to a GPU, TPU, or CPU that will be running
/// compiled XLA MLIR
pub struct ClientStore;
impl WorldStore for ClientStore {
    type Column = PjRtBuffer;
    type EntityBuffer = PjRtBuffer;
}

/// A dummy struct that implements WorldStore, for the host-side, i.e the cpu
///
/// Host here refers to the CPU that is calling the "client" (i.e a GPU / TPU). Not
/// to be confused with a host over the network.
#[derive(Debug)]
pub struct HostStore;

impl WorldStore for HostStore {
    type Column = HostColumn;
    type EntityBuffer = HostColumn;
}

pub struct HostColumnRef<'a, S: WorldStore = HostStore> {
    pub column: &'a S::Column,
    pub entities: &'a S::EntityBuffer,
}

impl HostColumnRef<'_> {
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, ComponentValue<'_>)> {
        self.entities
            .iter::<u64>()
            .map(EntityId)
            .zip(self.column.values_iter())
    }

    pub fn typed_iter<T: conduit::Component>(&self) -> impl Iterator<Item = (EntityId, T)> + '_ {
        self.entities
            .iter::<u64>()
            .map(EntityId)
            .zip(self.column.iter())
    }

    pub fn typed_buf<T: AnyBitPattern>(&self) -> Option<&[T]> {
        bytemuck::try_cast_slice(self.column.buf.as_slice()).ok()
    }

    pub fn ndarray<T: ArrayElement + Pod>(&self) -> Option<ndarray::ArrayViewD<'_, T>> {
        self.column.ndarray()
    }
}

pub struct ColumnRefMut<'a, S: WorldStore = HostStore> {
    pub column: &'a mut S::Column,
    pub entities: &'a mut S::EntityBuffer,
}

impl ColumnRefMut<'_, HostStore> {
    pub fn update(&mut self, offset: usize, value: ComponentValue<'_>) -> Result<(), Error> {
        let size = self.column.metadata.component_type.size();
        let offset = offset * size;
        let out = &mut self.column.buf[offset..offset + size];
        if let Some(bytes) = value.bytes() {
            if bytes.len() != out.len() {
                return Err(Error::ValueSizeMismatch);
            }
            out.copy_from_slice(bytes);
        }
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityId, ComponentValue<'_>)> {
        self.entities
            .iter::<u64>()
            .map(EntityId)
            .zip(self.column.values_iter())
    }

    pub fn typed_buf_mut<T: ArrayElement + Pod>(&mut self) -> Option<&mut [T]> {
        self.column.typed_buf_mut()
    }
}

pub trait Archetype {
    fn name() -> ArchetypeName;
    fn components() -> Vec<Metadata>;
    fn insert_into_table(self, table: &mut Table<HostStore>);
}

impl<T: Component + 'static> Archetype for T {
    fn name() -> ArchetypeName {
        ArchetypeName::from(T::name().as_str())
    }

    fn components() -> Vec<Metadata> {
        vec![T::metadata()]
    }

    fn insert_into_table(self, table: &mut Table<HostStore>) {
        let col = table.columns.get_mut(&T::component_id()).unwrap();
        col.push(self);
    }
}

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

impl<T: Component + FromOp> ComponentArray<T> {
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

impl<T: Component + 'static> SystemParam for ComponentArray<T> {
    type Item = ComponentArray<T>;

    fn init(builder: &mut PipelineBuilder) -> Result<(), Error> {
        let id = T::component_id();
        if builder.vars.contains_key(&id) {
            return Ok(());
        }
        let column = builder
            .world
            .column_mut::<T>()
            .ok_or(Error::ComponentNotFound)?;
        let len = column.column.len();
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
            entity_map: column.entities.entity_map(),
        };
        builder.vars.insert(id, array.into());
        Ok(())
    }

    fn from_builder(builder: &PipelineBuilder) -> Self::Item {
        builder.vars[&T::component_id()].borrow().clone().cast()
    }

    fn insert_into_builder(self, builder: &mut PipelineBuilder) {
        if let Some(var) = builder.vars.get_mut(&T::component_id()) {
            let mut var = var.borrow_mut();
            if var.entity_map != self.entity_map {
                var.buffer =
                    update_var(&var.entity_map, &self.entity_map, &var.buffer, &self.buffer);
                return;
            }
        }
        builder
            .vars
            .insert(T::component_id(), self.erase_ty().into());
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
    pub world: World<HostStore>,
}

impl PipelineBuilder {
    pub fn from_world(world: World<HostStore>) -> Self {
        PipelineBuilder {
            vars: BTreeMap::default(),
            param_ids: vec![],
            param_ops: vec![],
            world,
        }
    }
}

pub trait SystemParam {
    type Item;

    fn init(builder: &mut PipelineBuilder) -> Result<(), Error>;
    fn from_builder(builder: &PipelineBuilder) -> Self::Item;
    fn insert_into_builder(self, builder: &mut PipelineBuilder);
}

pub trait IntoSystem<Marker, Arg, Ret> {
    type System: System<Arg = Arg, Ret = Ret>;
    fn into_system(self) -> Self::System;
    fn pipe<M2, A2, R2, B: IntoSystem<M2, A2, R2>>(self, other: B) -> Pipe<Self::System, B::System>
    where
        Self: Sized,
    {
        Pipe {
            a: self.into_system(),
            b: other.into_system(),
        }
    }

    fn world(self) -> WorldBuilder<Self::System>
    where
        Self: Sized,
        Self::System: Sized,
    {
        World::default().builder().tick_pipeline(self.into_system())
    }
}

pub trait System {
    type Arg;
    type Ret;

    fn init_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error>;
    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error>;
}

impl<Sys: System> System for Arc<Sys> {
    type Arg = Sys::Arg;
    type Ret = Sys::Arg;

    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.as_ref().add_to_builder(builder)
    }

    fn init_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.as_ref().init_builder(builder)
    }
}

impl System for Arc<dyn System<Arg = (), Ret = ()> + Send + Sync> {
    type Arg = ();
    type Ret = ();

    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.as_ref().add_to_builder(builder)
    }

    fn init_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.as_ref().init_builder(builder)
    }
}

pub struct SystemFn<M, F> {
    func: F,
    phantom_data: PhantomData<M>,
}

macro_rules! impl_system_param {
      ($($ty:tt),+) => {
          #[allow(non_snake_case)]
          impl< $($ty,)* > SystemParam for ($($ty,)*)
            where $($ty: SystemParam,)*
          {
            type Item = ($($ty::Item,)*);

            fn init(builder: &mut PipelineBuilder) -> Result<(), Error> {
                $(
                    $ty::init(builder)?;
                )*
                Ok(())
            }

            fn from_builder(builder: &PipelineBuilder) -> Self::Item {
                ($(
                    $ty::from_builder(builder),
                )*)
            }

            fn insert_into_builder(self, builder: &mut PipelineBuilder) {
                let ($($ty,)*) = self;
                $(
                    $ty.insert_into_builder(builder);
                )*
            }
          }


            impl<$($ty,)* Ret, F> IntoSystem<F, ($($ty,)*), Ret> for F
            where
                F: Fn($($ty,)*) -> Ret,
                F: for<'a> Fn($($ty::Item, )*) -> Ret,
                $($ty: SystemParam,)*
                Ret: SystemParam,
            {
                type System = SystemFn<($($ty,)* Ret,), F>;
                fn into_system(self) -> Self::System {
                    SystemFn {
                        func: self,
                        phantom_data: PhantomData,
                    }
                }
            }


            impl<$($ty,)* Ret, F> System for SystemFn<($($ty,)* Ret,), F>
            where
                F: Fn($($ty,)*) -> Ret,
                F: for<'a> Fn($($ty::Item, )*) -> Ret,
                $($ty: SystemParam,)*
                Ret: SystemParam,
            {
                type Arg = ($($ty,)*);
                type Ret = Ret;
                fn init_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
                    $(
                        $ty::init(builder)?;
                    )*
                    Ok(())
                }
                fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
                    let ret = (self.func)(
                        $(
                            $ty::from_builder(builder),
                        )*
                    );
                    ret.insert_into_builder(builder);
                    Ok(())
                }
            }

      }
 }

impl_system_param!(T1);
impl_system_param!(T1, T2);
impl_system_param!(T1, T2, T3);
impl_system_param!(T1, T2, T3, T4);
impl_system_param!(T1, T2, T3, T4, T5);
impl_system_param!(T1, T2, T3, T4, T5, T6);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18);

impl<Ret, F> System for SystemFn<(Ret,), F>
where
    F: Fn() -> Ret,
    Ret: SystemParam,
{
    type Arg = ();
    type Ret = Ret;

    fn init_builder(&self, _: &mut PipelineBuilder) -> Result<(), Error> {
        Ok(())
    }

    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        let ret = (self.func)();
        ret.insert_into_builder(builder);
        Ok(())
    }
}

struct FnMarker;

impl<Ret, F> IntoSystem<FnMarker, (), Ret> for F
where
    F: Fn() -> Ret,
    Ret: SystemParam,
{
    type System = SystemFn<(Ret,), F>;

    fn into_system(self) -> Self::System {
        SystemFn {
            func: self,
            phantom_data: PhantomData,
        }
    }
}

pub struct SysMarker<S>(S);

impl<Arg, Ret, Sys> IntoSystem<SysMarker<Sys>, Arg, Ret> for Sys
where
    Sys: System<Arg = Arg, Ret = Ret>,
{
    type System = Sys;

    fn into_system(self) -> Self::System {
        self
    }
}

pub struct Pipe<A: System, B: System> {
    a: A,
    b: B,
}

impl<A: System, B: System> Pipe<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: System, B: System> System for Pipe<A, B> {
    type Arg = (A::Arg, B::Arg);
    type Ret = (A::Ret, B::Ret);

    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.a.add_to_builder(builder)?;
        self.b.add_to_builder(builder)
    }

    fn init_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.a.init_builder(builder)?;
        self.b.init_builder(builder)
    }
}

#[derive(Default)]
pub struct WorldBuilder<Sys = (), StartupSys = ()> {
    world: World<HostStore>,
    pipe: Sys,
    startup_sys: StartupSys,
    time_step: Option<Duration>,
}

impl<Sys, StartupSys> WorldBuilder<Sys, StartupSys>
where
    Sys: System,
    StartupSys: System,
{
    pub fn world(mut self, world: World<HostStore>) -> Self {
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
            time_step: self.time_step,
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
            time_step: self.time_step,
        }
    }

    pub fn time_step(mut self, time_step: Duration) -> Self {
        self.time_step = Some(time_step);
        self
    }

    pub fn spawn(&mut self, archetype: impl Archetype + 'static) -> Entity<'_> {
        self.world.spawn(archetype)
    }

    pub fn insert_with_id(&mut self, archetype: impl Archetype + 'static, entity_id: EntityId) {
        self.world.insert_with_id(archetype, entity_id);
    }

    pub fn build(mut self) -> Result<WorldExec, Error> {
        let mut tick_exec = self.pipe.build(&mut self.world)?;
        tick_exec.metadata.time_step = self.time_step;
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
        exec.world.host
    }
}

pub trait SystemExt {
    fn build(self, world: &mut World) -> Result<Exec, Error>;
}

impl<S: System> SystemExt for S {
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
            time_step: None,
            arg_ids: builder.param_ids,
            ret_ids,
        };
        Ok(Exec::new(metadata, comp.to_hlo_module()))
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ExecMetadata {
    pub time_step: Option<Duration>,
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
    fn run(&mut self, world: &mut SharedWorld) -> Result<(), Error> {
        let mut buffers = BufferArgsRef::default().untuple_result(true);
        for id in &self.metadata.arg_ids {
            let col = world
                .client
                .column_by_id(*id)
                .ok_or(Error::ComponentNotFound)?;
            buffers.push(col.column);
        }
        let ret_bufs = self.state.exec.execute_buffers(buffers)?;
        for (buf, comp_id) in ret_bufs.into_iter().zip(self.metadata.ret_ids.iter()) {
            let col = world
                .client
                .column_by_id_mut(*comp_id)
                .ok_or(Error::ComponentNotFound)?;
            *col.column = buf;
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct SharedWorld {
    host: World,
    client: World<ClientStore>,
    dirty_components: HashSet<ComponentId>,
}

impl SharedWorld {
    pub fn from_host(host: World) -> Self {
        let archetypes = host
            .archetypes
            .iter()
            .map(|(id, table)| {
                let columns = table
                    .columns
                    .keys()
                    .map(|id| (*id, PjRtBuffer::default()))
                    .collect::<BTreeMap<_, _>>();
                let table = Table::<ClientStore> {
                    columns,
                    entity_buffer: PjRtBuffer::default(),
                };
                (*id, table)
            })
            .collect();
        let client = World {
            archetypes,
            component_map: host.component_map.clone(),
            assets: AssetStore::default(),
            tick: host.tick,
            entity_len: host.entity_len,
        };
        let dirty_components = host.component_map.keys().copied().collect();
        SharedWorld {
            host,
            client,
            dirty_components,
        }
    }

    fn fork(&self) -> Self {
        Self::from_host(self.host.clone())
    }

    fn copy_to_client(&mut self, client: &Client) -> Result<(), Error> {
        for id in self.dirty_components.drain() {
            let client_column = self
                .client
                .column_by_id_mut(id)
                .ok_or(Error::ComponentNotFound)?;
            let host_column = self
                .host
                .column_by_id_mut(id)
                .ok_or(Error::ComponentNotFound)?;
            *client_column.column = host_column.column.copy_to_client(client)?;
        }
        Ok(())
    }

    fn copy_to_host(&mut self) -> Result<(), Error> {
        for (id, host_table) in &mut self.host.archetypes {
            let client_table = self
                .client
                .archetypes
                .get_mut(id)
                .ok_or(Error::ComponentNotFound)?;
            for (host, client) in host_table
                .columns
                .values_mut()
                .zip(client_table.columns.values_mut())
            {
                let literal = client.to_literal_sync()?;
                host.buf.copy_from_slice(literal.raw_buf());
            }
        }
        Ok(())
    }
}

pub struct WorldExec<S: ExecState = Uncompiled> {
    pub world: SharedWorld,
    pub tick_exec: Exec<S>,
    pub startup_exec: Option<Exec<S>>,
    pub history: History,
    pub profiler: Profiler,
}

impl<S: ExecState> std::ops::Deref for WorldExec<S> {
    type Target = World;
    fn deref(&self) -> &Self::Target {
        &self.world.host
    }
}

impl<S: ExecState> WorldExec<S> {
    pub fn new(world: World, tick_exec: Exec<S>, startup_exec: Option<Exec<S>>) -> Self {
        let mut world = Self {
            world: SharedWorld::from_host(world),
            tick_exec,
            startup_exec,
            history: Default::default(),
            profiler: Default::default(),
        };
        world.push_world().unwrap();
        world
    }

    fn push_world(&mut self) -> Result<(), Error> {
        let start = &mut Instant::now();
        let world = self.world.host.to_polars()?;
        self.history.worlds.push(world);
        self.profiler.add_to_history.observe(start);
        Ok(())
    }

    pub fn time_step(&self) -> Duration {
        self.tick_exec
            .metadata
            .time_step
            .unwrap_or(DEFAULT_TIME_STEP)
    }

    pub fn fork(&self) -> Self {
        Self {
            world: self.world.fork(),
            tick_exec: self.tick_exec.clone(),
            startup_exec: self.startup_exec.clone(),
            history: self.history.clone(),
            profiler: self.profiler.clone(),
        }
    }

    pub fn column_mut(&mut self, component_id: ComponentId) -> Result<ColumnRefMut<'_>, Error> {
        self.world
            .host
            .column_by_id_mut(component_id)
            .inspect(|_| {
                self.world.dirty_components.insert(component_id);
            })
            .ok_or(Error::ComponentNotFound)
    }

    pub fn write_to_dir(&mut self, dir: impl AsRef<Path>) -> Result<(), Error> {
        let start = &mut Instant::now();
        let dir = dir.as_ref();
        self.tick_exec.write_to_dir(dir.join("tick_exec"))?;
        if let Some(startup_exec) = &self.startup_exec {
            startup_exec.write_to_dir(dir.join("startup_exec"))?;
        }
        let mut polars_world = self.world.host.to_polars()?;
        polars_world.write_to_dir(dir.join("world"))?;
        self.history.write_to_dir(dir.join("history"))?;
        self.profiler.write_to_dir.observe(start);
        Ok(())
    }

    pub fn last_world(&self) -> &PolarsWorld {
        self.history.worlds.last().unwrap()
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
            tick_exec,
            startup_exec,
            history: self.history,
            profiler: self.profiler,
        })
    }

    pub fn read_from_dir(dir: impl AsRef<Path>) -> Result<WorldExec, Error> {
        let dir = dir.as_ref();
        let tick_exec = Exec::read_from_dir(dir.join("tick_exec"))?;
        let startup_exec_path = dir.join("startup_exec");
        let startup_exec = if startup_exec_path.exists() {
            Some(Exec::read_from_dir(&startup_exec_path)?)
        } else {
            None
        };
        let polars_world = PolarsWorld::read_from_dir(dir.join("world"))?;
        let world = World::try_from(polars_world)?;
        let world_exec = WorldExec::new(world, tick_exec, startup_exec);
        Ok(world_exec)
    }
}

impl WorldExec<Compiled> {
    pub fn run(&mut self) -> Result<(), Error> {
        let start = &mut Instant::now();
        let client = &self.tick_exec.state.client;
        self.world.copy_to_client(client)?;
        self.profiler.copy_to_client.observe(start);
        if let Some(mut startup_exec) = self.startup_exec.take() {
            startup_exec.run(&mut self.world)?;
        }
        self.tick_exec.run(&mut self.world)?;
        self.profiler.execute_buffers.observe(start);
        self.world.copy_to_host()?;
        self.profiler.copy_to_host.observe(start);
        self.world.host.tick += 1;
        self.push_world()?;
        Ok(())
    }

    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.profiler.profile(self.time_step())
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

impl System for () {
    type Arg = ();
    type Ret = ();
    fn add_to_builder(&self, _builder: &mut PipelineBuilder) -> Result<(), Error> {
        Ok(())
    }

    fn init_builder(&self, _builder: &mut PipelineBuilder) -> Result<(), Error> {
        Ok(())
    }
}

impl SystemParam for () {
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

impl<Sys, Arg, Ret> System for ErasedSystem<Sys, Arg, Ret>
where
    Sys: System<Arg = Arg, Ret = Ret>,
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

pub struct JoinSystem {
    systems: Vec<Box<dyn System<Arg = (), Ret = ()>>>,
}

impl System for JoinSystem {
    type Arg = ();
    type Ret = ();
    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        for system in &self.systems {
            system.add_to_builder(builder)?;
        }
        Ok(())
    }

    fn init_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        for system in &self.systems {
            system.init_builder(builder)?;
        }
        Ok(())
    }
}

pub trait ColumnStore {
    type Column<'a>: ColumnRef
    where
        Self: 'a;
    fn column(&self, id: ComponentId) -> Result<Self::Column<'_>, Error>;
    fn assets(&self) -> Option<&AssetStore>;
    fn tick(&self) -> u64;
}

impl ColumnStore for World {
    type Column<'a> = HostColumnRef<'a>;

    fn column(&self, id: ComponentId) -> Result<Self::Column<'_>, Error> {
        self.column_by_id(id).ok_or(Error::ComponentNotFound)
    }

    fn assets(&self) -> Option<&AssetStore> {
        Some(&self.assets)
    }

    fn tick(&self) -> u64 {
        self.tick
    }
}

pub trait ColumnRef {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn entity_buf(&self) -> Cow<'_, [u8]>;
    fn value_buf(&self) -> Cow<'_, [u8]>;
    fn is_asset(&self) -> bool;
}

impl ColumnRef for HostColumnRef<'_> {
    fn len(&self) -> usize {
        self.column.len
    }

    fn entity_buf(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.entities.buf)
    }

    fn value_buf(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.column.buf)
    }

    fn is_asset(&self) -> bool {
        self.column.metadata.asset
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
    #[error("asset not found")]
    AssetNotFound,
    #[error("channel closed")]
    ChannelClosed,
    #[error("invalid query")]
    InvalidQuery,
    #[error("entity not found")]
    EntityNotFound,
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("polars {0}")]
    Polars(#[from] ::polars::error::PolarsError),
    #[error("arrow {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("invalid component id")]
    InvalidComponentId,
    #[error("serde_json {0}")]
    Json(#[from] serde_json::Error),
    #[error("postcard {0}")]
    Postcard(#[from] postcard::Error),
    #[error("world not found")]
    WorldNotFound,
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
    use conduit::well_known::Glb;
    use nox::nalgebra::{self, vector};
    use nox::{Scalar, ScalarExt, Vector};

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
        let c = exec.column::<A>().unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[4.0]);
    }
}
