extern crate self as nox_ecs;

use bytemuck::{AnyBitPattern, Pod};
use conduit::well_known::EntityMetadata;
use conduit::{Asset, ComponentId, ComponentType, ComponentValue, EntityId, Metadata};
use history::History;
use nox::xla::{ArrayElement, BufferArgsRef, HloModuleProto, PjRtBuffer, PjRtLoadedExecutable};
use nox::{ArrayTy, Client, CompFn, FromOp, Noxpr, NoxprFn};
use once_cell::sync::OnceCell;
use polars::PolarsWorld;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::iter::once;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::{collections::BTreeMap, marker::PhantomData};

pub use conduit;
pub use nox;

mod assets;
mod component;
mod conduit_exec;
mod dyn_array;
mod host_column;
mod integrator;
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
    pub columns: BTreeMap<ComponentId, Column<S>>,
    pub entity_buffer: S::EntityBuffer,
    pub entity_map: BTreeMap<EntityId, usize>,
}

impl Default for Table<HostStore> {
    fn default() -> Self {
        Self {
            columns: Default::default(),
            entity_buffer: HostColumn::entity_ids(),
            entity_map: Default::default(),
        }
    }
}

impl Clone for Table<HostStore> {
    fn clone(&self) -> Self {
        Self {
            columns: self.columns.clone(),
            entity_buffer: self.entity_buffer.clone(),
            entity_map: self.entity_map.clone(),
        }
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
            .field("entity_map", &self.entity_map)
            .finish()
    }
}

pub struct Column<S: WorldStore> {
    pub buffer: S::Column,
    pub metadata: Metadata,
}

impl Clone for Column<HostStore> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

impl<S: WorldStore> std::fmt::Debug for Column<S>
where
    S::Column: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Column")
            .field("buffer", &self.buffer)
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
            entity_map: &mut archetype.entity_map,
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
            entity_map: &archetype.entity_map,
        })
    }

    pub fn column_by_id_mut(&mut self, id: ComponentId) -> Option<ColumnRefMut<'_, S>> {
        let table_id = self.component_map.get(&id)?;
        let archetype = self.archetypes.get_mut(table_id)?;
        let column = archetype.columns.get_mut(&id)?;
        Some(ColumnRefMut {
            column,
            entities: &mut archetype.entity_buffer,
            entity_map: &mut archetype.entity_map,
        })
    }

    pub fn insert_asset<C: Asset + Send + Sync + 'static>(
        //
        &mut self,
        asset: C,
    ) -> Handle<C> {
        self.assets.insert(asset)
    }
}

pub struct Entity<'a> {
    id: EntityId,
    world: &'a mut World<HostStore>,
}

impl Entity<'_> {
    pub fn metadata(self, metadata: EntityMetadata) -> Self {
        let metadata = self.world.insert_asset(metadata);
        self.world.spawn_with_id(metadata, self.id);
        self
    }

    pub fn insert(self, archetype: impl Archetype + 'static) -> Self {
        self.world.spawn_with_id(archetype, self.id);
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
            let component_ids = A::component_ids();
            let columns = component_ids
                .iter()
                .zip(A::component_tys().iter())
                .map(|(id, ty)| {
                    (
                        *id,
                        Column {
                            buffer: HostColumn::new(ty.clone(), *id),
                            metadata: Metadata {
                                component_id: *id,
                                component_type: ty.clone(),
                                tags: HashMap::new(),
                            },
                        },
                    )
                })
                .collect();
            for id in component_ids {
                self.component_map.insert(id, archetype_name);
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
        use nox::ScalarExt;
        let table = self.get_or_insert_archetype::<A>();
        table
            .entity_map
            .insert(entity_id, table.entity_buffer.len());
        table.entity_buffer.push(entity_id.0.constant());
        archetype.insert_into_table(table);
        self.entity_len += 1;
    }

    pub fn copy_to_client(&self, client: &Client) -> Result<World<ClientStore>, Error> {
        let archetypes = self
            .archetypes
            .iter()
            .map(|(id, table)| {
                let columns = table
                    .columns
                    .iter()
                    .map(|(id, column)| {
                        Ok((
                            *id,
                            Column {
                                buffer: column.buffer.copy_to_client(client)?,
                                metadata: column.metadata.clone(),
                            },
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, Error>>()?;
                let table = Table {
                    columns,
                    entity_buffer: table.entity_buffer.copy_to_client(client)?,
                    entity_map: table.entity_map.clone(),
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
        let literal = client_column.column.buffer.to_literal_sync()?;
        host_column
            .column
            .buffer
            .buf
            .copy_from_slice(literal.raw_buf());
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
    pub column: &'a Column<S>,
    pub entities: &'a S::EntityBuffer,
    pub entity_map: &'a BTreeMap<EntityId, usize>,
}

impl HostColumnRef<'_> {
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, ComponentValue<'_>)> {
        self.entities
            .iter::<u64>()
            .map(EntityId)
            .zip(self.column.buffer.values_iter())
    }

    pub fn typed_iter<T: conduit::Component>(&self) -> impl Iterator<Item = (EntityId, T)> + '_ {
        self.entities
            .iter::<u64>()
            .map(EntityId)
            .zip(self.column.buffer.iter())
    }

    pub fn typed_buf<T: AnyBitPattern>(&self) -> Option<&[T]> {
        bytemuck::try_cast_slice(self.column.buffer.buf.as_slice()).ok()
    }

    pub fn ndarray<T: ArrayElement + Pod>(&self) -> Option<ndarray::ArrayViewD<'_, T>> {
        self.column.buffer.ndarray()
    }
}

pub struct ColumnRefMut<'a, S: WorldStore = HostStore> {
    pub column: &'a mut Column<S>,
    pub entities: &'a mut S::EntityBuffer,
    pub entity_map: &'a mut BTreeMap<EntityId, usize>,
}

impl ColumnRefMut<'_, HostStore> {
    pub fn entity_buf(&mut self, entity_id: EntityId) -> Option<&mut [u8]> {
        let offset = self.entity_map.get(&entity_id)?;
        let size = self.column.buffer.component_type.size();
        let offset = *offset * size;
        self.column.buffer.buf.get_mut(offset..offset + size)
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityId, ComponentValue<'_>)> {
        self.entities
            .iter::<u64>()
            .map(EntityId)
            .zip(self.column.buffer.values_iter())
    }

    pub fn typed_buf_mut<T: ArrayElement + Pod>(&mut self) -> Option<&mut [T]> {
        self.column.buffer.typed_buf_mut()
    }
}

pub trait Archetype {
    fn name() -> ArchetypeName;
    fn component_ids() -> Vec<ComponentId>;
    fn component_tys() -> Vec<ComponentType>;
    fn insert_into_table(self, table: &mut Table<HostStore>);
}

impl<T: Component + 'static> Archetype for T {
    fn name() -> ArchetypeName {
        ArchetypeName::from(T::name().as_str())
    }

    fn component_ids() -> Vec<ComponentId> {
        vec![T::component_id()]
    }

    fn insert_into_table(self, table: &mut Table<HostStore>) {
        let col = table.columns.get_mut(&T::component_id()).unwrap();
        col.buffer.push(self);
    }

    fn component_tys() -> Vec<ComponentType> {
        vec![T::component_type()]
    }
}

impl<S: WorldStore> Column<S> {
    pub fn new(buffer: S::Column, metadata: Metadata) -> Self {
        Self { buffer, metadata }
    }
}

impl Column<ClientStore> {
    fn copy_from_host(&mut self, host: &Column<HostStore>, client: &Client) -> Result<(), Error> {
        self.buffer = host.buffer.copy_to_client(client)?;
        Ok(())
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
        let len = column.column.buffer.len();
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
            entity_map: column.entity_map.clone(),
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

    pub fn spawn_with_id(&mut self, archetype: impl Archetype + 'static, entity_id: EntityId) {
        self.world.spawn_with_id(archetype, entity_id);
    }

    pub fn build(mut self) -> Result<WorldExec, Error> {
        let mut tick_exec = self.pipe.build(&mut self.world)?;
        tick_exec.metadata.time_step = self.time_step;
        let startup_exec = self.startup_sys.build(&mut self.world)?;
        let world = SharedWorld::from_host(self.world);
        let world_exec = WorldExec::new(world, tick_exec, Some(startup_exec));
        Ok(world_exec)
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
        Ok(Exec {
            metadata: ExecMetadata {
                time_step: None,
                arg_ids: builder.param_ids,
                ret_ids,
            },
            exec: OnceCell::new(),
            hlo_module: comp.to_hlo_module(),
        })
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ExecMetadata {
    pub time_step: Option<Duration>,
    pub arg_ids: Vec<ComponentId>,
    pub ret_ids: Vec<ComponentId>,
}

#[derive(Clone)]
pub struct Exec {
    pub metadata: ExecMetadata,
    pub hlo_module: HloModuleProto,
    pub exec: OnceCell<PjRtLoadedExecutable>,
}

impl Exec {
    fn run(&self, world: &mut SharedWorld, client: &Client) -> Result<(), Error> {
        world.clear_cache();
        world.load_dirty_components(client)?;
        let client_world = world.copy_to_client(client)?;
        let mut buffers = BufferArgsRef::default().untuple_result(true);
        for id in &self.metadata.arg_ids {
            let col = client_world
                .column_by_id(*id)
                .ok_or(Error::ComponentNotFound)?;
            buffers.push(&col.column.buffer);
        }
        let exec = self.exec.get_or_try_init(|| {
            let comp = self.hlo_module.computation();
            let exec = client.0.compile(&comp)?;
            Ok::<_, Error>(exec)
        })?;
        let ret_bufs = exec.execute_buffers(buffers)?;
        for (buf, comp_id) in ret_bufs.into_iter().zip(self.metadata.ret_ids.iter()) {
            let col = world
                .client
                .get_mut()
                .and_then(|c| c.column_by_id_mut(*comp_id))
                .ok_or(Error::ComponentNotFound)?;
            col.column.buffer = buf;
        }
        Ok(())
    }

    pub fn write_to_dir(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let mut metadata = File::create(path.join("metadata.json"))?;
        serde_json::to_writer(&mut metadata, &self.metadata)?;
        std::fs::write(path.join("hlo.binpb"), self.hlo_module.to_bytes())?;
        Ok(())
    }

    pub fn read_from_dir(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let mut metadata = File::open(path.join("metadata.json"))?;
        let metadata: ExecMetadata = serde_json::from_reader(&mut metadata)?;
        let hlo_module_data = std::fs::read(path.join("hlo.binpb"))?;
        let hlo_module = HloModuleProto::parse_binary(&hlo_module_data)?;
        Ok(Self {
            metadata,
            hlo_module,
            exec: OnceCell::default(),
        })
    }
}

#[derive(Default)]
pub struct SharedWorld {
    pub host: World,
    pub client: OnceCell<World<ClientStore>>,
    pub loaded_components: HashSet<ComponentId>,
    pub dirty_components: HashSet<ComponentId>,
}

impl SharedWorld {
    pub fn from_host(host: World) -> Self {
        SharedWorld {
            host,
            ..Default::default()
        }
    }

    fn fork(&self) -> Self {
        Self {
            host: self.host.clone(),
            ..Default::default()
        }
    }

    fn copy_to_client(&self, client: &Client) -> Result<&World<ClientStore>, Error> {
        self.client
            .get_or_try_init(|| self.host.copy_to_client(client))
    }

    fn clear_cache(&mut self) {
        self.loaded_components.clear();
    }

    fn load_dirty_components(&mut self, client: &Client) -> Result<(), Error> {
        let Some(client_world) = self.client.get_mut() else {
            return Ok(());
        };
        for id in self.dirty_components.drain() {
            let client_column = client_world
                .column_by_id_mut(id)
                .ok_or(Error::ComponentNotFound)?;
            let host_column = self
                .host
                .column_by_id_mut(id)
                .ok_or(Error::ComponentNotFound)?;
            client_column
                .column
                .copy_from_host(host_column.column, client)?;
        }
        Ok(())
    }

    fn copy_all_columns(&mut self) -> Result<(), Error> {
        let Some(client_world) = self.client.get_mut() else {
            return Ok(());
        };
        for (id, host_table) in &mut self.host.archetypes {
            let client_table = client_world
                .archetypes
                .get_mut(id)
                .ok_or(Error::ComponentNotFound)?;
            for (host, client) in host_table
                .columns
                .values_mut()
                .zip(client_table.columns.values_mut())
            {
                let literal = client.buffer.to_literal_sync()?;
                host.buffer.buf.copy_from_slice(literal.raw_buf());
                self.loaded_components.insert(host.buffer.component_id);
            }
        }
        Ok(())
    }
}

pub struct WorldExec {
    pub world: SharedWorld,
    pub tick_exec: Exec,
    pub startup_exec: Option<Exec>,
    pub history: History,
}

impl WorldExec {
    pub fn new(world: SharedWorld, tick_exec: Exec, startup_exec: Option<Exec>) -> Self {
        let mut history = History::default();
        history.push_world(&world.host).unwrap();
        Self {
            world,
            tick_exec,
            startup_exec,
            history: History::default(),
        }
    }

    pub fn run(&mut self, client: &Client) -> Result<(), Error> {
        if let Some(startup_exec) = self.startup_exec.take() {
            startup_exec.run(&mut self.world, client)?;
        }
        self.tick_exec.run(&mut self.world, client)?;
        self.world.copy_all_columns()?;
        self.world.host.tick += 1;
        self.history.push_world(&self.world.host)?;
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
        }
    }

    pub fn column_mut(&mut self, component_id: ComponentId) -> Result<ColumnRefMut<'_>, Error> {
        if !self.world.loaded_components.contains(&component_id) {
            if let Some(client_world) = self.world.client.get() {
                self.world
                    .host
                    .load_column_from_client(component_id, client_world)?;
            }
        }
        self.world
            .host
            .column_by_id_mut(component_id)
            .inspect(|_| {
                self.world.dirty_components.insert(component_id);
            })
            .ok_or(Error::ComponentNotFound)
    }

    pub fn column(&mut self, component_id: ComponentId) -> Result<HostColumnRef<'_>, Error> {
        if !self.world.loaded_components.contains(&component_id) {
            if let Some(client_world) = self.world.client.get() {
                self.world
                    .host
                    .load_column_from_client(component_id, client_world)?;
            }
            self.world.loaded_components.insert(component_id);
        }
        self.world
            .host
            .column_by_id(component_id)
            .ok_or(Error::ComponentNotFound)
    }

    pub fn cached_column(&self, component_id: ComponentId) -> Result<HostColumnRef<'_>, Error> {
        if !self.world.loaded_components.contains(&component_id) {
            return Err(Error::ComponentNotFound);
        }
        self.world
            .host
            .column_by_id(component_id)
            .ok_or(Error::ComponentNotFound)
    }

    pub fn write_to_dir(&self, dir: impl AsRef<Path>) -> Result<(), Error> {
        let dir = dir.as_ref();
        self.tick_exec.write_to_dir(dir.join("tick_exec"))?;
        if let Some(startup_exec) = &self.startup_exec {
            startup_exec.write_to_dir(dir.join("startup_exec"))?;
        }
        let mut polars_world = self.world.host.to_polars()?;
        polars_world.write_to_dir(dir.join("world"))?;
        self.history.write_to_dir(dir.join("history"))?;
        Ok(())
    }

    pub fn read_from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
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
        let world = SharedWorld::from_host(world);
        let world_exec = WorldExec::new(world, tick_exec, startup_exec);
        Ok(world_exec)
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
    fn transfer_column(&mut self, id: ComponentId) -> Result<(), Error>;
    fn column(&self, id: ComponentId) -> Result<Self::Column<'_>, Error>;
    fn assets(&self) -> Option<&AssetStore>;
    fn tick(&self) -> u64;
}

impl ColumnStore for WorldExec {
    type Column<'a> = HostColumnRef<'a>;

    fn transfer_column(&mut self, id: ComponentId) -> Result<(), Error> {
        let _ = WorldExec::column(self, id)?;
        Ok(())
    }

    fn column(&self, id: ComponentId) -> Result<Self::Column<'_>, Error> {
        self.cached_column(id)
    }

    fn assets(&self) -> Option<&AssetStore> {
        Some(&self.world.host.assets)
    }

    fn tick(&self) -> u64 {
        self.world.host.tick
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
        self.column.buffer.len
    }

    fn entity_buf(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.entities.buf)
    }

    fn value_buf(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.column.buffer.buf)
    }

    fn is_asset(&self) -> bool {
        self.column.buffer.asset
    }
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
    use conduit::well_known::Pbr;
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
        let client = nox::Client::cpu().unwrap();
        let mut exec = world.build().unwrap();
        exec.run(&client).unwrap();
        let c = exec.column(C::component_id()).unwrap();
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
        let client = nox::Client::cpu().unwrap();
        let mut exec = world.build().unwrap();
        exec.run(&client).unwrap();
        let v = exec.column(Value::component_id()).unwrap();
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
        let client = nox::Client::cpu().unwrap();
        let mut exec = world.build().unwrap();
        exec.run(&client).unwrap();
        let v = exec.column(Value::component_id()).unwrap();
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
            pbr: Handle<Pbr>,
            a: A,
        }
        let mut world = World::default();
        let body = Body {
            pbr: world.insert_asset(Pbr::Url("foo-bar".to_string())),
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
        let client = nox::Client::cpu().unwrap();
        let mut exec = world
            .builder()
            .tick_pipeline(tick)
            .startup_pipeline(startup)
            .build()
            .unwrap();
        exec.run(&client).unwrap();
        let c = exec.column(A::component_id()).unwrap();
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
        let client = nox::Client::cpu().unwrap();
        let exec = world
            .builder()
            .tick_pipeline(tick)
            .startup_pipeline(startup)
            .build()
            .unwrap();
        let tempdir = tempfile::tempdir().unwrap();
        let tempdir = tempdir.path();
        exec.write_to_dir(&tempdir).unwrap();
        let mut exec = WorldExec::read_from_dir(&tempdir).unwrap();
        exec.run(&client).unwrap();
        let c = exec.column(A::component_id()).unwrap();
        assert_eq!(c.typed_buf::<f64>().unwrap(), &[4.0]);
    }
}
