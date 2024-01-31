use bytemuck::AnyBitPattern;
use elodin_conduit::{ComponentType, ComponentValue, EntityId};
use nox::xla::{BufferArgsRef, PjRtBuffer, PjRtLoadedExecutable};
use nox::{ArrayTy, Client, CompFn, Noxpr, NoxprFn, NoxprNode};
use smallvec::smallvec;
use std::any::TypeId;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::Arc;
use std::{collections::BTreeMap, marker::PhantomData};

pub use elodin_conduit;
pub use elodin_conduit::ComponentId;
pub use nox;

mod component;
mod conduit;
mod integrator;
mod query;

pub use component::*;
pub use conduit::*;
pub use integrator::*;
pub use query::*;

pub use nox_ecs_macros::{Archetype, Component};

pub struct Table<S: WorldStore> {
    pub columns: BTreeMap<ComponentId, Column<S>>,
    pub entity_buffer: S::EntityBuffer,
    pub entity_map: BTreeMap<EntityId, usize>,
}

pub struct Column<S: WorldStore> {
    pub buffer: S::Column,
}

pub struct World<S: WorldStore = HostStore> {
    pub archetypes: Vec<Table<S>>,
    pub component_map: HashMap<ComponentId, usize>,
    pub archetype_id_map: HashMap<ArchetypeId, usize>,
}

impl<S: WorldStore> Default for World<S> {
    fn default() -> Self {
        Self {
            archetypes: Default::default(),
            component_map: Default::default(),
            archetype_id_map: Default::default(),
        }
    }
}

impl<S: WorldStore> World<S> {
    pub fn column_mut<C: Component + 'static>(&mut self) -> Option<ColumnRefMut<'_, S>> {
        let Some(id) = self.component_map.get(&C::component_id()) else {
            return None;
        };
        let archetype = self.archetypes.get_mut(*id)?;
        let column = archetype.columns.get_mut(&C::component_id())?;
        Some(ColumnRefMut {
            column,
            entities: &mut archetype.entity_buffer,
            entity_map: &mut archetype.entity_map,
        })
    }

    pub fn column<C: Component + 'static>(&self) -> Option<ColumnRef<'_, S>> {
        self.column_by_id(C::component_id())
    }

    pub fn column_by_id(&self, id: ComponentId) -> Option<ColumnRef<'_, S>> {
        let table_id = self.component_map.get(&id)?;
        let archetype = self.archetypes.get(*table_id)?;
        let column = archetype.columns.get(&id)?;
        Some(ColumnRef {
            column,
            entities: &archetype.entity_buffer,
            entity_map: &archetype.entity_map,
        })
    }

    pub fn column_by_id_mut(&mut self, id: ComponentId) -> Option<ColumnRefMut<'_, S>> {
        let Some(table_id) = self.component_map.get(&id) else {
            return None;
        };
        let archetype = self.archetypes.get_mut(*table_id)?;
        let column = archetype.columns.get_mut(&id)?;
        Some(ColumnRefMut {
            column,
            entities: &mut archetype.entity_buffer,
            entity_map: &mut archetype.entity_map,
        })
    }
}

impl World<HostStore> {
    pub fn get_or_insert_archetype<A: Archetype + 'static>(&mut self) -> &mut Table<HostStore> {
        if let Some(id) = self
            .archetype_id_map
            .get(&ArchetypeId::TypeId(TypeId::of::<A>()))
        {
            &mut self.archetypes[*id]
        } else {
            self.insert_archetype::<A>()
        }
    }

    pub fn insert_archetype<A: Archetype + 'static>(&mut self) -> &mut Table<HostStore> {
        let component_ids = A::component_ids();
        let archetype_id = self.archetypes.len();
        let columns = component_ids
            .iter()
            .zip(A::component_tys().iter())
            .map(|(id, ty)| {
                (
                    *id,
                    Column {
                        buffer: HostColumn::from_ty(*ty),
                    },
                )
            })
            .collect();
        self.archetypes.push(Table {
            columns,
            entity_buffer: HostColumn::from_ty(ComponentType::U64),
            entity_map: BTreeMap::default(),
        });
        for id in component_ids {
            self.component_map.insert(id, archetype_id);
        }
        self.archetype_id_map
            .insert(ArchetypeId::TypeId(TypeId::of::<A>()), archetype_id);
        &mut self.archetypes[archetype_id]
    }

    pub fn spawn(&mut self, archetype: impl Archetype + 'static) -> EntityId {
        let entity_id = EntityId::rand();
        self.spawn_with_id(archetype, entity_id);
        entity_id
    }

    pub fn spawn_with_id<A: Archetype + 'static>(&mut self, archetype: A, entity_id: EntityId) {
        use nox::ScalarExt;
        let table = self.get_or_insert_archetype::<A>();
        table
            .entity_map
            .insert(entity_id, table.entity_buffer.len());
        table
            .entity_buffer
            .push((table.entity_buffer.len() as u64).constant());
        archetype.insert_into_table(table);
    }

    pub fn copy_to_client(&self, client: &Client) -> Result<World<ClientStore>, Error> {
        let archetypes = self
            .archetypes
            .iter()
            .map(|table| {
                let columns = table
                    .columns
                    .iter()
                    .map(|(id, column)| {
                        Ok((
                            *id,
                            Column {
                                buffer: column.buffer.copy_to_client(client)?,
                            },
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, Error>>()?;
                Ok(Table {
                    columns,
                    entity_buffer: table.entity_buffer.copy_to_client(client)?,
                    entity_map: table.entity_map.clone(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(World {
            archetypes,
            component_map: self.component_map.clone(),
            archetype_id_map: self.archetype_id_map.clone(),
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
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArchetypeId {
    Raw(u64),
    TypeId(TypeId),
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
pub struct HostStore;

impl WorldStore for HostStore {
    type Column = HostColumn;
    type EntityBuffer = HostColumn;
}

/// A type erased columnar data store located on the host CPU
pub struct HostColumn {
    buf: Vec<u8>,
    len: usize,
    component_type: ComponentType,
}

impl HostColumn {
    pub fn from_ty(ty: ComponentType) -> Self {
        HostColumn {
            buf: vec![],
            component_type: ty,
            len: 0,
        }
    }

    pub fn push<T: Component + 'static>(&mut self, val: T) {
        assert_eq!(self.component_type, T::component_type());
        let op = val.into_op();
        let NoxprNode::Constant(c) = op.deref() else {
            panic!("push into host column must be constant expr");
        };
        self.push_raw(c.data.raw_buf());
    }

    pub fn push_raw(&mut self, raw: &[u8]) {
        self.buf.extend_from_slice(raw);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn copy_to_client(&self, client: &Client) -> Result<PjRtBuffer, Error> {
        let mut dims: heapless::Vec<usize, 3> = heapless::Vec::default();
        dims.extend(self.component_type.dims().iter().map(|d| *d as usize));
        dims.push(self.len).unwrap();
        client
            .0
            .copy_raw_host_buffer(self.component_type.element_type(), &self.buf, &dims[..])
            .map_err(Error::from)
    }

    pub fn values_iter(&self) -> impl Iterator<Item = ComponentValue<'_>> + '_ {
        let mut buf_offset = 0;
        std::iter::from_fn(move || {
            let buf = self.buf.get(buf_offset..)?;
            let (offset, value) = self.component_type.parse(buf)?;
            buf_offset += offset;
            Some(value)
        })
    }

    pub fn iter<T: elodin_conduit::Component>(&self) -> impl Iterator<Item = T> + '_ {
        assert_eq!(self.component_type, T::component_type());
        self.values_iter()
            .filter_map(|v| T::from_component_value(v))
    }

    pub fn component_type(&self) -> ComponentType {
        self.component_type
    }

    pub fn raw_buf(&self) -> &[u8] {
        &self.buf
    }
}

pub struct ColumnRef<'a, S: WorldStore = HostStore> {
    pub column: &'a Column<S>,
    pub entities: &'a S::EntityBuffer,
    pub entity_map: &'a BTreeMap<EntityId, usize>,
}

impl ColumnRef<'_> {
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, ComponentValue<'_>)> {
        self.entities
            .iter::<u64>()
            .map(EntityId)
            .zip(self.column.buffer.values_iter())
    }

    pub fn typed_iter<T: elodin_conduit::Component>(
        &self,
    ) -> impl Iterator<Item = (EntityId, T)> + '_ {
        self.entities
            .iter::<u64>()
            .map(EntityId)
            .zip(self.column.buffer.iter())
    }

    pub fn typed_buf<T: AnyBitPattern>(&self) -> Option<&[T]> {
        bytemuck::try_cast_slice(self.column.buffer.buf.as_slice()).ok()
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
        let size = self.column.buffer.component_type.size()?;
        let offset = *offset * size;
        self.column.buffer.buf.get_mut(offset..offset + size)
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityId, ComponentValue<'_>)> {
        self.entities
            .iter::<u64>()
            .map(EntityId)
            .zip(self.column.buffer.values_iter())
    }
}

pub trait Archetype {
    fn component_ids() -> Vec<ComponentId>;
    fn component_tys() -> Vec<ComponentType>;
    fn insert_into_table(self, table: &mut Table<HostStore>);
}

impl<T: Component + 'static> Archetype for T {
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
    pub fn new(buffer: S::Column) -> Self {
        Self { buffer }
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

impl ComponentArray<()> {
    // NOTE: this is not generlaly safe to run, you should only cast `ComponentArray`,
    // when you are sure the destination type is the actual type of the inner `Op`
    fn cast<D: Component>(self) -> ComponentArray<D> {
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
}

impl<T: Component> ComponentArray<T> {
    fn erase_ty(self) -> ComponentArray<()> {
        ComponentArray {
            buffer: self.buffer,
            phantom_data: PhantomData,
            len: self.len,
            entity_map: self.entity_map,
        }
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
        let shape = std::iter::once(len as i64)
            .chain(T::component_type().dims().iter().copied())
            .collect();
        let op = Noxpr::parameter(
            builder.param_ops.len() as i64,
            ArrayTy {
                element_type: T::component_type().element_type(),
                shape, // FIXME
            },
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
        use nox::NoxprScalarExt;
        if let Some(var) = builder.vars.get_mut(&T::component_id()) {
            let mut var = var.borrow_mut();
            if var.entity_map != self.entity_map {
                let (old, new, _) = intersect_ids(&var.entity_map, &self.entity_map);
                let shape = self.buffer.shape().unwrap();
                let updated_buffer = old.iter().zip(new.iter()).fold(
                    var.buffer.clone(),
                    |buffer, (existing_index, update_index)| {
                        let mut start = shape.clone();
                        start[0] = *update_index as i64;
                        for x in start.iter_mut().skip(1) {
                            *x = 0;
                        }
                        let mut stop = shape.clone();
                        stop[0] = *update_index as i64 + 1;
                        buffer.dynamic_update_slice(
                            vec![(*existing_index).constant()],
                            self.buffer.clone().slice(
                                smallvec![*update_index as i64],
                                stop,
                                shape.iter().map(|_| 1).collect(),
                            ),
                        )
                    },
                );
                var.buffer = updated_buffer;
                return;
            }
        }
        builder
            .vars
            .insert(T::component_id(), self.erase_ty().into());
    }
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

pub trait System<T, R> {
    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error>;

    fn pipe<ArgB, RetB, SystemB: System<ArgB, RetB>>(
        self,
        other: SystemB,
    ) -> Pipe<T, R, ArgB, RetB, Self, SystemB>
    where
        Self: Sized,
    {
        Pipe {
            a: self,
            b: other,
            phantom_data: PhantomData,
        }
    }

    fn world(self) -> WorldBuilder<T, R, Self>
    where
        Self: Sized,
    {
        WorldBuilder::from_pipeline(self)
    }
}

impl<Arg, Ret, Sys: System<Arg, Ret>> System<Arg, Ret> for Arc<Sys> {
    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.as_ref().add_to_builder(builder)
    }
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


            impl<$($ty,)* Ret, F> System<($($ty,)*), Ret> for F
            where
                F: Fn($($ty,)*) -> Ret,
                F: for<'a> Fn($($ty::Item, )*) -> Ret,
                $($ty: SystemParam,)*
                Ret: SystemParam,
            {
                fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
                    $(
                        $ty::init(builder)?;
                    )*
                    let ret = self(
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

impl<Ret, F> System<(), Ret> for F
where
    F: Fn() -> Ret,
    Ret: SystemParam,
{
    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        let ret = self();
        ret.insert_into_builder(builder);
        Ok(())
    }
}

pub struct Pipe<TA, RA, TB, RB, A: System<TA, RA>, B: System<TB, RB>> {
    a: A,
    b: B,
    phantom_data: PhantomData<(TA, RA, TB, RB)>,
}

impl<TA, RA, TB, RB, A: System<TA, RA>, B: System<TB, RB>> System<(TA, TB), (RA, RB)>
    for Pipe<TA, RA, TB, RB, A, B>
{
    fn add_to_builder(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.a.add_to_builder(builder)?;
        self.b.add_to_builder(builder)
    }
}

pub struct WorldBuilder<Arg, Ret, Sys> {
    world: World<HostStore>,
    pipe: Sys,
    phantom_data: PhantomData<(Arg, Ret, Sys)>,
}

impl<Arg, Ret, Sys> WorldBuilder<Arg, Ret, Sys>
where
    Sys: System<Arg, Ret>,
{
    pub fn new(world: World<HostStore>, pipe: Sys) -> Self {
        WorldBuilder {
            world,
            pipe,
            phantom_data: PhantomData,
        }
    }

    pub fn from_pipeline(pipe: Sys) -> Self {
        WorldBuilder {
            world: World::default(),
            pipe,
            phantom_data: PhantomData,
        }
    }

    pub fn spawn(&mut self, archetype: impl Archetype + 'static) -> EntityId {
        self.world.spawn(archetype)
    }

    pub fn spawn_with_id(&mut self, archetype: impl Archetype + 'static, entity_id: EntityId) {
        self.world.spawn_with_id(archetype, entity_id);
    }

    pub fn build(self, client: &Client) -> Result<Exec, Error> {
        let mut builder = PipelineBuilder {
            vars: BTreeMap::default(),
            param_ids: vec![],
            param_ops: vec![],
            world: self.world,
        };
        self.pipe.add_to_builder(&mut builder)?;
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
        let exec = client.0.compile(&comp)?;
        let world = builder.world.copy_to_client(client)?;
        // ret_ids
        //     .iter()
        //     .filter_map(|id| world.column_by_id(*id).map(|c| c.buffer).clone())
        //     .collect::<Vec<_>>();
        Ok(Exec {
            client_world: world,
            arg_ids: builder.param_ids,
            ret_ids,
            exec,
            host_world: builder.world,
            loaded_components: HashSet::default(),
            dirty_components: HashSet::default(),
        })
    }
}

pub struct Exec {
    pub arg_ids: Vec<ComponentId>,
    pub ret_ids: Vec<ComponentId>,
    pub client_world: World<ClientStore>,
    pub host_world: World,
    pub loaded_components: HashSet<ComponentId>,
    pub exec: PjRtLoadedExecutable,
    pub dirty_components: HashSet<ComponentId>,
}

impl Exec {
    pub fn run(&mut self, client: &Client) -> Result<(), Error> {
        self.clear_cache();
        self.load_dirty_components(client)?;
        let mut buffers = BufferArgsRef::default().untuple_result(true);
        for id in &self.arg_ids {
            let col = self
                .client_world
                .column_by_id(*id)
                .ok_or(Error::ComponentNotFound)?;
            buffers.push(&col.column.buffer);
        }
        let ret_bufs = self.exec.execute_buffers(buffers)?;
        for (buf, comp_id) in ret_bufs.into_iter().zip(self.ret_ids.iter()) {
            let col = self
                .client_world
                .column_by_id_mut(*comp_id)
                .ok_or(Error::ComponentNotFound)?;
            col.column.buffer = buf;
        }
        Ok(())
    }

    fn load_dirty_components(&mut self, client: &Client) -> Result<(), Error> {
        for id in self.dirty_components.drain() {
            let client_column = self
                .client_world
                .column_by_id_mut(id)
                .ok_or(Error::ComponentNotFound)?;
            let host_column = self
                .host_world
                .column_by_id_mut(id)
                .ok_or(Error::ComponentNotFound)?;
            client_column
                .column
                .copy_from_host(host_column.column, client)?;
        }
        Ok(())
    }

    pub fn column_mut(&mut self, component_id: ComponentId) -> Result<ColumnRefMut<'_>, Error> {
        if !self.loaded_components.contains(&component_id) {
            self.host_world
                .load_column_from_client(component_id, &self.client_world)?;
        }
        self.dirty_components.insert(component_id);
        self.host_world
            .column_by_id_mut(component_id)
            .ok_or(Error::ComponentNotFound)
    }

    pub fn column(&mut self, component_id: ComponentId) -> Result<ColumnRef<'_>, Error> {
        if !self.loaded_components.contains(&component_id) {
            self.host_world
                .load_column_from_client(component_id, &self.client_world)?;
        }
        self.host_world
            .column_by_id(component_id)
            .ok_or(Error::ComponentNotFound)
    }

    fn clear_cache(&mut self) {
        self.loaded_components.clear();
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

impl System<(), ()> for () {
    fn add_to_builder(&self, _builder: &mut PipelineBuilder) -> Result<(), Error> {
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

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("nox {0}")]
    Nox(#[from] nox::Error),
    #[error("component not found")]
    ComponentNotFound,
    #[error("component value had wrong size")]
    ValueSizeMismatch,
    #[error("conduit error")]
    Conduit(#[from] elodin_conduit::Error),
}

impl From<nox::xla::Error> for Error {
    fn from(value: nox::xla::Error) -> Self {
        Error::Nox(nox::Error::Xla(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nox::Scalar;

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

        fn add_system(a: Query<(A, B)>) -> ComponentArray<C> {
            a.map(|a: A, b: B| C(a.0 + b.0)).unwrap()
        }

        let mut world = add_system.world();
        world.spawn(Body {
            a: A::host(1.0),
            b: B::host(2.0),
            c: C::host(-1.0),
        });

        world.spawn(Body {
            a: A::host(2.0),
            b: B::host(2.0),
            c: C::host(-1.0),
        });
        let client = nox::Client::cpu().unwrap();
        let mut exec = world.build(&client).unwrap();
        exec.run(&client).unwrap();
        let c = exec.client_world.column::<C>().unwrap();
        let lit = c.column.buffer.to_literal_sync().unwrap();
        assert_eq!(lit.typed_buf::<f64>().unwrap(), &[3.0, 4.0])
    }
}
