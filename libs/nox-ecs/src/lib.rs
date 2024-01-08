use elodin_conduit::{ComponentId, ComponentType, EntityId};
use nox::xla::{BufferArgsRef, PjRtBuffer, PjRtLoadedExecutable};
use nox::{ArrayTy, Client, CompFn, Noxpr, NoxprFn, NoxprNode};
use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use std::{collections::BTreeMap, marker::PhantomData};

pub use elodin_conduit;
pub use nox;

mod component;
mod integrator;
mod query;

pub use component::*;
pub use integrator::*;
pub use query::*;

pub use nox_ecs_macros::{Archetype, Component};

pub struct Table<S: WorldStore> {
    columns: BTreeMap<ComponentId, Column<S>>,
    entity_buffer: S::EntityBuffer,
    entity_map: BTreeMap<EntityId, usize>,
}

pub struct Column<S: WorldStore> {
    buffer: S::Column,
}

pub struct World<S: WorldStore = HostStore> {
    archetypes: Vec<Table<S>>,
    component_map: HashMap<ComponentId, usize>,
    archetype_id_map: HashMap<TypeId, usize>,
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
    pub fn column_mut<C: Component + 'static>(&mut self) -> Option<&mut Column<S>> {
        let Some(id) = self.component_map.get(&C::component_id()) else {
            return None;
        };
        let archetype = self.archetypes.get_mut(*id)?;
        archetype.columns.get_mut(&C::component_id())
    }

    pub fn column<C: Component + 'static>(&self) -> Option<&Column<S>> {
        let Some(id) = self.component_map.get(&C::component_id()) else {
            return None;
        };
        let archetype = self.archetypes.get(*id)?;
        archetype.columns.get(&C::component_id())
    }

    pub fn column_by_id(&self, id: ComponentId) -> Option<&Column<S>> {
        let Some(table_id) = self.component_map.get(&id) else {
            return None;
        };
        let archetype = self.archetypes.get(*table_id)?;
        archetype.columns.get(&id)
    }

    pub fn column_by_id_mut(&mut self, id: ComponentId) -> Option<&mut Column<S>> {
        let Some(table_id) = self.component_map.get(&id) else {
            return None;
        };
        let archetype = self.archetypes.get_mut(*table_id)?;
        archetype.columns.get_mut(&id)
    }
}

impl World<HostStore> {
    pub fn get_or_insert_archetype<A: Archetype + 'static>(&mut self) -> &mut Table<HostStore> {
        if let Some(id) = self.archetype_id_map.get(&TypeId::of::<A>()) {
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
            .insert(TypeId::of::<A>(), archetype_id);
        &mut self.archetypes[archetype_id]
    }

    pub fn spawn<A: Archetype + 'static>(&mut self, archetype: A) {
        let table = self.get_or_insert_archetype::<A>();
        archetype.insert_into_table(table);
    }

    fn copy_to_client(&self, client: &Client) -> Result<World<ClientStore>, Error> {
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
}

pub trait WorldStore {
    type Column;
    type EntityBuffer;
}

pub struct ClientStore;
impl WorldStore for ClientStore {
    type Column = PjRtBuffer;
    type EntityBuffer = PjRtBuffer;
}

pub struct HostStore;

impl WorldStore for HostStore {
    type Column = HostColumn;
    type EntityBuffer = HostColumn;
}

pub struct HostColumn {
    buf: Vec<u8>,
    len: usize,
    component_type: ComponentType,
}

impl HostColumn {
    fn from_ty(ty: ComponentType) -> Self {
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
        self.buf.extend_from_slice(c.data.raw_buf());
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

#[derive(Clone)]
pub struct ComponentArray<T> {
    buffer: Noxpr,
    view_ty: ViewTy,
    len: usize,
    phantom_data: PhantomData<T>,
}

impl ComponentArray<()> {
    // NOTE: this is not generlaly safe to run, you should only cast `ComponentArray`,
    // when you are sure the destination type is the actual type of the inner `Op`
    fn cast<D: Component>(self) -> ComponentArray<D> {
        ComponentArray {
            buffer: self.buffer,
            view_ty: self.view_ty,
            phantom_data: PhantomData,
            len: self.len,
        }
    }
}

impl<T: Component> ComponentArray<T> {
    fn erase_ty(self) -> ComponentArray<()> {
        ComponentArray {
            buffer: self.buffer,
            view_ty: self.view_ty,
            phantom_data: PhantomData,
            len: self.len,
        }
    }
}

#[derive(Clone)]
pub enum ViewTy {
    Slice { entities: Noxpr },
    Full,
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
        let len = column.buffer.len();
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
            view_ty: ViewTy::Full,
            phantom_data: PhantomData,
            len,
        };
        builder.vars.insert(id, array.into());
        Ok(())
    }

    fn from_builder(builder: &PipelineBuilder) -> Self::Item {
        builder.vars[&T::component_id()].borrow().clone().cast()
    }

    fn insert_into_builder(self, builder: &mut PipelineBuilder) {
        builder
            .vars
            .insert(T::component_id(), self.erase_ty().into());
    }
}

pub struct PipelineBuilder {
    vars: BTreeMap<ComponentId, RefCell<ComponentArray<()>>>,
    param_ids: Vec<ComponentId>,
    param_ops: Vec<Noxpr>,
    world: World<HostStore>,
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
            world,
            arg_ids: builder.param_ids,
            ret_ids,
            exec,
        })
    }
}

pub struct Exec {
    arg_ids: Vec<ComponentId>,
    ret_ids: Vec<ComponentId>,
    world: World<ClientStore>,
    exec: PjRtLoadedExecutable,
}

impl Exec {
    pub fn run(&mut self) -> Result<(), Error> {
        let mut buffers = BufferArgsRef::default().untuple_result(true);
        for id in &self.arg_ids {
            let col = self
                .world
                .column_by_id(*id)
                .ok_or(Error::ComponentNotFound)?;
            buffers.push(&col.buffer);
        }
        let ret_bufs = self.exec.execute_buffers(buffers)?;
        for (buf, comp_id) in ret_bufs.into_iter().zip(self.ret_ids.iter()) {
            let col = self
                .world
                .column_by_id_mut(*comp_id)
                .ok_or(Error::ComponentNotFound)?;
            col.buffer = buf;
        }
        Ok(())
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
            view_ty: self.view_ty.clone(),
            len: self.len,
            phantom_data: PhantomData,
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

        let mut world = World::default();
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
        let builder = WorldBuilder::new(world, add_system);
        let client = nox::Client::cpu().unwrap();
        let mut exec = builder.build(&client).unwrap();
        exec.run().unwrap();
        let c = exec.world.column::<C>().unwrap();
        let lit = c.buffer.to_literal_sync().unwrap();
        assert_eq!(lit.typed_buf::<f64>().unwrap(), &[3.0, 4.0])
    }
}
