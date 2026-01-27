#![doc = include_str!("../README.md")]
extern crate self as nox_ecs;

use crate::utils::SchemaExt;
use impeller2::types::{ComponentId, EntityId};
use impeller2_wkt::EntityMetadata;
use nox::{ArrayTy, CompFn, Noxpr};
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};
use std::iter::once;
use std::time::Duration;
use std::{collections::BTreeMap, marker::PhantomData};

pub use crate::archetype::ComponentExt;
pub use crate::component::Component;
pub use nox;
pub use nox::{DefaultRepr, Op, OwnedRepr};

mod archetype;
mod component;
mod dyn_array;
mod globals;
mod integrator;
pub mod profile;
mod query;
mod system;
pub mod utils;

pub mod graph;
// Note: impeller2_server functionality has been moved to nox-py/src/iree_server.rs
// for IREE-based execution
pub mod six_dof;
pub mod world;

pub use archetype::*;
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
        if let Some(var) = builder.vars.get_mut(&T::COMPONENT_ID)
            && var.entity_map != self.entity_map
        {
            return Ok(update_var(
                &var.entity_map,
                &self.entity_map,
                &var.buffer,
                &self.buffer,
            ));
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
                .chain(std::iter::repeat_n(0, shape.len() - 1))
                .collect();
            let existing_index = std::iter::once((*existing_index as i64).constant())
                .chain(std::iter::repeat_n(0i64.constant(), shape.len() - 1))
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

    pub fn build(mut self) -> Result<CompiledWorld, Error> {
        self.world.set_globals();
        let tick_exec = increment_sim_tick.pipe(self.pipe).compile(&self.world)?;
        let startup_exec = self.startup_sys.compile(&self.world)?;
        Ok(CompiledWorld {
            world: self.world,
            tick_exec,
            startup_exec: Some(startup_exec),
        })
    }
}

/// Holds a compiled world ready for execution.
/// The actual execution is handled by the IREE runtime in nox-py.
pub struct CompiledWorld {
    pub world: World,
    pub tick_exec: CompiledSystem,
    pub startup_exec: Option<CompiledSystem>,
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

/// Metadata for compiled system execution.
#[derive(Serialize, Deserialize, Clone)]
pub struct ExecMetadata {
    pub arg_ids: Vec<ComponentId>,
    pub ret_ids: Vec<ComponentId>,
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


// Note: Tests temporarily disabled during IREE migration.
// These tests relied on XLA execution which has been replaced by IREE.
// Tests should be re-enabled once IREE execution is fully working.
// See ai-context/iree-migration-progress.md for details.
