mod assets;
mod entity;

use bevy_ecs::{
    entity::Entities,
    prelude::Entity,
    system::{CommandQueue, Insert, Spawn},
};
use std::cell::RefMut;
use std::marker::PhantomData;

pub use assets::*;
pub use entity::*;

use crate::{effector::concrete_effector, sensor::Sensor, Time};

use super::{
    components::*,
    constraints::{DistanceConstraint, GravityConstriant, RevoluteJoint},
    editor::traces::TraceAnchor,
};

concrete_effector!(ConcreteEffector, XpbdEffector, EntityStateRef<'s>, Effect);

struct ConcreteSensor<ER, E> {
    sensor: ER,
    _phantom: PhantomData<(E,)>,
}

impl<ER, E> ConcreteSensor<ER, E> {
    fn new(sensor: ER) -> Self {
        Self {
            sensor,
            _phantom: PhantomData,
        }
    }
}

impl<ER, T> XpbdSensor for ConcreteSensor<ER, T>
where
    ER: for<'a> Sensor<T, EntityStateRef<'a>>,
{
    fn sense(&mut self, time: crate::Time, state: EntityStateRef<'_>) {
        self.sensor.sense(time, &state)
    }
}

pub trait XpbdSensor {
    fn sense(&mut self, time: Time, state: EntityStateRef<'_>);
}

pub struct XpbdBuilder<'a> {
    pub(crate) queue: RefMut<'a, CommandQueue>,
    pub(crate) entities: &'a Entities,
}

impl<'a> XpbdBuilder<'a> {
    pub fn entity(&mut self, mut entity_builder: EntityBuilder) -> Entity {
        let entity = self.entities.reserve_entity();
        if let Some(anchor) = entity_builder.trace {
            self.queue.push(Insert {
                entity,
                bundle: TraceAnchor { anchor },
            });
        }
        if let Some(pbr) = entity_builder.editor_bundle.take() {
            self.queue.push(Insert {
                entity,
                bundle: pbr,
            });
        }
        self.queue.push(Insert {
            entity,
            bundle: entity_builder.bundle(),
        });
        entity
    }

    pub fn distance_constraint(&mut self, distance_constriant: DistanceConstraint) {
        self.queue.push(Spawn {
            bundle: distance_constriant,
        });
    }

    pub fn revolute_join(&mut self, revolute_join: RevoluteJoint) {
        self.queue.push(Spawn {
            bundle: revolute_join,
        });
    }

    pub fn gravity_constraint(&mut self, gravity: GravityConstriant) {
        self.queue.push(Spawn { bundle: gravity });
    }
}

pub trait FromEnv<E: Env> {
    type Item<'a>
    where
        E: 'a;

    fn init(env: &mut E);

    fn from_env(env: <E as Env>::Param<'_>) -> Self::Item<'_>;
}

pub trait Env {
    type Param<'a>: Clone
    where
        Self: 'a;

    fn param(&mut self) -> Self::Param<'_>;
}

pub trait SimFunc<T, E: Env> {
    fn build(self, env: &mut E);
}

macro_rules! impl_sim_builder {
     ($($ty:tt),+) => {
         #[allow(non_snake_case)]
         impl<F, $($ty,)* E> SimFunc<($($ty, )*), E> for F
         where
             E: Env,
             F: Fn($($ty, )*),
             F: for<'a> Fn($(<$ty as FromEnv<E>>::Item<'a>, )*) ,
             $($ty: FromEnv<E>, )*
         {

             fn build(self, env: &mut E)  {

                 $(
                         $ty::init(env);
                 )*
                 let param = env.param();
                 $(
                     let $ty = $ty::from_env(param.clone());
                 )*
                 let res = (self)($($ty,)*);
                 drop(param);
                 res
             }
         }
     };
 }

impl_sim_builder!(T1);
impl_sim_builder!(T1, T2);
impl_sim_builder!(T1, T2, T3);
impl_sim_builder!(T1, T2, T3, T4);
impl_sim_builder!(T1, T2, T3, T4, T5);
impl_sim_builder!(T1, T2, T3, T4, T5, T6);
impl_sim_builder!(T1, T2, T3, T4, T5, T6, T7);
impl_sim_builder!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_sim_builder!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_sim_builder!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_sim_builder!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_sim_builder!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);
