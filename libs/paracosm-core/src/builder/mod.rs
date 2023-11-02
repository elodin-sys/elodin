mod assets;
mod entity;

use bevy::{prelude::AddChild, scene::SceneBundle};
use bevy_ecs::{
    entity::Entities,
    prelude::Entity,
    system::{CommandQueue, Insert, Query, Spawn},
};
use std::cell::RefMut;
use std::marker::PhantomData;

pub use assets::*;
use bevy_mod_picking::prelude::*;
pub use entity::*;

use crate::{bevy_transform::NoPropagate, effector::concrete_effector, sensor::Sensor, Time};

use super::{constraints::GravityConstraint, editor::traces::TraceAnchor, types::*};

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
        if entity_builder.editor_bundle.is_some() || entity_builder.scene.is_some() {
            self.queue.push(Insert {
                entity,
                bundle: (
                    PickableBundle::default(),
                    RaycastPickTarget::default(),
                    On::<Pointer<Click>>::run(move |mut query: Query<&mut Picked>| {
                        if let Ok(mut picked) = query.get_mut(entity) {
                            picked.0 = !picked.0;
                        }
                    }),
                ),
            })
        }
        if let Some(pbr) = entity_builder.editor_bundle.take() {
            self.queue.push(Insert {
                entity,
                bundle: pbr,
            });
        }
        if let Some(scene) = entity_builder.scene.take() {
            self.queue.push(Insert {
                entity,
                bundle: SceneBundle {
                    scene,
                    ..Default::default()
                },
            });
        }
        if let Some(parent) = entity_builder.parent {
            self.queue.push(AddChild {
                parent,
                child: entity,
            });
        }
        self.queue.push(Insert {
            entity,
            bundle: (entity_builder.bundle(), NoPropagate),
        });
        entity
    }

    pub fn gravity_constraint(&mut self, gravity: GravityConstraint) {
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

pub trait SimFunc<T, E: Env, R = ()> {
    fn build(&self, env: &mut E) -> R;
}

macro_rules! impl_sim_builder {
     ($($ty:tt),+) => {
         #[allow(non_snake_case)]
         impl<F, $($ty,)* E, R> SimFunc<($($ty, )*), E, R> for F
         where
             E: Env,
             F: Fn($($ty, )*) -> R,
             F: for<'a> Fn($(<$ty as FromEnv<E>>::Item<'a>, )*) -> R ,
             $($ty: FromEnv<E>, )*
         {

             fn build(&self, env: &mut E) -> R{

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

pub struct ConcreteSimFunc<F, T, R> {
    func: F,
    _phantom_data: PhantomData<(T, fn() -> R)>,
}

impl<E, T, R> ConcreteSimFunc<E, T, R> {
    pub(crate) fn new(func: E) -> Self {
        Self {
            func,
            _phantom_data: PhantomData,
        }
    }
}
impl<F, T, E: Env, R> SimFunc<(), E, R> for ConcreteSimFunc<F, T, R>
where
    F: for<'s> SimFunc<T, E, R>,
{
    fn build(&self, env: &mut E) -> R {
        self.func.build(env)
    }
}
