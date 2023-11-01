mod assets;
mod entity;

use bevy::{
    prelude::{Assets, BuildWorldChildren, Deref, Handle, PbrBundle},
    scene::Scene,
};
use bevy_ecs::world::{Mut, World};
use std::marker::PhantomData;

pub use entity::*;

use crate::{
    bevy_transform::NoPropagate, effector::concrete_effector, sensor::Sensor, types::Time,
};

use super::{constraints::GravityConstraint, types::*};

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
    fn sense(&self, time: Time, state: EntityStateRef<'_>) {
        self.sensor.sense(time, &state)
    }
}

pub trait XpbdSensor {
    fn sense(&self, time: Time, state: EntityStateRef<'_>);
}

#[derive(Default, Clone)]
pub struct SimBuilder {
    entity_builders: Vec<EntityBuilder>,
    gravity_constraints: Vec<(RigidBodyHandle, RigidBodyHandle)>,
}

impl SimBuilder {
    pub fn entity(&mut self, entity: EntityBuilder) -> RigidBodyHandle {
        let handle = RigidBodyHandle(self.entity_builders.len());
        self.entity_builders.push(entity);
        handle
    }

    pub fn gravity(&mut self, a: RigidBodyHandle, b: RigidBodyHandle) {
        self.gravity_constraints.push((a, b));
    }

    pub fn apply(self, world: &mut World) {
        let entities = world
            .entities()
            .reserve_entities(self.entity_builders.len() as u32)
            .collect::<Vec<_>>();
        for (i, (mut builder, entity_id)) in self
            .entity_builders
            .into_iter()
            .zip(entities.iter().copied())
            .enumerate()
        {
            let pbr = if let Some(mesh) = builder.mesh.take() {
                let mut pbr = PbrBundle::default();
                let mut mesh_assets: Mut<'_, Assets<bevy::prelude::Mesh>> =
                    world.get_resource_mut().unwrap();
                pbr.mesh = mesh_assets.add(*mesh);

                if let Some(material) = builder.material.take() {
                    let mut material_assets: Mut<'_, Assets<bevy::prelude::StandardMaterial>> =
                        world.get_resource_mut().unwrap();
                    pbr.material = material_assets.add(*material);
                }
                Some(pbr)
            } else {
                None
            };

            let scene: Option<Handle<Scene>> = if let Some(scene) = builder.scene.take() {
                let server: Mut<'_, bevy::prelude::AssetServer> = world.get_resource_mut().unwrap();
                Some(server.load(&scene))
            } else {
                None
            };

            let mut entity = world.get_or_spawn(entity_id).expect("entity not found");

            if let Some(pbr) = pbr {
                entity.insert(pbr);
            }

            if let Some(anchor) = builder.trace {
                entity.insert(TraceAnchor { anchor });
            }

            if let Some(scene) = scene {
                entity.insert(scene);
            }
            // if entity_builder.editor_bundle.is_some() || entity_builder.scene.is_some() {
            //     self.queue.push(Insert {
            //         entity,
            //         bundle: (
            //             PickableBundle::default(),
            //             RaycastPickTarget::default(),
            //             On::<Pointer<Click>>::run(move |mut query: Query<&mut Picked>| {
            //                 if let Ok(mut picked) = query.get_mut(entity) {
            //                     picked.0 = !picked.0;
            //                 }
            //             }),
            //         ),
            //     })
            // }
            // TODO: Add this to editor module

            let parent = builder.parent.take();

            entity.insert((builder.bundle(), NoPropagate, Uuid(i as u128)));

            if let Some(parent) = parent {
                let parent = entities[parent.0];
                let mut parent = world.get_entity_mut(parent).expect("entity not found");
                parent.add_child(entity_id);
            }
        }

        for (a, b) in self.gravity_constraints.into_iter() {
            world.spawn(GravityConstraint::new(entities[*a], entities[*b]));
        }
    }
}

#[derive(Debug, Clone, Deref)]
pub struct RigidBodyHandle(usize);

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

pub trait SimFunc<T, E: Env, R = ()>: Send + Sync {
    fn build(&self, env: &mut E) -> R;
}

impl<F, R, E> SimFunc<(), E, R> for F
where
    E: Env,
    F: Send + Sync,
    F: Fn() -> R,
{
    fn build(&self, _env: &mut E) -> R {
        let res = (self)();
        res
    }
}

macro_rules! impl_sim_builder {
     ($($ty:tt),+) => {
         #[allow(non_snake_case)]
         impl<F, $($ty,)* E, R> SimFunc<($($ty, )*), E, R> for F
         where
             E: Env,
             F: Sync + Send,
             F: Fn($($ty, )*) -> R,
             F: for<'a> Fn($(<$ty as FromEnv<E>>::Item<'a>, )*) -> R ,
             $($ty: FromEnv<E>, )*
         {

             fn build(&self, env: &mut E) -> R {

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
impl<F, T: Send + Sync, E: Env, R> SimFunc<(), E, R> for ConcreteSimFunc<F, T, R>
where
    F: for<'s> SimFunc<T, E, R>,
{
    fn build(&self, env: &mut E) -> R {
        self.func.build(env)
    }
}
