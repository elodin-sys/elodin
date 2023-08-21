use bevy_ecs::world::World;

use crate::Time;

use self::{builder::XpbdBuilder, components::Config, systems::SubstepSchedule};

pub mod body;
pub mod builder;
pub mod components;
pub mod editor;
pub mod systems;

pub struct Xpbd {
    world: World,
}

impl Default for Xpbd {
    fn default() -> Self {
        let mut world = World::new();
        world.insert_resource(Time(0.0));
        world.insert_resource(Config { dt: 0.001 });
        world.add_schedule(systems::schedule(), SubstepSchedule);
        Self { world }
    }
}

impl Xpbd {
    pub fn tick(&mut self) {
        self.world.run_schedule(SubstepSchedule)
    }

    pub fn into_world(self) -> World {
        self.world
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

pub trait SimBuilder<T, E: Env> {
    fn build(self, env: &mut E) -> XpbdBuilder;
}

impl<F, E> SimBuilder<(), E> for F
where
    E: Env,
    F: FnOnce() -> XpbdBuilder,
{
    fn build(self, _env: &mut E) -> XpbdBuilder {
        (self)()
    }
}

macro_rules! impl_sim_builder {
     ($($ty:tt),+) => {
         #[allow(non_snake_case)]
         impl<F, $($ty,)* E> SimBuilder<($($ty, )*), E> for F
         where
             E: Env,
             F: Fn($($ty, )*) -> XpbdBuilder,
             F: for<'a> Fn($(<$ty as FromEnv<E>>::Item<'a>, )*) -> XpbdBuilder,
             $($ty: FromEnv<E>, )*
         {

             fn build(self, env: &mut E) -> XpbdBuilder {

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
