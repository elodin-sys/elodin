use nox::{SpatialForce, SpatialInertia, SpatialMotion};
use nox_ecs::{Archetype, Component};
use nox_ecs::{IntoSystem, Query, System, WorldPos};
use nox_ecs_macros::{ComponentGroup, FromBuilder, IntoOp};
use std::ops::{Add, Mul};
use std::sync::Arc;

use crate::{semi_implicit_euler_with_dt, ComponentArray, ErasedSystem, Integrator, Rk4Ext, Time};

#[derive(Clone, Component)]
pub struct WorldVel(pub SpatialMotion<f64>);
#[derive(Clone, Component)]
pub struct WorldAccel(pub SpatialMotion<f64>);

#[derive(FromBuilder, ComponentGroup, IntoOp)]
struct U {
    x: WorldPos,
    v: WorldVel,
}

#[derive(FromBuilder, ComponentGroup, IntoOp)]
struct DU {
    v: WorldVel,
    a: WorldAccel,
}

impl Add<DU> for U {
    type Output = U;

    fn add(self, v: DU) -> Self::Output {
        U {
            x: WorldPos(self.x.0 + v.v.0),
            v: WorldVel(self.v.0 + v.a.0),
        }
    }
}

impl Add for DU {
    type Output = DU;

    fn add(self, v: DU) -> Self::Output {
        DU {
            v: WorldVel(self.v.0 + v.v.0),
            a: WorldAccel(self.a.0 + v.a.0),
        }
    }
}

impl Mul<DU> for f64 {
    type Output = DU;

    fn mul(self, rhs: DU) -> Self::Output {
        DU {
            v: WorldVel(self * rhs.v.0),
            a: WorldAccel(self * rhs.a.0),
        }
    }
}

impl Add<WorldVel> for WorldPos {
    type Output = WorldPos;

    fn add(self, v: WorldVel) -> Self::Output {
        WorldPos(self.0 + v.0)
    }
}

impl Add<WorldAccel> for WorldVel {
    type Output = WorldVel;

    fn add(self, v: WorldAccel) -> Self::Output {
        WorldVel(self.0 + v.0)
    }
}

impl Mul<WorldVel> for f64 {
    type Output = WorldVel;

    fn mul(self, rhs: WorldVel) -> Self::Output {
        WorldVel(self * rhs.0)
    }
}

impl Mul<WorldAccel> for f64 {
    type Output = WorldAccel;

    fn mul(self, rhs: WorldAccel) -> Self::Output {
        WorldAccel(self * rhs.0)
    }
}

#[derive(Clone, Component)]
pub struct Force(pub SpatialForce<f64>);
#[derive(Clone, Component)]
pub struct Inertia(pub SpatialInertia<f64>);

fn calc_accel(q: Query<(Force, Inertia)>) -> Query<WorldAccel> {
    q.map(|force: Force, mass: Inertia| WorldAccel(force.0 / mass.0))
        .unwrap()
}

fn clear_forces(q: ComponentArray<Force>) -> ComponentArray<Force> {
    q.map(|_| Force(SpatialForce::zero())).unwrap()
}

#[derive(Archetype)]
pub struct Body {
    pub pos: WorldPos,
    pub vel: WorldVel,
    pub accel: WorldAccel,
    pub force: Force,
    pub mass: Inertia,
}

pub fn advance_time(time_step: f64) -> impl System {
    let increment_time = move |query: ComponentArray<Time>| -> ComponentArray<Time> {
        query.map(|time: Time| Time(time.0 + time_step)).unwrap()
    };
    increment_time.into_system()
}

pub fn six_dof<Sys, M, A, R>(
    effectors: impl FnOnce() -> Sys,
    time_step: f64,
    integrator: Integrator,
) -> Arc<dyn System<Arg = (), Ret = ()> + Send + Sync>
where
    M: 'static,
    A: 'static,
    R: 'static,
    Sys: IntoSystem<M, A, R> + 'static,
    <Sys as IntoSystem<M, A, R>>::System: Send + Sync,
{
    let sys = clear_forces.pipe(effectors()).pipe(calc_accel);
    match integrator {
        Integrator::Rk4 => Arc::new(ErasedSystem::new(sys.rk4_with_dt::<U, DU>(time_step))),
        Integrator::SemiImplicit => {
            let integrate =
                semi_implicit_euler_with_dt::<WorldPos, WorldVel, WorldAccel>(time_step);
            Arc::new(ErasedSystem::new(sys.pipe(integrate)))
        }
    }
}
