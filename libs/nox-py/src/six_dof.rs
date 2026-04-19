use crate::{Archetype, Component};
use crate::{Query, WorldPos, system::IntoSystem, system::System};
use core::ops::{Add, Mul};
use elodin_macros::{ComponentGroup, FromBuilder, ReprMonad};
use nox::{Op, OwnedRepr, Scalar, SpatialForce, SpatialInertia, SpatialMotion};
use std::sync::Arc;

use crate::integrator::{Integrator, Rk4Ext, semi_implicit_euler, semi_implicit_euler_with_dt};
use crate::{ComponentArray, ErasedSystem};

#[derive(Component, ReprMonad)]
pub struct WorldVel<R: OwnedRepr = Op>(pub SpatialMotion<f64, R>);
#[derive(Component, ReprMonad)]
pub struct WorldAccel<R: OwnedRepr = Op>(pub SpatialMotion<f64, R>);

impl Clone for WorldVel {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Clone for WorldAccel {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[derive(FromBuilder, ComponentGroup)]
struct U {
    x: WorldPos,
    v: WorldVel,
}

#[derive(FromBuilder, ComponentGroup)]
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

impl Mul<DU> for Scalar<f64> {
    type Output = DU;

    fn mul(self, rhs: DU) -> Self::Output {
        DU {
            v: WorldVel(&self * rhs.v.0),
            a: WorldAccel(&self * rhs.a.0),
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

impl Mul<WorldVel> for Scalar<f64> {
    type Output = WorldVel;

    fn mul(self, rhs: WorldVel) -> Self::Output {
        WorldVel(&self * rhs.0)
    }
}

impl Mul<WorldAccel> for f64 {
    type Output = WorldAccel;

    fn mul(self, rhs: WorldAccel) -> Self::Output {
        WorldAccel(self * rhs.0)
    }
}

impl Mul<WorldAccel> for Scalar<f64> {
    type Output = WorldAccel;

    fn mul(self, rhs: WorldAccel) -> Self::Output {
        WorldAccel(&self * rhs.0)
    }
}

#[derive(Clone, Component, ReprMonad)]
pub struct Force<R: OwnedRepr = Op>(pub SpatialForce<f64, R>);
#[derive(Clone, Component, ReprMonad)]
pub struct Inertia<R: OwnedRepr = Op>(pub SpatialInertia<f64, R>);

fn calc_accel(q: Query<(Force, Inertia, WorldPos)>) -> Query<WorldAccel> {
    q.map(|force: Force, inertia: Inertia, pos: WorldPos| {
        let q = pos.0.angular();
        let body_frame_force = q.inverse() * force.0;
        let body_frame_accel = body_frame_force / inertia.0;
        let world_frame_accel = q * body_frame_accel;
        WorldAccel(world_frame_accel)
    })
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

pub fn six_dof_with_dt<Sys, M, A, R>(
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
        Integrator::Rk4 => Arc::new(sys.rk4_with_dt::<U, DU>(time_step)),
        Integrator::SemiImplicit => {
            let integrate =
                semi_implicit_euler_with_dt::<WorldPos, WorldVel, WorldAccel>(time_step);
            Arc::new(ErasedSystem::new(sys.pipe(integrate)))
        }
    }
}

pub fn six_dof<Sys, M, A, R>(
    effectors: impl FnOnce() -> Sys,
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
        Integrator::Rk4 => Arc::new(sys.rk4::<U, DU>()),
        Integrator::SemiImplicit => {
            let integrate = semi_implicit_euler::<WorldPos, WorldVel, WorldAccel>();
            Arc::new(ErasedSystem::new(sys.pipe(integrate)))
        }
    }
}

// Six-DOF integration tests validated via end-to-end: `elodin run examples/ball/main.py` etc.
