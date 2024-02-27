use elodin_conduit::well_known::Pbr;
use nox::{SpatialForce, SpatialInertia, SpatialMotion};
use nox_ecs::{Archetype, Component};
use nox_ecs::{Handle, IntoSystem, Query, Rk4Ext, System, WorldPos};
use nox_ecs_macros::{ComponentGroup, FromBuilder, IntoOp};
use std::ops::{Add, Mul};

use crate::ComponentArray;

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
    pub pbr: Handle<Pbr>,
}

pub fn six_dof<Sys, M, A, R>(effectors: impl FnOnce() -> Sys, time_step: f64) -> impl System
where
    Sys: IntoSystem<M, A, R>,
{
    let effectors = effectors();
    clear_forces
        .pipe(effectors)
        .pipe(calc_accel)
        .rk4_with_dt::<U, DU>(time_step)
}
