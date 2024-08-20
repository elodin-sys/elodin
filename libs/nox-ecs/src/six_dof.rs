use nox::{Scalar, SpatialForce, SpatialInertia, SpatialMotion};
use nox_ecs::{system::IntoSystem, system::System, Query, WorldPos};
use nox_ecs::{Archetype, Component};
use nox_ecs_macros::{ComponentGroup, FromBuilder, IntoOp};
use std::ops::{Add, Mul};
use std::sync::Arc;

use crate::{
    semi_implicit_euler, semi_implicit_euler_with_dt, ComponentArray, ErasedSystem, Integrator,
    Rk4Ext,
};

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

#[derive(Clone, Component)]
pub struct Force(pub SpatialForce<f64>);
#[derive(Clone, Component)]
pub struct Inertia(pub SpatialInertia<f64>);

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
    //let sys = clear_forces.pipe(calc_accel);
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
    //let sys = clear_forces.pipe(calc_accel);
    match integrator {
        Integrator::Rk4 => Arc::new(sys.rk4::<U, DU>()),
        Integrator::SemiImplicit => {
            let integrate = semi_implicit_euler::<WorldPos, WorldVel, WorldAccel>();
            Arc::new(ErasedSystem::new(sys.pipe(integrate)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;
    use crate::WorldExt;
    use approx::assert_relative_eq;
    use conduit::ComponentExt;
    use conduit::ComponentId;
    use nox::nalgebra::{UnitQuaternion, Vector3};
    use nox::tensor;
    use nox::ArrayRepr;
    use nox::SpatialTransform;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_six_dof_ang_vel() {
        let mut world = World::default();
        world.spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: tensor![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: tensor![0.0, 0.0, 1.0, 0.0, 0.0, 0.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: tensor![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
        });

        let time_step = 1.0 / 120.0;
        let client = nox::Client::cpu().unwrap();
        let mut exec = world
            .builder()
            .tick_pipeline(six_dof(|| (), Integrator::Rk4))
            .sim_time_step(std::time::Duration::from_secs_f64(time_step))
            .build()
            .unwrap()
            .compile(client)
            .unwrap();
        for _ in 0..120 {
            exec.run().unwrap();
        }
        let column = exec
            .column_at_tick(ComponentId::new("world_pos"), 120)
            .unwrap();
        let (_, pos) = column
            .typed_iter::<SpatialTransform<f64, ArrayRepr>>()
            .next()
            .unwrap();
        // see test-gen/julia/six-dof.jl for the source of these values
        approx::assert_relative_eq!(
            pos.inner,
            tensor![
                0.0,
                0.0,
                0.479425538604203,
                0.8775825618903728,
                0.0,
                0.0,
                0.0
            ],
            epsilon = 1e-5
        )
    }

    fn expect_angular_accel(
        client: &nox::Client,
        rot: UnitQuaternion<f64>,
        inertia_diag: [f64; 3],
        torque: [f64; 3],
        angular_accel: [f64; 3],
    ) {
        let constant_torque = move |q: ComponentArray<WorldPos>| -> ComponentArray<Force> {
            q.map(|_: WorldPos| {
                Force(SpatialForce::from_torque(tensor![
                    torque[0], torque[1], torque[2]
                ]))
            })
            .unwrap()
        };

        let quat_vec = rot.vector();
        let body = Body {
            pos: WorldPos(SpatialTransform {
                inner: tensor![quat_vec[0], quat_vec[1], quat_vec[2], rot.w, 0.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: tensor![
                    inertia_diag[0],
                    inertia_diag[1],
                    inertia_diag[2],
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
                .into(),
            }),
        };

        let mut world = World::default();
        world.spawn(body);

        let time_step = 1.0 / 120.0;
        let mut exec = world
            .builder()
            .tick_pipeline(six_dof(|| constant_torque, Integrator::Rk4))
            .sim_time_step(std::time::Duration::from_secs_f64(time_step))
            .build()
            .unwrap()
            .compile(client.clone())
            .unwrap();
        for _ in 0..120 {
            exec.run().unwrap();
        }
        let actual = exec
            .column_at_tick(WorldAccel::COMPONENT_ID, 120)
            .unwrap()
            .typed_buf::<[f64; 6]>()
            .unwrap()[0];
        approx::assert_relative_eq!(&actual[..3], &angular_accel[..], epsilon = 1e-5)
    }

    #[test]
    fn test_inertia_frame() {
        let client = nox::Client::cpu().unwrap();

        // test setup:
        // - body with inertia diag that only allows angular acceleration along x-axis (in body frame)

        // body is not rotated
        // 1 Nm torque along x-axis
        // = 1 rad/s^2 angular acceleration along x-axis
        expect_angular_accel(
            &client,
            UnitQuaternion::default(),
            [1.0, 1e6, 1e6],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        );
        // body is rotated 90 degrees around y-axis
        // 1 Nm torque along x-axis
        // = 0 angular acceleration because inertia frame is rotated
        expect_angular_accel(
            &client,
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0 * FRAC_PI_2),
            [1.0, 1e6, 1e6],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        );
        // body is rotated 90 degrees around y-axis
        // 1 Nm torque along z-axis
        // = 1 rad/s^2 angular acceleration along z-axis (in world frame)
        expect_angular_accel(
            &client,
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0 * FRAC_PI_2),
            [1.0, 1e6, 1e6],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        );
        // same as above, but with a torque along multiple axes
        expect_angular_accel(
            &client,
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0 * FRAC_PI_2),
            [1.0, 1e6, 1e6],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn test_six_dof_constant_force() {
        let mut world = World::default();
        world.spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: tensor![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            accel: WorldAccel(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            force: Force(SpatialForce {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            mass: Inertia(SpatialInertia {
                inner: tensor![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
            }),
        });

        fn constant_force(query: ComponentArray<Force>) -> ComponentArray<Force> {
            query
                .map(|_: Force| -> Force {
                    Force(SpatialForce {
                        inner: tensor![0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
                    })
                })
                .unwrap()
        }

        let time_step = 1.0 / 1.0;
        let world = world
            .builder()
            .tick_pipeline(six_dof(|| constant_force, Integrator::Rk4))
            .sim_time_step(std::time::Duration::from_secs_f64(time_step))
            .run();
        let column = world
            .column_at_tick(ComponentId::new("world_pos"), 1)
            .unwrap();
        let (_, pos) = column
            .typed_iter::<SpatialTransform<f64, ArrayRepr>>()
            .next()
            .unwrap();

        let vel = world
            .column_at_tick(ComponentId::new("world_vel"), 1)
            .unwrap();
        let (_, vel) = vel
            .typed_iter::<SpatialMotion<f64, ArrayRepr>>()
            .next()
            .unwrap();

        let accel = world
            .column_at_tick(ComponentId::new("world_accel"), 1)
            .unwrap();
        let (_, accel) = accel
            .typed_iter::<SpatialMotion<f64, ArrayRepr>>()
            .next()
            .unwrap();

        assert_eq!(accel.linear(), tensor![1.0, 0.0, 0.0]);
        assert_relative_eq!(vel.linear(), tensor![1.0, 0.0, 0.0], epsilon = 1e-6);
        assert_eq!(pos.linear(), tensor![0.5, 0.0, 0.0]);
    }
}
