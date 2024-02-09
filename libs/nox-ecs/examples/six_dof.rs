use elodin_conduit::well_known::{Material, Mesh};
use nox::{nalgebra, SpatialTransform};
use nox::{nalgebra::vector, SpatialMotion};
use nox_ecs::{spawn_tcp_server, Handle, Rk4Ext, WorldPos};
use nox_ecs::{Archetype, Component, World, WorldBuilder};
use nox_ecs_macros::{ComponentGroup, FromBuilder, IntoOp};
use std::ops::{Add, Mul};

#[derive(Clone, Component)]
struct V(SpatialMotion<f64>);
#[derive(Clone, Component)]
struct A(SpatialMotion<f64>);

#[derive(FromBuilder, ComponentGroup, IntoOp)]
struct U {
    x: WorldPos,
    v: V,
}

#[derive(FromBuilder, ComponentGroup, IntoOp)]
struct DU {
    v: V,
    a: A,
}

impl Add<DU> for U {
    type Output = U;

    fn add(self, v: DU) -> Self::Output {
        U {
            x: WorldPos(self.x.0 + v.v.0),
            v: V(self.v.0 + v.a.0),
        }
    }
}

impl Add for DU {
    type Output = DU;

    fn add(self, v: DU) -> Self::Output {
        DU {
            v: V(self.v.0 + v.v.0),
            a: A(self.a.0 + v.a.0),
        }
    }
}

impl Mul<DU> for f64 {
    type Output = DU;

    fn mul(self, rhs: DU) -> Self::Output {
        DU {
            v: V(self * rhs.v.0),
            a: A(self * rhs.a.0),
        }
    }
}

#[derive(Archetype)]
struct Body {
    x: WorldPos,
    v: V,
    a: A,
    model: Handle<Mesh>,
    material: Handle<Material>,
}

fn main() {
    tracing_subscriber::fmt::init();
    let mut world = World::default();
    let model = world.insert_asset(Mesh::bachs(1.0, 1.0, 1.0));
    let material = world.insert_asset(Material::color(1.0, 1.0, 1.0));

    world.spawn(Body {
        x: WorldPos(SpatialTransform {
            inner: vector![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
        }),
        v: V(SpatialMotion {
            inner: vector![1.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
        }),
        a: A(SpatialMotion {
            inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
        }),
        model,
        material,
    });

    world.spawn(Body {
        x: WorldPos(SpatialTransform {
            inner: vector![1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0].into(),
        }),
        v: V(SpatialMotion {
            inner: vector![1.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
        }),
        a: A(SpatialMotion {
            inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
        }),
        model,
        material,
    });
    let builder = WorldBuilder::new(world, ().rk4::<U, DU>());
    let client = nox::Client::cpu().unwrap();
    let exec = builder.build(&client).unwrap();
    spawn_tcp_server(
        "0.0.0.0:3104".parse().unwrap(),
        exec,
        &client,
        std::time::Duration::from_secs_f64(1.0 / 60.0),
    )
    .unwrap();
}
