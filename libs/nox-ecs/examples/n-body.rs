use std::f64;

use conduit::well_known::{Material, Mesh, Shape};
use nox::{tensor, SpatialMotion};
use nox::{SpatialForce, SpatialInertia, SpatialTransform};
use nox_ecs::graph::GraphQuery;
use nox_ecs::graph::TotalEdge;
use nox_ecs::{
    six_dof::*, spawn_tcp_server, ComponentArray, Integrator, Query, World, WorldExt, WorldPos,
};

const G: f64 = 6.6743e-11;

fn gravity(g: GraphQuery<TotalEdge>, q: Query<WorldPos>) -> ComponentArray<Force> {
    g.edge_fold(
        &q,
        &q,
        Force(SpatialForce::zero()),
        |acc: Force, (pos_a, pos_b): (WorldPos, WorldPos)| {
            let r = pos_a.0.linear() - pos_b.0.linear();
            let m = 1.0 / G;
            let M = 1.0 / G;
            let norm = r.norm();
            let f = G * M * m * r / (&norm * &norm * norm);
            Force(SpatialForce::from_linear(acc.0.force() - f))
        },
    )
}

fn main() {
    tracing_subscriber::fmt::init();
    let mut world = World::default();
    let mesh = world.insert_asset(Mesh::sphere(0.2, 36, 18));
    let dim = 10;
    for x in 0..dim {
        for y in 0..dim {
            for z in 0..dim {
                let dim = dim as f64;
                let theta = 2.0 * 3.14 * ((y as f64) - dim) / dim;
                let phi = 2.0 * 3.14 * ((x as f64) - dim) / dim;
                let rho = (z as f64) - dim;
                let x = rho * phi.sin() * theta.cos();
                let y = rho * phi.sin() * theta.sin();
                let z = rho * phi.cos();
                let material = world.insert_asset(Material::color(
                    2.0 * (x / dim + 1.0) as f32,
                    2.0 * (y / dim + 1.0) as f32,
                    2.0 * (z / dim + 1.0) as f32,
                ));
                world
                    .spawn(Body {
                        pos: WorldPos(SpatialTransform {
                            inner: tensor![0.0, 0.0, 0.0, 1.0, x, y, z].into(),
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
                            inner: tensor![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0 / G].into(),
                        }),
                    })
                    .insert(Shape {
                        mesh: mesh.clone(),
                        material: material.clone(),
                    });
            }
        }
    }

    let time_step = 1.0 / 240.0;
    let exec = world
        .builder()
        .tick_pipeline(six_dof(|| gravity, time_step, Integrator::SemiImplicit))
        .run_time_step(std::time::Duration::from_secs_f64(1.0 / 240.0))
        .build()
        .unwrap();
    let client = nox::Client::cpu().unwrap();
    spawn_tcp_server("0.0.0.0:2240".parse().unwrap(), exec, client, || false).unwrap();
}
