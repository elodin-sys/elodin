use elodin_conduit::well_known::{Material, Mesh};
use nox::{nalgebra, SpatialForce, SpatialInertia, SpatialTransform};
use nox::{nalgebra::vector, SpatialMotion};
use nox_ecs::World;
use nox_ecs::{six_dof::*, spawn_tcp_server, Query, WorldPos};

fn gravity(pos: Query<(WorldPos, Inertia, Force)>) -> Query<Force> {
    const G: f64 = 6.649e-11;
    let big_m: f64 = 1.0 / G;
    pos.map(|world_pos: WorldPos, inertia: Inertia, force: Force| {
        let mass = inertia.0.mass();
        let r = world_pos.0.linear();
        let norm = r.clone().norm();
        let force = force.0
            + SpatialForce::from_linear(
                -r / (norm.clone() * norm.clone() * norm) * G * big_m * mass,
            );
        Force(force)
    })
    .unwrap()
}

fn main() {
    tracing_subscriber::fmt::init();
    let mut world = World::default();
    let model = world.insert_asset(Mesh::sphere(0.1, 36, 18));
    let material = world.insert_asset(Material::color(1.0, 1.0, 1.0));

    let model_b = world.insert_asset(Mesh::sphere(0.1, 36, 18));
    let material_b = world.insert_asset(Material::color(0.5, 10.0, 1.0));

    world.spawn(Body {
        pos: WorldPos(SpatialTransform {
            inner: vector![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
        }),
        vel: WorldVel(SpatialMotion {
            inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 1.0].into(),
        }),
        accel: WorldAccel(SpatialMotion {
            inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
        }),
        model,
        material,
        force: Force(SpatialForce {
            inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
        }),
        mass: Inertia(SpatialInertia {
            inner: vector![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
        }),
    });

    world.spawn(Body {
        pos: WorldPos(SpatialTransform {
            inner: vector![2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
        }),
        vel: WorldVel(SpatialMotion {
            inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 2.0].into(),
        }),
        accel: WorldAccel(SpatialMotion {
            inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
        }),
        model: model_b,
        material: material_b,
        force: Force(SpatialForce {
            inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
        }),
        mass: Inertia(SpatialInertia {
            inner: vector![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0].into(),
        }),
    });

    let time_step = 1.0 / 60.0;
    let exec = world
        .builder()
        .tick_pipeline(six_dof(|| gravity, time_step))
        .time_step(std::time::Duration::from_secs_f64(time_step))
        .build()
        .unwrap();
    let client = nox::Client::cpu().unwrap();
    spawn_tcp_server("0.0.0.0:2240".parse().unwrap(), exec, &client, || false).unwrap();
}
