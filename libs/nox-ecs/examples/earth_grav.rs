use elodin_conduit::well_known::{Material, Mesh};
use nox::{nalgebra, SpatialForce, SpatialInertia, SpatialTransform};
use nox::{nalgebra::vector, SpatialMotion};
use nox_ecs::World;
use nox_ecs::{six_dof::*, spawn_tcp_server, Query, WorldPos};

fn earth_gravity(pos: Query<(WorldPos, Inertia, Force)>) -> Query<Force> {
    pos.map(|_, _, _| {
        let force = SpatialForce::from_linear(vector![0.0f64, -9.8, 0.0]);
        Force(force)
    })
    .unwrap()
}

fn main() {
    tracing_subscriber::fmt::init();
    let mut world = World::default();
    let model = world.insert_asset(Mesh::sphere(0.1, 36, 18));
    let material = world.insert_asset(Material::color(1.0, 1.0, 1.0));

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
    let builder = world
        .builder()
        .tick_pipeline(six_dof(|| earth_gravity, 1.0 / 60.0));
    let client = nox::Client::cpu().unwrap();
    let exec = builder.build().unwrap();
    spawn_tcp_server(
        "0.0.0.0:2240".parse().unwrap(),
        exec,
        &client,
        std::time::Duration::from_secs_f64(1.0 / 60.0),
        || false,
    )
    .unwrap();
}
