use impeller::well_known::{Material, Mesh};
use nox::{tensor, SpatialMotion};
use nox::{SpatialForce, SpatialInertia, SpatialTransform};
use nox_ecs::{six_dof::*, spawn_tcp_server, Integrator, Query, World, WorldExt, WorldPos};

fn earth_gravity(pos: Query<(WorldPos, Inertia, Force)>) -> Query<Force> {
    pos.map(|_, _, _| {
        let force = SpatialForce::from_linear(tensor![0.0f64, -9.8, 0.0]);
        Force(force)
    })
    .unwrap()
}

fn main() {
    tracing_subscriber::fmt::init();
    let mut world = World::default();
    let shape = world.insert_shape(Mesh::sphere(0.1, 36, 18), Material::color(1.0, 1.0, 1.0));
    world
        .spawn(Body {
            pos: WorldPos(SpatialTransform {
                inner: tensor![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            vel: WorldVel(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 1.0].into(),
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
        })
        .insert(shape);

    let time_step = std::time::Duration::from_secs_f64(1.0 / 240.0);
    let exec = world
        .builder()
        .tick_pipeline(six_dof(|| earth_gravity, Integrator::Rk4))
        .sim_time_step(time_step)
        .build()
        .unwrap();
    let client = nox::Client::cpu().unwrap();
    spawn_tcp_server("0.0.0.0:2240".parse().unwrap(), exec, client, || false).unwrap();
}
