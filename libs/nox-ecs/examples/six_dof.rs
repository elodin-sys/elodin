use impeller2::component::Component;
use impeller2_wkt::{Color, Line3d, Material, Mesh, Panel, Viewport};
use nox::{
    tensor, ArrayRepr, SpatialForce, SpatialInertia, SpatialMotion, SpatialTransform, Vector3,
};
use nox_ecs::{six_dof::*, Integrator, Query, World, WorldExt, WorldPos};

fn gravity(pos: Query<(WorldPos, Inertia, Force)>) -> Query<Force> {
    const G: f64 = 6.649e-11;
    let big_m: f64 = 1.0 / G;
    pos.map(|world_pos: WorldPos, inertia: Inertia, force: Force| {
        let mass = inertia.0.mass();
        let r = world_pos.0.linear();
        let mask: Vector3<f64> = tensor![1.0, 1.0, 0.0].into();
        let r = r * mask;
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
    stellarator::run(|| async {
        tracing_subscriber::fmt::init();
        let mut world = World::default();
        let shape = world.insert_shape(Mesh::sphere(0.1), Material::color(1.0, 0.0, 0.0));

        let shape_b = world.insert_shape(Mesh::sphere(0.1), Material::color(0.5, 1.0, 1.0));

        let a = world
            .spawn(Body {
                pos: WorldPos(SpatialTransform {
                    inner: tensor![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0].into(),
                }),
                vel: WorldVel(SpatialMotion {
                    inner: tensor![0.0, 0.0, 0.0, 0.0, -1.0, 0.0].into(),
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
            .insert(shape)
            .id();

        let b = world
            .spawn(Body {
                pos: WorldPos(SpatialTransform {
                    inner: tensor![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0].into(),
                }),
                vel: WorldVel(SpatialMotion {
                    inner: tensor![0.0, 0.0, 0.0, 0.0, 1.0, 2.0].into(),
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
            .insert(shape_b)
            .id();
        let panel = world.insert_asset(Panel::Viewport(Viewport {
            ..Default::default()
        }));
        world.spawn(panel);
        let asset = world.insert_asset(Line3d {
            entity: b,
            component_id: <WorldPos<ArrayRepr>>::COMPONENT_ID,
            index: [4, 5, 6],
            line_width: 10.0,
            color: Color::YOLK,
            perspective: true,
        });
        world.spawn(asset);

        let asset = world.insert_asset(Line3d {
            entity: a,
            component_id: <WorldPos<ArrayRepr>>::COMPONENT_ID,
            index: [4, 5, 6],
            line_width: 10.0,
            color: Color::MINT,
            perspective: true,
        });
        world.spawn(asset);

        let exec = world
            .builder()
            .tick_pipeline(six_dof(|| gravity, Integrator::Rk4))
            .build()
            .unwrap();
        let client = nox::Client::cpu().unwrap();
        let exec = exec.compile(client).unwrap();
        nox_ecs::impeller2_server::Server::new(
            elodin_db::Server::new("./test", "0.0.0.0:2240".parse().unwrap()).unwrap(),
            exec,
        )
        .run()
        .await
        .unwrap()
    })
}
