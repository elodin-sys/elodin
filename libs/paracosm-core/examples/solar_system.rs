use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::xpbd::{
    builder::{Assets, EntityBuilder, XpbdBuilder},
    constraints::GravityConstraint,
    editor::editor,
    runner::IntoSimRunner,
};

fn main() {
    editor(sim.substep_count(64).scale(2.).delta_t(100. / 60.))
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets) {
    let sun = builder.entity(
        EntityBuilder::default()
            .fixed()
            .mass(1.99e30)
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.2,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 188.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );
    let earth = builder.entity(
        EntityBuilder::default()
            .mass(5.972e24)
            .pos(helio_to_bevy(vector![
                9.932794033922092e-1,
                -8.115895094412964e-2,
                2.123378844575767e-4,
            ]))
            .vel(helio_to_bevy(vector![
                1.072468396698360e-3,
                1.708345322037895e-2,
                -3.545157932769615e-7
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(65.0 / 255.0 * 20., 187.0 / 255.0 * 20., 20.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let jupiter = builder.entity(
        EntityBuilder::default()
            .mass(1.89e27)
            .pos(helio_to_bevy(vector![
                4.006423068610660e00,
                2.921866864588122e00,
                -1.017570096026684e-01
            ]))
            .vel(helio_to_bevy(vector![
                -4.529782018388927e-03,
                6.452946144697471e-03,
                7.457109084988199e-05
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 145.0 / 255.0 * 20., 44.0 / 255.0 * 20.),
                base_color: Color::hex("FF912C").unwrap(),
                ..Default::default()
            })),
    );

    let mars = builder.entity(
        EntityBuilder::default()
            .mass(6.39e23)
            .pos(helio_to_bevy(vector![
                -1.455093533090440e+00,
                -7.026325579976354e-01,
                2.101911966287818e-02,
            ]))
            .vel(helio_to_bevy(vector![
                6.625189920311338e-03,
                -1.140793077770015e-02,
                -4.013568238263378e-04
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 15.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let mercury = builder.entity(
        EntityBuilder::default()
            .mass(4.867e24)
            .pos(helio_to_bevy(vector![
                2.028847101112924e-01,
                2.308731142232934e-01,
                -1.276223148323567e-04
            ]))
            .vel(helio_to_bevy(vector![
                -2.640829017559176e-02,
                2.006600272464892e-02,
                4.063076271928278e-03
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 188.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let venus = builder.entity(
        EntityBuilder::default()
            .mass(4.867e24)
            .pos(helio_to_bevy(vector![
                6.798017112259706e-01,
                2.237111260436163e-01,
                -3.639561594712085e-02
            ]))
            .vel(helio_to_bevy(vector![
                -6.371222592894960e-03,
                1.912020851757406e-02,
                6.305071804394223e-04
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 188.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let saturn = builder.entity(
        EntityBuilder::default()
            .mass(5.683e26)
            .pos(helio_to_bevy(vector![
                8.783226102061334e+00,
                -4.247296553985379e+00,
                -2.758526575085964e-01
            ]))
            .vel(helio_to_bevy(vector![
                2.116571054761369e-03,
                5.012229319169293e-03,
                -1.716983622031672e-04,
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 188.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let neptune = builder.entity(
        EntityBuilder::default()
            .mass(1.024e26)
            .pos(helio_to_bevy(vector![
                2.981579078855878e+01,
                -2.121513644468658e+00,
                -6.434483800948522e-01,
            ]))
            .vel(helio_to_bevy(vector![
                2.021264053090830e-04,
                3.149741242740432e-03,
                -6.967831155254172e-05,
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(65.0 / 255.0 * 20., 187.0 / 255.0 * 20., 20.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let uranus = builder.entity(
        EntityBuilder::default()
            .mass(1.024e26)
            .pos(helio_to_bevy(vector![
                1.258255102521142e+01,
                1.505737113095830e+01,
                -1.070858413350803e-01,
            ]))
            .vel(helio_to_bevy(vector![
                -3.046918944187991e-03,
                2.338765156267450e-03,
                4.819887062168329e-05,
            ]))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.05,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(65.0 / 255.0 * 20., 187.0 / 255.0 * 20., 20.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    builder.gravity_constraint(GravityConstraint::new(earth, sun).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(earth, jupiter).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, jupiter).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, mars).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, venus).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, saturn).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, neptune).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, mercury).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, uranus).constant(1.4883e-34));
}

fn helio_to_bevy(vec: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(vec.x, vec.z, vec.y)
}
