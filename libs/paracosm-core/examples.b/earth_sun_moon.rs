use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::{
    builder::{Assets, EntityBuilder, Free, XpbdBuilder},
    constraints::GravityConstraint,
    editor::editor,
    runner::IntoSimRunner,
    spatial::{SpatialMotion, SpatialPos},
};

fn main() {
    editor(sim.substep_count(2).scale(50.))
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets) {
    let sun = builder.entity(
        EntityBuilder::default()
            .mass(1.99e30)
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 1.0,
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
            .joint(
                Free::default()
                    .pos(SpatialPos::linear(vector![
                        9.932794033922092e-1,
                        -8.115895094412964e-2,
                        2.123378844575767e-4,
                    ]))
                    .vel(SpatialMotion::linear(vector![
                        1.072468396698360e-3,
                        1.708345322037895e-2,
                        -3.545157932769615e-7
                    ])),
            )
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

    let moon = builder.entity(
        EntityBuilder::default()
            .mass(7.349e22)
            .joint(
                Free::default()
                    .pos(SpatialPos::linear(vector![
                        9.912496065985930e-01,
                        -8.283334150161940e-02,
                        1.509441471371588e-04
                    ]))
                    .vel(SpatialMotion::linear(vector![
                        1.457499139321606e-03,
                        1.665678883540840e-02,
                        -5.053951154093518e-05
                    ])),
            )
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.01,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 15.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    builder.gravity_constraint(GravityConstraint::new(earth, sun).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(earth, moon).constant(1.4883e-34));
    builder.gravity_constraint(GravityConstraint::new(sun, moon).constant(1.4883e-34));
}
