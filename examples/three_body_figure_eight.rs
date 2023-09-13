use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::xpbd::{
    builder::{Assets, EntityBuilder, XpbdBuilder},
    constraints::GravityConstraint,
    editor::editor,
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets) {
    let figure_eight_pos = vector![0.97000436, -0.24308753, 0.0];
    let figure_eight_vel = vector![-0.93240737, -0.86473146, 0.0];

    let a = builder.entity(
        EntityBuilder::default()
            .mass(1.0 / 6.649e-11)
            .pos(figure_eight_pos)
            .vel(-0.5 * figure_eight_vel)
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
    let b = builder.entity(
        EntityBuilder::default()
            .mass(1.0 / 6.649e-11)
            .pos(-1.0 * figure_eight_pos)
            .vel(-0.5 * figure_eight_vel)
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.2,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(65.0 / 255.0 * 20., 187.0 / 255.0 * 20., 20.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    let c = builder.entity(
        EntityBuilder::default()
            .mass(1.0 / 6.649e-11)
            .pos(Vector3::zeros())
            .vel(figure_eight_vel)
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.2,
                ..Default::default()
            })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                emissive: Color::rgb_linear(20., 15.0 / 255.0 * 20., 0.),
                base_color: Color::hex("FFB800").unwrap(),
                ..Default::default()
            })),
    );

    builder.gravity_constraint(GravityConstraint::new(a, b));
    builder.gravity_constraint(GravityConstraint::new(a, c));
    builder.gravity_constraint(GravityConstraint::new(b, c));
}
