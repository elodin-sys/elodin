use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::{
    builder::{Assets, EntityBuilder, XpbdBuilder},
    constraints::DistanceConstraint,
    editor::{editor, Input},
    Force, Time,
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets, input: Input) {
    let a = builder.entity(
        EntityBuilder::default()
            .mass(10.0)
            .pos(vector![1.0, 0.0, 0.0])
            .vel(vector![0.0, 0.0, 0.0])
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .effector(move |Time(_)| Force(Vector3::new(*input.0.load() * 100.0, 0.0, 0.0)))
            .material(assets.material(Color::rgb(1.0, 0.0, 0.0).into())),
    );
    let b = builder.entity(
        EntityBuilder::default()
            .fixed()
            .mass(1.0)
            .pos(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .material(assets.material(Color::rgb(0.0, 0.2, 1.0).into())),
    );
    builder.distance_constraint(
        DistanceConstraint::new(a, b)
            .distance_target(1.0)
            .compliance(0.001),
    );
}
