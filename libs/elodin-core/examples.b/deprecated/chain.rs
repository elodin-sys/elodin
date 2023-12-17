use bevy::prelude::{shape, Color, Mesh};
use nalgebra::vector;
use elodin::{
    builder::{Assets, EntityBuilder, XpbdBuilder},
    constraints::DistanceConstraint,
    editor::editor,
    Force, Time,
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets) {
    let mut previous_link = builder.entity(
        EntityBuilder::default()
            .mass(10.0)
            .fixed()
            .pos(vector![0.0, 0.0, 0.0])
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .material(assets.material(Color::rgb(1.0, 0.0, 0.0).into())),
    );
    for i in 1..=20 {
        let link = builder.entity(
            EntityBuilder::default()
                .mass(1.0)
                .pos(vector![(i as f64) / 4.0, 0.0, 0.0])
                .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                    radius: 0.1,
                    ..Default::default()
                })))
                .effector(|Time(_)| Force(vector![0.0, -5.0, 0.0]))
                .material(assets.material(Color::rgb(0.0, 0.2, 1.0).into())),
        );
        builder.distance_constraint(
            DistanceConstraint::new(link, previous_link)
                .distance_target(0.25)
                .compliance(0.0),
        );
        previous_link = link
    }
}
