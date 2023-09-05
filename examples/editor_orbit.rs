use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use paracosm::{
    forces::gravity,
    xpbd::{
        builder::{Assets, EntityBuilder, XpbdBuilder},
        editor::{editor, Input},
    },
    Force, Pos,
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder<'_>, mut assets: Assets, input: Input) {
    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(vector![0.0, 0.0, 1.0])
            .vel(vector![1.0, 0.0, 0.0])
            .effector(gravity(1.0 / 6.649e-11, Vector3::zeros()))
            .effector(move |Pos(pos)| Force(*input.0.load() * pos.normalize()))
            .trace(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .material(assets.material(Color::rgb(1.0, 0.0, 0.0).into())),
    );
    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .pos(Vector3::zeros())
            .mesh(assets.mesh(Mesh::from(shape::UVSphere {
                radius: 0.1,
                ..Default::default()
            })))
            .material(assets.material(Color::rgb(0.0, 0.2, 1.0).into())),
    );
}
