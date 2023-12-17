use bevy::prelude::{shape, Color, Mesh};
use nalgebra::{vector, Vector3};
use elodin::{
    builder::{Assets, EntityBuilder, Free, XpbdBuilder},
    editor::{editor, Input},
    spatial::SpatialPos,
    Force,
};

fn main() {
    editor(sim)
}

fn sim(mut builder: XpbdBuilder, mut assets: Assets, input: Input) {
    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .joint(Free::default().pos(SpatialPos::linear(vector![0.0, 0.75, 0.0])))
            .effector(move || {
                let torque = *input.0.load();
                Force(Vector3::new(0.0, torque, 0.0))
            })
            .mesh(assets.mesh(Mesh::from(shape::Cube { size: 1.5 })))
            .material(assets.material(bevy::prelude::StandardMaterial {
                base_color: Color::hex("#0085FF").unwrap(),
                metallic: 0.6,
                perceptual_roughness: 0.3,
                ..Default::default()
            })),
    );
}
